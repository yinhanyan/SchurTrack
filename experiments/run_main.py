"""Unified reproducibility runner for Schur tracking experiments.

Examples
--------
Centralized experiments:

    python -m experiments.run_main --tasks sw,attp

Ray distributed experiments (the default distributed backend):

    python -m experiments.run_main \
        --tasks dist,dsw --distributed-backend ray
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pathlib
import platform
import socket
import sys
import time
from typing import Callable

import numpy as np


PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from algorithm.sw.schur_fd_sw import SchurFdSw
from base.fd import reduce_rows_fd
from base.schur_fd import EnergyAdaptiveSchurATTP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        default="sw,attp,dist,dsw",
        help="comma-separated subset of sw,attp,dist,dsw",
    )
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--dimensions", default="32,64,128")
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        help="optional .npy, .npz, .mat, or GloVe-style text matrix",
    )
    parser.add_argument(
        "--matrix-key",
        default="A",
        help="array key for .npz/.mat input (default: A)",
    )
    parser.add_argument("--ell", type=int, default=8)
    parser.add_argument(
        "--sw-ells",
        help=(
            "optional comma-separated sliding-window ell sweep; "
            "defaults to --ell"
        ),
    )
    parser.add_argument(
        "--schur-query-width-multiplier",
        type=int,
        choices=(2,),
        default=2,
        help=(
            "apply a deterministic final SchurTrack query reduction to "
            "multiplier * ell rows for sliding windows; ATTP always uses "
            "the formal 2 * ell output contract"
        ),
    )
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--query-step", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="run only SchurTrack methods (useful for remote smoke tests)",
    )
    parser.add_argument(
        "--distributed-backend",
        choices=("ray",),
        default="ray",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=PROJECT / "results" / "schur",
    )
    return parser.parse_args()


def validate_rows(rows: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("input must be a non-empty two-dimensional matrix")
    norm_sq = np.sum(matrix * matrix, axis=1)
    if (
        not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(norm_sq))
        or np.any(norm_sq <= 0.0)
    ):
        raise ValueError("every input row must have a finite nonzero norm")
    return matrix


def make_rows(seed: int, count: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return validate_rows(rng.normal(size=(count, d)))


def scale_rows_for_sliding_window(
    rows: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Apply the legacy Aero scaling min ||a_i||^2 = 1."""

    matrix = validate_rows(rows)
    norm_sq = np.sum(matrix * matrix, axis=1)
    minimum = float(np.min(norm_sq))
    scaled = matrix / np.sqrt(minimum)
    scaled_norm_sq = np.sum(scaled * scaled, axis=1)
    norm_sq_upper = max(1.0, float(np.max(scaled_norm_sq)))
    return scaled, norm_sq_upper


def sliding_window_ells(args: argparse.Namespace) -> list[int]:
    if args.sw_ells is None:
        values = [args.ell]
    else:
        values = [
            int(value.strip())
            for value in args.sw_ells.split(",")
            if value.strip()
        ]
    if not values or any(value < 1 for value in values):
        raise ValueError("sliding-window ell values must be positive")
    if len(set(values)) != len(values):
        raise ValueError("sliding-window ell values must be unique")
    return values


def load_input_rows(
    path: pathlib.Path, matrix_key: str, limit: int
) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = matrix_key if matrix_key in archive else archive.files[0]
            matrix = archive[key]
    elif suffix == ".mat":
        from scipy.io import loadmat

        payload = loadmat(path)
        if matrix_key not in payload:
            raise KeyError(f"{matrix_key!r} is not present in {path}")
        matrix = payload[matrix_key]
    else:
        parsed = []
        expected_columns = None
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                fields = line.strip().split()
                if not fields:
                    continue
                if expected_columns is None:
                    try:
                        vector = [float(value) for value in fields]
                    except ValueError:
                        vector = [float(value) for value in fields[1:]]
                    expected_columns = len(vector)
                else:
                    if len(fields) == expected_columns:
                        vector_fields = fields
                    elif len(fields) == expected_columns + 1:
                        # A label can itself look numeric (common in GloVe),
                        # so column count, not float conversion, identifies it.
                        vector_fields = fields[1:]
                    else:
                        raise ValueError(
                            f"line {line_number} of {path} has "
                            f"{len(fields)} fields; expected "
                            f"{expected_columns} values with an optional label"
                        )
                    try:
                        vector = [float(value) for value in vector_fields]
                    except ValueError as error:
                        raise ValueError(
                            f"line {line_number} of {path} contains a "
                            "non-numeric matrix value"
                        ) from error
                parsed.append(vector)
                if len(parsed) >= limit:
                    break
        matrix = np.asarray(parsed, dtype=np.float64)
    return validate_rows(np.asarray(matrix)[:limit])


def covariance_error_from_covariance(
    exact_covariance: np.ndarray, sketch: np.ndarray
) -> float:
    if sketch.shape[0]:
        difference = exact_covariance - sketch.T @ sketch
    else:
        difference = exact_covariance
    symmetric = 0.5 * (difference + difference.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return float(np.max(np.abs(eigenvalues), initial=0.0))


def schur_query_method_name(args: argparse.Namespace) -> str:
    multiplier = args.schur_query_width_multiplier
    return f"SchurTrack-{multiplier}ell-query" if multiplier else "SchurTrack"


def latency_metrics(
    update_latencies_ns: list[int],
    query_latencies_ns: list[int],
) -> dict:
    update = np.asarray(update_latencies_ns, dtype=np.float64) / 1e3
    query = np.asarray(query_latencies_ns, dtype=np.float64) / 1e3
    return {
        "update_us": float(np.median(update)) if update.size else 0.0,
        "update_mean_us": float(np.mean(update)) if update.size else 0.0,
        "update_p95_us": (
            float(np.percentile(update, 95)) if update.size else 0.0
        ),
        "update_p99_us": (
            float(np.percentile(update, 99)) if update.size else 0.0
        ),
        "query_us": float(np.median(query)) if query.size else 0.0,
    }


def error_metrics(errors: list[float]) -> dict:
    return {
        "relative_error_max": max(errors, default=0.0),
        "relative_error_mean": (
            float(np.mean(errors)) if errors else 0.0
        ),
    }


def benchmark_sw_method(
    args: argparse.Namespace,
    d: int,
    rows: np.ndarray,
    ell: int,
    norm_sq_upper: float,
    method: str,
    update: Callable[[np.ndarray], None],
    query: Callable[[], np.ndarray],
    persistent_words: Callable[[], int],
    counters: Callable[[], dict],
) -> dict:
    update_latencies = []
    query_latencies = []
    errors = []
    maximum_size_words = 0
    exact_covariance = np.zeros((d, d), dtype=np.float64)
    exact_energy = 0.0
    for time_value, row in enumerate(rows, 1):
        start = time.process_time_ns()
        update(row)
        update_latencies.append(time.process_time_ns() - start)
        maximum_size_words = max(
            maximum_size_words, int(persistent_words())
        )
        exact_covariance += np.outer(row, row)
        exact_energy += float(row @ row)
        if time_value > args.window:
            expired = rows[time_value - args.window - 1]
            exact_covariance -= np.outer(expired, expired)
            exact_energy -= float(expired @ expired)
        if time_value % args.query_step:
            continue
        start = time.process_time_ns()
        output = query()
        query_latencies.append(time.process_time_ns() - start)
        errors.append(
            covariance_error_from_covariance(exact_covariance, output)
            / exact_energy
        )
    return {
        "task": "sw",
        "method": method,
        "d": d,
        "ell": ell,
        "epsilon_nominal": 2.0 / ell,
        "rows": len(rows),
        "window_size": args.window,
        "query_step": args.query_step,
        "norm_sq_upper": norm_sq_upper,
        "timing_clock": "process_time_ns",
        "relative_error_denominator": "window_frobenius_sq",
        **latency_metrics(update_latencies, query_latencies),
        **error_metrics(errors),
        "max_size_kb": maximum_size_words * 8.0 / 1024.0,
        "communication_floats": 0,
        **counters(),
    }


def run_sw(
    args: argparse.Namespace,
    d: int,
    rows: np.ndarray,
    norm_sq_upper: float,
) -> list[dict]:
    results = []

    for ell in sliding_window_ells(args):
        schur = SchurFdSw(args.window, norm_sq_upper, d, ell)
        schur_query_svd_calls = [0]

        def query_schur() -> np.ndarray:
            output = schur.get()[0]
            multiplier = args.schur_query_width_multiplier
            if not multiplier:
                return output
            query_width = min(d, multiplier * ell)
            if output.shape[0] > query_width:
                schur_query_svd_calls[0] += 1
            return reduce_rows_fd(output, query_width)

        results.append(
            benchmark_sw_method(
                args,
                d,
                rows,
                ell,
                norm_sq_upper,
                schur_query_method_name(args),
                schur.update,
                query_schur,
                schur.size,
                lambda: {
                    "svd_calls": schur.aggregate_stats().svd_calls,
                    "query_svd_calls": schur_query_svd_calls[0],
                    "power_calls": 0,
                    "simultaneous_calls": 0,
                    "query_reducer_width": (
                        min(d, args.schur_query_width_multiplier * ell)
                        if args.schur_query_width_multiplier else 0
                    ),
                },
            )
        )
        if args.skip_baselines:
            continue

        from algorithm.sw.sifd_sw import SiFdSw

        np.random.seed(args.seed + d + ell + 1_000)
        aero = SiFdSw(
            args.window, norm_sq_upper, d, ell, beta=1.0
        )
        results.append(
            benchmark_sw_method(
                args,
                d,
                rows,
                ell,
                norm_sq_upper,
                "AeroSketch",
                aero.fit,
                lambda: aero.get()[0],
                lambda: sum(
                    level.size() for level in aero.levels.values()
                ),
                lambda: {
                    "svd_calls": sum(
                        level.C.svd_calls + level.query_svd_calls
                        for level in aero.levels.values()
                    ),
                    "power_calls": sum(
                        level.power_calls for level in aero.levels.values()
                    ),
                    "simultaneous_calls": sum(
                        level.simultaneous_calls
                        for level in aero.levels.values()
                    ),
                },
            )
        )

        from algorithm.sw.fast_ds_fd import FastDsFd

        ds_query_svd_calls = [0]
        ds_fd = FastDsFd(
            args.window, norm_sq_upper, d, ell, beta=1.0
        )

        def query_ds_fd() -> np.ndarray:
            selected = 0
            for level_index in range(ds_fd.logR):
                queue = ds_fd.levels[level_index].fd.queue
                if not queue:
                    break
                head = queue[0]
                if ds_fd.time - head.s >= min(
                    ds_fd.N - 1, ds_fd.time - 1
                ):
                    selected = level_index
                    break
            if ds_fd.levels[selected].fd.queue:
                ds_query_svd_calls[0] += 1
            return ds_fd.get()[0]

        results.append(
            benchmark_sw_method(
                args,
                d,
                rows,
                ell,
                norm_sq_upper,
                "Fast-DS-FD",
                lambda row: ds_fd.fit(row.reshape(1, -1)),
                query_ds_fd,
                lambda: ds_fd.get_size() * d,
                lambda: {
                    "svd_calls": ds_query_svd_calls[0],
                    "power_calls": 0,
                    "simultaneous_calls": 0,
                },
            )
        )
    return results


def benchmark_attp_method(
    args: argparse.Namespace,
    d: int,
    rows: np.ndarray,
    method: str,
    update: Callable[[np.ndarray], None],
    query: Callable[[int], np.ndarray],
    persistent_words: Callable[[], int],
    counters: Callable[[], dict],
) -> dict:
    update_latencies = []
    query_latencies = []
    errors = []
    maximum_size_words = 0
    exact_covariance = np.zeros((d, d), dtype=np.float64)
    frobenius_sq = 0.0
    for time_value, row in enumerate(rows, 1):
        start = time.process_time_ns()
        update(row)
        update_latencies.append(time.process_time_ns() - start)
        maximum_size_words = max(
            maximum_size_words, int(persistent_words())
        )
        exact_covariance += np.outer(row, row)
        frobenius_sq += float(row @ row)
        if time_value % args.query_step:
            continue
        start = time.process_time_ns()
        output = query(time_value)
        query_latencies.append(time.process_time_ns() - start)
        errors.append(
            covariance_error_from_covariance(exact_covariance, output)
            / frobenius_sq
        )
    return {
        "task": "attp",
        "method": method,
        "d": d,
        "ell": args.ell,
        "rows": len(rows),
        "timing_clock": "process_time_ns",
        **latency_metrics(update_latencies, query_latencies),
        **error_metrics(errors),
        "max_size_kb": maximum_size_words * 8.0 / 1024.0,
        "communication_floats": 0,
        **counters(),
    }


def run_attp(
    args: argparse.Namespace, d: int, rows: np.ndarray
) -> list[dict]:
    results = []

    schur = EnergyAdaptiveSchurATTP(
        d, args.ell, retain_sealed_residual=False
    )
    schur_query_svd_calls = [0]

    def query_schur(query_time: int) -> np.ndarray:
        output = schur.get(query_time)
        query_width = min(d, 2 * args.ell)
        if output.shape[0] > query_width:
            schur_query_svd_calls[0] += 1
        return reduce_rows_fd(output, query_width)

    results.append(
        benchmark_attp_method(
            args,
            d,
            rows,
            "SchurTrack-2ell-query",
            schur.update,
            query_schur,
            schur.size,
            lambda: {
                "svd_calls": schur.aggregate_stats().svd_calls,
                "query_svd_calls": schur_query_svd_calls[0],
                "power_calls": 0,
                "simultaneous_calls": 0,
                "energy_epochs": schur.epoch_count,
                "query_reducer_width": min(d, 2 * args.ell),
            },
        )
    )
    if args.skip_baselines:
        return results

    from algorithm.attp.fd_attp import FdAttp
    from algorithm.attp.sifd_attp import SiFdAttp

    np.random.seed(args.seed + d + 2_000)
    aero = SiFdAttp(d, args.ell)
    results.append(
        benchmark_attp_method(
            args,
            d,
            rows,
            "AeroSketch",
            aero.fit,
            lambda _: aero.get(),
            aero.size,
            lambda: {
                "svd_calls": aero.C.svd_calls + aero.query_svd_calls,
                "power_calls": aero.power_calls,
                "simultaneous_calls": aero.simultaneous_calls,
                "power_iterations": aero.power_iterations,
                "simultaneous_iterations": aero.simultaneous_iterations,
            },
        )
    )

    for method, fast_fd in (("PFD", False),):
        baseline = FdAttp(d, args.ell, fast_fd=fast_fd)
        results.append(
            benchmark_attp_method(
                args,
                d,
                rows,
                method,
                baseline.fit,
                lambda _, baseline=baseline: baseline.get(),
                baseline.size,
                lambda baseline=baseline: {
                    "svd_calls": (
                        baseline.C.svd_calls + baseline.B.svd_calls
                    ),
                    "power_calls": 0,
                    "simultaneous_calls": 0,
                },
            )
        )
    return results


def distributed_classes(backend: str):
    if importlib.util.find_spec("ray") is None:
        raise RuntimeError(
            "Ray backend requested but ray is not installed; install the "
            "distributed requirements"
        )
    from algorithm.distributed.ray_schur_fd import RaySchurFdDist
    from algorithm.dsw.ray_schur_fd_dsw import RaySchurFdDsw

    return RaySchurFdDist, RaySchurFdDsw


def distributed_row_counts(stats: dict) -> tuple[int, int, int]:
    coordinator_rows = int(stats["coordinator"]["row_num"])
    site_rows = [int(site["row_num"]) for site in stats["sites"]]
    return (
        coordinator_rows,
        max(site_rows, default=0),
        coordinator_rows + sum(site_rows),
    )


def benchmark_distributed_method(
    args: argparse.Namespace,
    d: int,
    rows: np.ndarray,
    assignments: np.ndarray,
    task: str,
    method: str,
    factory: Callable[[], object],
) -> dict:
    sketch = factory()
    update_latencies = []
    query_latencies = []
    errors = []
    maximum_stored_rows = 0
    exact_covariance = np.zeros((d, d), dtype=np.float64)
    exact_energy = 0.0
    total_rows_to_process = len(rows)
    progress_interval = max(1, total_rows_to_process // 20)
    benchmark_start = time.perf_counter()
    print(
        f"[{task}] {method}: 0/{total_rows_to_process} (0.0%)",
        flush=True,
    )
    try:
        for time_value, (row, site_id) in enumerate(
            zip(rows, assignments), 1
        ):
            start = time.perf_counter_ns()
            sketch.fit(row, int(site_id))
            update_latencies.append(time.perf_counter_ns() - start)
            exact_covariance += np.outer(row, row)
            exact_energy += float(row @ row)
            if task == "dsw" and time_value > args.window:
                expired = rows[time_value - args.window - 1]
                exact_covariance -= np.outer(expired, expired)
                exact_energy -= float(expired @ expired)
            if (
                time_value % progress_interval == 0
                or time_value == total_rows_to_process
            ):
                elapsed = time.perf_counter() - benchmark_start
                rate = time_value / elapsed if elapsed > 0.0 else 0.0
                remaining = total_rows_to_process - time_value
                eta = remaining / rate if rate > 0.0 else float("inf")
                print(
                    f"[{task}] {method}: {time_value}/"
                    f"{total_rows_to_process} "
                    f"({100.0 * time_value / total_rows_to_process:.1f}%) "
                    f"elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                    flush=True,
                )
            if task == "dsw" and time_value < args.window:
                continue
            if time_value % args.query_step:
                continue
            if hasattr(sketch, "query_with_stats"):
                output, query_stats, query_time_ns = sketch.query_with_stats()
                query_latencies.append(query_time_ns)
            else:
                start = time.perf_counter_ns()
                output = sketch.get()
                query_latencies.append(time.perf_counter_ns() - start)
                query_stats = sketch.get_stats()
            errors.append(
                covariance_error_from_covariance(exact_covariance, output)
                / exact_energy
            )
            _, _, total_rows = distributed_row_counts(query_stats)
            maximum_stored_rows = max(
                maximum_stored_rows, total_rows
            )
        stats = sketch.get_stats()
        coordinator_rows, _, total_rows = distributed_row_counts(stats)
        maximum_stored_rows = max(maximum_stored_rows, total_rows)
    finally:
        sketch.shutdown()
    site_stats = stats["sites"]
    svd_calls = int(stats["coordinator"].get("svd_calls", 0)) + sum(
        int(site.get("svd_calls", 0)) for site in site_stats
    )
    power_calls = sum(
        int(site.get("power_calls", 0)) for site in site_stats
    )
    simultaneous_calls = sum(
        int(site.get("simultaneous_calls", 0)) for site in site_stats
    )
    return {
        "task": task,
        "method": method,
        "d": d,
        "ell": args.ell,
        "rows": len(rows),
        **latency_metrics(update_latencies, query_latencies),
        **error_metrics(errors),
        "stored_rows": maximum_stored_rows,
        "svd_calls": svd_calls,
        "power_calls": power_calls,
        "simultaneous_calls": simultaneous_calls,
        "communication_floats": stats["coordinator"]["communication_cost"],
        "coordinator_rows": coordinator_rows,
        "coordinator_processing_ms": float(
            stats["coordinator"].get("processing_time", 0.0)
        ),
        "site_processing_ms": sum(
            float(site.get("processing_time", 0.0)) for site in site_stats
        ),
        "actor_processing_ms": (
            float(stats["coordinator"].get("processing_time", 0.0))
            + sum(
                float(site.get("processing_time", 0.0))
                for site in site_stats
            )
        ),
        "actor_processing_ns": (
            int(stats["coordinator"].get("processing_time_ns", 0))
            + sum(
                int(site.get("processing_time_ns", 0))
                for site in site_stats
            )
        ),
        "actor_nodes": json.dumps(
            {
                "coordinator": stats["coordinator"].get(
                    "node_id", "local"
                ),
                "sites": [
                    site.get("node_id", "local") for site in site_stats
                ],
            }
        ),
    }


def run_dist(
    args: argparse.Namespace,
    d: int,
    rows: np.ndarray,
    assignments: np.ndarray,
) -> list[dict]:
    Dist, _ = distributed_classes(args.distributed_backend)
    results = [
        benchmark_distributed_method(
            args,
            d,
            rows,
            assignments,
            "dist",
            f"SchurTrack-{args.distributed_backend}",
            lambda: Dist(d, args.ell, args.sites),
        )
    ]
    if args.skip_baselines or args.distributed_backend != "ray":
        return results
    from algorithm.distributed.ray_aero_fd import RayAeroFdDist
    from algorithm.distributed.ray_p2_fd import RayP2FdDist

    results.append(
        benchmark_distributed_method(
            args,
            d,
            rows,
            assignments,
            "dist",
            "P2-ray",
            lambda: RayP2FdDist(
                d, args.ell, args.sites, seed=args.seed + d + 3_500
            ),
        )
    )
    results.append(
        benchmark_distributed_method(
            args,
            d,
            rows,
            assignments,
            "dist",
            "AeroSketch-ray",
            lambda: RayAeroFdDist(
                d, args.ell, args.sites, seed=args.seed + d + 3_000
            ),
        )
    )
    return results


def run_dsw(
    args: argparse.Namespace,
    d: int,
    rows: np.ndarray,
    assignments: np.ndarray,
) -> list[dict]:
    _, Dsw = distributed_classes(args.distributed_backend)
    results = [
        benchmark_distributed_method(
            args,
            d,
            rows,
            assignments,
            "dsw",
            f"SchurTrack-raw-epoch-{args.distributed_backend}",
            lambda: Dsw(
                d,
                args.ell,
                args.window,
                args.sites,
                **(
                    {"seed": args.seed + d + 4_000}
                    if args.distributed_backend == "ray"
                    else {}
                ),
            ),
        )
    ]
    if args.skip_baselines or args.distributed_backend != "ray":
        return results
    from algorithm.dsw.ray_aero_fd_dsw import RayAeroFdDsw

    results.append(
        benchmark_distributed_method(
            args,
            d,
            rows,
            assignments,
            "dsw",
            "AeroSketch-raw-epoch-ray",
            lambda: RayAeroFdDsw(
                d,
                args.ell,
                args.window,
                args.sites,
                seed=args.seed + d + 5_000,
            ),
        )
    )
    from algorithm.dsw.ray_da2_baseline import RayDa2Dsw

    results.append(
        benchmark_distributed_method(
            args,
            d,
            rows,
            assignments,
            "dsw",
            "DA2-raw-epoch-ray",
            lambda: RayDa2Dsw(
                d,
                args.ell,
                args.window,
                args.sites,
                seed=args.seed + d + 6_000,
            ),
        )
    )
    return results


def write_metadata(args: argparse.Namespace, path: pathlib.Path) -> None:
    metadata = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "distributed_backend": args.distributed_backend,
        "sites": args.sites,
        "seed": args.seed,
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else []
        ),
        "ray_num_cpus": os.environ.get("SCHUR_RAY_NUM_CPUS", ""),
        "blas_threads": {
            name: os.environ.get(name, "")
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }
    try:
        import scipy

        metadata["scipy"] = scipy.__version__
    except ImportError:
        metadata["scipy"] = "unavailable"
    if importlib.util.find_spec("ray") is not None:
        import ray

        metadata["ray"] = ray.__version__
    else:
        metadata["ray"] = "unavailable"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = {task.strip() for task in args.tasks.split(",") if task.strip()}
    dimensions = [int(value) for value in args.dimensions.split(",")]
    results: list[dict] = []

    if args.input is None:
        datasets = [
            (
                f"synthetic-d={dimension}",
                make_rows(args.seed + dimension, args.rows, dimension),
            )
            for dimension in dimensions
        ]
    else:
        input_rows = load_input_rows(
            args.input, args.matrix_key, args.rows
        )
        datasets = [(args.input.stem, input_rows)]

    owns_ray = False
    if args.distributed_backend == "ray" and tasks & {"dist", "dsw"}:
        from algorithm.distributed.ray_runtime import initialize_ray

        owns_ray = initialize_ray()
    try:
        for dataset_name, raw_rows in datasets:
            dimension = int(raw_rows.shape[1])
            sw_rows, norm_sq_upper = scale_rows_for_sliding_window(raw_rows)
            # Match the legacy distributed-SWFD experiments, which reset
            # NumPy's RandomState to the configured seed for every dataset.
            assignments = np.random.RandomState(args.seed).randint(
                0, args.sites, size=len(raw_rows)
            )
            first_result = len(results)
            if "sw" in tasks:
                results.extend(
                    run_sw(args, dimension, sw_rows, norm_sq_upper)
                )
            if "attp" in tasks:
                results.extend(run_attp(args, dimension, raw_rows))
            if "dist" in tasks:
                results.extend(
                    run_dist(
                        args,
                        dimension,
                        sw_rows,
                        assignments,
                    )
                )
            if "dsw" in tasks:
                results.extend(
                    run_dsw(
                        args,
                        dimension,
                        sw_rows,
                        assignments,
                    )
                )
            for result in results[first_result:]:
                result["dataset"] = dataset_name
    finally:
        if owns_ray:
            import ray

            ray.shutdown()

    csv_path = args.output_dir / "results.csv"
    fields = sorted({key for result in results for key in result})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_metadata(args, args.output_dir / "metadata.json")
    print(json.dumps(results, indent=2))
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()

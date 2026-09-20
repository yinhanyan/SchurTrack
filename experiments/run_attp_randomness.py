"""Reproduce the fixed-width Random Noisy ATTP 100-seed study."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import pathlib
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.linalg


PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from algorithm.attp.instrumented_aero_attp import InstrumentedAeroAttp
from base.fd import reduce_covariance_fd, reduce_rows_fd
from base.schur_fd import EnergyAdaptiveSchurATTP
from experiments.run_main import load_input_rows


ROWS: np.ndarray
CALIBRATED_UPDATE_MS: dict[str, float]
SCHUR_UPDATE_MS: float
D = 500
ELL = 64


def initialize_worker() -> None:
    allowed = tuple(sorted(os.sched_getaffinity(0)))
    identity = multiprocessing.current_process()._identity
    index = (identity[0] - 1) % len(allowed) if identity else 0
    os.sched_setaffinity(0, {allowed[index]})



def timed_updates(builder, rows: np.ndarray) -> float:
    sketch = builder()
    start = time.process_time_ns()
    for row in rows:
        sketch.fit(row)
    return (time.process_time_ns() - start) / len(rows) / 1e6


def calibrate(rows: np.ndarray, qs: list[int], repeats: int, seed: int) -> dict:
    warmup = rows[: min(32, len(rows))]
    timed_updates(
        lambda: EnergyAdaptiveSchurATTP(
            D, ELL, retain_sealed_residual=False
        ),
        warmup,
    )
    timed_updates(
        lambda: InstrumentedAeroAttp(
            D, ELL, power_iterations=qs[0],
            simultaneous_iterations=qs[0], seed=seed
        ),
        warmup,
    )
    samples = {"SchurTrack": []}
    samples.update({f"q={q}": [] for q in qs})
    order_rng = np.random.default_rng(seed + 9173)
    entries = [None, *qs]
    for repeat in range(repeats):
        order = list(entries)
        order_rng.shuffle(order)
        for q in order:
            if q is None:
                builder = lambda: EnergyAdaptiveSchurATTP(
                    D, ELL, retain_sealed_residual=False
                )
                key = "SchurTrack"
            else:
                builder = lambda q=q, repeat=repeat: InstrumentedAeroAttp(
                    D, ELL, power_iterations=q,
                    simultaneous_iterations=q, seed=seed + repeat
                )
                key = f"q={q}"
            samples[key].append(timed_updates(builder, rows))
    calibrated = {
        key: float(np.median(values)) for key, values in samples.items()
    }
    return {
        "clock": "process_time_ns",
        "query_and_error_costs_excluded": True,
        "repeats": repeats,
        "samples_ms": samples,
        "calibrated_update_ms": calibrated,
    }


def spectral_error(exact: np.ndarray, approx: np.ndarray, energy: float) -> float:
    difference = exact - approx
    difference = 0.5 * (difference + difference.T)
    eigenvalues = scipy.linalg.eigvalsh(difference, check_finite=False)
    return float(np.max(np.abs(eigenvalues)) / energy)


def run_aero(job: tuple[int, int]) -> dict:
    q, seed = job
    sketch = InstrumentedAeroAttp(
        D,
        ELL,
        power_iterations=q,
        simultaneous_iterations=q,
        seed=seed,
        mode="randomized",
    )
    exact = np.zeros((D, D), dtype=np.float64)
    ledger = np.zeros((D, D), dtype=np.float64)
    energy = 0.0
    error_sum = 0.0
    maximum_error = 0.0
    seen = 0
    for row in ROWS:
        exact += np.outer(row, row)
        energy += float(row @ row)
        sketch.fit(row)
        while seen < len(sketch.Zs):
            z = sketch.Zs[seen]
            zcc = sketch.ZCCs[seen]
            product = z @ zcc
            ledger += product + product.T - z @ (zcc @ z) @ z.T
            seen += 1
        query_rows = reduce_covariance_fd(
            ledger, min(D, 2 * ELL)
        )
        error = spectral_error(exact, query_rows.T @ query_rows, energy)
        error_sum += error
        maximum_error = max(maximum_error, error)

    update_ms = float(CALIBRATED_UPDATE_MS[f"q={q}"])
    return {
        "method": "AeroSketch-random-noisy-quality",
        "mode": "randomized",
        "ell": ELL,
        "seed": seed,
        "power_iterations": q,
        "simultaneous_iterations": q,
        "calibrated_update_ms": update_ms,
        "mean_error": error_sum / len(ROWS),
        "max_error": maximum_error,
        "time_ratio_to_schur": update_ms / SCHUR_UPDATE_MS,
        "power_calls": sketch.power_calls,
        "simultaneous_calls": sketch.simultaneous_calls,
        "cancellation_blocks": len(sketch.Zs),
        "query_reducer_width": min(D, 2 * ELL),
    }


def run_schur() -> dict:
    sketch = EnergyAdaptiveSchurATTP(
        D, ELL, retain_sealed_residual=False
    )
    exact = np.zeros((D, D), dtype=np.float64)
    energy = 0.0
    errors = []
    cancellation_blocks = 0
    query_width = min(D, 2 * ELL)
    for row in ROWS:
        exact += np.outer(row, row)
        energy += float(row @ row)
        cancellation_blocks += len(sketch.fit(row))
        query_rows = reduce_rows_fd(sketch.get(), query_width)
        errors.append(
            spectral_error(exact, query_rows.T @ query_rows, energy)
        )

    return {
        "method": "SchurTrack-no-seal",
        "mode": "deterministic",
        "ell": ELL,
        "seed": -1,
        "power_iterations": 0,
        "simultaneous_iterations": 0,
        "calibrated_update_ms": SCHUR_UPDATE_MS,
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "time_ratio_to_schur": 1.0,
        "power_calls": 0,
        "simultaneous_calls": 0,
        "cancellation_blocks": cancellation_blocks,
        "query_reducer_width": query_width,
    }


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: pathlib.Path, records: list[dict]) -> None:
    fields: list[str] = []
    for record in records:
        for field in record:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--matrix-key", default="A")
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=130000)
    parser.add_argument("--qs", default="1,2,4,8,10")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timing-json", type=pathlib.Path)
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="write single-process update timing and stop",
    )
    args = parser.parse_args()
    qs = sorted({int(value) for value in args.qs.split(",") if value})
    if (
        args.rows < 1 or args.seeds < 1 or args.workers < 1
        or args.calibration_repeats < 1
    ):
        raise ValueError(
            "rows, seeds, workers, and calibration repeats must be positive"
        )
    if not qs or any(q < 1 for q in qs):
        raise ValueError("all q values must be positive")

    args.result_dir.mkdir(parents=True, exist_ok=True)
    global ROWS, CALIBRATED_UPDATE_MS, SCHUR_UPDATE_MS
    ROWS = load_input_rows(args.dataset, args.matrix_key, args.rows)
    timing_path = (
        args.result_dir / "timing.json"
        if args.timing_json is None else args.timing_json
    )
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    else:
        print("Calibrating query-free update time on one process", flush=True)
        timing = calibrate(
            ROWS, qs, args.calibration_repeats, args.seed_start
        )
        timing_path.write_text(
            json.dumps(timing, indent=2), encoding="utf-8"
        )
    calibrated = {
        key: float(value)
        for key, value in timing["calibrated_update_ms"].items()
    }
    if ROWS.shape != (args.rows, D):
        raise ValueError(f"expected rows with shape {(args.rows, D)}")
    CALIBRATED_UPDATE_MS = calibrated
    SCHUR_UPDATE_MS = calibrated["SchurTrack"]
    if args.calibrate_only:
        print(timing_path.resolve(), flush=True)
        return

    checkpoint = args.result_dir / "quality.jsonl"
    records = read_jsonl(checkpoint)
    query_width = min(D, 2 * ELL)
    if records and any(
        int(record.get("query_reducer_width", -1)) != query_width
        for record in records
    ):
        raise ValueError(
            "existing checkpoint does not use the 2ell query contract"
        )
    if not any(record["mode"] == "deterministic" for record in records):
        schur = run_schur()
        records.insert(0, schur)
        with checkpoint.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(schur) + "\n")
    completed = {
        (int(record["power_iterations"]), int(record["seed"]))
        for record in records
        if record["mode"] == "randomized"
    }
    jobs = [
        (q, seed)
        for q in qs
        for seed in range(args.seed_start, args.seed_start + args.seeds)
        if (q, seed) not in completed
    ]
    print(
        f"existing={len(records)}, missing={len(jobs)}, "
        f"workers={args.workers}, affinity={sorted(os.sched_getaffinity(0))}",
        flush=True,
    )

    start = time.perf_counter()
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=initialize_worker,
    ) as executor:
        futures = [executor.submit(run_aero, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            with checkpoint.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            if done % max(1, args.workers) == 0 or done == len(jobs):
                elapsed = time.perf_counter() - start
                rate = done / elapsed
                eta = (len(jobs) - done) / rate if rate else 0.0
                counts = Counter(
                    int(row["power_iterations"])
                    for row in records
                    if row["mode"] == "randomized"
                )
                print(
                    f"completed={done}/{len(jobs)} counts={dict(counts)} "
                    f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                    flush=True,
                )

    records.sort(
        key=lambda record: (
            0 if record["mode"] == "deterministic" else 1,
            int(record["power_iterations"]),
            int(record["seed"]),
        )
    )
    requested_seeds = set(
        range(args.seed_start, args.seed_start + args.seeds)
    )
    expected = {q: args.seeds for q in qs}
    counts = Counter(
        int(record["power_iterations"])
        for record in records
        if (
            record["mode"] == "randomized"
            and int(record["power_iterations"]) in qs
            and int(record["seed"]) in requested_seeds
        )
    )
    if any(counts[q] != count for q, count in expected.items()):
        raise RuntimeError(f"incomplete seed counts: {dict(counts)}")

    (args.result_dir / "quality.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    write_csv(args.result_dir / "quality.csv", records)
    metadata = {
        "protocol": "attp-random-noisy-ell64-gap1-q-sweep-seeds100-v2",
        "query_output_contract": "FD-min(d,2ell)",
        "query_reducer_width": min(D, 2 * ELL),
        "timing_source": str(timing_path),
        "calibration_repeats": int(timing["repeats"]),
        "dataset": str(args.dataset),
        "matrix_key": args.matrix_key,
        "d": D,
        "ell": ELL,
        "stream_rows": len(ROWS),
        "query_gap": 1,
        "query_count": len(ROWS),
        "qs": qs,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.seeds - 1,
        "seeds_per_config": args.seeds,
        "workers": args.workers,
        "worker_affinity": sorted(os.sched_getaffinity(0)),
        "calibrated_update_ms": calibrated,
        "timing_excludes_queries_and_exact_error": True,
        "paired_seed_reuse_across_configs": True,
    }
    (args.result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(args.result_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()

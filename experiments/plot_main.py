"""Generate the sixteen paper panels from freshly rerun JSON results."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


METHOD_ALIASES = {
    "SchurTrack-2ell-query": "SchurTrack",
    "SchurTrack-ray": "SchurTrack",
    "SchurTrack-raw-epoch-ray": "SchurTrack",
    "AeroSketch": "AeroSketch",
    "AeroSketch-ray": "AeroSketch",
    "AeroSketch-raw-epoch-ray": "AeroSketch",
    "Fast-DS-FD": "Fast-DS-FD",
    "PFD": "PFD",
    "P2-ray": "P2",
    "DA2-raw-epoch-ray": "DA2",
}
METHOD_ORDER = {
    "sw": ("Fast-DS-FD", "AeroSketch", "SchurTrack"),
    "attp": ("PFD", "AeroSketch", "SchurTrack"),
    "dist": ("P2", "AeroSketch", "SchurTrack"),
    "dsw": ("DA2", "AeroSketch", "SchurTrack"),
}
STYLES = {
    "Fast-DS-FD": ("^", "#00BFBF"),
    "PFD": ("^", "#00BFBF"),
    "P2": ("^", "#00BFBF"),
    "DA2": ("^", "#00BFBF"),
    "AeroSketch": ("p", "#BF00BF"),
    "SchurTrack": ("D", "#E69F00"),
}


def load_series(root: pathlib.Path, task: str) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    paths = sorted((root / task).glob("ell-*/results.json"))
    if not paths:
        raise FileNotFoundError(f"no results under {root / task}")
    for path in paths:
        for record in json.loads(path.read_text(encoding="utf-8")):
            if record.get("task") != task:
                continue
            alias = METHOD_ALIASES.get(str(record.get("method")))
            if alias is not None:
                grouped[alias].append(record)
    missing = set(METHOD_ORDER[task]).difference(grouped)
    if missing:
        raise ValueError(f"{task} results miss {sorted(missing)}")

    series = {}
    for method in METHOD_ORDER[task]:
        records = sorted(grouped[method], key=lambda row: int(row["ell"]))
        series[method] = {
            "ell": np.asarray([int(row["ell"]) for row in records]),
            "avg_error": np.asarray(
                [float(row["relative_error_mean"]) for row in records]
            ),
            "max_error": np.asarray(
                [float(row["relative_error_max"]) for row in records]
            ),
            "update_ms": np.asarray(
                [float(row["update_mean_us"]) / 1000.0 for row in records]
            ),
            "cost_kb": np.asarray(
                [
                    (
                        float(row["communication_floats"]) * 8.0 / 1024.0
                        if task in {"dist", "dsw"}
                        else float(row["max_size_kb"])
                    )
                    for row in records
                ]
            ),
        }
    return series


def interpolated_ratio(
    series: dict[str, dict[str, np.ndarray]],
    x_key: str,
    y_key: str,
    numerator: str,
    denominator: str,
) -> tuple[np.ndarray, np.ndarray]:
    def ordered(method: str) -> tuple[np.ndarray, np.ndarray]:
        x = series[method][x_key]
        y = series[method][y_key]
        finite = np.isfinite(x) & np.isfinite(y)
        order = np.argsort(x[finite])
        return x[finite][order], y[finite][order]

    x_num, y_num = ordered(numerator)
    x_den, y_den = ordered(denominator)
    lower = max(float(x_num.min()), float(x_den.min()))
    upper = min(float(x_num.max()), float(x_den.max()))
    if lower > upper:
        raise ValueError("AeroSketch and SchurTrack have no error overlap")
    x = (
        np.asarray([lower], dtype=np.float64)
        if lower == upper
        else np.linspace(lower, upper, 100)
    )
    return x, np.interp(x, x_num, y_num) / np.interp(x, x_den, y_den)


def plot_panel(
    series: dict[str, dict[str, np.ndarray]],
    task: str,
    error_key: str,
    cost_key: str,
    output: pathlib.Path,
) -> None:
    figure, axis = plt.subplots(figsize=(4.4, 2.92))
    for method in METHOD_ORDER[task]:
        values = series[method]
        order = np.argsort(values[error_key])
        marker, color = STYLES[method]
        axis.plot(
            values[error_key][order],
            values[cost_key][order],
            marker=marker,
            color=color,
            label=method,
            linewidth=2.0,
            markersize=7.0,
        )

    if cost_key == "update_ms":
        axis.set_yscale("log")
        ylabel = "Amortized update time\n(ms)"
        numerator, denominator = "AeroSketch", "SchurTrack"
        ratio_label, ratio_color = "Speedup", "#0000FF"
    else:
        axis.set_ylim(bottom=0)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.yaxis.offsetText.set_fontsize(9)
        ylabel = (
            "Messages (KB)" if task in {"dist", "dsw"}
            else "Maximum size (KB)"
        )
        numerator, denominator = "SchurTrack", "AeroSketch"
        ratio_label = (
            "Msg. ratio" if task in {"dist", "dsw"} else "Space ratio"
        )
        ratio_color = "#008000"

    xlabel = (
        "Average relative cov. error"
        if error_key == "avg_error"
        else "Maximum relative cov. error"
    )
    axis.set_xlabel(xlabel, fontsize=12)
    axis.set_ylabel(ylabel, fontsize=11)
    axis.tick_params(axis="both", which="major", labelsize=10)
    axis.grid(True, which="both", linestyle="--", linewidth=0.5)
    axis.set_axisbelow(True)

    ratio_x, ratio_y = interpolated_ratio(
        series, error_key, cost_key, numerator, denominator
    )
    ratio_axis = axis.twinx()
    ratio_axis.plot(
        ratio_x,
        ratio_y,
        linestyle="--",
        color=ratio_color,
        linewidth=1.8,
        alpha=0.8,
        label=ratio_label,
    )
    ratio_axis.set_ylabel(ratio_label, fontsize=11)
    ratio_axis.tick_params(axis="y", which="major", labelsize=10)
    ratio_axis.set_ylim(bottom=0)
    # Invisible proxy makes ``loc=best`` consider the twin-axis curve.
    axis.plot(ratio_x, ratio_y, transform=ratio_axis.transData, alpha=0.0)
    handles, labels = axis.get_legend_handles_labels()
    ratio_handles, ratio_labels = ratio_axis.get_legend_handles_labels()
    axis.legend(
        handles + ratio_handles,
        labels + ratio_labels,
        fontsize=8,
        loc="best",
        framealpha=0.9,
    )
    figure.subplots_adjust(left=0.17, right=0.83, bottom=0.20, top=0.95)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root", type=pathlib.Path, default=pathlib.Path("results/main")
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=pathlib.Path("figures")
    )
    args = parser.parse_args()
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    for task in METHOD_ORDER:
        series = load_series(args.result_root, task)
        for error_name, error_key in (("avg", "avg_error"), ("max", "max_error")):
            for cost_name, cost_key in (("time", "update_ms"), ("cost", "cost_kb")):
                output = args.output_dir / f"{task}-{cost_name}-{error_name}-error.pdf"
                plot_panel(series, task, error_key, cost_key, output)
                print(output)


if __name__ == "__main__":
    main()

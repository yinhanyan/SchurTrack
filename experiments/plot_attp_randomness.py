"""Plot the 100-seed Random Noisy ATTP iteration-budget sweep."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


SCHUR_COLOR = "#E69F00"
AERO_COLOR = "m"
AERO_LOW_COLOR = "#CC79C6"
GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0
COMPACT_MEDIAN_SIZE = 38.0
COMPACT_SEED_SIZE = COMPACT_MEDIAN_SIZE / GOLDEN_RATIO**2


def read_records(result_dir: pathlib.Path) -> tuple[dict, dict[int, list[dict]]]:
    metadata = json.loads(
        (result_dir / "metadata.json").read_text(encoding="utf-8")
    )
    records = json.loads(
        (result_dir / "quality.json").read_text(encoding="utf-8")
    )
    schur = [record for record in records if record["mode"] == "deterministic"]
    if len(schur) != 1:
        raise ValueError("expected exactly one deterministic SchurTrack record")

    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        if record["mode"] == "randomized":
            grouped[int(record["power_iterations"])].append(record)
    expected_qs = [int(q) for q in metadata["qs"]]
    expected_runs = int(metadata["seeds_per_config"])
    if sorted(grouped) != expected_qs:
        raise ValueError("AeroSketch configurations do not match metadata")
    for q, group in grouped.items():
        group.sort(key=lambda record: int(record["seed"]))
        if len(group) != expected_runs or len(
            {int(record["seed"]) for record in group}
        ) != expected_runs:
            raise ValueError(f"q={q} does not contain {expected_runs} seeds")
    return schur[0], grouped


def plot_panel(
    axis: plt.Axes,
    schur: dict,
    grouped: dict[int, list[dict]],
    metric: str,
    metric_label: str,
) -> tuple[object, object, object]:
    schur_handle = axis.scatter(
        [float(schur[metric])],
        [float(schur["calibrated_update_ms"])],
        marker="D",
        facecolors=SCHUR_COLOR,
        edgecolors=SCHUR_COLOR,
        linewidths=1.2,
        s=COMPACT_MEDIAN_SIZE,
        zorder=4,
    )

    recommended_q = max(grouped)
    aero_low_handle = None
    aero_recommended_handle = None
    for q in sorted(grouped):
        recommended = q == recommended_q
        distribution_color = AERO_COLOR if recommended else AERO_LOW_COLOR
        distribution_alpha = 1.0 if recommended else 0.5
        errors = np.asarray(
            [float(record[metric]) for record in grouped[q]],
            dtype=np.float64,
        )
        median = float(np.median(errors))
        fifth, ninety_fifth = np.percentile(errors, [5.0, 95.0])
        update_ms = float(grouped[q][0]["calibrated_update_ms"])
        axis.errorbar(
            [median],
            [update_ms],
            xerr=[[median - fifth], [ninety_fifth - median]],
            fmt="none",
            color=distribution_color,
            linewidth=0.9,
            capsize=3,
            capthick=0.9,
            alpha=distribution_alpha,
            zorder=1,
        )
        axis.scatter(
            errors,
            np.full(errors.shape, update_ms),
            marker="o",
            color=distribution_color,
            s=COMPACT_SEED_SIZE,
            alpha=0.28 if recommended else 0.12,
            linewidths=0,
            zorder=2,
        )
        median_handle = axis.scatter(
            [median],
            [update_ms],
            marker="o",
            facecolors=distribution_color,
            edgecolors=distribution_color,
            linewidths=1.2,
            s=COMPACT_MEDIAN_SIZE,
            zorder=3,
        )
        if recommended:
            aero_recommended_handle = median_handle
        elif aero_low_handle is None:
            aero_low_handle = median_handle
        axis.annotate(
            rf"$q={q}$",
            (float(np.min(errors)), update_ms),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=10,
            color=distribution_color,
            fontweight="bold" if recommended else "normal",
        )

    if aero_low_handle is None or aero_recommended_handle is None:
        raise ValueError("missing AeroSketch legend handles")
    axis.set_xlabel(metric_label, fontsize=11)
    axis.tick_params(axis="both", which="major", labelsize=10)
    axis.grid(True, linestyle="--", linewidth=0.6)
    axis.set_axisbelow(True)
    return schur_handle, aero_low_handle, aero_recommended_handle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    schur, grouped = read_records(args.result_dir)
    figure, axes = plt.subplots(1, 2, figsize=(4.5, 2.55), sharey=True)
    legend_handles = plot_panel(
        axes[0], schur, grouped, "max_error", "Maximum error"
    )
    plot_panel(axes[1], schur, grouped, "mean_error", "Average error")

    update_times = [float(schur["calibrated_update_ms"])] + [
        float(group[0]["calibrated_update_ms"])
        for group in grouped.values()
    ]
    span = max(update_times) - min(update_times)
    margin = 0.08 * span
    axes[0].set_ylim(min(update_times) - margin, max(update_times) + margin)
    axes[1].tick_params(axis="y", labelleft=True)
    figure.supylabel(
        "Amortized update time\n(ms)",
        x=0.02,
        y=0.49,
        fontsize=10,
        va="center",
        multialignment="center",
    )
    figure.legend(
        legend_handles,
        [
            "SchurTrack",
            rf"AeroSketch $q<{max(grouped)}$",
            rf"AeroSketch $q={max(grouped)}$",
        ],
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        columnspacing=0.8,
        handletextpad=0.35,
        framealpha=0.9,
        markerscale=1.0,
    )
    figure.subplots_adjust(
        left=0.14,
        right=0.985,
        bottom=0.22,
        top=0.76,
        wspace=0.24,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".pdf"))
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()

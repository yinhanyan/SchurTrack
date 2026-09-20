"""Summarize the ATTP deterministic-versus-probabilistic experiment."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict

import numpy as np


def summarize(result_dir: pathlib.Path) -> list[dict]:
    records = json.loads(
        (result_dir / "quality.json").read_text(encoding="utf-8")
    )
    schur = [row for row in records if row["mode"] == "deterministic"]
    if len(schur) != 1:
        raise ValueError("expected one deterministic SchurTrack run")

    result = [
        {
            "method": "SchurTrack",
            "config": "(0,0)",
            "runs": 1,
            "update_ms": float(schur[0]["calibrated_update_ms"]),
            "time_ratio": 1.0,
            "median_avg_error": float(schur[0]["mean_error"]),
            "median_max_error": float(schur[0]["max_error"]),
        }
    ]
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in records:
        if row["mode"] == "randomized":
            grouped[int(row["power_iterations"])].append(row)
    schur_ms = result[0]["update_ms"]
    for q in sorted(grouped):
        rows = grouped[q]
        update_ms = float(rows[0]["calibrated_update_ms"])
        result.append(
            {
                "method": "AeroSketch",
                "config": f"({q},{q})",
                "runs": len(rows),
                "update_ms": update_ms,
                "time_ratio": update_ms / schur_ms,
                "median_avg_error": float(
                    np.median([float(row["mean_error"]) for row in rows])
                ),
                "median_max_error": float(
                    np.median([float(row["max_error"]) for row in rows])
                ),
            }
        )
    return result


def write_tsv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)


def format_markdown(rows: list[dict]) -> str:
    headers = (
        "Method",
        "Config (q_PI, q_SI)",
        "Runs",
        "Update (ms)",
        "Time / Schur",
        "Median avg. error",
        "Median max. error",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :-- | :--: | --: | --: | --: | --: | --: |",
    ]
    for row in rows:
        values = (
            row["method"],
            row["config"],
            str(row["runs"]),
            f"{row['update_ms']:.3f}",
            f"{row['time_ratio']:.3f}x",
            f"{row['median_avg_error']:.5f}",
            f"{row['median_max_error']:.5f}",
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=pathlib.Path,
        default=pathlib.Path("results/attp-randomness/table"),
    )
    args = parser.parse_args()
    rows = summarize(args.result_dir)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_prefix.with_suffix(".tsv"), rows)
    markdown = format_markdown(rows)
    args.output_prefix.with_suffix(".md").write_text(
        markdown, encoding="utf-8"
    )
    print(markdown, end="")


if __name__ == "__main__":
    main()

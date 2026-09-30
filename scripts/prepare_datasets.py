"""Prepare the two compact matrices used by the paper experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import urllib.request
import zipfile

import numpy as np


PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from experiments.run_main import load_input_rows


GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ARCHIVE = "glove.6B.zip"
GLOVE_MEMBER = "glove.6B.300d.txt"


def sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(matrix.tobytes(order="C")).hexdigest()


def discard_normals(rng: np.random.RandomState, count: int) -> None:
    chunk = 1_000_000
    while count:
        step = min(chunk, count)
        rng.standard_normal(step)
        count -= step


def generate_random_noisy(
    *, seed: int, source_rows: int, kept_rows: int, d: int, zeta: float
) -> np.ndarray:
    """Generate a deterministic instance of the paper's random-noisy model.

    Random draws are ordered like the original generator: all of ``S``, then
    the matrix used for ``U``, then ``G``.  Unused rows of ``S`` are consumed
    without being retained, so preparation does not require the full matrix.
    """

    if kept_rows > source_rows:
        raise ValueError("kept_rows cannot exceed source_rows")
    rng = np.random.RandomState(seed)
    signal = rng.standard_normal((kept_rows, d))
    discard_normals(rng, (source_rows - kept_rows) * d)
    random_basis = rng.standard_normal((d, d))
    q, _ = np.linalg.qr(random_basis, mode="reduced")
    right_basis = q.T
    diagonal = 1.0 - np.arange(d, dtype=np.float64) / d
    noise = rng.standard_normal((kept_rows, d))
    return signal @ (diagonal[:, None] * right_basis) + noise / zeta


def download_file(url: str, destination: pathlib.Path) -> None:
    """Download a file atomically while reporting coarse progress."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "SchurTrack-artifact/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            downloaded = 0
            next_report = 64 * 1024 * 1024
            with partial.open("wb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        if total:
                            percent = 100.0 * downloaded / total
                            print(
                                f"Downloaded {downloaded / 2**20:.0f} MiB "
                                f"of {total / 2**20:.0f} MiB "
                                f"({percent:.1f}%)",
                                file=sys.stderr,
                            )
                        else:
                            print(
                                f"Downloaded {downloaded / 2**20:.0f} MiB",
                                file=sys.stderr,
                            )
                        next_report += 64 * 1024 * 1024
    except Exception as error:
        raise RuntimeError(
            f"failed to download {url}; use --glove with a local file"
        ) from error
    partial.replace(destination)


def extract_glove_archive(
    archive: pathlib.Path, destination: pathlib.Path
) -> None:
    """Extract only the 300-dimensional GloVe member atomically."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with zipfile.ZipFile(archive) as bundle:
        try:
            member = bundle.getinfo(GLOVE_MEMBER)
        except KeyError as error:
            raise ValueError(
                f"{archive} does not contain {GLOVE_MEMBER}"
            ) from error
        with bundle.open(member) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    partial.replace(destination)


def resolve_glove_source(
    supplied: pathlib.Path | None,
    cache_dir: pathlib.Path,
    url: str,
) -> pathlib.Path:
    """Return a local GloVe text file, downloading and extracting if needed."""

    if supplied is not None:
        if not supplied.is_file():
            raise FileNotFoundError(supplied)
        return supplied

    text_path = cache_dir / GLOVE_MEMBER
    if text_path.is_file():
        print(f"Using cached {text_path}", file=sys.stderr)
        return text_path

    archive_path = cache_dir / GLOVE_ARCHIVE
    if not archive_path.is_file():
        print(f"Downloading {url} to {archive_path}", file=sys.stderr)
        download_file(url, archive_path)
    else:
        print(f"Using cached {archive_path}", file=sys.stderr)
    print(f"Extracting {GLOVE_MEMBER}", file=sys.stderr)
    extract_glove_archive(archive_path, text_path)
    return text_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glove",
        type=pathlib.Path,
        help=(
            "optional path to glove.6B.300d.txt; when omitted, download and "
            "extract the official GloVe 6B archive"
        ),
    )
    parser.add_argument("--glove-url", default=GLOVE_URL)
    parser.add_argument(
        "--random-noisy-source",
        type=pathlib.Path,
        help=(
            "optional original .mat/.npy matrix; when omitted, generate a "
            "fixed-seed instance of the model described in the paper"
        ),
    )
    parser.add_argument("--matrix-key", default="A")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=pathlib.Path, default=PROJECT / "data")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    glove_path = resolve_glove_source(
        args.glove, args.output_dir / "raw", args.glove_url
    )
    glove = load_input_rows(glove_path, "A", 10_000)
    if glove.shape != (10_000, 300):
        raise ValueError(f"expected GloVe shape (10000, 300), got {glove.shape}")

    if args.random_noisy_source is None:
        random_noisy = generate_random_noisy(
            seed=args.seed,
            source_rows=50_000,
            kept_rows=16_000,
            d=500,
            zeta=5.0,
        )
        source = "generated random-noisy model"
    else:
        random_noisy = load_input_rows(
            args.random_noisy_source, args.matrix_key, 16_000
        )
        source = str(args.random_noisy_source.resolve())
    if random_noisy.shape != (16_000, 500):
        raise ValueError(
            f"expected random-noisy shape (16000, 500), got {random_noisy.shape}"
        )

    outputs = {
        "glove": args.output_dir / "glove-10000x300.npy",
        "random_noisy": args.output_dir / "random-noisy-16000x500.npy",
    }
    np.save(outputs["glove"], np.asarray(glove, dtype=np.float64))
    np.save(
        outputs["random_noisy"],
        np.asarray(random_noisy, dtype=np.float64),
    )
    metadata = {
        "glove": {
            "source": str(glove_path.resolve()),
            "download_url": None if args.glove else args.glove_url,
            "shape": list(glove.shape),
            "sha256_float64": sha256(glove),
        },
        "random_noisy": {
            "source": source,
            "shape": list(random_noisy.shape),
            "seed_if_generated": args.seed,
            "model": "A = S D U + G / 5",
            "sha256_float64": sha256(random_noisy),
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

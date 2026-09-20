from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from algorithm.attp.schur_fd_attp import SchurFdAttp
from algorithm.sw.fast_ds_fd import FastDsFd
from algorithm.sw.schur_fd_sw import SchurFdSw
from base.fd import reduce_rows_fd
from experiments.run_main import load_input_rows
from scripts.prepare_datasets import extract_glove_archive


class ReproductionSmokeTests(unittest.TestCase):
    def test_extracts_only_300d_glove_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "glove.6B.zip"
            output = root / "glove.6B.300d.txt"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("glove.6B.50d.txt", "unused\n")
                bundle.writestr("glove.6B.300d.txt", "word 1 2 3\n")
            extract_glove_archive(archive, output)
            self.assertEqual(
                output.read_text(encoding="utf-8"), "word 1 2 3\n"
            )

    def test_labeled_text_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.txt"
            path.write_text("word 1 2 3\n42 4 5 6\n", encoding="utf-8")
            matrix = load_input_rows(path, "A", 2)
        np.testing.assert_allclose(matrix, [[1, 2, 3], [4, 5, 6]])

    def test_attp_two_ell_query(self) -> None:
        rng = np.random.default_rng(1)
        tracker = SchurFdAttp(d=12, ell=3)
        for row in rng.normal(size=(32, 12)):
            tracker.fit(row)
        output = reduce_rows_fd(tracker.get(), 6)
        self.assertLessEqual(output.shape[0], 6)
        self.assertEqual(output.shape[1], 12)

    def test_sliding_window_implementations(self) -> None:
        rng = np.random.default_rng(2)
        rows = rng.normal(size=(24, 10))
        norm_sq = np.sum(rows * rows, axis=1)
        rows /= np.sqrt(norm_sq.min())
        upper = float(np.max(np.sum(rows * rows, axis=1)))
        schur = SchurFdSw(12, upper, 10, 3)
        baseline = FastDsFd(12, upper, 10, 3)
        for row in rows:
            schur.fit(row)
            baseline.fit(row.reshape(1, -1))
        self.assertEqual(schur.get()[0].shape[1], 10)
        self.assertEqual(baseline.get()[0].shape[1], 10)


if __name__ == "__main__":
    unittest.main()

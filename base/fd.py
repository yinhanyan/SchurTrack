import numpy as np
import numpy.typing as npt
from scipy import linalg


def reduce_rows_fd(rows: npt.ArrayLike, sketch_dim: int) -> np.ndarray:
    """Return the deterministic FD reduction of a row-factor matrix.

    If the input already has at most ``sketch_dim`` rows, no spectral
    reduction is needed.  Otherwise this applies the standard FD shrink to
    the squared singular values and returns at most ``sketch_dim`` rows.
    """

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("rows must be a two-dimensional matrix")
    if sketch_dim < 1:
        raise ValueError("sketch_dim must be positive")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rows contain non-finite values")
    if matrix.shape[0] <= sketch_dim:
        return matrix

    _, singular_values, Vt = linalg.svd(
        matrix,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesvd",
    )
    retained = min(int(sketch_dim), singular_values.size)
    if retained == 0:
        return np.empty((0, matrix.shape[1]), dtype=np.float64)
    delta = (
        float(singular_values[retained] ** 2)
        if singular_values.size > retained
        else 0.0
    )
    retained_sq = np.maximum(
        singular_values[:retained] ** 2 - delta, 0.0
    )
    positive = retained_sq > 0.0
    if not np.any(positive):
        return np.empty((0, matrix.shape[1]), dtype=np.float64)
    return np.sqrt(retained_sq[positive])[:, None] * Vt[:retained][positive]


def reduce_covariance_fd(
    covariance: npt.ArrayLike, sketch_dim: int
) -> np.ndarray:
    """Return the deterministic FD row factor of a symmetric covariance.

    Coordinator ledgers are only symmetric up to roundoff.  Working with the
    explicitly symmetrized matrix avoids the less stable general ``gesdd``
    path, and clipping negative eigenvalues implements the PSD projection
    required before an FD shrink.
    """

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    if sketch_dim < 1:
        raise ValueError("sketch_dim must be positive")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance contains non-finite values")

    symmetric = 0.5 * (matrix + matrix.T)
    try:
        eigenvalues, eigenvectors = linalg.eigh(
            symmetric,
            check_finite=False,
            driver="evr",
        )
    except linalg.LinAlgError:
        eigenvalues, eigenvectors = linalg.eigh(
            symmetric,
            check_finite=False,
            driver="evd",
        )
    eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
    eigenvectors = eigenvectors[:, ::-1]
    retained = min(int(sketch_dim), eigenvalues.size)
    if retained == 0:
        return np.empty((0, matrix.shape[0]), dtype=np.float64)
    if eigenvalues.size > retained:
        eigenvalues = np.maximum(
            eigenvalues[:retained] - eigenvalues[retained], 0.0
        )
    else:
        eigenvalues = eigenvalues[:retained]
    positive = eigenvalues > 0.0
    if not np.any(positive):
        return np.empty((0, matrix.shape[0]), dtype=np.float64)
    return (
        np.sqrt(eigenvalues[positive])[:, None]
        * eigenvectors[:, :retained][:, positive].T
    )


class FrequentDirections:
    def __init__(self, d: int, sketch_dim: int, fast_fd: bool = False, **kwargs):
        """
        Class wrapper for all FD-type methods

        __rotate_and_reduce__ is not defined for the standard FrequentDirections but is for the
        subsequent subclasses which inherit from FrequentDirections.
        """
        self.d = d
        # self.delta = 0.0  # For RFD

        self.sketch_dim = sketch_dim
        self.fast_fd = fast_fd
        self.max_row_num = 2 * self.sketch_dim if fast_fd else self.sketch_dim + 1
        self.sketch = np.zeros((self.max_row_num, self.d), dtype=np.float64)

        if "start_row" in kwargs:
            self.row = kwargs["start_row"]
        else:
            self.row = 0
        self.sigma = np.zeros(self.max_row_num, dtype=np.float64)
        self.Vt = np.zeros((self.max_row_num, self.d), dtype=np.float64)
        self.svd_computed = False
        self.svd_calls = 0
        # self.energy = 0.0

    def fit(self, X: npt.NDArray):
        """
        Fits the FD transform to dataset X
        """
        # self.energy += np.sum(X**2)

        # n = X.shape[0] if X.ndim == 2 else 1
        if X.ndim == 1:
            X = X.reshape((1, -1))
        n = X.shape[0]
        cursor = 0
        while cursor < n:
            if self.row < self.max_row_num:
                step = min(n - cursor, self.max_row_num - self.row)
                self.sketch[self.row : self.row + step, :] = X[
                    cursor : cursor + step, :
                ]
                self.row += step
                cursor += step
                self.svd_computed = False
            if self.row >= self.max_row_num:
                try:
                    _, s, Vt = linalg.svd(
                        self.sketch[: self.row],
                        full_matrices=False,
                        check_finite=False,
                    )
                except np.linalg.LinAlgError:
                    # gesdd can fail on strongly scaled but finite streams.
                    # The QR-based gesvd driver is slower but more robust and
                    # preserves the same exact FD reduction.
                    _, s, Vt = linalg.svd(
                        self.sketch[: self.row],
                        full_matrices=False,
                        check_finite=False,
                        lapack_driver="gesvd",
                    )
                self.svd_calls += 1
                sigma_squared = s**2
                if len(sigma_squared) > self.sketch_dim:
                    sigma_squared = (
                        sigma_squared[: self.sketch_dim]
                        - sigma_squared[self.sketch_dim]
                    )
                    Vt = Vt[: self.sketch_dim]

                s = np.sqrt(sigma_squared)
                self.sigma = s
                self.Vt = Vt

                self.sketch[: self.sketch_dim] = Vt * s.reshape(-1, 1)
                self.sketch[self.sketch_dim :] = 0
                self.row = self.sketch_dim
                self.svd_computed = True

    def get(self):
        return self.sketch

    def row_num(self):
        return self.sketch.shape[0]

    def size(self):
        return self.sketch.size

    def clear(self):
        self.sketch.fill(0.0)
        self.sigma.fill(0.0)
        self.Vt.fill(0.0)
        self.row = 0
        self.svd_computed = False


class FdDump(FrequentDirections):
    def __init__(self, *args, **kwargs):
        self.iteration_steps = kwargs.pop("iteration_steps", None)
        super().__init__(*args, **kwargs)
        self.power_calls = 0
        self.simultaneous_calls = 0

    def minus(self, Z):
        ZCC = (Z.T @ self.sketch[: self.row].T) @ self.sketch[: self.row]
        self.sketch[: self.row] = (
            self.sketch[: self.row] - (self.sketch[: self.row] @ Z) @ Z.T
        )
        return ZCC

    def dump(self, threshold: float):
        from base.sim_iter import simultaneous_iteration
        from base.utils import power_iteration

        if self.row == 0:
            return np.zeros((self.d, 0)), np.zeros((0, self.d))
        iter_step = (
            int(np.ceil(np.log2(max(self.d, 1)))) + 1
            if self.iteration_steps is None
            else int(self.iteration_steps)
        )
        sigma_squared, _ = power_iteration(
            self.sketch[: self.row], max_iter=iter_step
        )
        self.power_calls += 1

        if sigma_squared > threshold / 2:
            rank = 1
            while True:
                rank = min(rank * 2, self.row)
                Z, sigmas_squared = simultaneous_iteration(
                    self.sketch[: self.row].T, rank, iter_step
                )
                self.simultaneous_calls += 1
                if sigmas_squared[-1] < threshold or rank == self.row:
                    selected = len(sigmas_squared) - np.searchsorted(
                        sigmas_squared[::-1], threshold, side="left"
                    )
                    Z = Z[:, :selected]
                    if selected:
                        return Z, self.minus(Z)
                    break

        return np.zeros((self.d, 0)), np.zeros((0, self.d))


class SlowFdDump(FrequentDirections):
    """Exact SVD threshold dumper used by the migrated distributed baselines."""

    def dump(self, threshold: float):
        if self.row == 0:
            return np.zeros((0, self.d))
        if not self.svd_computed:
            _, self.sigma, self.Vt = linalg.svd(
                self.sketch[: self.row], full_matrices=False
            )
            self.svd_calls += 1
        squared = self.sigma**2
        selected = len(squared) - np.searchsorted(
            squared[::-1], threshold, side="right"
        )
        self.sketch[selected : self.row] = self.Vt[
            selected : self.row
        ] * self.sigma[selected : self.row, None]
        self.svd_computed = True
        if selected == 0:
            return np.zeros((0, self.d))
        result = self.Vt[:selected] * self.sigma[:selected, None]
        retained_sigma = self.sigma[selected : self.row].copy()
        retained_vt = self.Vt[selected : self.row].copy()
        retained = retained_sigma.size
        self.sketch.fill(0.0)
        if retained:
            self.sketch[:retained] = (
                retained_vt * retained_sigma[:, None]
            )
        self.row = retained
        self.sigma = retained_sigma
        self.Vt = retained_vt
        return result


class FdRestore:
    """Restore AeroSketch subspace snapshots at a coordinator."""

    def __init__(self, d: int, sketch_dim: int):
        self.d = int(d)
        self.sketch_dim = int(sketch_dim)
        self.sketch = np.zeros((self.sketch_dim, self.d), dtype=np.float64)
        self.Zs = []
        self.ZCCs = []
        self.snapshot_rank = 0
        self.svd_calls = 0

    def fit(self, Z, ZCC):
        self.snapshot_rank += Z.shape[1]
        self.Zs.append(Z)
        self.ZCCs.append(ZCC)
        if self.snapshot_rank > self.sketch_dim:
            self.sketch = self.get()
            self.Zs.clear()
            self.ZCCs.clear()
            self.snapshot_rank = 0

    def get(self):
        covariance = self.sketch.T @ self.sketch
        for Z, ZCC in zip(self.Zs, self.ZCCs):
            product = Z @ ZCC
            covariance += (
                product
                + product.T
                - Z @ (ZCC @ Z) @ Z.T
            )
        self.svd_calls += 1
        return reduce_covariance_fd(covariance, self.sketch_dim)

    def row_num(self):
        return self.sketch.shape[0] + 2 * self.snapshot_rank


class RowFdRestore:
    """Lazy coordinator FD for immutable row-factor messages."""

    def __init__(self, d: int, sketch_dim: int):
        self.d = int(d)
        self.sketch_dim = int(sketch_dim)
        if self.d < 1 or self.sketch_dim < 1:
            raise ValueError("d and sketch_dim must be positive")
        self.sketch = np.empty((0, self.d), dtype=np.float64)
        self.blocks: list[np.ndarray] = []
        self.pending_rows = 0
        self.svd_calls = 0

    def fit(self, rows: npt.ArrayLike) -> None:
        block = np.asarray(rows, dtype=np.float64)
        if block.ndim == 1:
            block = block.reshape(1, -1)
        if block.ndim != 2 or block.shape[1] != self.d:
            raise ValueError("row-factor block has the wrong shape")
        if not np.all(np.isfinite(block)):
            raise ValueError("row-factor block contains non-finite values")
        if block.shape[0] == 0:
            return
        self.blocks.append(np.array(block, copy=True))
        self.pending_rows += int(block.shape[0])
        if self.pending_rows > self.sketch_dim:
            self.sketch = self.get()
            self.blocks.clear()
            self.pending_rows = 0

    def get(self) -> np.ndarray:
        covariance = self.sketch.T @ self.sketch
        for block in self.blocks:
            covariance += block.T @ block
        self.svd_calls += 1
        return reduce_covariance_fd(covariance, self.sketch_dim)

    def row_num(self) -> int:
        return int(self.sketch.shape[0] + self.pending_rows)

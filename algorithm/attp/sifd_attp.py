from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from base.fd import FdDump
from base.sim_iter import simultaneous_iteration
from base.utils import power_iteration
import scipy.linalg


class SiFdAttp:
    def __init__(self, d: int, ell: int):
        self.time = 0
        self.d, self.ell = d, ell

        self.C = FdDump(d, ell, fast_fd=True)

        self.p = 0
        self.energy: float = 0.0

        # Ordered cancellation factors and their creation times.
        self.Zs = []
        self.ZCCs = []
        self.snapshot_time = []
        self.rows_of_snapshots = 0
        # Theory-scale AeroSketch protocol: both randomized subroutines use
        # an ambient-dimension budget, independent of the sketch width.
        theory_iterations = int(np.ceil(np.log2(max(self.d, 1)))) + 1
        self.power_iterations = theory_iterations
        self.simultaneous_iterations = theory_iterations
        # Retain the old public name for compatibility with external runners.
        self.iter_step = theory_iterations
        self.power_calls = 0
        self.simultaneous_calls = 0
        self.query_svd_calls = 0

    def fit(self, X: npt.NDArray):
        self.energy += np.sum(X**2)

        self.C.fit(X)

        sigma_squared, _ = power_iteration(
            self.C.sketch, max_iter=self.power_iterations
        )
        self.power_calls += 1

        dump_threshold = self.energy / self.ell

        if sigma_squared > dump_threshold / 2:
            k = 1
            l = self.C.max_row_num
            while True:
                k = min(k * 2, l)
                Z, sigmas_squared = simultaneous_iteration(
                    self.C.sketch.T, k, self.simultaneous_iterations
                )
                self.simultaneous_calls += 1
                if sigmas_squared[-1] < dump_threshold or k == l:
                    kappa = len(sigmas_squared) - np.searchsorted(
                        sigmas_squared[::-1], dump_threshold, side="left"
                    )
                    if kappa < 1:
                        break
                    Z = Z[:, :kappa]
                    self.Zs.append(Z)
                    ZCC = self.C.minus(Z)
                    self.ZCCs.append(ZCC)
                    self.rows_of_snapshots += 2 * kappa

                    self.snapshot_time.append(self.time)
                    break

        self.time += 1

    def get(self):
        """Return the reduced covariance sketch of immutable snapshots."""
        ret = np.zeros((self.d, self.d), dtype=np.float64)
        for i in range(len(self.snapshot_time)):
            Z = self.Zs[i]
            ZCC = self.ZCCs[i]
            ZZCC = Z @ ZCC
            ZZCCZZ = Z @ (ZCC @ Z) @ Z.T
            ret += ZZCC + ZZCC.T - ZZCCZZ

        _, sigma_squared, Vt = scipy.linalg.svd(ret, overwrite_a=True)
        self.query_svd_calls += 1
        sketch_dim = 2 * self.ell
        if len(sigma_squared) > sketch_dim:
            sigma_squared = sigma_squared[:sketch_dim] - sigma_squared[sketch_dim]
            Vt = Vt[:sketch_dim]

        s = np.sqrt(np.maximum(sigma_squared, 0.0))
        ret = Vt * s.reshape(-1, 1)
        return ret

    def row_num(self):
        return self.rows_of_snapshots + self.C.row_num()

    def size(self):
        return (
            sum([sketch.size for sketch in self.Zs])
            + sum([sketch.size for sketch in self.ZCCs])
            + self.C.size()
        )


@dataclass(frozen=True)
class AeroAttpSnapshot:
    """One immutable AeroSketch covariance-cancellation message."""

    Z: npt.NDArray[np.float64]
    ZCC: npt.NDArray[np.float64]

    @property
    def rank(self) -> int:
        return int(self.Z.shape[1])

    @property
    def communication_floats(self) -> int:
        return int(self.Z.size + self.ZCC.size)


class AeroAttpEmitter:
    """AeroSketch ATTP core with an explicit send-once block interface.

    The threshold remains the cumulative local energy divided by ``ell``, as
    in the legacy ATTP implementation.  Unlike :class:`SiFdAttp`, this class
    does not retain emitted blocks: a distributed wrapper owns each returned
    message and the site keeps only the mutable FD residual.
    """

    def __init__(self, d: int, ell: int):
        if d < 1 or ell < 1:
            raise ValueError("d and ell must be positive")
        self.d = int(d)
        self.ell = int(ell)
        self.time = 0
        self.energy = 0.0
        self.C = FdDump(self.d, self.ell, fast_fd=True)
        self.emitted_rank = 0

    def update(self, row: npt.ArrayLike) -> list[AeroAttpSnapshot]:
        vector = np.asarray(row, dtype=np.float64).reshape(-1)
        if vector.size != self.d:
            raise ValueError(f"expected a row of length {self.d}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("row contains a non-finite value")
        self.time += 1
        self.energy += float(vector @ vector)
        self.C.fit(vector)
        Z, ZCC = self.C.dump(self.energy / self.ell)
        if Z.shape[1] == 0:
            return []
        Z = np.array(Z, dtype=np.float64, copy=True)
        ZCC = np.array(ZCC, dtype=np.float64, copy=True)
        Z.setflags(write=False)
        ZCC.setflags(write=False)
        snapshot = AeroAttpSnapshot(Z=Z, ZCC=ZCC)
        self.emitted_rank += snapshot.rank
        return [snapshot]

    fit = update

    def row_num(self) -> int:
        return int(self.C.row)

    def size(self) -> int:
        return int(self.C.size())

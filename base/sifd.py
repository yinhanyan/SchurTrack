import numpy as np
import numpy.typing as npt
from base.fd import FdDump
from base.sim_iter import simultaneous_iteration
from base.utils import power_iteration
import scipy.linalg
from dataclasses import dataclass
from collections import deque


@dataclass
class SiFdSnapshot:
    s_time: int
    time: int
    Z: npt.NDArray
    ZCC: npt.NDArray


class SiFd:
    def __init__(self, d: int, ell: int, dump_threshold: float):
        self.time = 0
        self.d, self.ell = d, ell

        self.C = FdDump(d, ell, fast_fd=True)

        self.p = 0
        # self.energy: float = 0.0

        # Number of rows stored by the ordered snapshot sequence.
        self.rows_of_snapshots = 0
        # AeroSketch's theory-scale Theta(log d) budget.
        self.iter_step = int(np.ceil(np.log2(max(self.d, 1)))) + 1
        self.snapshots: deque[SiFdSnapshot] = deque()
        self.power_calls = 0
        self.simultaneous_calls = 0
        self.query_svd_calls = 0

        self.dump_threshold = dump_threshold
        self.last_dump_time = 1

    def fit(self, X: npt.NDArray):
        # self.energy += np.sum(X**2)

        self.C.fit(X)

        sigma_squared, _ = power_iteration(
            self.C.sketch, max_iter=self.iter_step
        )
        self.power_calls += 1

        dump_threshold = self.dump_threshold

        if sigma_squared > dump_threshold / 2:
            k = 1
            l = self.C.max_row_num
            while True:
                k = min(k * 2, l)
                Z, sigmas_squared = simultaneous_iteration(
                    self.C.sketch.T, k, self.iter_step
                )
                self.simultaneous_calls += 1
                if sigmas_squared[-1] < dump_threshold or k == l:
                    kappa = len(sigmas_squared) - np.searchsorted(
                        sigmas_squared[::-1], dump_threshold, side="left"
                    )
                    if kappa < 1:
                        break
                    Z = Z[:, :kappa]
                    ZCC = self.C.minus(Z)
                    self.rows_of_snapshots += 2 * kappa

                    self.snapshots.append(
                        SiFdSnapshot(self.last_dump_time, self.time, Z, ZCC)
                    )
                    self.last_dump_time = self.time + 1
                    break

        self.time += 1

    def get(self):
        """Return the reduced covariance sketch for the current prefix."""
        ret = self.C.get()
        ret = ret.T @ ret
        for i in range(len(self.snapshots)):
            Z = self.snapshots[i].Z
            ZCC = self.snapshots[i].ZCC
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
        self.sigma = s
        self.Vt = Vt

        ret = Vt * s.reshape(-1, 1)
        return ret

    def row_num(self):
        return self.rows_of_snapshots + self.C.row_num()

    def size(self):
        return (
            sum(
                snapshot.Z.size + snapshot.ZCC.size
                for snapshot in self.snapshots
            )
            + self.C.size()
        )

    # def popleft(self):
    #     s = self.snapshots.popleft()
    #     self.rows_of_snapshots -= 2 * s.Z.shape[1]

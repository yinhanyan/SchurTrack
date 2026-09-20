"""Instrumented AeroSketch ATTP core for randomized-reliability studies.

This module exposes independent Power/Simultaneous Iteration budgets, an owned
random generator, an exact-SVD diagnostic mode, and exact residual audits at
selected updates.

Exact audits and the oracle mode are experiment diagnostics.  Their costs must
not be included in the reported Aero update latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

import numpy as np
import numpy.typing as npt
import scipy.linalg

from base.fd import FdDump, reduce_covariance_fd
from base.sim_iter import simultaneous_iteration
from base.utils import power_iteration


Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class AeroAttpAuditRecord:
    """Exact residual state around one pre-registered update."""

    time: int
    threshold: float
    pre_residual_norm_sq: float
    estimated_norm_sq: float
    triggered: bool
    selected_rank: int
    post_residual_norm_sq: float
    power_miss: bool
    simultaneous_failure: bool

    @property
    def residual_violation(self) -> bool:
        return self.post_residual_norm_sq > 4.0 * self.threshold

    @property
    def pre_ratio(self) -> float:
        return self.pre_residual_norm_sq / self.threshold

    @property
    def post_ratio(self) -> float:
        return self.post_residual_norm_sq / self.threshold


class InstrumentedAeroAttp:
    """AeroSketch ATTP with explicit random budgets and exact audit hooks."""

    def __init__(
        self,
        d: int,
        ell: int,
        *,
        power_iterations: int,
        simultaneous_iterations: int,
        seed: int = 0,
        mode: str = "randomized",
        audit_times: Collection[int] = (),
    ) -> None:
        if d < 1 or ell < 1:
            raise ValueError("d and ell must be positive")
        if power_iterations < 0 or simultaneous_iterations < 0:
            raise ValueError("iteration counts must be nonnegative")
        if mode not in {"randomized", "oracle"}:
            raise ValueError("mode must be 'randomized' or 'oracle'")

        self.time = 0
        self.d = int(d)
        self.ell = int(ell)
        self.power_iterations = int(power_iterations)
        self.simultaneous_iterations = int(simultaneous_iterations)
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.audit_times = frozenset(int(value) for value in audit_times)

        self.C = FdDump(self.d, self.ell, fast_fd=True)
        self.energy = 0.0
        self.Zs: list[Array] = []
        self.ZCCs: list[Array] = []
        self.snapshot_time: list[int] = []
        self.rows_of_snapshots = 0

        self.power_calls = 0
        self.simultaneous_calls = 0
        self.oracle_svd_calls = 0
        self.query_svd_calls = 0
        self.audit_svd_calls = 0
        self.audit_records: list[AeroAttpAuditRecord] = []

    @staticmethod
    def _spectral_norm_sq(rows: Array) -> float:
        if rows.shape[0] == 0:
            return 0.0
        singular_values = scipy.linalg.svdvals(
            rows,
            overwrite_a=False,
            check_finite=False,
        )
        return float(singular_values[0] ** 2)

    def _store_cancellation(self, Z: Array) -> int:
        rank = int(Z.shape[1])
        if rank == 0:
            return 0
        self.Zs.append(np.array(Z, dtype=np.float64, copy=True))
        ZCC = self.C.minus(Z)
        self.ZCCs.append(np.array(ZCC, dtype=np.float64, copy=True))
        self.rows_of_snapshots += 2 * rank
        # fit advances self.time after the randomized update, so the
        # arriving row has the one-based timestamp self.time + 1 here.
        self.snapshot_time.append(self.time + 1)
        return rank

    def _randomized_cancel(self, threshold: float) -> tuple[float, bool, int]:
        estimated, _ = power_iteration(
            self.C.sketch,
            max_iter=self.power_iterations,
            rng=self.rng,
        )
        self.power_calls += 1
        estimated = float(estimated)
        triggered = estimated > threshold / 2.0
        selected_rank = 0
        if not triggered:
            return estimated, triggered, selected_rank

        rank = 1
        limit = self.C.max_row_num
        while True:
            rank = min(rank * 2, limit)
            Z, estimated_values = simultaneous_iteration(
                self.C.sketch.T,
                rank,
                self.simultaneous_iterations,
                rng=self.rng,
            )
            self.simultaneous_calls += 1
            if estimated_values[-1] < threshold or rank == limit:
                selected = len(estimated_values) - np.searchsorted(
                    estimated_values[::-1],
                    threshold,
                    side="left",
                )
                if selected:
                    selected_rank = self._store_cancellation(
                        Z[:, :selected]
                    )
                break
        return estimated, triggered, selected_rank

    def _oracle_cancel(self, threshold: float) -> tuple[float, bool, int]:
        live = self.C.sketch[: self.C.row]
        if live.shape[0] == 0:
            return 0.0, False, 0
        _, singular_values, Vt = scipy.linalg.svd(
            live,
            full_matrices=False,
            overwrite_a=False,
            check_finite=False,
        )
        self.oracle_svd_calls += 1
        squared = singular_values**2
        estimated = float(squared[0])
        triggered = estimated > threshold / 2.0
        if not triggered:
            return estimated, False, 0
        selected = int(np.count_nonzero(squared > threshold))
        if selected == 0:
            return estimated, True, 0
        return (
            estimated,
            True,
            self._store_cancellation(Vt[:selected].T),
        )

    def fit(self, X: npt.ArrayLike) -> None:
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != self.d:
            raise ValueError("input has the wrong shape")
        if matrix.shape[0] != 1:
            raise ValueError("instrumented ATTP updates accept one row")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("input contains a non-finite value")

        self.energy += float(np.sum(matrix * matrix))
        self.C.fit(matrix)
        threshold = self.energy / self.ell
        update_time = self.time + 1
        should_audit = update_time in self.audit_times

        pre_norm_sq = float("nan")
        if should_audit and self.mode == "randomized":
            pre_norm_sq = self._spectral_norm_sq(
                self.C.sketch[: self.C.row]
            )
            self.audit_svd_calls += 1

        if self.mode == "oracle":
            estimated, triggered, selected_rank = self._oracle_cancel(
                threshold
            )
            if should_audit:
                pre_norm_sq = estimated
        else:
            estimated, triggered, selected_rank = self._randomized_cancel(
                threshold
            )

        if should_audit:
            post_norm_sq = self._spectral_norm_sq(
                self.C.sketch[: self.C.row]
            )
            self.audit_svd_calls += 1
            tolerance = 1.0 + 128.0 * np.finfo(np.float64).eps
            pre_large = pre_norm_sq > 4.0 * threshold * tolerance
            post_large = post_norm_sq > 4.0 * threshold * tolerance
            self.audit_records.append(
                AeroAttpAuditRecord(
                    time=update_time,
                    threshold=threshold,
                    pre_residual_norm_sq=pre_norm_sq,
                    estimated_norm_sq=estimated,
                    triggered=triggered,
                    selected_rank=selected_rank,
                    post_residual_norm_sq=post_norm_sq,
                    power_miss=(
                        pre_large and estimated <= threshold / 2.0
                    ),
                    simultaneous_failure=(
                        pre_large and triggered and post_large
                    ),
                )
            )

        self.time = update_time

    update = fit

    def _validate_query_time(self, query_time: int | None) -> int:
        if query_time is None:
            return self.time
        query_time = int(query_time)
        if query_time < 0 or query_time > self.time:
            raise ValueError("query_time must lie in [0, current time]")
        return query_time

    def persistent_covariance_ledger(
        self, query_time: int | None = None
    ) -> Array:
        """Return the snapshot-only ledger for a historical prefix.

        The mutable live residual is deliberately excluded: after later
        updates the ATTP data structure no longer owns the residual that was
        live at an earlier query time. Timestamped cancellation snapshots are
        immutable and therefore form the persistent query state.
        """

        query_time = self._validate_query_time(query_time)
        covariance = np.zeros((self.d, self.d), dtype=np.float64)
        for Z, ZCC, snapshot_time in zip(
            self.Zs, self.ZCCs, self.snapshot_time
        ):
            if snapshot_time > query_time:
                break
            product = Z @ ZCC
            covariance += (
                product
                + product.T
                - Z @ (ZCC @ Z) @ Z.T
            )
        return 0.5 * (covariance + covariance.T)

    def covariance_ledger(self, query_time: int | None = None) -> Array:
        """Compatibility name for the persistent snapshot-only ledger."""

        return self.persistent_covariance_ledger(query_time)

    def current_augmented_covariance_ledger(self) -> Array:
        """Diagnostic current-prefix ledger including the live residual."""

        covariance = self.persistent_covariance_ledger(self.time)
        residual = self.C.sketch[: self.C.row]
        covariance += residual.T @ residual
        return 0.5 * (covariance + covariance.T)

    def get(self, query_time: int | None = None) -> Array:
        covariance = self.persistent_covariance_ledger(query_time)
        self.query_svd_calls += 1
        return reduce_covariance_fd(covariance, 2 * self.ell)

    def row_num(self) -> int:
        return int(self.rows_of_snapshots + self.C.row)

    def size(self) -> int:
        return int(
            sum(block.size for block in self.Zs)
            + sum(block.size for block in self.ZCCs)
            + self.C.size()
        )

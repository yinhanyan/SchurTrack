"""Deterministic Schur--Householder tracking sketches.

The algebra follows the exact-RAM append-or-cancel construction in
``exact_ram_schur_tracking(1).tex``.  This module is a floating-point research
prototype: it uses a small comparison tolerance and periodically checks the
barrier, but it does not claim a finite-precision error theorem.

Only NumPy is required.  In particular, the update path contains no power
iteration or simultaneous iteration.  A thin SVD is used only by the batched
Frequent-Directions reduction after the residual reaches ``2 * width`` rows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SchurSnapshot:
    """An immutable one- or two-row covariance snapshot."""

    time: int
    rows: Array


@dataclass
class SchurStats:
    updates: int = 0
    appends: int = 0
    cancels: int = 0
    zero_certificate_cancels: int = 0
    batch_reductions: int = 0
    svd_calls: int = 0
    snapshot_rows: int = 0


class SchurFdCore:
    """Variable-row Schur append-or-cancel core.

    Parameters
    ----------
    d:
        Ambient row dimension.
    width:
        Batch-FD target width ``k``.  The residual always has fewer than
        ``2 * k`` rows after an update.
    threshold:
        Spectral barrier ``tau``.  It must be positive.
    comparison_rtol:
        Relative tolerance used only by the floating-point implementation when
        deciding whether a Schur remainder is safely positive.
    audit:
        If true, check the spectral barrier after every update.  This costs an
        additional small SVD and is intended only for tests.
    """

    def __init__(
        self,
        d: int,
        width: int,
        threshold: float,
        *,
        comparison_rtol: float = 64.0 * np.finfo(np.float64).eps,
        audit: bool = False,
    ) -> None:
        if d < 1:
            raise ValueError("d must be positive")
        if width < 1 or width > d:
            raise ValueError("width must lie in [1, d]")
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive")

        self.d = int(d)
        self.width = int(width)
        self.threshold = float(threshold)
        self.comparison_rtol = float(comparison_rtol)
        self.audit = bool(audit)

        self.C: Array = np.empty((0, self.d), dtype=np.float64)
        self.J: Array = np.empty((0, 0), dtype=np.float64)
        self.snapshots: list[SchurSnapshot] = []
        self.stats = SchurStats()

    @property
    def row_count(self) -> int:
        return int(self.C.shape[0])

    def _decision_tolerance(
        self, row_norm_sq: float, coupling: float
    ) -> float:
        scale = max(self.threshold, row_norm_sq, abs(coupling), 1.0)
        return self.comparison_rtol * scale

    @staticmethod
    def _symmetrize(matrix: Array) -> Array:
        return 0.5 * (matrix + matrix.T)

    def _append(self, row: Array, z: Array, delta: float) -> None:
        q = self.row_count
        new_C = np.empty((q + 1, self.d), dtype=np.float64)
        if q:
            new_C[:q] = self.C
        new_C[q] = row

        new_J = np.empty((q + 1, q + 1), dtype=np.float64)
        inv_delta = 1.0 / delta
        if q:
            np.multiply(z[:, None], z[None, :], out=new_J[:q, :q])
            new_J[:q, :q] *= inv_delta
            new_J[:q, :q] += self.J
            new_J[:q, q] = inv_delta * z
            new_J[q, :q] = inv_delta * z
        new_J[q, q] = inv_delta

        self.C = new_C
        # Every block above is constructed symmetrically.  Avoid another
        # O(q^2) allocation and copy on the common append path.
        self.J = new_J
        self.stats.appends += 1

    def _store_snapshot(self, time: int, rows: Array) -> SchurSnapshot:
        block = np.array(rows, dtype=np.float64, copy=True).reshape((-1, self.d))
        block.setflags(write=False)
        snapshot = SchurSnapshot(int(time), block)
        self.snapshots.append(snapshot)
        self.stats.snapshot_rows += int(block.shape[0])
        return snapshot

    def _cancel(
        self, row: Array, z: Array, time: int, zero_tolerance: float
    ) -> SchurSnapshot:
        self.stats.cancels += 1
        z_norm = float(np.linalg.norm(z))
        if self.row_count == 0 or z_norm <= zero_tolerance:
            self.stats.zero_certificate_cancels += 1
            return self._store_snapshot(time, row.reshape(1, -1))

        w = z / z_norm
        e1 = np.zeros(self.row_count, dtype=np.float64)
        e1[0] = 1.0

        # Apply H = I - 2 u u^T without materializing a dense Householder
        # matrix.  H is chosen so that its first row is w^T.
        if np.array_equal(w, e1):
            rotated_C = self.C.copy()
            rotated_J = self.J.copy()
        else:
            direction = e1 - w
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm == 0.0:
                rotated_C = self.C.copy()
                rotated_J = self.J.copy()
            else:
                u = direction / direction_norm
                uTC = u @ self.C
                rotated_C = self.C - 2.0 * np.outer(u, uTC)

                Ju = self.J @ u
                uJu = float(u @ Ju)
                companion = -2.0 * Ju + 2.0 * uJu * u
                rotated_J = self.J + np.outer(u, companion)
                rotated_J += np.outer(companion, u)

        extracted = rotated_C[0].copy()
        snapshot = self._store_snapshot(
            time, np.vstack((row.reshape(1, -1), extracted.reshape(1, -1)))
        )

        if self.row_count == 1:
            self.C = np.empty((0, self.d), dtype=np.float64)
            self.J = np.empty((0, 0), dtype=np.float64)
            return snapshot

        rho = float(rotated_J[0, 0])
        g = rotated_J[1:, 0]
        if rho <= 0.0 or not np.isfinite(rho):
            raise np.linalg.LinAlgError(
                "floating-point principal-block inverse lost positivity"
            )
        self.C = rotated_C[1:].copy()
        self.J = self._symmetrize(
            rotated_J[1:, 1:] - np.outer(g, g) / rho
        )
        return snapshot

    def _batch_reduce(self) -> None:
        if self.row_count != 2 * self.width:
            return

        _, singular_values, Vt = np.linalg.svd(self.C, full_matrices=False)
        self.stats.batch_reductions += 1
        self.stats.svd_calls += 1

        if self.width < singular_values.size:
            mu = float(singular_values[self.width] ** 2)
        else:
            mu = 0.0
        retained_sq = np.maximum(
            singular_values[: self.width] ** 2 - mu, 0.0
        )
        retained = np.sqrt(retained_sq)
        self.C = retained[:, None] * Vt[: self.width]

        diagonal = self.threshold - retained_sq
        if np.any(diagonal <= 0.0):
            # This should not occur in exact arithmetic because Batch FD is a
            # contraction.  Rebuilding from the Gram matrix gives a clearer
            # failure mode than silently continuing with a non-SPD barrier.
            self._rebuild_inverse()
        else:
            self.J = np.diag(1.0 / diagonal)

    def _rebuild_inverse(self) -> None:
        q = self.row_count
        if q == 0:
            self.J = np.empty((0, 0), dtype=np.float64)
            return
        barrier = self.threshold * np.eye(q) - self.C @ self.C.T
        barrier = self._symmetrize(barrier)
        np.linalg.cholesky(barrier)
        self.J = self._symmetrize(np.linalg.inv(barrier))

    def _update_vector(
        self, vector: Array, row_norm_sq: float, time: int | None
    ) -> list[SchurSnapshot]:
        """Update from an already validated vector and its squared norm."""

        if time is None:
            time = self.stats.updates + 1
        q = self.row_count
        if q:
            b = self.C @ vector
            z = self.J @ b
            coupling = float(b @ z)
        else:
            z = np.empty(0, dtype=np.float64)
            coupling = 0.0
        delta = self.threshold - row_norm_sq - coupling
        tolerance = self._decision_tolerance(row_norm_sq, coupling)

        before = len(self.snapshots)
        if delta > tolerance:
            self._append(vector, z, delta)
        else:
            self._cancel(vector, z, int(time), tolerance)

        self._batch_reduce()
        self.stats.updates += 1
        if self.audit:
            self.check_invariants()
        return self.snapshots[before:]

    def update(
        self, row: npt.ArrayLike, time: int | None = None
    ) -> list[SchurSnapshot]:
        """Process one row and return any snapshot created by this update."""

        vector = np.asarray(row, dtype=np.float64).reshape(-1)
        if vector.size != self.d:
            raise ValueError(f"expected a row of length {self.d}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("row contains a non-finite value")
        return self._update_vector(vector, float(vector @ vector), time)

    def fit(self, rows: npt.ArrayLike, *, start_time: int | None = None) -> None:
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != self.d:
            raise ValueError(f"expected a matrix with {self.d} columns")
        first_time = self.stats.updates + 1 if start_time is None else start_time
        for offset, row in enumerate(matrix):
            self.update(row, first_time + offset)

    def snapshot_rows(
        self, start_time: int | None = None, end_time: int | None = None
    ) -> Array:
        blocks = []
        for snapshot in self.snapshots:
            if start_time is not None and snapshot.time < start_time:
                continue
            if end_time is not None and snapshot.time > end_time:
                continue
            blocks.append(snapshot.rows)
        if not blocks:
            return np.empty((0, self.d), dtype=np.float64)
        return np.vstack(blocks)

    def sketch(
        self,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        include_residual: bool = True,
    ) -> Array:
        blocks: list[Array] = []
        snapshots = self.snapshot_rows(start_time, end_time)
        if snapshots.shape[0]:
            blocks.append(snapshots)
        if include_residual and self.row_count:
            blocks.append(self.C.copy())
        if not blocks:
            return np.empty((0, self.d), dtype=np.float64)
        return np.vstack(blocks)

    def size(self) -> int:
        """Number of stored float scalars (timestamps excluded)."""

        return int(
            self.C.size
            + self.J.size
            + sum(snapshot.rows.size for snapshot in self.snapshots)
        )

    def stored_rows(self) -> int:
        return int(
            self.row_count
            + sum(snapshot.rows.shape[0] for snapshot in self.snapshots)
        )

    def release_snapshot(self, snapshot: SchurSnapshot) -> None:
        """Drop a snapshot after an external coordinator has taken ownership."""

        if not self.snapshots or self.snapshots[-1] is not snapshot:
            raise ValueError("only the most recently emitted snapshot can be released")
        self.snapshots.pop()

    def check_invariants(self) -> None:
        if self.row_count >= 2 * self.width:
            raise AssertionError("residual row bound violated")
        if self.J.shape != (self.row_count, self.row_count):
            raise AssertionError("inverse dimension mismatch")
        if self.row_count == 0:
            return
        barrier = self.threshold * np.eye(self.row_count) - self.C @ self.C.T
        smallest = float(np.linalg.eigvalsh(self._symmetrize(barrier))[0])
        tolerance = 1024.0 * np.finfo(np.float64).eps * max(
            self.threshold, 1.0
        )
        if smallest <= -tolerance:
            raise AssertionError(
                f"spectral barrier violated: lambda_min={smallest}"
            )
        inverse_error = float(
            np.linalg.norm(barrier @ self.J - np.eye(self.row_count), ord=2)
        )
        if inverse_error > 1e-6 * max(1.0, np.linalg.cond(barrier)):
            raise AssertionError(
                f"barrier inverse drifted: residual={inverse_error}"
            )


@dataclass
class SchurWindowLevel:
    """One persistent Schur core and its externally owned block queue."""

    core: SchurFdCore
    snapshots: deque[SchurSnapshot]
    evicted_until: int = 0


class MultiLevelWindowSchurFD:
    """Aero-compatible multilevel Schur tracker for a sequence window."""

    def __init__(
        self,
        window_size: int,
        norm_sq_upper: float,
        d: int,
        ell: int,
        *,
        block_cap: int | None = None,
        audit: bool = False,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if d < 1:
            raise ValueError("d must be positive")
        if ell < 1:
            raise ValueError("ell must be positive")
        if (
            not np.isfinite(norm_sq_upper)
            or float(norm_sq_upper) < 1.0
        ):
            raise ValueError("norm_sq_upper must be finite and at least one")

        self.window_size = int(window_size)
        self.norm_sq_upper = float(norm_sq_upper)
        self.d = int(d)
        self.ell = int(ell)
        self.width = min(self.d, self.ell)
        self.block_cap = (
            4 * self.ell if block_cap is None else int(block_cap)
        )
        if self.block_cap < 1:
            raise ValueError("block_cap must be positive")
        self.audit = bool(audit)
        self.time = 0

        self.level_count = int(np.floor(np.log2(self.norm_sq_upper))) + 1
        self.levels: dict[int, SchurWindowLevel] = {}
        for level in range(self.level_count):
            threshold = (2.0**level) * self.window_size / self.ell
            self.levels[level] = SchurWindowLevel(
                core=SchurFdCore(
                    self.d,
                    self.width,
                    threshold,
                    audit=self.audit,
                ),
                snapshots=deque(),
            )

    def _window_start(self) -> int:
        return max(1, self.time - self.window_size + 1)

    def _prune_level(self, state: SchurWindowLevel) -> None:
        window_start = self._window_start()
        while (
            state.snapshots
            and state.snapshots[0].time < window_start
        ):
            state.snapshots.popleft()
        while len(state.snapshots) > self.block_cap:
            evicted = state.snapshots.popleft()
            state.evicted_until = max(
                state.evicted_until, evicted.time
            )

    def update_at(
        self, row: npt.ArrayLike, time: int
    ) -> list[tuple[int, SchurSnapshot]]:
        if int(time) != self.time + 1:
            raise ValueError("time must advance by exactly one")
        vector = np.asarray(row, dtype=np.float64).reshape(-1)
        if vector.size != self.d:
            raise ValueError(f"expected a row of length {self.d}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("row contains a non-finite value")
        row_norm_sq = float(vector @ vector)
        self.time = int(time)
        emitted: list[tuple[int, SchurSnapshot]] = []
        for level, state in self.levels.items():
            created = state.core._update_vector(
                vector, row_norm_sq, self.time
            )
            # The window wrapper owns timestamped blocks; the core only needs
            # its live residual for future Schur updates.
            for snapshot in reversed(created):
                state.core.release_snapshot(snapshot)
            state.snapshots.extend(created)
            emitted.extend((level, snapshot) for snapshot in created)
            self._prune_level(state)
        return emitted

    def update(
        self, row: npt.ArrayLike
    ) -> list[tuple[int, SchurSnapshot]]:
        return self.update_at(row, self.time + 1)

    fit = update

    def selected_level(self) -> int:
        if self.time == 0:
            return 0
        window_start = self._window_start()
        for level, state in self.levels.items():
            if state.evicted_until < window_start:
                return level
        raise RuntimeError(
            "no Schur level covers the current window; increase R or the "
            "cancellation-block cap"
        )

    def get(self) -> tuple[Array, None, None, None]:
        if self.time == 0:
            return np.empty((0, self.d), dtype=np.float64), None, None, None
        state = self.levels[self.selected_level()]
        blocks: list[Array] = [
            snapshot.rows for snapshot in state.snapshots
        ]
        if state.core.row_count:
            blocks.append(state.core.C.copy())
        return stack_nonempty(blocks, self.d), None, None, None

    def row_num(self) -> int:
        return sum(
            state.core.row_count
            + sum(snapshot.rows.shape[0] for snapshot in state.snapshots)
            for state in self.levels.values()
        )

    get_size = row_num

    def size(self) -> int:
        return sum(
            state.core.C.size
            + state.core.J.size
            + sum(snapshot.rows.size for snapshot in state.snapshots)
            for state in self.levels.values()
        )

    def aggregate_stats(self) -> SchurStats:
        result = SchurStats()
        for state in self.levels.values():
            for field in result.__dataclass_fields__:
                setattr(
                    result,
                    field,
                    getattr(result, field) + getattr(state.core.stats, field),
                )
        return result


class FixedWindowSchurFD:
    """Two-epoch wrapper for a normalized, sequence-based fixed window."""

    def __init__(
        self,
        window_size: int,
        d: int,
        ell: int,
        *,
        width: int | None = None,
        threshold: float | None = None,
        audit: bool = False,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if ell < 1:
            raise ValueError("ell must be positive")
        self.window_size = int(window_size)
        self.d = int(d)
        self.ell = int(ell)
        self.width = min(self.d, 4 * self.ell) if width is None else int(width)
        self.threshold = (
            self.window_size / (2.0 * self.ell)
            if threshold is None
            else float(threshold)
        )
        self.audit = bool(audit)
        self.time = 0
        self.epochs: dict[int, SchurFdCore] = {}

    def _epoch_start(self, time: int) -> int:
        return 1 + self.window_size * ((time - 1) // self.window_size)

    def advance(self, time: int) -> None:
        if time < self.time:
            raise ValueError("time must be monotone")
        self.time = int(time)
        start = self._epoch_start(self.time)
        if start not in self.epochs:
            self.epochs[start] = SchurFdCore(
                self.d,
                self.width,
                self.threshold,
                audit=self.audit,
            )
        expired = [
            epoch_start
            for epoch_start in self.epochs
            if self.time > epoch_start + 2 * self.window_size - 1
        ]
        for epoch_start in expired:
            del self.epochs[epoch_start]

    def update_at(
        self, row: npt.ArrayLike, time: int
    ) -> list[tuple[int, SchurSnapshot]]:
        self.advance(time)
        emitted: list[tuple[int, SchurSnapshot]] = []
        for epoch_start, core in self.epochs.items():
            if epoch_start <= time <= epoch_start + 2 * self.window_size - 1:
                emitted.extend(
                    (epoch_start, snapshot)
                    for snapshot in core.update(row, time)
                )
        return emitted

    def update(self, row: npt.ArrayLike) -> list[tuple[int, SchurSnapshot]]:
        return self.update_at(row, self.time + 1)

    fit = update

    def get(self) -> tuple[Array, None, None, None]:
        if self.time == 0:
            return np.empty((0, self.d), dtype=np.float64), None, None, None
        start_time = max(1, self.time - self.window_size + 1)
        epoch_start = self._epoch_start(start_time)
        core = self.epochs[epoch_start]
        result = core.sketch(
            start_time=start_time,
            end_time=self.time,
            include_residual=True,
        )
        return result, None, None, None

    def snapshot_sketch(self) -> Array:
        """Return the coordinator-visible snapshot part of the current window."""

        if self.time == 0:
            return np.empty((0, self.d), dtype=np.float64)
        start_time = max(1, self.time - self.window_size + 1)
        epoch_start = self._epoch_start(start_time)
        return self.epochs[epoch_start].sketch(
            start_time=start_time,
            end_time=self.time,
            include_residual=False,
        )

    def get_size(self) -> int:
        return sum(core.size() for core in self.epochs.values())

    def row_num(self) -> int:
        return sum(core.stored_rows() for core in self.epochs.values())

    def aggregate_stats(self) -> SchurStats:
        result = SchurStats()
        for core in self.epochs.values():
            for field in result.__dataclass_fields__:
                setattr(result, field, getattr(result, field) + getattr(core.stats, field))
        return result



class EnergyAdaptiveSchurATTP:
    """Deterministic persistent prefix sketch for arbitrary row norms.

    The Aero-compatible outer policy tracks cumulative squared Frobenius
    energy, but a Schur barrier cannot be changed after every update without
    invalidating its maintained inverse.  Instead, this wrapper keeps the
    threshold fixed inside an energy epoch.  When the cumulative energy
    doubles, it discards the old residual and starts an empty core at the new
    threshold.  Retaining completed residuals remains available only as an
    explicit legacy option.

    Historical queries return only blocks whose timestamps are no later than
    the requested time.  In particular, they omit the mutable active residual.
    In exact arithmetic this gives an ``O(1) / ell`` relative covariance
    guarantee without a known horizon or unit-row assumption.
    """

    def __init__(
        self,
        d: int,
        ell: int,
        *,
        retain_sealed_residual: bool = False,
        audit: bool = False,
    ) -> None:
        if d < 1:
            raise ValueError("d must be positive")
        if ell < 1:
            raise ValueError("ell must be positive")
        self.d = int(d)
        self.ell = int(ell)
        self.width = min(self.d, self.ell)
        self.retain_sealed_residual = bool(retain_sealed_residual)
        self.audit = bool(audit)

        self.time = 0
        self.energy = 0.0
        self.epoch_energy: float | None = None
        self.epoch_count = 0
        self.core: SchurFdCore | None = None
        self.snapshots: list[SchurSnapshot] = []
        self.sealed_residual_rows = 0
        self.discarded_residual_rows = 0
        self._completed_stats = SchurStats()

    @staticmethod
    def _immutable_snapshot(time: int, rows: Array) -> SchurSnapshot:
        block = np.array(rows, dtype=np.float64, copy=True)
        block.setflags(write=False)
        return SchurSnapshot(int(time), block)

    def _accumulate_stats(self, stats: SchurStats) -> None:
        for field in self._completed_stats.__dataclass_fields__:
            setattr(
                self._completed_stats,
                field,
                getattr(self._completed_stats, field) + getattr(stats, field),
            )

    def _start_epoch(self, energy: float) -> None:
        self.epoch_energy = float(energy)
        self.epoch_count += 1
        self.core = SchurFdCore(
            self.d,
            self.width,
            self.epoch_energy / self.ell,
            audit=self.audit,
        )

    def _seal_epoch(self, time: int) -> list[SchurSnapshot]:
        if self.core is None:
            return []
        sealed: list[SchurSnapshot] = []
        if self.core.row_count and self.retain_sealed_residual:
            snapshot = self._immutable_snapshot(time, self.core.C)
            self.snapshots.append(snapshot)
            self.sealed_residual_rows += snapshot.rows.shape[0]
            sealed.append(snapshot)
        elif self.core.row_count:
            self.discarded_residual_rows += self.core.row_count
        self._accumulate_stats(self.core.stats)
        self.core = None
        return sealed

    def update_at(
        self, row: npt.ArrayLike, time: int
    ) -> list[SchurSnapshot]:
        if int(time) != self.time + 1:
            raise ValueError("time must advance by exactly one")
        vector = np.asarray(row, dtype=np.float64).reshape(-1)
        if vector.size != self.d:
            raise ValueError(f"expected a row of length {self.d}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("row contains a non-finite value")
        row_energy = float(vector @ vector)
        new_energy = self.energy + row_energy
        if not np.isfinite(new_energy):
            raise ValueError("cumulative row energy is not finite")

        self.time = int(time)
        self.energy = new_energy
        emitted: list[SchurSnapshot] = []

        # Zero rows change neither the covariance ledger nor the energy scale.
        # Their timestamps still advance so historical-query semantics remain
        # unchanged.
        if row_energy == 0.0:
            return emitted

        if self.core is None:
            self._start_epoch(new_energy)
        elif (
            self.epoch_energy is not None
            and new_energy >= 2.0 * self.epoch_energy
        ):
            # The old residual contains rows only through the preceding
            # update.  Keeping that support endpoint exact is essential when
            # an external distributed wrapper maps local order to global
            # forward/reverse timestamps.
            emitted.extend(self._seal_epoch(self.time - 1))
            self._start_epoch(new_energy)

        assert self.core is not None
        created = self.core._update_vector(
            vector, row_energy, self.time
        )
        # The wrapper owns all persistent blocks.  Releasing them from the
        # active core prevents double-counting state while leaving the
        # immutable arrays valid.
        for snapshot in reversed(created):
            self.core.release_snapshot(snapshot)
        self.snapshots.extend(created)
        emitted.extend(created)
        return emitted

    def update(self, row: npt.ArrayLike) -> list[SchurSnapshot]:
        return self.update_at(row, self.time + 1)

    fit = update

    def release_snapshot(self, snapshot: SchurSnapshot) -> None:
        """Release an immutable block after an external owner acknowledges it."""

        for index in range(len(self.snapshots) - 1, -1, -1):
            if self.snapshots[index] is snapshot:
                del self.snapshots[index]
                return
        raise ValueError("snapshot is not owned by this ATTP wrapper")

    def get(self, query_time: int | None = None) -> Array:
        if query_time is None:
            query_time = self.time
        if query_time < 0 or query_time > self.time:
            raise ValueError("query_time must lie in [0, current time]")
        blocks = [
            snapshot.rows
            for snapshot in self.snapshots
            if snapshot.time <= query_time
        ]
        return stack_nonempty(blocks, self.d)

    def row_num(self) -> int:
        residual_rows = 0 if self.core is None else self.core.row_count
        return int(
            residual_rows
            + sum(snapshot.rows.shape[0] for snapshot in self.snapshots)
        )

    def size(self) -> int:
        active_size = 0 if self.core is None else self.core.size()
        return int(
            active_size
            + sum(snapshot.rows.size for snapshot in self.snapshots)
        )

    get_size = size

    def aggregate_stats(self) -> SchurStats:
        result = SchurStats(
            **{
                field: getattr(self._completed_stats, field)
                for field in self._completed_stats.__dataclass_fields__
            }
        )
        if self.core is not None:
            for field in result.__dataclass_fields__:
                setattr(
                    result,
                    field,
                    getattr(result, field) + getattr(self.core.stats, field),
                )
        return result


def covariance(rows: npt.ArrayLike, d: int | None = None) -> Array:
    """Return ``rows.T @ rows`` while handling an empty row stack."""

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.size == 0:
        if d is None:
            if matrix.ndim != 2:
                raise ValueError("d is required for a non-matrix empty input")
            d = int(matrix.shape[1])
        return np.zeros((d, d), dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix.T @ matrix


def stack_nonempty(blocks: Iterable[Array], d: int) -> Array:
    kept = [np.asarray(block) for block in blocks if np.asarray(block).shape[0]]
    if not kept:
        return np.empty((0, d), dtype=np.float64)
    return np.vstack(kept)

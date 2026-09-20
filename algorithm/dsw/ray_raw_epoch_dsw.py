"""Matched raw-epoch Ray protocol for distributed sequence windows.

Sites retain one epoch of original timestamped rows.  A persistent prefix
emitter runs online in forward order; at the boundary the same emitter runs on
the original rows in reverse order.  The coordinator combines the current
forward prefix with a historical prefix of the preceding reversed epoch.
"""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
import ray

from algorithm.distributed.ray_runtime import initialize_ray
from base.fd import FdDump, SlowFdDump, reduce_covariance_fd
from base.schur_fd import SchurFdCore, SchurStats


EncodedBlock = tuple[str, np.ndarray, np.ndarray]
WireBlock = tuple[int, str, np.ndarray, np.ndarray]


def _communication_floats(block: EncodedBlock | WireBlock) -> int:
    if len(block) == 3:
        return int(block[1].size + block[2].size)
    return int(block[2].size + block[3].size)


def _storage_rows(block: EncodedBlock | WireBlock) -> int:
    if len(block) == 3:
        encoding, first, _ = block
    else:
        _, encoding, first, _ = block
    if encoding == "rows":
        return int(first.shape[0])
    if encoding == "aero":
        return int(2 * first.shape[1])
    raise ValueError(f"unknown block encoding {encoding!r}")


def _block_covariance(block: EncodedBlock | WireBlock, d: int) -> np.ndarray:
    if len(block) == 3:
        encoding, first, second = block
    else:
        _, encoding, first, second = block
    if encoding == "rows":
        rows = np.asarray(first, dtype=np.float64).reshape((-1, d))
        return rows.T @ rows
    if encoding == "aero":
        Z = np.asarray(first, dtype=np.float64)
        ZCC = np.asarray(second, dtype=np.float64)
        product = Z @ ZCC
        return product + product.T - Z @ (ZCC @ Z) @ Z.T
    raise ValueError(f"unknown block encoding {encoding!r}")


def _row_block(rows: npt.ArrayLike, d: int) -> EncodedBlock:
    matrix = np.asarray(rows, dtype=np.float64).reshape((-1, d))
    return (
        "rows",
        np.array(matrix, dtype=np.float64, copy=True),
        np.empty((0, d), dtype=np.float64),
    )


def _wire(endpoint: int, block: EncodedBlock) -> WireBlock:
    encoding, first, second = block
    return (
        int(endpoint),
        str(encoding),
        np.array(first, dtype=np.float64, copy=True),
        np.array(second, dtype=np.float64, copy=True),
    )


def _copy_wire(block: WireBlock) -> WireBlock:
    endpoint, encoding, first, second = block
    return (
        int(endpoint),
        str(encoding),
        np.array(first, dtype=np.float64, copy=True),
        np.array(second, dtype=np.float64, copy=True),
    )


class _PersistentDumpEmitter:
    """No-seal dynamic-threshold wrapper for SlowFD or AeroSketch."""

    def __init__(self, method: str, d: int, ell: int):
        if method not in {"da2", "aero"}:
            raise ValueError("method must be 'da2' or 'aero'")
        self.method = str(method)
        self.d = int(d)
        self.ell = int(ell)
        self.energy = 0.0
        self.sketch: SlowFdDump | FdDump | None = self._new_sketch()
        self.completed = {
            "svd_calls": 0,
            "power_calls": 0,
            "simultaneous_calls": 0,
        }
        self.snapshot_rows = 0
        self.discarded_residual_rows = 0

    def _new_sketch(self) -> SlowFdDump | FdDump:
        if self.method == "da2":
            return SlowFdDump(self.d, self.ell, fast_fd=False)
        return FdDump(self.d, self.ell, fast_fd=True)

    def _accumulate_sketch(self) -> None:
        if self.sketch is None:
            return
        self.completed["svd_calls"] += int(self.sketch.svd_calls)
        if isinstance(self.sketch, FdDump):
            self.completed["power_calls"] += int(self.sketch.power_calls)
            self.completed["simultaneous_calls"] += int(
                self.sketch.simultaneous_calls
            )
        self.sketch = None

    def _discard_residual(self) -> None:
        if self.sketch is not None:
            self.discarded_residual_rows += int(self.sketch.row)
        self._accumulate_sketch()

    def update(self, row: npt.ArrayLike, current_time: int) -> list[WireBlock]:
        vector = np.asarray(row, dtype=np.float64).reshape(-1)
        row_energy = float(vector @ vector)
        new_energy = self.energy + row_energy
        if not np.isfinite(new_energy):
            raise ValueError("cumulative row energy is not finite")
        self.energy = new_energy
        if row_energy == 0.0:
            return []

        emitted: list[WireBlock] = []
        if self.sketch is None:
            raise RuntimeError("cannot update a finalized dump emitter")

        assert self.sketch is not None
        self.sketch.fit(vector.reshape(1, self.d))
        threshold = max(new_energy / self.ell, np.finfo(np.float64).tiny)
        if self.method == "da2":
            assert isinstance(self.sketch, SlowFdDump)
            rows = self.sketch.dump(threshold)
            if rows.shape[0]:
                block = _row_block(rows, self.d)
                emitted.append(_wire(current_time, block))
                self.snapshot_rows += _storage_rows(block)
        else:
            assert isinstance(self.sketch, FdDump)
            Z, ZCC = self.sketch.dump(threshold)
            if Z.shape[1]:
                block = (
                    "aero",
                    np.array(Z, dtype=np.float64, copy=True),
                    np.array(ZCC, dtype=np.float64, copy=True),
                )
                emitted.append(_wire(current_time, block))
                self.snapshot_rows += _storage_rows(block)
        return emitted

    def finalize(self) -> list[WireBlock]:
        self._discard_residual()
        return []

    def counters(self) -> dict[str, int]:
        result = dict(self.completed)
        if self.sketch is not None:
            result["svd_calls"] += int(self.sketch.svd_calls)
            if isinstance(self.sketch, FdDump):
                result["power_calls"] += int(self.sketch.power_calls)
                result["simultaneous_calls"] += int(
                    self.sketch.simultaneous_calls
                )
        result["snapshot_rows"] = int(self.snapshot_rows)
        result["sealed_residual_rows"] = 0
        result["discarded_residual_rows"] = int(
            self.discarded_residual_rows
        )
        result["energy_epochs"] = 0
        return result

    def row_num(self) -> int:
        return 0 if self.sketch is None else int(self.sketch.row)


class _PersistentSchurEmitter:
    """No-seal energy-doubling prefix wrapper for the Schur core."""

    def __init__(self, d: int, ell: int, audit: bool):
        self.d = int(d)
        self.ell = int(ell)
        self.width = min(self.d, self.ell)
        self.audit = bool(audit)
        self.energy = 0.0
        self.anchor: float | None = None
        self.core: SchurFdCore | None = None
        self.completed = SchurStats()
        self.discarded_residual_rows = 0
        self.energy_epochs = 0

    def _accumulate_core(self) -> None:
        if self.core is None:
            return
        for field in self.completed.__dataclass_fields__:
            setattr(
                self.completed,
                field,
                getattr(self.completed, field)
                + getattr(self.core.stats, field),
            )
        self.core = None

    def _start_epoch(self, anchor: float) -> None:
        self.anchor = float(anchor)
        self.energy_epochs += 1
        self.core = SchurFdCore(
            self.d,
            self.width,
            max(self.anchor / self.ell, np.finfo(np.float64).tiny),
            audit=self.audit,
        )

    def _discard_residual(self) -> None:
        if self.core is not None:
            self.discarded_residual_rows += self.core.row_count
        self._accumulate_core()

    def update(self, row: npt.ArrayLike, current_time: int) -> list[WireBlock]:
        vector = np.asarray(row, dtype=np.float64).reshape(-1)
        row_energy = float(vector @ vector)
        new_energy = self.energy + row_energy
        if not np.isfinite(new_energy):
            raise ValueError("cumulative row energy is not finite")
        self.energy = new_energy
        if row_energy == 0.0:
            return []

        emitted: list[WireBlock] = []
        if self.core is None:
            self._start_epoch(new_energy)
        elif self.anchor is not None and new_energy >= 2.0 * self.anchor:
            self._discard_residual()
            self._start_epoch(new_energy)

        assert self.core is not None
        created = self.core.update(vector, int(current_time))
        for snapshot in created:
            emitted.append(
                _wire(snapshot.time, _row_block(snapshot.rows, self.d))
            )
        for snapshot in reversed(created):
            self.core.release_snapshot(snapshot)
        return emitted

    def finalize(self) -> list[WireBlock]:
        self._discard_residual()
        return []

    def counters(self) -> dict[str, int]:
        stats = SchurStats(
            **{
                field: getattr(self.completed, field)
                for field in self.completed.__dataclass_fields__
            }
        )
        if self.core is not None:
            for field in stats.__dataclass_fields__:
                setattr(
                    stats,
                    field,
                    getattr(stats, field) + getattr(self.core.stats, field),
                )
        return {
            "svd_calls": int(stats.svd_calls),
            "power_calls": 0,
            "simultaneous_calls": 0,
            "snapshot_rows": int(stats.snapshot_rows),
            "sealed_residual_rows": 0,
            "discarded_residual_rows": int(self.discarded_residual_rows),
            "energy_epochs": int(self.energy_epochs),
        }

    def row_num(self) -> int:
        return 0 if self.core is None else int(self.core.row_count)


def _new_emitter(
    method: str,
    d: int,
    ell: int,
    audit: bool,
) -> _PersistentDumpEmitter | _PersistentSchurEmitter:
    if method in {"da2", "aero"}:
        return _PersistentDumpEmitter(method, d, ell)
    if method == "schur":
        return _PersistentSchurEmitter(d, ell, audit)
    raise ValueError("method must be 'da2', 'aero', or 'schur'")


@ray.remote(num_cpus=1)
class RayRawEpochWindowCoordinator:
    def __init__(self, d: int, ell: int, window_size: int, sites: int):
        self.d = int(d)
        self.ell = int(ell)
        self.window_size = int(window_size)
        self.num_sites = int(sites)
        self.time = 0
        self.communication_cost = 0
        self.processing_time_ns = 0
        self.query_svd_calls = 0
        self.forward: dict[tuple[int, int], list[WireBlock]] = {}
        self.reverse: dict[tuple[int, int], list[WireBlock]] = {}
        self._forward_covariance: dict[tuple[int, int], np.ndarray] = {}
        self._forward_cached_blocks: dict[tuple[int, int], int] = {}
        self._reverse_covariance: dict[tuple[int, int], np.ndarray] = {}
        self._reverse_cached_cutoff: dict[
            tuple[int, int], int
        ] = {}

    def add_forward(
        self,
        site_id: int,
        epoch_id: int,
        payload: list[WireBlock],
        current_time: int,
    ) -> None:
        start = time.process_time_ns()
        key = (int(site_id), int(epoch_id))
        copied = [_copy_wire(block) for block in payload]
        self.forward.setdefault(key, []).extend(copied)
        self.communication_cost += sum(
            _communication_floats(block) for block in copied
        )
        self.time = max(self.time, int(current_time))
        self.processing_time_ns += time.process_time_ns() - start

    def install_reverse(
        self,
        site_id: int,
        epoch_id: int,
        payload: list[WireBlock],
        current_time: int,
    ) -> None:
        start = time.process_time_ns()
        key = (int(site_id), int(epoch_id))
        copied = [_copy_wire(block) for block in payload]
        self.reverse[key] = copied
        self._reverse_covariance.pop(key, None)
        self._reverse_cached_cutoff.pop(key, None)
        self.forward.pop(key, None)
        self._forward_covariance.pop(key, None)
        self._forward_cached_blocks.pop(key, None)
        self.communication_cost += sum(
            _communication_floats(block) for block in copied
        )
        self.time = max(self.time, int(current_time))
        self.processing_time_ns += time.process_time_ns() - start

    def _advance(self, current_time: int) -> None:
        self.time = max(self.time, int(current_time))
        current_epoch = self.time // self.window_size
        previous_epoch = current_epoch - 1
        for key in list(self.forward):
            if key[1] != current_epoch:
                del self.forward[key]
                self._forward_covariance.pop(key, None)
                self._forward_cached_blocks.pop(key, None)
        for key in list(self.reverse):
            if key[1] != previous_epoch:
                del self.reverse[key]
                self._reverse_covariance.pop(key, None)
                self._reverse_cached_cutoff.pop(key, None)

    def finish_epoch(self, current_time: int | None = None) -> None:
        start = time.process_time_ns()
        self.communication_cost += self.num_sites
        if current_time is not None:
            self._advance(current_time)
        self.processing_time_ns += time.process_time_ns() - start

    def tick(self, current_time: int) -> None:
        start = time.process_time_ns()
        self._advance(current_time)
        self.processing_time_ns += time.process_time_ns() - start

    def _active_covariance(self) -> np.ndarray:
        """Return the active covariance using query-only lazy caches."""

        covariance = np.zeros((self.d, self.d), dtype=np.float64)
        if self.time == 0:
            return covariance
        current_epoch = self.time // self.window_size
        previous_epoch = current_epoch - 1
        cutoff = self.time - self.window_size
        for key, blocks in self.forward.items():
            if key[1] != current_epoch:
                continue
            cached = self._forward_covariance.setdefault(
                key, np.zeros((self.d, self.d), dtype=np.float64)
            )
            cached_blocks = self._forward_cached_blocks.get(key, 0)
            for block in blocks[cached_blocks:]:
                cached += _block_covariance(block, self.d)
            self._forward_cached_blocks[key] = len(blocks)
            covariance += cached
        for key, blocks in self.reverse.items():
            if key[1] != previous_epoch:
                continue
            cached = self._reverse_covariance.get(key)
            cached_cutoff = self._reverse_cached_cutoff.get(key)
            if cached is None or cached_cutoff is None:
                cached = np.zeros((self.d, self.d), dtype=np.float64)
                for block in blocks:
                    if block[0] > cutoff:
                        cached += _block_covariance(block, self.d)
            elif cutoff > cached_cutoff:
                for block in blocks:
                    if cached_cutoff < block[0] <= cutoff:
                        cached -= _block_covariance(block, self.d)
            self._reverse_covariance[key] = cached
            self._reverse_cached_cutoff[key] = cutoff
            covariance += cached
        return 0.5 * (covariance + covariance.T)

    def _direct_active_covariance(self) -> np.ndarray:
        """Reference implementation for cache-equivalence audits."""

        covariance = np.zeros((self.d, self.d), dtype=np.float64)
        if self.time == 0:
            return covariance
        current_epoch = self.time // self.window_size
        previous_epoch = current_epoch - 1
        cutoff = self.time - self.window_size
        for (_, epoch_id), blocks in self.forward.items():
            if epoch_id == current_epoch:
                for block in blocks:
                    covariance += _block_covariance(block, self.d)
        for (_, epoch_id), blocks in self.reverse.items():
            if epoch_id == previous_epoch:
                for block in blocks:
                    if block[0] > cutoff:
                        covariance += _block_covariance(block, self.d)
        return 0.5 * (covariance + covariance.T)

    def audit_covariance_cache(self, current_time: int | None = None) -> float:
        if current_time is not None:
            self._advance(current_time)
        cached = self._active_covariance()
        direct = self._direct_active_covariance()
        return float(np.max(np.abs(cached - direct), initial=0.0))

    def get_covariance(self, current_time: int | None = None) -> np.ndarray:
        if current_time is not None:
            self._advance(current_time)
        return self._active_covariance()

    def get_result(self, current_time: int | None = None) -> np.ndarray:
        start = time.process_time_ns()
        if current_time is not None:
            self._advance(current_time)
        result = reduce_covariance_fd(
            self._active_covariance(), min(self.d, 2 * self.ell)
        )
        self.query_svd_calls += 1
        self.processing_time_ns += time.process_time_ns() - start
        return result

    def get_result_and_stats(
        self, current_time: int
    ) -> tuple[np.ndarray, dict]:
        return self.get_result(current_time), self.get_stats()

    def get_stats(self, current_time: int | None = None) -> dict:
        if current_time is not None:
            self._advance(current_time)
        blocks = [
            block
            for archive in (self.forward, self.reverse)
            for values in archive.values()
            for block in values
        ]
        return {
            "row_num": int(sum(_storage_rows(block) for block in blocks)),
            "node_id": ray.get_runtime_context().get_node_id(),
            "processing_time": self.processing_time_ns / 1e6,
            "processing_time_ns": int(self.processing_time_ns),
            "communication_cost": int(self.communication_cost),
            "svd_calls": int(self.query_svd_calls),
            "pending_reverse_rows": int(
                sum(
                    _storage_rows(block)
                    for values in self.reverse.values()
                    for block in values
                )
            ),
            "live_forward_archives": len(self.forward),
            "live_reverse_archives": len(self.reverse),
        }


@ray.remote(num_cpus=1)
class RayRawEpochWindowSite:
    def __init__(
        self,
        site_id: int,
        d: int,
        ell: int,
        window_size: int,
        method: str,
        coordinator,
        audit: bool,
        seed: int,
    ):
        self.site_id = int(site_id)
        self.d = int(d)
        self.ell = int(ell)
        self.window_size = int(window_size)
        self.method = str(method)
        self.coordinator = coordinator
        self.audit = bool(audit)
        np.random.seed(int(seed) + self.site_id)
        self.forward_epoch = 0
        self.forward = _new_emitter(method, d, ell, self.audit)
        self.raw_buffer: list[tuple[int, np.ndarray]] = []
        self.max_raw_buffer_rows = 0
        self.processing_time_ns = 0
        self.communication_cost = 0
        self.epochs_finalized = 0
        self.completed = {
            "svd_calls": 0,
            "power_calls": 0,
            "simultaneous_calls": 0,
            "snapshot_rows": 0,
            "sealed_residual_rows": 0,
            "discarded_residual_rows": 0,
            "energy_epochs": 0,
        }

    def _accumulate(self, emitter) -> None:
        for key, value in emitter.counters().items():
            self.completed[key] += int(value)

    async def fit(self, row: npt.ArrayLike, current_time: int) -> None:
        epoch_id = (int(current_time) - 1) // self.window_size
        if epoch_id != self.forward_epoch:
            raise RuntimeError("site epoch was not finalized before update")
        vector = np.asarray(row, dtype=np.float64).reshape(-1)
        if vector.size != self.d:
            raise ValueError(f"expected a row of length {self.d}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("row contains a non-finite value")
        start = time.process_time_ns()
        self.raw_buffer.append(
            (int(current_time), np.array(vector, dtype=np.float64, copy=True))
        )
        self.max_raw_buffer_rows = max(
            self.max_raw_buffer_rows, len(self.raw_buffer)
        )
        payload = self.forward.update(vector, int(current_time))
        self.processing_time_ns += time.process_time_ns() - start
        if payload:
            self.communication_cost += sum(
                _communication_floats(block) for block in payload
            )
            await self.coordinator.add_forward.remote(
                self.site_id, epoch_id, payload, current_time
            )

    async def finalize_epoch(
        self, epoch_id: int, current_time: int
    ) -> None:
        if int(epoch_id) != self.forward_epoch:
            raise RuntimeError("attempted to finalize the wrong epoch")
        start = time.process_time_ns()
        self.forward.finalize()
        self._accumulate(self.forward)
        reverse = _new_emitter(
            self.method, self.d, self.ell, self.audit
        )
        payload: list[WireBlock] = []
        for timestamp, row in reversed(self.raw_buffer):
            payload.extend(reverse.update(row, timestamp))
        payload.extend(reverse.finalize())
        self._accumulate(reverse)
        self.processing_time_ns += time.process_time_ns() - start
        self.communication_cost += sum(
            _communication_floats(block) for block in payload
        )
        await self.coordinator.install_reverse.remote(
            self.site_id, epoch_id, payload, current_time
        )
        self.raw_buffer.clear()
        self.forward_epoch += 1
        self.epochs_finalized += 1
        self.forward = _new_emitter(
            self.method, self.d, self.ell, self.audit
        )

    def get_stats(self) -> dict:
        active = self.forward.counters()
        counters = {
            key: int(self.completed[key] + active[key])
            for key in self.completed
        }
        return {
            "site_id": self.site_id,
            "node_id": ray.get_runtime_context().get_node_id(),
            "row_num": int(
                self.forward.row_num() + len(self.raw_buffer)
            ),
            "raw_buffer_rows": int(len(self.raw_buffer)),
            "max_raw_buffer_rows": int(self.max_raw_buffer_rows),
            "meh_rows": 0,
            "processing_time": self.processing_time_ns / 1e6,
            "processing_time_ns": int(self.processing_time_ns),
            "communication_cost": int(self.communication_cost),
            "epochs_finalized": int(self.epochs_finalized),
            **counters,
        }


class RayRawEpochFdDsw:
    """Ray facade for the matched raw-epoch DSW comparison."""

    def __init__(
        self,
        d: int,
        l: int,
        N: int,
        sites: int,
        *,
        method: str,
        audit: bool = False,
        seed: int = 0,
    ):
        if min(d, l, N, sites) < 1:
            raise ValueError("d, l, N, and sites must be positive")
        if l > d:
            raise ValueError("l must not exceed d")
        if method not in {"da2", "aero", "schur"}:
            raise ValueError("method must be 'da2', 'aero', or 'schur'")
        self._owns_ray = initialize_ray()
        self.d = int(d)
        self.l = int(l)
        self.N = int(N)
        self.num_sites = int(sites)
        self.method = str(method)
        self.time = 0
        self.coordinator = RayRawEpochWindowCoordinator.remote(
            d, l, N, sites
        )
        self.sites = [
            RayRawEpochWindowSite.remote(
                site_id,
                d,
                l,
                N,
                method,
                self.coordinator,
                audit,
                seed,
            )
            for site_id in range(sites)
        ]

    def fit(self, row: npt.ArrayLike, site_id: int) -> None:
        if site_id < 0 or site_id >= self.num_sites:
            raise ValueError(f"invalid site_id {site_id}")
        self.time += 1
        ray.get(self.sites[site_id].fit.remote(row, self.time))
        if self.time % self.N == 0:
            epoch_id = self.time // self.N - 1
            ray.get(
                [
                    site.finalize_epoch.remote(epoch_id, self.time)
                    for site in self.sites
                ]
            )
            ray.get(self.coordinator.finish_epoch.remote(self.time))

    def get(self, query_time: int | None = None) -> np.ndarray:
        if query_time is not None and int(query_time) != self.time:
            raise ValueError(
                "raw-epoch DSW exposes only the current distributed window"
            )
        return ray.get(self.coordinator.get_result.remote(self.time))

    def get_covariance(self) -> np.ndarray:
        return ray.get(self.coordinator.get_covariance.remote(self.time))

    def covariance_cache_error(self) -> float:
        return float(
            ray.get(self.coordinator.audit_covariance_cache.remote(self.time))
        )

    def query_with_stats(self) -> tuple[np.ndarray, dict, int]:
        start = time.perf_counter_ns()
        coordinator_ref = self.coordinator.get_result_and_stats.remote(
            self.time
        )
        site_refs = [site.get_stats.remote() for site in self.sites]
        result, coordinator = ray.get(coordinator_ref)
        query_time_ns = time.perf_counter_ns() - start
        sites = ray.get(site_refs)
        return (
            result,
            {"coordinator": coordinator, "sites": sites},
            query_time_ns,
        )

    def get_stats(self) -> dict:
        refs = [
            self.coordinator.get_stats.remote(self.time),
            *[site.get_stats.remote() for site in self.sites],
        ]
        coordinator, *sites = ray.get(refs)
        return {
            "coordinator": coordinator,
            "sites": sites,
        }

    def row_num(self) -> tuple[int, int, int]:
        stats = self.get_stats()
        coordinator_rows = int(stats["coordinator"]["row_num"])
        site_rows = [int(site["row_num"]) for site in stats["sites"]]
        max_site_rows = max(site_rows, default=0)
        return (
            coordinator_rows,
            max_site_rows,
            coordinator_rows + sum(site_rows),
        )

    def shutdown(self) -> None:
        if ray.is_initialized():
            for site in self.sites:
                ray.kill(site, no_restart=True)
            ray.kill(self.coordinator, no_restart=True)
        if self._owns_ray and ray.is_initialized():
            ray.shutdown()

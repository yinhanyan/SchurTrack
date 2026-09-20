"""Ray implementation of deterministic distributed Schur prefix tracking.

Each site runs an independent energy-adaptive Schur ATTP stream.  Immutable
cancellation blocks are sent once to a lazy Batch-FD coordinator.  Live
residuals remain local, and completed residuals are discarded rather than
sealed and transmitted.  For every global prefix, the local ATTP errors add
because the site row sets are disjoint.  This avoids a changing global Schur
barrier and does not require a known horizon.
"""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
import ray

from algorithm.distributed.ray_runtime import initialize_ray
from base.fd import RowFdRestore
from base.schur_fd import EnergyAdaptiveSchurATTP


@ray.remote(num_cpus=1)
class RaySchurPrefixCoordinator:
    def __init__(self, d: int, ell: int, sites: int):
        self.d = int(d)
        self.ell = int(ell)
        self.num_sites = int(sites)
        self.communication_cost = 0
        self.processing_time_ns = 0
        self.sketch = RowFdRestore(self.d, 2 * self.ell)

    def add_snapshots(
        self,
        site_id: int,
        payload: list[tuple[int, np.ndarray]],
        current_time: int,
    ) -> None:
        start = time.process_time_ns()
        for created_at, rows in payload:
            if int(created_at) > int(current_time):
                raise ValueError("snapshot endpoint exceeds current time")
            block = np.asarray(rows, dtype=np.float64)
            self.sketch.fit(block)
            self.communication_cost += int(block.size)
        self.processing_time_ns += time.process_time_ns() - start

    def get_result(self) -> np.ndarray:
        return self.sketch.get()

    def get_result_and_stats(self) -> tuple[np.ndarray, dict]:
        return self.sketch.get(), self.get_stats()

    def get_stats(self) -> dict:
        return {
            "row_num": self.sketch.row_num(),
            "node_id": ray.get_runtime_context().get_node_id(),
            "processing_time": self.processing_time_ns / 1e6,
            "processing_time_ns": int(self.processing_time_ns),
            "communication_cost": self.communication_cost,
            "svd_calls": self.sketch.svd_calls,
        }


@ray.remote(num_cpus=1)
class RaySchurPrefixSite:
    def __init__(
        self,
        site_id: int,
        d: int,
        ell: int,
        coordinator,
        audit: bool,
    ):
        self.site_id = int(site_id)
        self.coordinator = coordinator
        self.processing_time_ns = 0
        self.communication_cost = 0
        self.sketch = EnergyAdaptiveSchurATTP(
            d=d,
            ell=ell,
            audit=audit,
            retain_sealed_residual=False,
        )

    async def fit(self, row: npt.ArrayLike, current_time: int) -> None:
        start = time.process_time_ns()
        emitted = self.sketch.update(row)
        local_time = self.sketch.time
        self.processing_time_ns += time.process_time_ns() - start
        payload = []
        for snapshot in emitted:
            if snapshot.time != local_time:
                raise RuntimeError("unexpected Schur snapshot support time")
            payload.append((int(current_time), snapshot.rows))
        if payload:
            sent = sum(rows.size for _, rows in payload)
            self.communication_cost += int(sent)
            await self.coordinator.add_snapshots.remote(
                self.site_id, payload, current_time
            )
            for snapshot in emitted:
                self.sketch.release_snapshot(snapshot)

    def get_stats(self) -> dict:
        stats = self.sketch.aggregate_stats()
        return {
            "site_id": self.site_id,
            "node_id": ray.get_runtime_context().get_node_id(),
            "row_num": self.sketch.row_num(),
            "processing_time": self.processing_time_ns / 1e6,
            "processing_time_ns": int(self.processing_time_ns),
            "communication_cost": self.communication_cost,
            "svd_calls": stats.svd_calls,
            "snapshot_rows": stats.snapshot_rows,
            "sealed_residual_rows": self.sketch.sealed_residual_rows,
            "discarded_residual_rows": self.sketch.discarded_residual_rows,
            "energy_epochs": self.sketch.epoch_count,
        }


class RaySchurFdDist:
    """Ray-backed facade compatible with the legacy distributed experiments."""

    def __init__(
        self,
        d: int,
        l: int,
        sites: int,
        *,
        audit: bool = False,
    ):
        self._owns_ray = initialize_ray()
        self.d = int(d)
        self.l = int(l)
        self.num_sites = int(sites)
        self.time = 0
        self.coordinator = RaySchurPrefixCoordinator.remote(
            self.d, self.l, self.num_sites
        )
        self.sites = [
            RaySchurPrefixSite.remote(
                site_id,
                self.d,
                self.l,
                self.coordinator,
                audit,
            )
            for site_id in range(self.num_sites)
        ]

    def fit(self, X: npt.ArrayLike, site_id: int) -> None:
        if site_id < 0 or site_id >= self.num_sites:
            raise ValueError(f"invalid site_id {site_id}")
        self.time += 1
        ray.get(self.sites[site_id].fit.remote(X, self.time))

    def get(self, query_time: int | None = None) -> np.ndarray:
        if query_time is None:
            query_time = self.time
        if int(query_time) != self.time:
            raise ValueError("only the current online prefix can be queried")
        return ray.get(self.coordinator.get_result.remote())

    def query_with_stats(
        self, query_time: int | None = None
    ) -> tuple[np.ndarray, dict, int]:
        if query_time is None:
            query_time = self.time
        if int(query_time) != self.time:
            raise ValueError("only the current online prefix can be queried")
        start = time.perf_counter_ns()
        coordinator_ref = self.coordinator.get_result_and_stats.remote()
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
            self.coordinator.get_stats.remote(),
            *[site.get_stats.remote() for site in self.sites],
        ]
        coordinator, *sites = ray.get(refs)
        return {"coordinator": coordinator, "sites": sites}

    def row_num(self) -> tuple[int, int, int]:
        stats = self.get_stats()
        coordinator_rows = int(stats["coordinator"]["row_num"])
        max_site_rows = max(
            (int(site["row_num"]) for site in stats["sites"]), default=0
        )
        total_rows = coordinator_rows + sum(
            int(site["row_num"]) for site in stats["sites"]
        )
        return coordinator_rows, max_site_rows, total_rows

    def shutdown(self) -> None:
        if ray.is_initialized():
            for site in self.sites:
                ray.kill(site, no_restart=True)
            ray.kill(self.coordinator, no_restart=True)
        if self._owns_ray and ray.is_initialized():
            ray.shutdown()

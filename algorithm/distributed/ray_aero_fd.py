"""Ray AeroSketch distributed baseline."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import numpy.typing as npt
import ray

from algorithm.distributed.ray_runtime import initialize_ray
from base.fd import FdDump, FdRestore


@ray.remote(num_cpus=1)
class RayAeroCoordinator:
    def __init__(self, d: int, ell: int, sites: int):
        self.d = int(d)
        self.ell = int(ell)
        self.num_sites = int(sites)
        self.sites = []
        self.sketch = FdRestore(d, 2 * ell)
        self.global_energy = 0.0
        self.energy_messages = 0
        self.processing_time_ns = 0
        self.communication_cost = 0

    def set_sites(self, sites):
        self.sites = sites

    async def update_energy(self, local_energy: float):
        self.communication_cost += 1
        self.global_energy += float(local_energy)
        self.energy_messages += 1
        if self.energy_messages >= self.num_sites:
            self.energy_messages = 0
            await asyncio.gather(
                *[
                    site.set_global_energy.remote(self.global_energy)
                    for site in self.sites
                ]
            )
            self.communication_cost += self.num_sites

    def update(self, Z: np.ndarray, ZCC: np.ndarray):
        self.communication_cost += int(Z.size + ZCC.size)
        start = time.process_time_ns()
        self.sketch.fit(Z, ZCC)
        self.processing_time_ns += time.process_time_ns() - start

    def get_result(self):
        return self.sketch.get()

    def get_result_and_stats(self):
        return self.sketch.get(), self.get_stats()

    def get_stats(self):
        return {
            "row_num": self.sketch.row_num(),
            "node_id": ray.get_runtime_context().get_node_id(),
            "processing_time": self.processing_time_ns / 1e6,
            "processing_time_ns": int(self.processing_time_ns),
            "communication_cost": self.communication_cost,
            "svd_calls": self.sketch.svd_calls,
        }


@ray.remote(num_cpus=1)
class RayAeroSite:
    def __init__(
        self,
        site_id: int,
        d: int,
        ell: int,
        sites: int,
        coordinator,
        seed: int,
    ):
        np.random.seed(int(seed) + int(site_id))
        self.site_id = int(site_id)
        self.ell = int(ell)
        self.num_sites = int(sites)
        self.coordinator = coordinator
        self.sketch = FdDump(d, ell, fast_fd=True)
        self.local_energy = 0.0
        self.global_energy = 0.0
        self.threshold = 1.0 / (sites * ell)
        self.processing_time_ns = 0
        self.communication_cost = 0

    async def fit(self, row: npt.ArrayLike, current_time: int):
        del current_time
        vector = np.asarray(row, dtype=np.float64)
        start = time.process_time_ns()
        self.local_energy += float(np.sum(vector**2))
        self.processing_time_ns += time.process_time_ns() - start

        if self.local_energy >= self.threshold * self.global_energy:
            await self.coordinator.update_energy.remote(self.local_energy)
            self.communication_cost += 1
            self.local_energy = 0.0

        start = time.process_time_ns()
        self.sketch.fit(vector)
        Z, ZCC = self.sketch.dump(self.threshold * self.global_energy)
        self.processing_time_ns += time.process_time_ns() - start
        if Z.shape[1]:
            self.communication_cost += int(Z.size + ZCC.size)
            await self.coordinator.update.remote(Z, ZCC)

    def set_global_energy(self, global_energy: float):
        self.global_energy = float(global_energy)

    def get_stats(self):
        return {
            "site_id": self.site_id,
            "node_id": ray.get_runtime_context().get_node_id(),
            "row_num": self.sketch.row_num(),
            "processing_time": self.processing_time_ns / 1e6,
            "processing_time_ns": int(self.processing_time_ns),
            "communication_cost": self.communication_cost,
            "svd_calls": self.sketch.svd_calls,
            "power_calls": self.sketch.power_calls,
            "simultaneous_calls": self.sketch.simultaneous_calls,
        }


class RayAeroFdDist:
    def __init__(self, d: int, l: int, sites: int, *, seed: int = 0):
        self._owns_ray = initialize_ray()
        self.d = int(d)
        self.l = int(l)
        self.num_sites = int(sites)
        self.time = 0
        self.coordinator = RayAeroCoordinator.remote(d, l, sites)
        self.sites = [
            RayAeroSite.remote(
                site_id, d, l, sites, self.coordinator, seed
            )
            for site_id in range(sites)
        ]
        ray.get(self.coordinator.set_sites.remote(self.sites))

    def fit(self, row: npt.ArrayLike, site_id: int):
        if site_id < 0 or site_id >= self.num_sites:
            raise ValueError(f"invalid site_id {site_id}")
        self.time += 1
        ray.get(self.sites[site_id].fit.remote(row, self.time))

    def get(self):
        return ray.get(self.coordinator.get_result.remote())

    def query_with_stats(self):
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

    def get_stats(self):
        refs = [
            self.coordinator.get_stats.remote(),
            *[site.get_stats.remote() for site in self.sites],
        ]
        coordinator, *sites = ray.get(refs)
        return {"coordinator": coordinator, "sites": sites}

    def row_num(self):
        stats = self.get_stats()
        coordinator_rows = int(stats["coordinator"]["row_num"])
        max_site_rows = max(
            (int(site["row_num"]) for site in stats["sites"]), default=0
        )
        total_rows = coordinator_rows + sum(
            int(site["row_num"]) for site in stats["sites"]
        )
        return coordinator_rows, max_site_rows, total_rows

    def shutdown(self):
        if ray.is_initialized():
            for site in self.sites:
                ray.kill(site, no_restart=True)
            ray.kill(self.coordinator, no_restart=True)
        if self._owns_ray and ray.is_initialized():
            ray.shutdown()

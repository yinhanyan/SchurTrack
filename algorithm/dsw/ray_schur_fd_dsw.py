"""SchurTrack adapter for the shared raw-epoch Ray DSW protocol."""

from __future__ import annotations

from algorithm.dsw.ray_raw_epoch_dsw import RayRawEpochFdDsw


class RaySchurFdDsw(RayRawEpochFdDsw):
    def __init__(
        self,
        d: int,
        l: int,
        N: int,
        sites: int,
        *,
        audit: bool = False,
        seed: int = 0,
    ):
        super().__init__(
            d,
            l,
            N,
            sites,
            method="schur",
            audit=audit,
            seed=seed,
        )

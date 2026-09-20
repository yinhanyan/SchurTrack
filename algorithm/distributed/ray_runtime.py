"""Shared Ray initialization for the unified distributed experiments."""

from __future__ import annotations

import os

import ray


def initialize_ray() -> bool:
    """Initialize or attach to Ray and return whether this process did it."""

    if ray.is_initialized():
        return False
    address = os.environ.get("SCHUR_RAY_ADDRESS") or os.environ.get(
        "RAY_ADDRESS"
    )
    if address:
        ray.init(address=address)
    else:
        init_kwargs = {
            "include_dashboard": False,
            "_metrics_export_port": 0,
        }
        requested_cpus = os.environ.get("SCHUR_RAY_NUM_CPUS")
        if requested_cpus:
            num_cpus = int(requested_cpus)
            if num_cpus < 1:
                raise ValueError("SCHUR_RAY_NUM_CPUS must be positive")
            init_kwargs["num_cpus"] = num_cpus
        ray.init(**init_kwargs)
    return True

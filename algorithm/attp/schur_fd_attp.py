"""Compatibility adapter for energy-adaptive ATTP experiments."""

from base.schur_fd import EnergyAdaptiveSchurATTP


class SchurFdAttp(EnergyAdaptiveSchurATTP):
    def __init__(self, d: int, ell: int):
        super().__init__(d=d, ell=ell, retain_sealed_residual=False)


class SchurFdAttpNoSeal(EnergyAdaptiveSchurATTP):
    """Backward-compatible name for the default no-seal ATTP tracker."""

    def __init__(self, d: int, ell: int):
        super().__init__(d=d, ell=ell, retain_sealed_residual=False)

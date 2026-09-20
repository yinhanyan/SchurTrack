"""Multilevel SchurTrack adapter for sliding-window experiment drivers."""

from base.schur_fd import MultiLevelWindowSchurFD


class SchurFdSw(MultiLevelWindowSchurFD):
    def __init__(self, N, R, d, l, beta=1.0):
        del beta
        super().__init__(
            window_size=N,
            norm_sq_upper=R,
            d=d,
            ell=l,
        )

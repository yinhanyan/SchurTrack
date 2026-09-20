from algorithm.sw.legacy_fast_fd import FrequentDirectionsWithDump
from algorithm.sw.legacy_window_fd import SlidingWindowFD
import numpy as np
import numpy.typing as npt
from typing import Dict
class FastDsFd:
    def __init__(
        self,
        N: int,
        R: float,
        d: int,
        sketch_dim: int,
        beta: float = 1.0,
        ty=SlidingWindowFD,
        faster=FrequentDirectionsWithDump,
        **kwargs,
    ):
        """Sliding Window on Frequent Directions

        Args:
            N (int): Sliding window size.
            R (float): Upper bound of square of 2-norm of row vectors.
            d (int): Vector dimension.
            sketch_dim (int): Sketch dimension.
            beta (float): Additional coefficient of error, default as 1.0.
        """
        self.N = np.uint64(N)
        self.d = d
        self.R = R
        self.logR = int(np.floor(np.log2(R))) + 1
        # print(self.logR)
        # exit(0)
        self.beta = beta
        self.sketch_dim = sketch_dim
        self.ty = ty
        self.faster = faster

        self.levels: Dict[int, ty] = {}

        if "upper_F_norm" in kwargs:
            self.logR = int(np.ceil(np.log2(kwargs["upper_F_norm"] / self.N)))
            base = self.N / self.sketch_dim
            for j in range(self.logR):
                self.levels[j] = ty(
                    self.N,
                    self.d,
                    self.sketch_dim,
                    error_threshold=base * (2**j),
                    faster=faster,
                )

        else:
            for j in range(self.logR):
                self.levels[j] = ty(
                    self.N,
                    self.d,
                    self.sketch_dim,
                    error_threshold=(2**j) * self.N / self.sketch_dim,
                    faster=faster,
                )

        self.time = np.uint64(0)

    # @profile
    def fit(self, X: npt.NDArray):
        """Handle the input vector

        Args:
            X (npt.NDArray): Arriving vector at the time. (row vector: (1, n)-shape)
        """
        self.time += np.uint64(1)

        for j in range(self.logR):
            sq = self.levels[j].fd
            q = sq.queue
            C = sq.C
            while len(q) != 0:
                head_snapshot = q[0]
                if (
                    len(q) > (2 + 8 / self.beta) * self.sketch_dim
                    or head_snapshot.t + self.N <= self.time
                ):
                    q.popleft()
                else:
                    break

            sq = self.levels[j].fd_aux
            q = sq.queue
            C = sq.C
            while len(q) != 0:
                head_snapshot = q[0]
                if (
                    len(q) > (2 + 8 / self.beta) * self.sketch_dim
                    or head_snapshot.t + self.N <= self.time
                ):
                    q.popleft()
                else:
                    break

            error = C.get_error()
            if X @ X.T >= error:
                self.levels[j].append(X)
            else:
                self.levels[j].fit(X)

    # @profile
    def get(self):
        j = 0
        rj = 0
        while j < self.logR:
            sq = self.levels[j].fd
            q = sq.queue
            if len(q) != 0:
                head_snapshot = q[0]
                if self.time - head_snapshot.s >= min(self.N - 1, self.time - 1):
                    rj = j
                    break
            else:
                break
            j += 1

        j = rj

        return self.levels[j].get()

    def get_size(self):
        return sum([self.levels[j].get_size() for j in self.levels])



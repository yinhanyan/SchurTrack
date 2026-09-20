import numpy as np
from base.sifd import SiFd
from typing import Dict


class SiFdSw:
    def __init__(self, N, R, d, l, beta):
        self.N = N
        self.d = d
        self.R = R
        self.logR = int(np.floor(np.log2(R))) + 1
        self.levels: Dict[int, SiFd] = {}
        self.sketch_dim = l
        self.beta = beta

        for j in range(self.logR):
            self.levels[j] = SiFd(
                self.d,
                self.sketch_dim,
                dump_threshold=(2**j) * self.N / self.sketch_dim,
            )

        self.time = 0

    def fit(self, X):
        self.time += 1

        for j in range(self.logR):
            sq = self.levels[j]
            q = sq.snapshots
            while len(q) != 0:
                head_snapshot = q[0]
                if (
                    len(q) > 2 * self.sketch_dim
                    or head_snapshot.time + self.N <= self.time
                ):
                    s = q.popleft()
                    sq.rows_of_snapshots -= 2 * s.Z.shape[1]
                else:
                    break

            self.levels[j].fit(X)

    def get(self):
        j = 0
        rj = 0
        while j < self.logR:
            sq = self.levels[j]
            q = sq.snapshots
            if len(q) != 0:
                head_snapshot = q[0]
                if self.time - head_snapshot.s_time >= min(self.N - 1, self.time - 1):
                    rj = j
                    break
            else:
                break
            j += 1

        j = rj

        return self.levels[j].get(), None, None, None

    def get_size(self):
        return sum([self.levels[j].row_num() for j in self.levels])

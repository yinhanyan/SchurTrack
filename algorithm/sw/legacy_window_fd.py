from __future__ import annotations

from algorithm.sw.legacy_fast_fd import FrequentDirectionsWithDump
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from scipy import linalg
from collections import deque
from abc import ABC, abstractmethod
@dataclass
class FDSnapshot:
    v: npt.NDArray
    s: np.uint64
    t: np.uint64


@dataclass
class DumpFDwithSnapshotQueue:
    C: FrequentDirectionsWithDump
    queue: deque[FDSnapshot] = field(default_factory=deque)
    last_dump_time: np.uint64 = np.uint64(1)
    size: float = 0.0


class SlidingWindowFdBase(ABC):
    @abstractmethod
    def fit(self, X: npt.NDArray, time: int | None) -> None:
        pass

    @abstractmethod
    def get_sketch(self) -> npt.NDArray:
        pass


class SlidingWindowFD(SlidingWindowFdBase):
    def __init__(
        self,
        N: int,
        d: int,
        sketch_dim: int,
        C: int = 1,
        faster=FrequentDirectionsWithDump,
        **kwargs,
    ):
        """Sliding Window on Frequent Directions

        Args:
            N (int): Sliding window size.
            d (int): Vector dimension.
            sketch_dim (int): Sketch dimension.
            error_threshold (float, optional): Dump threshold, default as `1.0`.
        """
        if "error_threshold" in kwargs:
            self.error = kwargs["error_threshold"]
        else:
            self.error = N * 1.0 / sketch_dim

        self.faster = faster

        self.fd = DumpFDwithSnapshotQueue(C=self.faster(d, sketch_dim * C, self.error))
        self.fd_aux = DumpFDwithSnapshotQueue(
            C=self.faster(d, sketch_dim * C, self.error)
        )
        self.period = 0

        self.N = np.uint64(N)
        self.d = d
        self.C = C
        self.sketch_dim = sketch_dim
        self.time = np.uint64(0)
        self.size = 0.0
        self.real_size = 0.0

    def append(self, X: npt.NDArray, t=None):
        if t != None:
            self.time = np.uint64(t)
        else:
            self.time += np.uint64(1)

        while self.period < self.time // self.N:
            self.fd = self.fd_aux
            self.fd_aux = DumpFDwithSnapshotQueue(
                C=self.faster(self.d, self.sketch_dim * self.C, self.error),
                last_dump_time=self.time,
            )
            self.period += 1

        s = self.fd.last_dump_time
        self.fd.last_dump_time = self.time + 1
        self.fd.queue.append(FDSnapshot(v=X, s=s, t=self.time))

        s = self.fd_aux.last_dump_time
        self.fd_aux.last_dump_time = self.time + 1
        self.fd_aux.queue.append(FDSnapshot(v=X, s=s, t=self.time))

    # @profile
    def fit(self, X: npt.NDArray, t=None):
        """Handle the input vector

        Args:
            X (npt.NDArray): Arriving vector at the time. (row vector: (1, n)-shape)
        """

        if t != None:
            self.time = np.uint64(t)
        else:
            self.time += np.uint64(1)
        
        while self.period < self.time // self.N:
            self.fd = self.fd_aux
            self.fd_aux = DumpFDwithSnapshotQueue(
                C=self.faster(self.d, self.sketch_dim * self.C, self.error),
                last_dump_time=self.time,
            )
            self.period += 1

        while len(self.fd.queue) != 0:
            head_snapshot = self.fd.queue[0]
            if head_snapshot.t + self.N <= self.time:
                self.fd.queue.popleft()
            else:
                break

        while len(self.fd_aux.queue) != 0:
            head_snapshot = self.fd_aux.queue[0]
            if head_snapshot.t + self.N <= self.time:
                self.fd_aux.queue.popleft()
            else:
                break

        # with energy optimization
        self.fd.C.fit(X)

        dumped = self.fd.C.dump()
        if dumped is not None:
            s = self.fd.last_dump_time
            self.fd.last_dump_time = self.time + 1
            self.fd.queue.append(FDSnapshot(v=dumped, s=s, t=self.time))

        self.fd_aux.C.fit(X)

        dumped = self.fd_aux.C.dump()
        if dumped is not None:
            s = self.fd_aux.last_dump_time
            self.fd_aux.last_dump_time = self.time + 1
            self.fd_aux.queue.append(FDSnapshot(v=dumped, s=s, t=self.time))

    def update(self, X: npt.NDArray):
        sq = self.fd
        q = sq.queue
        C = sq.C
        while len(q) != 0:
            head_snapshot = q[0]
            if len(q) > self.queue_capacity or head_snapshot.t + self.N <= self.time:
                q.popleft()
            else:
                break

        sq = self.fd_aux
        q = sq.queue
        C = sq.C
        while len(q) != 0:
            head_snapshot = q[0]
            if len(q) > self.queue_capacity or head_snapshot.t + self.N <= self.time:
                q.popleft()
            else:
                break

        error = C.get_error()
        if X @ X.T >= error:
            self.append(X)
        else:
            self.fit(X)

    def get(self):
        q = self.fd.queue
        C = self.fd.C.get_sketch()
        if len(q) != 0:
            ret = np.vstack([C, *(s.v for s in q)])

            _, s, Vt = linalg.svd(ret, full_matrices=False, lapack_driver="gesvd")
            s = s**2
            if s.shape[0] > self.sketch_dim:
                s = s[: self.sketch_dim] - s[self.sketch_dim]
            Vt = Vt[: self.sketch_dim]
            sketch = Vt * np.sqrt(s).reshape(-1, 1)

            return sketch, s, Vt, 0.0
        else:
            return self.fd.C.get()

    # @profile
    def get_sketch(self):
        q = self.fd.queue
        ret = self.fd.C.sketch
        if len(q) != 0:
            ret = np.vstack([ret, *(s.v for s in q)])

        if self.fd.C.buffer is not None:
            ret = np.vstack([ret, self.fd.C.buffer])

        if len(ret) > self.sketch_dim:
            _, s, Vt = linalg.svd(ret, full_matrices=False, lapack_driver="gesvd")
            s = s**2
            s = s[: self.sketch_dim] - s[self.sketch_dim]
            Vt = Vt[: self.sketch_dim]
            sketch = Vt * np.sqrt(s).reshape(-1, 1)

            return sketch
        else:
            return ret

    def get_size(self):
        return len(self.fd.queue) + len(self.fd_aux.queue) + 2 * self.sketch_dim



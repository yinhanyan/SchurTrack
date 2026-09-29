import numpy as np
import numpy.typing as npt
from base.fd import FrequentDirections
import bisect


class FdAttp:
    def __init__(self, d: int, ell: int, fast_fd: bool = False):
        self.time = 0
        self.d, self.ell = d, ell
        self.fast_fd = fast_fd

        self.C = FrequentDirections(d, ell, fast_fd=fast_fd, start_row=ell)
        self.B = FrequentDirections(d, ell, fast_fd=True)

        self.p = 0
        self.energy: float = 0.0

        # Ordered full and partial sketches with their creation times.
        self.full_sketch = []
        self.full_sketch_time = []
        self.partial_sketch = []
        self.partial_sketch_time = []
        self.rows_of_snapshots = 0

    def fit(self, X: npt.NDArray):
        self.energy += np.sum(X**2)

        self.C.fit(X)

        i = 0
        while i < len(self.C.sigma) and self.C.sigma[i] ** 2 > self.energy / self.ell:
            c_1 = self.C.sigma[i] * self.C.Vt[i, :]
            self.partial_sketch.append(c_1)
            self.partial_sketch_time.append(self.time)
            self.B.fit(c_1)
            self.rows_of_snapshots += 1
            self.C.sigma[i] = 0
            self.C.sketch[i, :] = 0
            self.p += 1
            i += 1

            if self.p >= self.ell:
                # Store the accumulated sketch and its current timestamp.
                # Freeze this timestamped checkpoint before the live FD changes.
                B = self.B.get().copy()
                self.full_sketch.append(B)
                self.full_sketch_time.append(self.time)
                self.rows_of_snapshots += B.shape[0]
                self.p = 0

        self.time += 1

    def get(self):
        """Return all full and partial sketches for the current prefix."""
        if len(self.full_sketch_time):
            t = self.full_sketch_time[-1]
            B = self.full_sketch[-1]
            index = bisect.bisect_right(self.partial_sketch_time, t)
            partial_sketch = self.partial_sketch[index:]
            # Append partial sketches created after the last full sketch.
            if partial_sketch:
                return np.vstack([B, *partial_sketch])
            else:
                return B
        else:
            # No full sketch has been created yet.
            if self.partial_sketch:
                return np.vstack(self.partial_sketch)
            else:
                return np.zeros((1, self.d))

    def row_num(self):
        return self.rows_of_snapshots + self.B.row_num() + self.C.row_num()

    def size(self):
        return (
            sum([sketch.size for sketch in self.full_sketch])
            + sum([sketch.size for sketch in self.partial_sketch])
            + self.B.size()
            + self.C.size()
        )

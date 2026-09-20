import numpy as np
import numpy.typing as npt
import scipy.linalg


def power_iteration(
    A: npt.NDArray,
    max_iter: int = 0,
    *,
    rng=None,
) -> npt.NDArray:
    """Computes the dominant eigenvector of a matrix

    Args:
        A (npt.NDArray): matrix
        eps (float, optional): threshold for convergence. Defaults to 1e-6.

    Returns:
        npt.NDArray: dominant eigenvector
    """
    l, d = A.shape
    random_source = np.random if rng is None else rng
    x = random_source.standard_normal(d)
    x /= scipy.linalg.norm(x)

    for _ in range(max_iter):
        Ax = A @ x
        x = A.T @ Ax
        x /= scipy.linalg.norm(x)

    Ax = A @ x
    sigma_squared = np.inner(Ax, Ax)

    return sigma_squared, x


def asym_power_iteration(
    A: npt.NDArray, B: npt.NDArray, max_iter: int = 0
) -> npt.NDArray:
    """Computes the dominant eigenvector of a matrix

    Args:
        A (npt.NDArray): matrix
        eps (float, optional): threshold for convergence. Defaults to 1e-6.

    Returns:
        npt.NDArray: dominant eigenvector
    """
    m_x, l = A.shape
    m_y, _ = B.shape

    x = np.random.standard_normal(m_x)
    x /= scipy.linalg.norm(x)

    for _ in range(max_iter):
        x = B @ (A.T @ x)
        x = A @ (B.T @ x)
        x /= scipy.linalg.norm(x)

    ABx = B @ (A.T @ x)
    sigma_squared = np.inner(ABx, ABx)

    return sigma_squared, x


def sampler(norm, rand, sampler):
    if sampler == "priority":
        return norm / rand
    if sampler == "ess":
        return rand ** (1 / norm)


def cod_shrink(X, Y, l):
    """
    X: m_x * n
    Y: m_y * n
    """
    Q_x, R_x = scipy.linalg.qr(X, mode="economic")
    Q_y, R_y = scipy.linalg.qr(Y, mode="economic")
    # print(R_x.shape, R_y.shape)
    U, S, Vh = scipy.linalg.svd(R_x @ R_y.T, full_matrices=False)
    if len(S) > l:
        S = S[:l] - S[l]

    A = (Q_x @ U[:, :l]) * np.sqrt(S)
    B = (Q_y @ Vh[:l, :].T) * np.sqrt(S)

    return A, B

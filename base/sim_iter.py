import numpy as np
import scipy.linalg


def simultaneous_iteration(
    A: np.ndarray,
    k: int,
    max_iter: int,
    *,
    rng=None,
):
    """
    Simultaneous Iteration Algorithm for approximating top-k eigenvectors/values
    of A A^T where A ∈ R^{d×ℓ}.

    Parameters:
        A (np.ndarray): The input matrix of shape (d, ℓ)
        k (int): Number of top eigenvectors/eigenvalues to approximate
        epsilon_si (float): Error tolerance parameter (0 < epsilon_si < 1)

    Returns:
        Z (np.ndarray): Approximate top-k eigenvectors of shape (d, k)
        Sigma (np.ndarray): Approximate top-k eigenvalues (length k)
    """
    d, ell = A.shape

    # Step 1: Random Gaussian matrix Π ~ N(0,1)^{ℓ × k}
    random_source = np.random if rng is None else rng
    Pi = random_source.standard_normal((ell, k))

    # Step 2: Compute K = (A A^T)^q A Π
    K = A @ Pi  # Start with AΠ
    for _ in range(max_iter):
        K = A @ (A.T @ K)

    # Step 3: QR decomposition of K
    Q, _ = scipy.linalg.qr(K, overwrite_a=True, mode="economic")  # Q ∈ R^{d × k}

    A_T_Q = A.T @ Q  # Compute A^T Q.
    S = np.sum(A_T_Q**2, axis=0)  # Squared norm of each column.

    sorted_indices = np.argsort(S)[::-1]
    S = S[sorted_indices]
    Q = Q[:, sorted_indices]

    return Q, S


def asym_simultaneous_iteration(A: np.ndarray, B: np.ndarray, k: int, max_iter: int):
    m_x, l = A.shape
    m_y, _ = B.shape

    # Step 1: Random Gaussian matrix Π ~ N(0,1)^{ℓ × k}
    Pi = np.random.randn(m_y, k)

    # Step 2: Compute K = (A A^T)^q A Π
    K = A @ (B.T @ Pi)  # Start with AΠ
    for _ in range(max_iter):
        K = B @ (A.T @ K)
        K = A @ (B.T @ K)

    # Step 3: QR decomposition of K
    Q, _ = scipy.linalg.qr(K, overwrite_a=True, mode="economic")  # Q ∈ R^{m_x × k}

    A_T_Q = B @ (A.T @ Q)  # Compute B A^T Q.
    M = A_T_Q.T @ A_T_Q
    Sigma, U = scipy.linalg.eigh(M, overwrite_a=True)
    Z = Q @ U  # Top-k eigenvectors
    W = (Z.T @ A) @ B.T
    W = W / np.linalg.norm(W, axis=1, keepdims=True).clip(min=1.0)

    return np.flip(Z, axis=1), Sigma[::-1], np.flip(W, axis=0)

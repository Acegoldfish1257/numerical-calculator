import numpy as np
from core.validation import checked, matrix
from core.results import result


@checked
def qr_factorization(A):
    """Householder QR: H=I-2vv^T, R <- HR, Q <- QH; return reduced Q,R.

    Input: real m-by-n matrix with m>=n. Output: Q,R, reflector norms and
    reconstruction residual. Zero column tails need no reflection. QR exists
    for rank-deficient matrices, but using R to solve may then fail.
    """
    A = matrix(A)
    m, n = A.shape
    if m < n:
        raise ValueError("Reduced QR here requires at least as many rows as columns.")
    Q = np.eye(m)
    R = A.copy()
    history = []
    for k in range(n):
        v = R[k:, k].copy()
        length = np.linalg.norm(v)
        if length != 0:
            v[0] += np.copysign(length, v[0])
            v /= np.linalg.norm(v)
            R[k:, k:] -= 2 * np.outer(v, v @ R[k:, k:])
            Q[:, k:] -= 2 * np.outer(Q[:, k:] @ v, v)
        history.append(dict(column=k + 1, column_tail_norm=length))
    Q, R = Q[:, :n], R[:n, :]
    return result(
        dict(Q=Q, R=R),
        history,
        float(np.linalg.norm(A - Q @ R, np.inf)),
        error_kind="Reconstruction residual",
        orthogonality_error=float(np.linalg.norm(Q.T @ Q - np.eye(n), np.inf)),
    )

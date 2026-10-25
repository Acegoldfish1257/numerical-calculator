import numpy as np
from core.validation import checked, matrix
from core.results import result


@checked
def gram_schmidt(A, *, variant="modified"):
    """Build orthonormal columns q_j from a_j minus prior projections.

    Inputs: m-by-n full-column-rank A, m>=n; classical/modified variant.
    Output: Q,R, column norms/projections and ||A-QR||_inf. Modified form
    projects the progressively updated vector; classical uses original a_j.
    Nearly dependent columns lose orthogonality; tiny norms are rejected.
    """
    A = matrix(A)
    m, n = A.shape
    if m < n or variant not in ("classical", "modified"):
        raise ValueError("Need m>=n and classical or modified variant.")
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    history = []
    threshold = np.finfo(float).eps * np.linalg.norm(A) * max(A.shape)
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ (A[:, j] if variant == "classical" else v)
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] <= threshold:
            raise ValueError(
                "Columns are linearly dependent or numerically indistinguishable."
            )
        Q[:, j] = v / R[j, j]
        history.append(dict(column=j + 1, remaining_norm=R[j, j]))
    return result(
        dict(Q=Q, R=R),
        history,
        float(np.linalg.norm(A - Q @ R, np.inf)),
        error_kind="Reconstruction residual",
        orthogonality_error=float(np.linalg.norm(Q.T @ Q - np.eye(n), np.inf)),
    )

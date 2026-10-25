import numpy as np
from core.validation import checked, matrix
from core.results import result


@checked
def cholesky_factorization(A):
    """A=L L^T, L_ii=sqrt(A_ii-sum L_ik^2), L_ij=(A_ij-sum L_ik L_jk)/L_jj.

    Input: symmetric positive-definite real square A. Output: L, entry steps,
    reconstruction residual. Nonsymmetric or nonpositive pivots rejected;
    near-semidefinite problems may fail because of rounding.
    """
    A = matrix(A, square=True)
    if not np.allclose(A, A.T, rtol=1e-12, atol=1e-14):
        raise ValueError("Cholesky requires a symmetric matrix.")
    L = np.zeros_like(A)
    history = []
    for i in range(len(A)):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError("Cholesky requires a positive-definite matrix.")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
            history.append(dict(row=i + 1, column=j + 1, value=L[i, j]))
    return result(
        L,
        history,
        float(np.linalg.norm(A - L @ L.T, np.inf)),
        error_kind="Factorization residual",
    )

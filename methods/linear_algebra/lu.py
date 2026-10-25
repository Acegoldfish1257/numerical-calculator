import numpy as np
from core.validation import checked, matrix
from core.results import result


@checked
def lu_factorization(A, *, pivoting=True):
    """Doolittle LU with unit diagonal L, optionally partial-pivoted: P A=L U.

    Input: real square matrix. Output: P,L,U, elimination history and infinity
    norm reconstruction residual. Swaps apply to completed columns of L.
    Zero/tiny pivots are rejected; without pivoting a valid matrix may fail.
    """
    A = matrix(A, square=True)
    n = len(A)
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    history = []
    threshold = np.finfo(float).eps * np.linalg.norm(A, np.inf) * n
    for k in range(n):
        p = k + int(np.argmax(np.abs(U[k:, k]))) if pivoting else k
        if abs(U[p, k]) <= threshold:
            raise ValueError("LU encountered a zero or numerically singular pivot.")
        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]
        history.append(dict(column=k + 1, swapped_row=p + 1, pivot=U[k, k]))
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    residual = float(np.linalg.norm(P @ A - L @ U, np.inf))
    return result(
        dict(P=P, L=L, U=U),
        history,
        residual,
        error_kind="Factorization residual",
        identity="P @ A = L @ U",
    )

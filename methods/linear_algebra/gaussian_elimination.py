import numpy as np
from core.validation import checked, system
from core.results import result


@checked
def gaussian_elimination(A, b, *, pivoting="partial"):
    """Eliminate A[i,k] with multiplier A[i,k]/A[k,k], then back-substitute.

    Inputs: square finite A, matching b, none/partial/scaled/complete pivoting. Output:
    solution, elimination steps, residual infinity norm. Scaled pivoting uses
    original row maxima. Zero or numerically tiny pivots are rejected; no
    pivoting may fail on a nonsingular matrix. Residual is not forward error.
    """
    A, b = system(A, b)
    original = A.copy()
    rhs = b.copy()
    n = len(b)
    if pivoting not in ("none", "partial", "scaled", "complete"):
        raise ValueError("Pivoting must be none, partial, scaled, or complete.")
    scale = np.max(np.abs(A), axis=1)
    if np.any(scale == 0):
        raise ValueError("Matrix has a zero row and is singular.")
    threshold = np.finfo(float).eps * np.max(scale) * n
    history = []
    column_order = np.arange(n)
    for k in range(n):
        if pivoting == "none":
            p = k
        elif pivoting == "partial":
            p = k + int(np.argmax(np.abs(A[k:, k])))
        elif pivoting == "scaled":
            p = k + int(np.argmax(np.abs(A[k:, k]) / scale[k:]))
        else:
            row_offset, col_offset = np.unravel_index(np.argmax(np.abs(A[k:, k:])), A[k:, k:].shape)
            p, q = k + row_offset, k + col_offset
            if q != k:
                A[:, [k, q]] = A[:, [q, k]]
                column_order[[k, q]] = column_order[[q, k]]
        if pivoting != "complete":
            q = k
        if abs(A[p, k]) <= threshold:
            raise ValueError(
                "Zero or numerically singular pivot; change pivoting or inspect conditioning."
            )
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
            scale[[k, p]] = scale[[p, k]]
        history.append(
            dict(
                column=k + 1,
                pivot_row=p + 1,
                pivot_column=q + 1,
                pivot=A[k, k],
                target_row=k + 1,
                multiplier=0.0,
            )
        )
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]
            history.append(
                dict(
                    column=k + 1,
                    pivot_row=k + 1,
                    pivot=A[k, k],
                    target_row=i + 1,
                    multiplier=m,
                )
            )
    x_working = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_working[i] = (b[i] - A[i, i + 1 :] @ x_working[i + 1 :]) / A[i, i]
    x = np.zeros(n)
    x[column_order] = x_working
    residual = float(np.linalg.norm(original @ x - rhs, np.inf))
    return result(
        x,
        history,
        residual,
        error_kind="Residual infinity norm",
        upper_triangular=A,
        transformed_rhs=b,
        column_order=column_order + 1,
    )

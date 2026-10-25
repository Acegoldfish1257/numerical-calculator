import numpy as np
from core.validation import checked, matrix, array
from core.results import result
from methods.best_approximation.qr import qr_factorization


@checked
def least_squares(A, b, *, solver="qr"):
    """Minimize ||Ax-b||_2 via reduced QR and R*x=Q^T*b, or NumPy SVD.

    Inputs: finite matrix A and matching b; solver qr/svd. Output: x, observed
    versus fitted rows, residual norm. QR needs m>=n and independent columns;
    SVD also supports rank deficiency and returns a minimum-norm solution.
    Residual norm measures fit, not coefficient error.
    """
    A = matrix(A)
    b = array(b)
    if A.shape[0] != len(b):
        raise ValueError("b must have one entry per row of A.")
    if solver == "qr":
        if np.linalg.matrix_rank(A) < A.shape[1]:
            raise ValueError(
                "QR least squares needs independent columns; use SVD for rank-deficient problems."
            )
        qr = qr_factorization(A)["result"]
        x = np.linalg.solve(qr["R"], qr["Q"].T @ b)
    elif solver == "svd":
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        raise ValueError("Solver must be qr or svd.")
    fitted = A @ x
    residual = fitted - b
    return result(
        x,
        [
            dict(row=i + 1, observed=bi, fitted=fi, residual=ri)
            for i, (bi, fi, ri) in enumerate(zip(b, fitted, residual))
        ],
        float(np.linalg.norm(residual)),
        error_kind="Residual 2-norm",
        normal_equation_residual=float(np.linalg.norm(A.T @ residual)),
        solver=solver,
    )

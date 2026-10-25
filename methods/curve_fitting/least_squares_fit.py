import numpy as np
from core.validation import checked, points, array, evaluate
from core.results import result


@checked
def least_squares_fit(x, y, *, basis, x_eval=None):
    """Minimize ||Ac-y||_2, with A[i,j]=basis[j](x[i]), using NumPy SVD.

    Inputs: data, independent real basis functions, optional evaluation grid.
    Output: fitted values, coefficients, residual table, RMSE (not a bound).
    Rank-deficient bases are rejected; extrapolation may be unreliable.
    """
    x, y, _ = points(x, y)
    if not 1 <= len(basis) <= min(len(x), 20):
        raise ValueError("Use 1 to min(number of points,20) basis functions.")
    A = np.array([[evaluate(fn, xi) for fn in basis] for xi in x])
    c, _, rank, s = np.linalg.lstsq(A, y, rcond=None)
    if rank < len(basis):
        raise ValueError(
            "Basis columns are linearly dependent or numerically rank deficient."
        )
    t = x if x_eval is None else array(x_eval)
    values = np.array([[evaluate(fn, ti) for fn in basis] for ti in t]) @ c
    fitted = A @ c
    residual = fitted - y
    return result(
        values,
        [
            dict(x=xi, observed=yi, fitted=fi, residual=ri)
            for xi, yi, fi, ri in zip(x, y, fitted, residual)
        ],
        float(np.sqrt(np.mean(residual**2))),
        error_kind="Training RMSE",
        coefficients=c,
        condition_number=float(s[0] / s[-1]),
        backend="numpy.linalg.lstsq (SVD)",
    )

import numpy as np
from core.validation import checked, system, array, positive, integer
from core.results import result


@checked
def jacobi(A, b, *, x0=None, tol=1e-8, max_iter=500):
    """x_i[k+1]=(b_i-sum_{j!=i}A_ij*x_j[k])/A_ii.

    Inputs: square A, b, optional initial vector, positive residual tolerance,
    limit. Output: solution, residual/iterate table, convergence flag. Assumes
    spectral radius of iteration matrix <1; strict diagonal dominance is
    sufficient. Zero diagonals are invalid and divergence is reported.
    """
    A, b = system(A, b)
    tol = positive(tol)
    max_iter = integer(max_iter, "Iteration limit")
    x = np.zeros_like(b) if x0 is None else array(x0)
    if len(x) != len(b) or np.any(np.diag(A) == 0):
        raise ValueError(
            "Initial vector must match b; diagonal entries must be nonzero."
        )
    D = np.diag(A)
    R = A - np.diag(D)
    history = []
    for k in range(1, max_iter + 1):
        new = (b - R @ x) / D
        error = float(np.linalg.norm(A @ new - b, np.inf))
        history.append(
            dict(
                iteration=k,
                residual=error,
                step=float(np.linalg.norm(new - x, np.inf)),
                **{f"x{i+1}": v for i, v in enumerate(new)},
            )
        )
        x = new
        if error <= tol:
            return result(
                x,
                history,
                error,
                True,
                "Converged",
                error_kind="Residual infinity norm",
            )
        if error > 1e100:
            break
    return result(
        x,
        history,
        error,
        False,
        "Iteration limit or divergence threshold reached",
        error_kind="Residual infinity norm",
    )

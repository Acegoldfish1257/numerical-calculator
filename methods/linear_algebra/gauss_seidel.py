import numpy as np
from core.validation import checked, system, array, positive, integer
from core.results import result


@checked
def gauss_seidel(A, b, *, x0=None, tol=1e-8, max_iter=500):
    """Update x_i=(b_i-sum_{j<i}A_ij*x_j[new]-sum_{j>i}A_ij*x_j[old])/A_ii.

    Inputs: square A,b, optional initial vector,tolerance,limit. Output:
    solution and residual history. SPD or strict diagonal dominance ensures
    convergence. Zero diagonal rejected; general matrices may diverge.
    """
    A, b = system(A, b)
    tol = positive(tol)
    max_iter = integer(max_iter, "Iteration limit")
    x = np.zeros_like(b) if x0 is None else array(x0)
    if len(x) != len(b) or np.any(np.diag(A) == 0):
        raise ValueError(
            "Initial vector must match b; diagonal entries must be nonzero."
        )
    history = []
    for k in range(1, max_iter + 1):
        old = x.copy()
        for i in range(len(x)):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i + 1 :] @ old[i + 1 :]) / A[i, i]
        error = float(np.linalg.norm(A @ x - b, np.inf))
        history.append(
            dict(
                iteration=k,
                residual=error,
                step=float(np.linalg.norm(x - old, np.inf)),
                **{f"x{i+1}": v for i, v in enumerate(x)},
            )
        )
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

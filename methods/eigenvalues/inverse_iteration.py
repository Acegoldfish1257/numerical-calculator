import numpy as np
from core.validation import checked, matrix, array, positive, integer, scalar
from core.results import result


@checked
def inverse_iteration(A, *, shift=0.0, x0=None, tol=1e-8, max_iter=1000):
    """Solve (A-shift*I)y=x, normalize y, estimate lambda=x^T*A*x.

    Inputs: real symmetric A, fixed shift, nonzero initial vector, tolerance
    and limit. Output: eigenpair and residual history. Targets an eigenvalue
    nearest the shift if the initial projection is nonzero. A shift equal to
    an eigenvalue makes the system singular; equal distances may prevent convergence.
    """
    A = matrix(A, square=True)
    shift = scalar(shift)
    tol = positive(tol)
    max_iter = integer(max_iter, "Iteration limit")
    if not np.allclose(A, A.T, rtol=1e-12, atol=1e-14):
        raise ValueError("This eigenpair page supports real symmetric matrices.")
    x = np.ones(len(A)) if x0 is None else array(x0)
    if len(x) != len(A) or np.linalg.norm(x) == 0:
        raise ValueError("Supply a nonzero vector matching the matrix size.")
    shifted = A - shift * np.eye(len(A))
    if np.linalg.matrix_rank(shifted) < len(A):
        raise ValueError(
            "Shifted matrix is singular; move the shift away from an eigenvalue."
        )
    x /= np.linalg.norm(x)
    history = []
    for k in range(max_iter + 1):
        eigenvalue = float(x @ A @ x)
        residual = float(np.linalg.norm(A @ x - eigenvalue * x))
        history.append(dict(iteration=k, eigenvalue=eigenvalue, residual=residual))
        if residual <= tol:
            return result(
                dict(eigenvalue=eigenvalue, eigenvector=x),
                history,
                residual,
                True,
                "Eigenpair residual converged",
                error_kind="Eigenpair residual norm",
            )
        if k == max_iter:
            break
        y = np.linalg.solve(shifted, x)
        x = y / np.linalg.norm(y)
    return result(
        dict(eigenvalue=eigenvalue, eigenvector=x),
        history,
        residual,
        False,
        "Iteration limit reached",
        error_kind="Eigenpair residual norm",
    )

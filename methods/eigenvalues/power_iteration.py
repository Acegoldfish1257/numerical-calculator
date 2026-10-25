import numpy as np
from core.validation import checked, matrix, array, positive, integer
from core.results import result


@checked
def power_iteration(A, *, x0=None, tol=1e-8, max_iter=1000):
    """Normalize x_next=A*x; estimate lambda=x^T*A*x (Rayleigh quotient).

    Inputs: real symmetric A, nonzero initial vector, tolerance and limit.
    Output: eigenvalue/vector, quotient and residual history. A unique largest
    magnitude eigenvalue and a nonzero initial component in its eigenspace
    are needed to find the dominant pair. A converged pair may be nondominant.
    """
    A = matrix(A, square=True)
    tol = positive(tol)
    max_iter = integer(max_iter, "Iteration limit")
    if not np.allclose(A, A.T, rtol=1e-12, atol=1e-14):
        raise ValueError("This eigenpair page supports real symmetric matrices.")
    x = np.ones(len(A)) if x0 is None else array(x0)
    if len(x) != len(A) or np.linalg.norm(x) == 0:
        raise ValueError("Supply a nonzero vector matching the matrix size.")
    x /= np.linalg.norm(x)
    history = []
    for k in range(max_iter + 1):
        Ax = A @ x
        eigenvalue = float(x @ Ax)
        residual = float(np.linalg.norm(Ax - eigenvalue * x))
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
        length = np.linalg.norm(Ax)
        if length == 0:
            raise ValueError("Iteration reached a zero vector.")
        x = Ax / length
    return result(
        dict(eigenvalue=eigenvalue, eigenvector=x),
        history,
        residual,
        False,
        "Iteration limit reached",
        error_kind="Eigenpair residual norm",
    )

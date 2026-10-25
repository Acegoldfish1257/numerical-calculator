import numpy as np
from core.validation import checked, system, array, positive, integer
from core.results import result


@checked
def conjugate_gradient(A, b, *, x0=None, tol=1e-8, max_iter=500):
    """Minimize q(x)=0.5*x^T*A*x-b^T*x for symmetric positive-definite A.

    alpha=(r^T r)/(p^T A p); x+=alpha*p; r-=alpha*A*p;
    beta=(r_new^T r_new)/(r_old^T r_old); p=r+beta*p. Inputs: SPD A,b,
    optional start,tolerance,limit. Output: x, residual/objective steps and
    convergence. This is linear CG for quadratics, not nonlinear CG.
    """
    A, b = system(A, b)
    tol = positive(tol)
    max_iter = integer(max_iter, "Iteration limit")
    if (
        not np.allclose(A, A.T, rtol=1e-12, atol=1e-14)
        or np.min(np.linalg.eigvalsh(A)) <= 0
    ):
        raise ValueError(
            "Conjugate gradient requires a symmetric positive-definite matrix."
        )
    x = np.zeros_like(b) if x0 is None else array(x0)
    if len(x) != len(b):
        raise ValueError("Initial vector dimension must match b.")
    r = b - A @ x
    p = r.copy()
    rr = float(r @ r)
    history = []
    for k in range(max_iter + 1):
        norm = float(np.linalg.norm(b - A @ x))
        objective = float(0.5 * x @ A @ x - b @ x)
        history.append(
            dict(
                iteration=k,
                residual=norm,
                objective=objective,
                **{f"x{i+1}": v for i, v in enumerate(x)},
            )
        )
        if norm <= tol:
            return result(
                x,
                history,
                norm,
                True,
                "Converged",
                error_kind="Residual 2-norm",
                objective=objective,
            )
        if k == max_iter:
            break
        Ap = A @ p
        denominator = float(p @ Ap)
        if denominator <= 0:
            raise ValueError("CG lost a positive search-direction curvature.")
        alpha = rr / denominator
        x += alpha * p
        r -= alpha * Ap
        new_rr = float(r @ r)
        if rr == 0:
            break
        p = r + (new_rr / rr) * p
        rr = new_rr
    return result(
        x,
        history,
        norm,
        False,
        "Iteration limit or numerical breakdown",
        error_kind="Residual 2-norm",
        objective=objective,
    )

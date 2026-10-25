import numpy as np
from core.validation import checked, array, scalar, positive, integer
from core.results import result


@checked
def gradient_descent(f, gradient, x0, *, initial_step=1.0, tol=1e-8, max_iter=1000):
    """x_next=x-alpha*grad(f), Armijo backtracking with c=1e-4 and factor 1/2.

    Inputs: real objective, gradient, starting vector, positive step/tolerance,
    iteration limit. Output: point, objective/gradient history and gradient
    norm. Convergence means stationary, not necessarily a global minimum.
    Poor scaling and nonconvexity can cause slow convergence or failure.
    """
    x = array(x0)
    initial_step = positive(initial_step)
    tol = positive(tol)
    max_iter = integer(max_iter, "Iteration limit")
    history = []
    for k in range(max_iter + 1):
        fx = scalar(f(x), "Objective")
        g = array(gradient(x), "Gradient")
        if g.shape != x.shape:
            raise ValueError("Gradient must match the initial vector dimensions.")
        norm = float(np.linalg.norm(g))
        history.append(
            dict(
                iteration=k,
                objective=fx,
                gradient_norm=norm,
                **{f"x{i+1}": v for i, v in enumerate(x)},
            )
        )
        if norm <= tol:
            return result(
                x,
                history,
                norm,
                True,
                "Stationary point reached",
                error_kind="Gradient norm",
                objective=fx,
            )
        if k == max_iter:
            break
        alpha = initial_step
        accepted = False
        for _ in range(50):
            candidate = x - alpha * g
            try:
                trial = scalar(f(candidate), "Trial objective")
            except ValueError:
                trial = np.inf
            if trial <= fx - 1e-4 * alpha * norm**2:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return result(
                x,
                history,
                norm,
                False,
                "Line search failed",
                error_kind="Gradient norm",
                objective=fx,
            )
        x = candidate
    return result(
        x,
        history,
        norm,
        False,
        "Iteration limit reached",
        error_kind="Gradient norm",
        objective=fx,
    )

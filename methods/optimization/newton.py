import numpy as np
from core.validation import checked, array, matrix, scalar, positive, integer
from core.results import result


@checked
def newton_optimization(f, gradient, hessian, x0, *, tol=1e-8, max_iter=100):
    """Solve H*p=-g, then use Armijo backtracking along Newton direction p.

    Inputs: real objective, gradient, positive-definite Hessian callback,
    initial vector, tolerance/limit. Output: point, iteration table, gradient
    norm. This version requires a positive-definite Hessian at each iterate;
    indefinite/singular Hessians are rejected. Global optimality not assured.
    """
    x = array(x0)
    tol = positive(tol)
    max_iter = integer(max_iter, "Iteration limit")
    history = []
    for k in range(max_iter + 1):
        fx = scalar(f(x))
        g = array(gradient(x))
        H = matrix(hessian(x), square=True)
        if g.shape != x.shape or H.shape != (len(x), len(x)):
            raise ValueError("Gradient/Hessian dimensions must match x.")
        if (
            not np.allclose(H, H.T, rtol=1e-12, atol=1e-14)
            or np.min(np.linalg.eigvalsh(H)) <= 0
        ):
            raise ValueError(
                "This Newton optimizer requires a positive-definite Hessian; try gradient descent."
            )
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
        direction = np.linalg.solve(H, -g)
        alpha = 1.0
        accepted = False
        for _ in range(50):
            candidate = x + alpha * direction
            try:
                trial = scalar(f(candidate))
            except ValueError:
                trial = np.inf
            if trial <= fx + 1e-4 * alpha * (g @ direction):
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

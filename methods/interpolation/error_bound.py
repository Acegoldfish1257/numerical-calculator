from math import factorial
import numpy as np
from core.validation import checked, array, scalar
from core.results import result


@checked
def interpolation_error_bound(x, x_eval, *, derivative_bound, kind="lagrange"):
    """Bound |f(t)-P(t)| by M/n! product|t-x_i| (Lagrange), or
    M/(2n)! product(t-x_i)^2 (first-derivative Hermite).

    Inputs: distinct nodes, evaluation array, nonnegative derivative bound M,
    kind. Output: bounds and product steps. M must bound the n-th or 2n-th
    derivative over the convex hull of nodes AND evaluation points. Without
    that assumption the number is not a certified bound.
    """
    x, t = array(x, "Nodes"), array(x_eval, "Evaluation points")
    M = scalar(derivative_bound, "Derivative bound")
    if M < 0 or len(x) > 30 or len(np.unique(x)) != len(x):
        raise ValueError(
            "Use distinct nodes (at most 30) and a nonnegative derivative bound."
        )
    if kind not in ("lagrange", "hermite"):
        raise ValueError("Kind must be lagrange or hermite.")
    power = 1 if kind == "lagrange" else 2
    n = power * len(x)
    product = np.prod(np.abs(t[:, None] - x[None, :]) ** power, axis=1)
    bounds = M * product / float(factorial(n))
    return result(
        bounds,
        [dict(x=a, product=b, bound=c) for a, b, c in zip(t, product, bounds)],
        float(max(bounds)),
        error_kind="Maximum conditional error bound",
        derivative_order=n,
    )

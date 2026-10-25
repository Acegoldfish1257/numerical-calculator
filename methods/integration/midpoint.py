import numpy as np
from core.validation import checked, interval, integer, evaluate
from core.results import result


@checked
def midpoint(f, a, b, n):
    """Composite midpoint Q=h sum f(a+(i+1/2)h), h=(b-a)/n.

    Inputs: finite real integrand, a<b, 1..5000 panels. Output: integral,
    midpoint contributions, estimated coarse-grid error |Q(2n)-Q(n)|/(1-1/4).
    Estimate assumes smooth O(h^2) error; singularities may invalidate it.
    """
    a, b = interval(a, b)
    n = integer(n, "Panels", 1, 5000)
    h = (b - a) / n
    x = a + (np.arange(n) + 0.5) * h
    y = np.array([evaluate(f, v) for v in x])
    q = float(h * np.sum(y))
    fine = (h / 2) * sum(evaluate(f, a + (i + 0.5) * h / 2) for i in range(2 * n))
    return result(
        q,
        [
            dict(node=i, x=xi, value=yi, contribution=h * yi)
            for i, (xi, yi) in enumerate(zip(x, y))
        ],
        abs(fine - q) * 4 / 3,
        error_kind="Estimated coarse-grid error (O(h²))",
        refined_integral=fine,
    )

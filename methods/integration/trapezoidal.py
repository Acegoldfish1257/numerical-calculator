import numpy as np
from core.validation import checked, interval, integer, evaluate
from core.results import result


@checked
def trapezoidal(f, a, b, n):
    """Composite trapezoids Q=h*(f(a)/2+sum interior f+f(b)/2).

    Inputs: real f, a<b, 1..5000 panels. Output: integral, weighted-node table,
    estimated coarse error (4/3)|Q(2n)-Q(n)| assuming smooth O(h^2) error.
    Discontinuities, poles and cancellation can invalidate the estimate.
    """
    a, b = interval(a, b)
    n = integer(n, "Panels", 1, 5000)
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([evaluate(f, v) for v in x])
    w = np.ones(n + 1) * h
    w[[0, -1]] /= 2
    q = float(w @ y)
    fine = q / 2 + (h / 2) * sum(evaluate(f, a + (i + 0.5) * h) for i in range(n))
    return result(
        q,
        [
            dict(node=i, x=xi, value=yi, weight=wi, contribution=wi * yi)
            for i, (xi, yi, wi) in enumerate(zip(x, y, w))
        ],
        abs(fine - q) * 4 / 3,
        error_kind="Estimated coarse-grid error (O(h²))",
        refined_integral=fine,
    )

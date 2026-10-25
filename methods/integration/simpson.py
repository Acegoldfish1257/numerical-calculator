import numpy as np
from core.validation import checked, interval, integer, evaluate
from core.results import result


@checked
def simpson(f, a, b, n):
    """Composite Simpson 1/3: Q=h/3*(f0+4*f1+2*f2+...+fn).

    Inputs: real f, a<b, even subinterval count 2..5000. Output: integral,
    weighted nodes, estimated coarse error (16/15)|Q(2n)-Q(n)| for smooth f.
    Odd n is invalid; nonsmooth integrands break the O(h^4) estimate.
    """
    a, b = interval(a, b)
    n = integer(n, "Subintervals", 2, 5000)
    if n % 2:
        raise ValueError("Simpson's 1/3 rule requires an even subinterval count.")

    def compute(count):
        x = np.linspace(a, b, count + 1)
        h = (b - a) / count
        y = np.array([evaluate(f, v) for v in x])
        w = np.ones(count + 1)
        w[1:-1:2] = 4
        w[2:-1:2] = 2
        w *= h / 3
        return float(w @ y), x, y, w

    q, x, y, w = compute(n)
    fine = compute(2 * n)[0]
    return result(
        q,
        [
            dict(node=i, x=xi, value=yi, weight=wi, contribution=wi * yi)
            for i, (xi, yi, wi) in enumerate(zip(x, y, w))
        ],
        abs(fine - q) * 16 / 15,
        error_kind="Estimated coarse-grid error (O(h⁴))",
        refined_integral=fine,
    )

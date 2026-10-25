from core.validation import checked, scalar, positive, evaluate
from core.results import result


@checked
def finite_difference(f, x, h, *, formula="central_first"):
    """First derivatives: forward (f(x+h)-f(x))/h, backward analogue,
    central (f(x+h)-f(x-h))/(2h); second: (f(x+h)-2f(x)+f(x-h))/h^2.

    Inputs: real f, x, positive h, named formula. Output: derivative, sampled
    terms, refinement estimate for coarse h. Smoothness assumed. Very small h
    amplifies rounding and cancellation; refinement estimates are asymptotic.
    """
    x = scalar(x)
    h = positive(h, "Step size")
    formulas = {
        "forward_first": ([0, 1], [-1, 1], 1, 1),
        "backward_first": ([-1, 0], [-1, 1], 1, 1),
        "central_first": ([-1, 1], [-0.5, 0.5], 1, 2),
        "central_second": ([-1, 0, 1], [1, -2, 1], 2, 2),
    }
    if formula not in formulas:
        raise ValueError("Unknown finite-difference formula.")
    offsets, weights, derivative_order, accuracy = formulas[formula]

    def compute(step):
        locations = [x + o * step for o in offsets]
        if len(set(locations)) != len(locations):
            raise ValueError(
                "Step is too small to distinguish sample points in floating-point arithmetic."
            )
        values = [evaluate(f, t) for t in locations]
        return (
            sum(w * y for w, y in zip(weights, values)) / step**derivative_order,
            locations,
            values,
        )

    q, locations, values = compute(h)
    fine, _, _ = compute(h / 2)
    error = abs(fine - q) / (1 - 2.0 ** (-accuracy))
    return result(
        q,
        [
            dict(x=t, value=y, weight=w / h**derivative_order)
            for t, y, w in zip(locations, values, weights)
        ],
        error,
        error_kind="Refinement error estimate",
        derivative_order=derivative_order,
        accuracy_order=accuracy,
        formula=formula,
    )

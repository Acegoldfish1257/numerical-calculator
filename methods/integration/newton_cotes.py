import numpy as np
from core.validation import checked, interval, integer, evaluate
from core.results import result


@checked
def newton_cotes(f, a, b, *, order=4, panels=1):
    """Closed equally spaced Newton-Cotes of degree/order m=1..6 per panel.

    Solve sum_j w_j*t_j^k=1/(k+1), k=0..m, for t_j=j/m on [0,1].
    Inputs: f, ordered bounds, order and panels. Output: integral, weights,
    contributions, refinement difference (indicator, not a bound).
    High-order equispaced quadrature can be unstable; order is capped at 6.
    """
    a, b = interval(a, b)
    order = integer(order, "Order", 1, 6)
    panels = integer(panels, "Panels", 1, 500)
    nodes = np.linspace(0, 1, order + 1)
    V = np.vstack([nodes**k for k in range(order + 1)])
    weights = np.linalg.solve(V, 1 / np.arange(1, order + 2))

    def compute(count, record=False):
        width = (b - a) / count
        total = 0.0
        history = []
        for i in range(count):
            for node, w in zip(nodes, weights):
                x = a + (i + node) * width
                fx = evaluate(f, x)
                contribution = width * w * fx
                total += contribution
                if record:
                    history.append(
                        dict(
                            panel=i + 1,
                            x=x,
                            value=fx,
                            weight=width * w,
                            contribution=contribution,
                        )
                    )
        return total, history

    q, history = compute(panels, True)
    fine, _ = compute(2 * panels)
    return result(
        q,
        history,
        abs(fine - q),
        error_kind="Panel-refinement difference",
        normalized_weights=weights,
        order=order,
        refined_integral=fine,
    )

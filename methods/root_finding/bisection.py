"""Bracketed bisection."""

import numpy as np
from core.validation import checked, interval, positive, integer, evaluate
from core.results import result


@checked
def bisection(f, a, b, *, xtol=1e-8, ftol=1e-8, max_iter=100):
    """Bisect c=a+(b-a)/2, preserving opposite endpoint signs.

    Inputs: continuous real f, a<b, positive tolerances and iteration limit.
    Output: root, step history, bracket half-width error bound and residual.
    Assumes continuity on [a,b]; a sign jump across a pole is not a root.
    """
    a, b = interval(a, b)
    xtol, ftol = positive(xtol), positive(ftol)
    max_iter = integer(max_iter, "Maximum iterations")
    fa, fb = evaluate(f, a), evaluate(f, b)
    for x, fx in ((a, fa), (b, fb)):
        if fx == 0:
            return result(
                x,
                [{"iteration": 0, "x": x, "f(x)": fx, "bound": 0.0}],
                0.0,
                True,
                "Exact endpoint root",
                residual=0.0,
            )
    # Same signs invalidate the bracket; do not multiply, which can overflow.
    if np.signbit(fa) == np.signbit(fb):
        raise ValueError("Bisection requires opposite signs at the endpoints.")
    history = []
    for k in range(1, max_iter + 1):
        c = a / 2 + b / 2
        fc = evaluate(f, c)
        bound = b / 2 - a / 2
        history.append(dict(iteration=k, a=a, b=b, x=c, **{"f(x)": fc, "bound": bound}))
        if fc == 0 or (bound <= xtol and abs(fc) <= ftol):
            return result(
                c,
                history,
                0.0 if fc == 0 else bound,
                True,
                "Converged",
                residual=abs(fc),
                error_kind="Bracket half-width",
            )
        if c == a or c == b:
            return result(
                c,
                history,
                bound,
                False,
                "Floating-point resolution reached",
                residual=abs(fc),
            )
        if np.signbit(fa) != np.signbit(fc):
            b = c
        else:
            a, fa = c, fc
    return result(
        c,
        history,
        bound,
        False,
        "Iteration limit reached",
        residual=abs(fc),
        error_kind="Bracket half-width",
    )

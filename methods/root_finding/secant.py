"""Two-point secant root finder."""

from core.validation import checked, scalar, positive, integer, evaluate
from core.results import result


@checked
def secant(f, x0, x1, *, xtol=1e-8, ftol=1e-8, max_iter=100):
    """Use x2=x1-f(x1)*(x1-x0)/(f(x1)-f(x0)).

    Inputs: real f, distinct initial guesses, tolerances and iteration limit.
    Output: root, steps, successive difference (not a bound), residual.
    Assumes useful initial guesses; equal function values cause division by
    zero, and extrapolation can leave the function's real domain.
    """
    x0, x1 = scalar(x0), scalar(x1)
    xtol, ftol = positive(xtol), positive(ftol)
    max_iter = integer(max_iter, "Maximum iterations")
    if x0 == x1:
        raise ValueError("Secant needs distinct initial guesses.")
    f0, f1 = evaluate(f, x0), evaluate(f, x1)
    for x, fx in ((x0, f0), (x1, f1)):
        if fx == 0:
            return result(
                x,
                [dict(iteration=0, x=x, step=0.0, residual=0.0)],
                0.0,
                True,
                "Exact initial root",
                residual=0.0,
            )
    history = []
    for k in range(1, max_iter + 1):
        if f1 == f0:
            raise ValueError(
                "Secant encountered equal function values; choose different guesses."
            )
        new = scalar(x1 - f1 * ((x1 - x0) / (f1 - f0)), "Next iterate")
        fn = evaluate(f, new)
        error = abs(new - x1)
        history.append(dict(iteration=k, x=new, step=error, residual=abs(fn)))
        if fn == 0 or (error <= xtol and abs(fn) <= ftol):
            return result(
                new,
                history,
                error,
                True,
                "Converged",
                residual=abs(fn),
                error_kind="Successive-iterate difference",
            )
        x0, x1, f0, f1 = x1, new, f1, fn
    return result(
        x1,
        history,
        error,
        False,
        "Iteration limit reached",
        residual=abs(f1),
        error_kind="Successive-iterate difference",
    )

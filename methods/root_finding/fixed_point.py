"""Fixed-point iteration."""

from core.validation import checked, scalar, positive, integer, evaluate
from core.results import result


@checked
def fixed_point(g, x0, *, f=None, xtol=1e-8, ftol=1e-8, max_iter=100):
    """Iterate x[k+1]=g(x[k]); locally convergent when |g'(x*)|<1.

    Inputs: real g, initial x0, optional original f, tolerances and limit.
    Output: root, steps, successive-iterate difference (not a bound), residual.
    A noncontractive mapping may diverge or cycle; a wrong g may solve another
    equation. When f is supplied its residual is also required to be small.
    """
    x = scalar(x0, "Initial guess")
    xtol, ftol = positive(xtol), positive(ftol)
    max_iter = integer(max_iter, "Maximum iterations")
    history = []
    for k in range(1, max_iter + 1):
        new = evaluate(g, x)
        error = abs(new - x)
        residual = abs(evaluate(f, new)) if f else abs(evaluate(g, new) - new)
        history.append(dict(iteration=k, x=new, step=error, residual=residual))
        if error <= xtol and residual <= ftol:
            return result(
                new,
                history,
                error,
                True,
                "Converged",
                residual=residual,
                error_kind="Successive-iterate difference",
            )
        x = new
    return result(
        x,
        history,
        error,
        False,
        "Iteration limit reached; the mapping may not be contractive",
        residual=residual,
        error_kind="Successive-iterate difference",
    )

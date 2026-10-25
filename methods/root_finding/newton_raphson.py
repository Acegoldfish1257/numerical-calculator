"""Newton-Raphson root finder."""

from core.validation import checked, scalar, positive, integer, evaluate
from core.results import result


@checked
def newton_raphson(f, df, x0, *, xtol=1e-8, ftol=1e-8, max_iter=100):
    """Use x[k+1]=x[k]-f(x[k])/f'(x[k]).

    Inputs: differentiable real f, df, initial guess, tolerances and limit.
    Output: root, iterations, step-size estimate (not a bound), residual.
    Assumes a suitable initial guess. Zero derivatives stop the method;
    poor guesses, multiple roots and cycles can prevent fast convergence.
    """
    x = scalar(x0, "Initial guess")
    xtol, ftol = positive(xtol), positive(ftol)
    max_iter = integer(max_iter, "Maximum iterations")
    history = []
    fx = evaluate(f, x)
    if fx == 0:
        return result(
            x,
            [dict(iteration=0, x=x, residual=0.0, step=0.0)],
            0.0,
            True,
            "Exact initial root",
            residual=0.0,
        )
    for k in range(1, max_iter + 1):
        derivative = evaluate(df, x)
        if derivative == 0:
            raise ValueError(
                "Newton-Raphson encountered a zero derivative; choose a different initial guess."
            )
        new = scalar(x - fx / derivative, "Next iterate")
        error = abs(new - x)
        fx = evaluate(f, new)
        history.append(dict(iteration=k, x=new, step=error, residual=abs(fx)))
        if fx == 0 or (error <= xtol and abs(fx) <= ftol):
            return result(
                new,
                history,
                error,
                True,
                "Converged",
                residual=abs(fx),
                error_kind="Successive-iterate difference",
            )
        x = new
    return result(
        x,
        history,
        error,
        False,
        "Iteration limit reached",
        residual=abs(fx),
        error_kind="Successive-iterate difference",
    )

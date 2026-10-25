import numpy as np
from core.validation import checked, interval, integer, array, evaluate
from methods.curve_fitting.least_squares_fit import least_squares_fit


@checked
def polynomial_approximation(f, a, b, degree, *, sample_count=50, x_eval=None):
    """Fit a degree-d polynomial to equally spaced samples of f on [a,b].

    Inputs: real f, ordered interval, degree 0..12, sample count, evaluation
    points. Output: fitted values, normalized-basis coefficients and training
    RMSE. This is discrete least squares, not continuous minimax approximation.
    High degree and extrapolation can be unstable; x is scaled to [-1,1].
    """
    a, b = interval(a, b)
    degree = integer(degree, "Degree", 0, 12)
    sample_count = integer(sample_count, "Sample count", max(2, degree + 1), 500)
    x = np.linspace(a, b, sample_count)
    y = np.array([evaluate(f, xi) for xi in x])
    basis = [lambda t, k=k: ((2 * t - a - b) / (b - a)) ** k for k in range(degree + 1)]
    r = least_squares_fit(x, y, basis=basis, x_eval=x if x_eval is None else x_eval)
    r["diagnostics"].update(basis="P(x)=sum c[k]*((2*x-a-b)/(b-a))^k", interval=[a, b])
    return r

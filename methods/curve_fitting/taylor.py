from math import factorial
import numpy as np
from core.validation import checked, scalar, integer, array
from core.results import result


@checked
def taylor_approximation(f, center, degree, x_eval, *, derivative_bound=None):
    """P_n(x)=sum_{k=0}^n f^(k)(c)*(x-c)^k/k!.

    Inputs: parsed differentiable function, center c, degree 0..12, points,
    optional bound M on |f^(n+1)| over the center-to-point intervals. Output:
    values, terms, actual sampled errors and optional Lagrange remainder bounds
    M*|x-c|^(n+1)/(n+1)!. Nonanalytic functions and distant points may fail.
    """
    c = scalar(center)
    degree = integer(degree, "Degree", 0, 12)
    t = array(x_eval)
    values = np.zeros_like(t)
    rows = []
    for k in range(degree + 1):
        derivative = f(c) if k == 0 else f.derivative(k)(c)
        coefficient = derivative / factorial(k)
        values += coefficient * (t - c) ** k
        rows.append(
            dict(
                order=k,
                derivative_at_center=derivative,
                coefficient=coefficient,
                first_partial_sum=values[0],
            )
        )
    actual = np.array([f(x) for x in t])
    errors = np.abs(values - actual)
    bounds = None
    if derivative_bound is not None:
        M = scalar(derivative_bound)
        if M < 0:
            raise ValueError("Derivative bound must be nonnegative.")
        bounds = M * np.abs(t - c) ** (degree + 1) / factorial(degree + 1)
    return result(
        values,
        rows,
        float(max(errors)),
        error_kind="Maximum sampled actual error",
        actual_values=actual,
        absolute_errors=errors,
        conditional_remainder_bounds=bounds,
    )

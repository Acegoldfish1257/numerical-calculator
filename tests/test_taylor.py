import numpy as np
from core.expressions import parse_function
from methods.curve_fitting.taylor import taylor_approximation


def test_exp_and_polynomial():
    r = taylor_approximation(
        parse_function("exp(x)"), 0, 4, [0, 0.5, 1], derivative_bound=np.e
    )
    assert np.all(
        r["diagnostics"]["absolute_errors"]
        <= r["diagnostics"]["conditional_remainder_bounds"]
    )
    r = taylor_approximation(parse_function("x^3"), 1, 3, [0, 1, 2])
    assert np.allclose(r["result"], [0, 1, 8], atol=1e-12)

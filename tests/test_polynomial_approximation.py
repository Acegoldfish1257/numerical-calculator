import numpy as np
import pytest
from methods.curve_fitting.polynomial_approximation import polynomial_approximation


def test_quadratic():
    t = np.linspace(-2, 3, 10)
    assert np.allclose(
        polynomial_approximation(lambda x: x * x + 2, -2, 3, 2, x_eval=t)["result"],
        t * t + 2,
        atol=1e-11,
    )
    with pytest.raises(ValueError):
        polynomial_approximation(lambda x: x, 0, 1, 4, sample_count=3)

import numpy as np
import pytest
from methods.curve_fitting.least_squares_fit import least_squares_fit


def test_line():
    r = least_squares_fit([0, 1, 2], [1, 3, 5], basis=[lambda x: 1, lambda x: x])
    assert np.allclose(r["diagnostics"]["coefficients"], [1, 2], atol=1e-12)
    assert r["error_estimate"] < 1e-12
    with pytest.raises(ValueError):
        least_squares_fit([0, 1], [1, 2], basis=[lambda x: 1, lambda x: 1])

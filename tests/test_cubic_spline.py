import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from methods.curve_fitting.cubic_spline import cubic_spline


@pytest.mark.parametrize("bc", ["natural", "clamped"])
def test_reference(bc):
    x = np.array([0.0, 0.5, 1.0, 2.0])
    y = x**3
    t = np.linspace(0, 2, 30)
    slopes = (0.0, 12.0)
    r = cubic_spline(x, y, t, boundary=bc, endpoint_slopes=slopes)
    ref = CubicSpline(x, y, bc_type=bc if bc == "natural" else ((1, 0.0), (1, 12.0)))(t)
    assert np.allclose(r["result"], ref, atol=1e-12)
    with pytest.raises(ValueError):
        cubic_spline(x, y, [-1])

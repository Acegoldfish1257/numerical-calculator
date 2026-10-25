import math
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from methods.ode_solvers.rk4 import rk4


def test_exponential_and_final_step():
    r = rk4(lambda t, y: y, 0, 1, 1, 0.1)
    assert abs(r["result"] - math.e) < 3e-6
    ref = solve_ivp(
        lambda t, y: y,
        (0, 1),
        [1.0],
        t_eval=r["diagnostics"]["t"],
        rtol=1e-11,
        atol=1e-13,
    )
    assert np.max(np.abs(ref.y[0] - r["diagnostics"]["y"])) < 3e-6
    assert rk4(lambda t, y: 1, 0, 0, 1, 0.3)["diagnostics"]["t"][-1] == 1


def test_invalid():
    with pytest.raises(ValueError):
        rk4(lambda t, y: y, 0, 1, 1, -0.1)
    with pytest.raises(ValueError):
        rk4(lambda t, y: y, 0, 1, 1, 1e-9)

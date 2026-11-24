import numpy as np

from methods.linear_algebra.norms import vector_norms
from methods.ode_solvers.rk2 import rk2


def test_rk2_y_equals_y():
    answer = rk2(lambda t, y: y, 0, 1, 1, 0.01)["result"]
    assert abs(answer - np.e) < 5e-5


def test_vector_norms():
    norms = vector_norms([3, -4, 12])["result"]
    assert norms == {"L1": 19.0, "L2": 13.0, "L∞": 12.0}

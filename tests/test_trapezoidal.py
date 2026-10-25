import math
import pytest
from methods.integration.trapezoidal import trapezoidal


def test_exp_and_linear():
    assert abs(trapezoidal(math.exp, 0, 1, 100)["result"] - math.expm1(1)) < 1.5e-5
    assert trapezoidal(lambda x: 2 * x + 1, 0, 1, 3)["result"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        trapezoidal(math.exp, 1, 0, 10)

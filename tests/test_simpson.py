import math
import pytest
from methods.integration.simpson import simpson


def test_exp_and_cubic():
    assert abs(simpson(math.exp, 0, 1, 20)["result"] - math.expm1(1)) < 1e-7
    assert simpson(lambda x: x**3, 0, 1, 2)["result"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        simpson(math.exp, 0, 1, 3)

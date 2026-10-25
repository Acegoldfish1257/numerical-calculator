import math
import pytest
from methods.integration.gauss_legendre import gauss_legendre


def test_exp_and_degree():
    assert abs(gauss_legendre(math.exp, 0, 1, nodes=4)["result"] - math.expm1(1)) < 1e-9
    assert gauss_legendre(lambda x: x**7, -1, 2, nodes=4)["result"] == pytest.approx(
        255 / 8, abs=1e-11
    )
    with pytest.raises(ValueError):
        gauss_legendre(math.exp, 0, 1, nodes=0)

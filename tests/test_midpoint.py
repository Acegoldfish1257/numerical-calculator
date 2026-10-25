import math
import pytest
from methods.integration.midpoint import midpoint


def test_exp():
    r = midpoint(math.exp, 0, 1, 100)
    assert abs(r["result"] - math.expm1(1)) < 8e-6
    assert r["error_estimate"] == pytest.approx(
        abs(r["result"] - math.expm1(1)), rel=1e-4
    )
    with pytest.raises(ValueError):
        midpoint(math.exp, 0, 1, 0)

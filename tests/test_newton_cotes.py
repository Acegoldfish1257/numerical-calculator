import math
import pytest
from methods.integration.newton_cotes import newton_cotes


@pytest.mark.parametrize("order", range(1, 7))
def test_exact_polynomial(order):
    assert newton_cotes(lambda x: x**order, 0, 1, order=order)[
        "result"
    ] == pytest.approx(1 / (order + 1), abs=1e-12)


def test_exp():
    assert (
        abs(newton_cotes(math.exp, 0, 1, order=4, panels=4)["result"] - math.expm1(1))
        < 1e-9
    )
    with pytest.raises(ValueError):
        newton_cotes(math.exp, 0, 1, order=0)

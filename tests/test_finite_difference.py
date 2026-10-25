import math
import pytest
from methods.differentiation.finite_difference import finite_difference


@pytest.mark.parametrize(
    "formula,tol",
    [
        ("forward_first", 2e-4),
        ("backward_first", 2e-4),
        ("central_first", 1e-7),
        ("central_second", 1e-6),
    ],
)
def test_exp(formula, tol):
    assert (
        abs(finite_difference(math.exp, 1, 1e-4, formula=formula)["result"] - math.e)
        < tol
    )


def test_invalid_and_refinement():
    with pytest.raises(ValueError):
        finite_difference(math.sin, 1, 0)
    with pytest.raises(ValueError):
        finite_difference(math.sin, 1, 1e-30)
    errors = [
        abs(finite_difference(math.sin, 0.5, h)["result"] - math.cos(0.5))
        for h in (0.1, 0.05)
    ]
    assert 3.9 < errors[0] / errors[1] < 4.1

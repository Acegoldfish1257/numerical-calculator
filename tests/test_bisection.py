import pytest
from scipy.optimize import brentq
from methods.root_finding.bisection import bisection


def test_known_root():
    f = lambda x: x**3 - x - 2
    r = bisection(f, 1, 2)
    assert r["converged"]
    assert abs(r["result"] - brentq(f, 1, 2)) <= r["error_estimate"] + 1e-14
    assert abs(f(r["result"])) <= 1e-8


def test_endpoint():
    assert bisection(lambda x: x, 0, 1)["result"] == 0


@pytest.mark.parametrize(
    "args,kwargs",
    [((1, 2), {}), ((2, 1), {}), ((0, 2), {"xtol": 0}), ((0, 2), {"max_iter": 0})],
)
def test_invalid(args, kwargs):
    with pytest.raises(ValueError):
        bisection(lambda x: x * x + 1, *args, **kwargs)


def test_nonconvergence_and_domain():
    assert not bisection(lambda x: x * x - 2, 0, 2, max_iter=1)["converged"]
    with pytest.raises(ValueError):
        bisection(lambda x: 1 / x, -1, 1)

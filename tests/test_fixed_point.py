import pytest
from scipy.optimize import brentq
from methods.root_finding.fixed_point import fixed_point


def test_cubic():
    f = lambda x: x**3 - x - 2
    r = fixed_point(lambda x: (x + 2) ** (1 / 3), 1.5, f=f)
    assert r["converged"] and r["result"] == pytest.approx(brentq(f, 1, 2), abs=1e-8)


def test_cycle_wrong_mapping_and_invalid():
    assert not fixed_point(lambda x: -x, 1, max_iter=5)["converged"]
    assert not fixed_point(lambda x: 2.0, 1, f=lambda x: x - 3, max_iter=5)["converged"]
    with pytest.raises(ValueError):
        fixed_point(lambda x: x, 1, xtol=-1)

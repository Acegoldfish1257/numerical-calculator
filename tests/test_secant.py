import pytest
from methods.root_finding.secant import secant


def test_root():
    r = secant(lambda x: x**3 - x - 2, 1, 2)
    assert r["converged"] and r["result"] == pytest.approx(
        1.5213797068045676, abs=1e-10
    )


def test_failures():
    with pytest.raises(ValueError):
        secant(lambda x: x * x + 1, -1, 1)
    with pytest.raises(ValueError):
        secant(lambda x: x, 1, 1)
    assert not secant(lambda x: x * x - 2, 1, 2, max_iter=1)["converged"]

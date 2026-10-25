import pytest
from methods.root_finding.newton_raphson import newton_raphson


def test_root():
    r = newton_raphson(lambda x: x**3 - x - 2, lambda x: 3 * x * x - 1, 1.5)
    assert r["converged"] and r["result"] == pytest.approx(
        1.5213797068045676, abs=1e-10
    )


def test_failures():
    with pytest.raises(ValueError):
        newton_raphson(lambda x: x * x + 1, lambda x: 2 * x, 0)
    assert not newton_raphson(
        lambda x: x**3 - 2 * x + 2, lambda x: 3 * x * x - 2, 0, max_iter=8
    )["converged"]
    assert newton_raphson(lambda x: x, lambda x: 1, 0)["converged"]

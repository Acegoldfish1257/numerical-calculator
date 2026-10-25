import numpy as np
import pytest
from methods.optimization.newton import newton_optimization


def test_quadratic():
    r = newton_optimization(
        lambda x: float(x @ x), lambda x: 2 * x, lambda x: 2 * np.eye(len(x)), [3, 4]
    )
    assert r["converged"] and np.allclose(r["result"], [0, 0], atol=1e-12)
    with pytest.raises(ValueError):
        newton_optimization(
            lambda x: -float(x @ x),
            lambda x: -2 * x,
            lambda x: -2 * np.eye(len(x)),
            [1],
        )

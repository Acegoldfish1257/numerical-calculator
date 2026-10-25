import numpy as np
import pytest
from methods.interpolation.hermite import hermite


def test_cubic_unsorted():
    t = np.linspace(0, 2, 9)
    assert np.allclose(hermite([2, 0], [8, 0], [12, 0], t)["result"], t**3, atol=1e-12)


def test_invalid():
    with pytest.raises(ValueError):
        hermite([0, 1], [0, 1], [1], [0.5])

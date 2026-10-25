import pytest
from methods.interpolation.error_bound import interpolation_error_bound


def test_known():
    assert interpolation_error_bound([0, 1], [0.5], derivative_bound=2)["result"][
        0
    ] == pytest.approx(0.25)
    assert interpolation_error_bound(
        [0, 1], [0.5], derivative_bound=24, kind="hermite"
    )["result"][0] == pytest.approx(0.0625)
    with pytest.raises(ValueError):
        interpolation_error_bound([0, 1], [0.5], derivative_bound=-1)

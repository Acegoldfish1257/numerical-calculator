import numpy as np
import pytest
from core.validation import scalar, array, integer, matrix
from services.common import parse_matrix, parse_vector


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1j, True, "3"])
def test_invalid_scalar(value):
    with pytest.raises(ValueError):
        scalar(value)


def test_shapes_and_types():
    for value in ([], [float("nan")], [[1, 2]], [1j]):
        with pytest.raises(ValueError):
            array(value)
    for value in (True, 1.5, 0, "3"):
        with pytest.raises(ValueError):
            integer(value)
    with pytest.raises(ValueError):
        matrix([[1, 2, 3]], square=True)
    with pytest.raises(ValueError):
        parse_matrix("1,2\n3")
    with pytest.raises(ValueError):
        parse_vector("1, nan")
    assert np.array_equal(parse_vector("1,2;3 4"), [1, 2, 3, 4])

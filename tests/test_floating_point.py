import pytest
from methods.computer_arithmetic.floating_point import floating_point_representation


def test_ieee():
    r = floating_point_representation("1.5", precision="float32")
    assert r["diagnostics"]["bits"] == "00111111110000000000000000000000"
    assert r["error_estimate"] == 0
    assert floating_point_representation("0.1")["error_estimate"] > 0
    with pytest.raises(ValueError):
        floating_point_representation("NaN")

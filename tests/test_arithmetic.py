import pytest
from methods.computer_arithmetic.arithmetic import floating_point_arithmetic


def test_decimal_difference():
    r = floating_point_arithmetic("0.1", "0.2")
    assert r["result"] == 0.1 + 0.2 and r["error_estimate"] > 0
    assert r["diagnostics"]["decimal_reference"] == "0.3"
    with pytest.raises(ValueError):
        floating_point_arithmetic("1", "0", operation="divide")

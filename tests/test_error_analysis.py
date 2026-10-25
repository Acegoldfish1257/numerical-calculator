import pytest
from methods.computer_arithmetic.error_analysis import numerical_error


def test_error():
    r = numerical_error("3.1", "3")
    assert r["result"] == pytest.approx(0.1)
    assert r["diagnostics"]["relative_error"] == pytest.approx(1 / 30)
    assert numerical_error("1", "0")["diagnostics"]["relative_error"] is None

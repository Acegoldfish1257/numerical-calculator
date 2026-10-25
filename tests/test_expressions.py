import math
import pytest
from core.expressions import parse_function


@pytest.mark.parametrize(
    "text, x, expected",
    [
        ("x^3-x-2", 2, 4),
        ("sin(x)+cos(x)", 0, 1),
        ("exp(x)", 1, math.e),
        ("log(x)", math.e, 1),
        ("sqrt(x)", 4, 2),
        ("2", 0, 2),
    ],
)
def test_supported(text, x, expected):
    assert parse_function(text)(x) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('whoami')",
        "x.__class__",
        "[x][0]",
        "lambda:x",
        "open('a')",
        "sum(x)",
        "x**999",
        "2**2**20",
        "((2**20)**20)**20",
        "x^(1/0)",
        "sin(x, y)",
        "nan",
        "1/0",
        "",
        "x" * 301,
    ],
)
def test_reject(text):
    with pytest.raises(ValueError):
        parse_function(text)


def test_domain_and_derivative():
    with pytest.raises(ValueError):
        parse_function("log(x)")(-1)
    assert parse_function("x^3").derivative()(2) == 12
    assert parse_function("y+t", ("t", "y"))(2, 3) == 5
    assert parse_function("(x+2)^(1/3)")(6) == pytest.approx(2)

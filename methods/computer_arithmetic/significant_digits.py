from decimal import Decimal, localcontext, ROUND_DOWN, ROUND_HALF_EVEN
from core.decimal_values import decimal_value
from core.validation import checked, integer
from core.results import result


@checked
def significant_digits(value, digits, *, mode="round"):
    """Round (ties-to-even) or truncate toward zero to d significant digits.

    Inputs: finite decimal string, digits 1..30, mode round/truncate. Output:
    decimal result text, normalized decimal scientific notation and absolute
    error. Decimal demonstration differs from binary IEEE-754 rounding.
    """
    d = decimal_value(value)
    digits = integer(digits, "Significant digits", 1, 30)
    if mode not in ("round", "truncate"):
        raise ValueError("Mode must be round or truncate.")
    with localcontext() as context:
        context.prec = 150
        exponent = d.adjusted() if d else 0
        quantum = Decimal(1).scaleb(exponent - digits + 1)
        q = d.quantize(
            quantum, rounding=ROUND_HALF_EVEN if mode == "round" else ROUND_DOWN
        )
        error = float(abs(q - d))
        normalized = str(d.scaleb(-exponent))
    return result(
        str(q),
        [
            dict(
                original=str(d),
                normalized_mantissa=normalized,
                exponent=exponent,
                quantum=str(quantum),
                output=str(q),
            )
        ],
        error,
        error_kind="Absolute decimal error",
        rounding="ties to even" if mode == "round" else "toward zero",
    )

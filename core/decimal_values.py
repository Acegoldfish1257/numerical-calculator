"""Bounded exact-decimal input for arithmetic demonstrations."""

from decimal import Decimal, InvalidOperation


def decimal_value(value):
    try:
        if len(str(value)) > 100:
            raise ValueError
        d = Decimal(str(value))
        if not d.is_finite() or (d != 0 and abs(d.adjusted()) > 100):
            raise ValueError
        return d
    except (ValueError, InvalidOperation) as exc:
        raise ValueError(
            "Enter a finite decimal with at most 100 characters and exponent between -100 and 100."
        ) from exc

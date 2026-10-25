from decimal import localcontext
from core.decimal_values import decimal_value
from core.validation import checked
from core.results import result


@checked
def numerical_error(approximate, reference):
    """Absolute error |a-r|; relative error |a-r|/|r| and percentage 100*rel.

    Inputs: finite decimal strings. Output: absolute error, error table and
    relative diagnostics using 100-digit Decimal arithmetic. A true reference
    must be supplied; relative error is undefined at reference zero.
    """
    a, r = decimal_value(approximate), decimal_value(reference)
    with localcontext() as context:
        context.prec = 100
        absolute = abs(a - r)
        relative = None if r == 0 else absolute / abs(r)
    rows = [
        dict(measure="Absolute error", value=float(absolute)),
        dict(
            measure="Relative error",
            value=None if relative is None else float(relative),
        ),
        dict(
            measure="Percentage error",
            value=None if relative is None else float(100 * relative),
        ),
    ]
    return result(
        float(absolute),
        rows,
        float(absolute),
        error_kind="Absolute error",
        relative_error=None if relative is None else float(relative),
        note=(
            "Relative error undefined at zero reference"
            if r == 0
            else "Reference supplied by user"
        ),
    )

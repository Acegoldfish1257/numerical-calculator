from decimal import Decimal, localcontext
import numpy as np
from core.decimal_values import decimal_value
from core.validation import checked
from core.results import result


@checked
def floating_point_arithmetic(a, b, *, operation="add", precision="float64"):
    """Perform +,-,*,/ in selected binary precision and compare to Decimal.

    Inputs: decimal strings, operation and float32/64. Output: binary result,
    operand conversion steps and absolute difference from a 100-digit decimal
    reference. Division reference may itself be rounded; zero division and
    overflow rejected. Cancellation can magnify relative errors.
    """
    da, db = decimal_value(a), decimal_value(b)
    if precision not in ("float32", "float64"):
        raise ValueError("Precision must be float32 or float64.")
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y,
    }
    if operation not in operations:
        raise ValueError("Unknown arithmetic operation.")
    if operation == "divide" and db == 0:
        raise ValueError("Division by zero.")
    dtype = np.float32 if precision == "float32" else np.float64
    x, y = dtype(str(da)), dtype(str(db))
    value = float(operations[operation](x, y))
    if not np.isfinite(value):
        raise ValueError("Result overflows the selected precision.")
    with localcontext() as context:
        context.prec = 100
        reference = operations[operation](da, db)
        error = float(abs(Decimal.from_float(value) - reference))
    return result(
        value,
        [
            dict(operand="a", entered=str(da), represented=float(x)),
            dict(operand="b", entered=str(db), represented=float(y)),
        ],
        error,
        error_kind="Absolute reference difference",
        decimal_reference=str(reference),
        precision=precision,
    )

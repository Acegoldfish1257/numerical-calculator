import struct
from decimal import Decimal
import numpy as np
from core.decimal_values import decimal_value
from core.validation import checked
from core.results import result


@checked
def floating_point_representation(value, *, precision="float64"):
    """Inspect IEEE-754 sign, exponent and fraction for float32 or float64.

    Inputs: decimal string and precision. Output: represented value, bit fields,
    decimal conversion error, machine epsilon and normalized binary value.
    Subnormals use exponent 1-bias and no implicit leading 1; out-of-range
    values are rejected rather than silently displaying infinity.
    """
    d = decimal_value(value)
    if precision not in ("float32", "float64"):
        raise ValueError("Precision must be float32 or float64.")
    dtype = np.float32 if precision == "float32" else np.float64
    v = dtype(str(d))
    if not np.isfinite(v):
        raise ValueError("Value overflows the selected precision.")
    bits = "".join(
        f"{byte:08b}"
        for byte in struct.pack(">f" if precision == "float32" else ">d", float(v))
    )
    e = 8 if precision == "float32" else 11
    exponent = int(bits[1 : 1 + e], 2)
    fraction = bits[1 + e :]
    bias = 127 if e == 8 else 1023
    error = float(abs(Decimal.from_float(float(v)) - d))
    info = np.finfo(dtype)
    rows = [
        dict(field="sign", bits=bits[:1], integer=int(bits[:1], 2)),
        dict(field="exponent", bits=bits[1 : 1 + e], integer=exponent),
        dict(field="fraction", bits=fraction, integer=int(fraction, 2)),
    ]
    return result(
        float(v),
        rows,
        error,
        error_kind="Decimal representation error",
        bits=bits,
        exponent_bias=bias,
        unbiased_exponent=(exponent - bias if exponent else 1 - bias),
        implicit_leading_bit=1 if exponent else 0,
        machine_epsilon=float(info.eps),
        smallest_normal=float(info.tiny),
        binary_hex=float(v).hex(),
        classification=(
            "zero" if v == 0 else ("subnormal" if exponent == 0 else "normal")
        ),
    )

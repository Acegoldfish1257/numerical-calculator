"""Reusable page descriptors and text-data parsing."""

from dataclasses import dataclass
import re
import numpy as np
from core.validation import array


@dataclass
class Field:
    key: str
    label: str
    default: object
    kind: str = "text"
    options: tuple = ()
    help: str = ""


@dataclass
class Method:
    topic: str
    name: str
    description: str
    fields: list[Field]
    run: object


def parse_vector(text):
    try:
        if len(text) > 10000:
            raise ValueError("Data input is too long.")
        return array(
            [float(v) for v in re.split(r"[,;\s]+", text.strip()) if v], "Numeric data"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter finite numbers separated by commas or spaces.") from exc


def parse_matrix(text):
    try:
        rows = [parse_vector(row) for row in text.strip().splitlines() if row.strip()]
        return array(rows, "Matrix", ndim=2)
    except ValueError as exc:
        raise ValueError(
            "Enter matrix rows on separate lines, with the same number of values in each row."
        ) from exc


def function_plot(f, a, b, label="f(x)"):
    x = np.linspace(a, b, 300)
    y = []
    for value in x:
        try:
            y.append(f(value))
        except ValueError:
            y.append(np.nan)
    return dict(x=x, series={label: y}, xlabel="x", ylabel="Function value")

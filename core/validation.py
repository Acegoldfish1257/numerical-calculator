"""Finite real data validation shared by all numerical modules."""

from functools import wraps
import numpy as np


def checked(function):
    """Normalize invalid numerical inputs and arithmetic failures to ValueError."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            with np.errstate(all="raise"):
                return function(*args, **kwargs)
        except ValueError:
            raise
        except (TypeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"{function.__name__}: {exc}") from exc

    return wrapped


def scalar(value, name="Value"):
    try:
        if np.iscomplexobj(value) or isinstance(value, (bool, str)):
            raise ValueError
        value = float(value)
        if not np.isfinite(value):
            raise ValueError
        return value
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc


def positive(value, name="Tolerance"):
    value = scalar(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def integer(value, name="Count", minimum=1, maximum=10000):
    if (
        isinstance(value, (bool, str))
        or not isinstance(value, (int, np.integer))
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}.")
    return int(value)


def array(value, name="Data", ndim=1):
    try:
        if np.iscomplexobj(value):
            raise ValueError
        a = np.asarray(value, dtype=float)
        if a.ndim != ndim or not a.size or a.size > 10000 or not np.all(np.isfinite(a)):
            raise ValueError
        return a.copy()
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{name} must be a nonempty, finite {ndim}-dimensional real array (at most 10000 values)."
        ) from exc


def points(x, y, minimum=2):
    x, y = array(x, "x"), array(y, "y")
    if len(x) != len(y) or len(x) < minimum:
        raise ValueError(f"x and y need matching lengths of at least {minimum}.")
    if len(np.unique(x)) != len(x):
        raise ValueError("Data nodes x must be distinct.")
    order = np.argsort(x)
    return x[order], y[order], order


def interval(a, b):
    a, b = scalar(a, "Lower bound"), scalar(b, "Upper bound")
    if a >= b:
        raise ValueError("Lower bound must be smaller than upper bound.")
    return a, b


def evaluate(f, x):
    try:
        return scalar(f(x), f"Function value at {x:g}")
    except Exception as exc:
        raise ValueError(
            f"Function is undefined or non-finite at {x:g}: {exc}"
        ) from exc


def matrix(A, square=False):
    A = array(A, "Matrix", ndim=2)
    if max(A.shape) > 50:
        raise ValueError("Matrices are limited to 50 rows and columns.")
    if square and A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    return A


def system(A, b):
    A, b = matrix(A, square=True), array(b, "Right-hand side")
    if len(A) != len(b):
        raise ValueError("Matrix and right-hand side dimensions must match.")
    return A, b

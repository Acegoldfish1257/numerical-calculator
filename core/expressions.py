"""Allowlisted math grammar before SymPy parsing; no direct eval calls.

Only numeric literals, named variables, arithmetic and listed unary functions
are accepted. SymPy itself uses code generation internally. This is a local
educational app, not a resource-isolated public execution service.
"""

import ast
from dataclasses import dataclass
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, auto_number
from core.validation import scalar, integer

FUNCTIONS = {
    name: getattr(sp, name)
    for name in ("sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh")
}
CONSTANTS = {"pi": sp.pi, "e": sp.E, "E": sp.E}


@dataclass
class ParsedFunction:
    expression: sp.Expr
    symbols: tuple
    function: object

    def __call__(self, *args):
        try:
            with np.errstate(all="raise"):
                return scalar(self.function(*args), "Function output")
        except Exception as exc:
            raise ValueError(
                f"Expression is outside its real finite domain: {exc}"
            ) from exc

    def derivative(self, order=1, variable=0):
        order = integer(order, "Derivative order", 0, 12)
        variable = integer(variable, "Variable index", 0, len(self.symbols) - 1)
        expr = self.expression
        for _ in range(order):
            expr = sp.diff(expr, self.symbols[variable])
            if sp.count_ops(expr) > 2000:
                raise ValueError(
                    "The symbolic derivative is too complex; simplify the function or reduce the order."
                )
        return ParsedFunction(
            expr, self.symbols, sp.lambdify(self.symbols, expr, "numpy", cse=False)
        )


def parse_function(text: str, variables=("x",)) -> ParsedFunction:
    """Parse <=300 characters of explicit real arithmetic, with ^ as power.

    No attributes, indexing, strings, comprehensions, unknown names or keyword
    arguments. Exponents must be numeric literals or fractions in [-20, 20]; use exp(x)
    for a variable exponential. Limits reduce accidental symbolic blowups.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 300:
        raise ValueError("Enter a function using at most 300 characters.")
    if not variables or any(v not in ("x", "y", "t", "z") for v in variables):
        raise ValueError("Unsupported variable names.")
    source = text.strip().replace("^", "**")
    try:
        tree = ast.parse(source, mode="eval")
        if len(list(ast.walk(tree))) > 100:
            raise ValueError("Expression is too complex.")

        def exponent_value(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return float(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)
            ):
                return exponent_value(node.operand) * (
                    -1 if isinstance(node.op, ast.USub) else 1
                )
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                denominator = exponent_value(node.right)
                if denominator == 0:
                    raise ValueError("Exponent denominator cannot be zero.")
                return exponent_value(node.left) / denominator
            raise ValueError(
                "Powers require a numeric exponent or fraction; use exp(x) for variable exponentials."
            )

        def validate(node, depth=0, power_cost=1):
            if depth > 14:
                raise ValueError("Expression is nested too deeply.")
            if isinstance(node, ast.Constant):
                if (
                    type(node.value) not in (int, float)
                    or not np.isfinite(float(node.value))
                    or abs(node.value) > 1e100
                ):
                    raise ValueError(
                        "Use finite numeric constants with magnitude at most 1e100."
                    )
            elif isinstance(node, ast.Name):
                if node.id not in (*variables, *CONSTANTS):
                    raise ValueError(f"Unknown symbol: {node.id}.")
            elif isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)
            ):
                validate(node.operand, depth + 1, power_cost)
            elif isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
            ):
                if isinstance(node.op, ast.Pow):
                    exponent = exponent_value(node.right)
                    if not np.isfinite(exponent) or abs(exponent) > 20:
                        raise ValueError("Exponents must be between -20 and 20.")
                    power_cost *= max(1, abs(exponent))
                    if power_cost > 64:
                        raise ValueError("Nested powers are too large.")
                validate(node.left, depth + 1, power_cost)
                validate(node.right, depth + 1, power_cost)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in FUNCTIONS
                and len(node.args) == 1
                and not node.keywords
            ):
                validate(node.args[0], depth + 1, power_cost)
            else:
                raise ValueError(
                    "Use only arithmetic, supported functions, and the listed variables."
                )

        validate(tree.body)
        symbols = tuple(sp.Symbol(v, real=True) for v in variables)
        local = dict(zip(variables, symbols)) | FUNCTIONS | CONSTANTS
        expr = parse_expr(
            source,
            local_dict=local,
            global_dict={
                "__builtins__": {},
                "Integer": sp.Integer,
                "Float": sp.Float,
                "Rational": sp.Rational,
            },
            transformations=(auto_number,),
            evaluate=True,
        )
        if not isinstance(expr, sp.Expr) or expr.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            raise ValueError(
                "Expression must be finite and real on its evaluation domain."
            )
        return ParsedFunction(
            expr, symbols, sp.lambdify(symbols, expr, modules="numpy", cse=False)
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Invalid mathematical expression: {exc}") from exc

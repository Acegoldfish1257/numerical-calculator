"""Common result contract. Estimates and residuals have distinct labels."""

from typing import Any


def result(
    value, history, error=None, converged=None, message="Completed", **diagnostics
) -> dict[str, Any]:
    return dict(
        result=value,
        history=history,
        error_estimate=error,
        converged=converged,
        message=message,
        diagnostics=diagnostics,
    )

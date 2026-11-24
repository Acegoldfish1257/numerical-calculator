"""Second-order Runge--Kutta (midpoint) solver for scalar initial-value ODEs."""

import numpy as np

from core.results import result
from core.validation import checked, positive, scalar


@checked
def rk2(f, t0, y0, t_end, h):
    """Solve y'=f(t,y) with midpoint RK2: y[k+1]=y[k]+h f(t+h/2,y+h*k1/2).

    Inputs are a scalar right-hand side, initial time/value, a later final time,
    and positive step size.  Returns the final value and every accepted step.
    The final step is shortened when needed.  Explicit RK2 is not suitable for
    stiff problems and has global O(h^2) error only for sufficiently smooth f.
    """
    t0, y0, t_end, h = scalar(t0, "Initial time"), scalar(y0, "Initial value"), scalar(t_end, "Final time"), positive(h, "Step size")
    if t_end <= t0:
        raise ValueError("Final time must be greater than initial time.")
    steps = int(np.ceil((t_end - t0) / h))
    if steps > 10000:
        raise ValueError("RK2 is limited to 10000 steps; increase h.")
    t, y = t0, y0
    history = [dict(step=0, t=t, y=y, step_size=0.0)]
    for step in range(1, steps + 1):
        dt = min(h, t_end - t)
        k1 = scalar(f(t, y), "ODE right-hand side")
        midpoint_value = y + dt * k1 / 2
        k2 = scalar(f(t + dt / 2, midpoint_value), "ODE right-hand side")
        y = scalar(y + dt * k2, "RK2 value")
        t = scalar(t + dt, "Time")
        history.append(dict(step=step, t=t, y=y, step_size=dt, k1=k1, k2=k2))
    return result(
        y,
        history,
        None,
        True,
        "RK2 completed.",
        t=np.array([row["t"] for row in history]),
        y=np.array([row["y"] for row in history]),
        error_kind="Error estimate",
        note="Fixed-step midpoint RK2 has global O(h^2) error for smooth nonstiff problems.",
    )

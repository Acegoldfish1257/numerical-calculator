import math
import numpy as np
from core.validation import checked, interval, positive, scalar
from core.results import result


@checked
def rk4(f, t0, y0, t_end, h):
    """Classical RK4: y_next=y+h*(k1+2*k2+2*k3+k4)/6.

    k1=f(t,y), k2=f(t+h/2,y+h*k1/2), k3=f(t+h/2,y+h*k2/2),
    k4=f(t+h,y+h*k3). Inputs: scalar real f(t,y), initial condition,
    t_end>t0, positive h. Output: final y, all steps and trajectory. Last step
    is shortened to hit t_end. No intrinsic error estimate. Explicit RK4 is
    not suitable for all stiff problems; excessive step counts are rejected.
    """
    t0, t_end = interval(t0, t_end)
    y = scalar(y0)
    h = positive(h, "Step size")
    count = math.ceil((t_end - t0) / h)
    if count > 10000:
        raise ValueError("RK4 is limited to 10000 steps; increase h.")
    if t0 + h == t0:
        raise ValueError("Step is too small at the starting time.")
    times = [t0]
    values = [y]
    history = []

    def rhs(t, v):
        try:
            return scalar(f(t, v), "ODE right-hand side")
        except Exception as exc:
            raise ValueError(f"ODE is undefined at t={t:g}, y={v:g}: {exc}") from exc

    for i in range(count):
        t = times[-1]
        end = min(t0 + (i + 1) * h, t_end)
        step = end - t
        if step <= 0:
            break
        k1 = rhs(t, y)
        k2 = rhs(t + step / 2, y + step * k1 / 2)
        k3 = rhs(t + step / 2, y + step * k2 / 2)
        k4 = rhs(end, y + step * k3)
        new = scalar(y + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6, "ODE solution")
        history.append(
            dict(
                step=i + 1, t_start=t, t=end, h=step, k1=k1, k2=k2, k3=k3, k4=k4, y=new
            )
        )
        y = new
        times.append(end)
        values.append(y)
    return result(
        y,
        history,
        t=np.array(times),
        y=np.array(values),
        note="Fixed-step RK4 has global O(h^4) error for sufficiently smooth nonstiff problems.",
    )

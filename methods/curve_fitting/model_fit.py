import numpy as np
from core.validation import checked, points, array
from core.results import result


@checked
def fit_model(x, y, model="exponential"):
    """Fit a named nonlinear model by its linearized form.

    ``exponential`` fits y=a*b**x using ln(y)=ln(a)+x ln(b), requiring y>0.
    ``power`` fits y=a*x**b using ln(y)=ln(a)+b ln(x), requiring x,y>0.
    Returns a,b, the equation, fitted values, RMSE and the transformed linear
    coefficients. Log-linearization weights relative error rather than y error.
    """
    x,y,_=points(x,y)
    if model=="exponential":
        if np.any(y<=0): raise ValueError("Exponential fitting requires every y value to be positive.")
        X=x; Y=np.log(y); label="y = a * b^x"
    elif model=="power":
        if np.any(x<=0) or np.any(y<=0): raise ValueError("Power fitting requires positive x and y values.")
        X=np.log(x); Y=np.log(y); label="y = a * x^b"
    else: raise ValueError("Choose exponential or power model.")
    intercept,slope=np.linalg.lstsq(np.column_stack((np.ones_like(X),X)),Y,rcond=None)[0]
    a=float(np.exp(intercept)); b=float(np.exp(slope) if model=="exponential" else slope)
    fitted=a*b**x if model=="exponential" else a*x**b
    residual=fitted-y
    equation=f"y = {a:.10g} * {b:.10g}^x" if model=="exponential" else f"y = {a:.10g} * x^{b:.10g}"
    return result(fitted,[dict(x=xi,observed=yi,fitted=fi,residual=ri) for xi,yi,fi,ri in zip(x,y,fitted,residual)],float(np.sqrt(np.mean(residual**2))),error_kind="Original-scale RMSE",coefficients={"a":a,"b":b},equation=equation,linear_form="ln(y) = ln(a) + x*ln(b)" if model=="exponential" else "ln(y) = ln(a) + b*ln(x)",linear_coefficients={"intercept":float(intercept),"slope":float(slope)},model=model)

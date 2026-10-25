import numpy as np
from core.validation import checked, scalar, positive, evaluate
from core.results import result


@checked
def theorem_check(f, a, b, theorem="bolzano", derivative=None):
    """Numerically check a theorem's hypotheses on [a,b] and report evidence.

    Supported theorem labels: continuity, bolzano/ivt, rolle, mvt. A numerical
    grid cannot prove continuity; it reports sampled finiteness and hypotheses.
    Rolle requires f(a)=f(b), MVT reports a point where f'(c) is close to the
    secant slope when a derivative is supplied. Results are evidence, not proof.
    """
    a,b=scalar(a),scalar(b)
    if a>=b: raise ValueError("a must be smaller than b.")
    x=np.linspace(a,b,201); values=np.array([evaluate(f,v) for v in x]); fa,fb=values[0],values[-1]
    if theorem in ("bolzano","ivt"):
        satisfied=fa==0 or fb==0 or fa*fb<0; message="Bolzano/IVT sign-change evidence" if satisfied else "No sign change: IVT does not guarantee a zero on this interval."
    elif theorem=="continuity":
        jumps=np.max(np.abs(np.diff(values))); satisfied=np.all(np.isfinite(values)); message="All sampled values are finite; continuity cannot be proved from samples."
    elif theorem=="rolle":
        satisfied=abs(fa-fb)<=1e-8; message="Endpoint equality supports Rolle's hypothesis." if satisfied else "Rolle requires f(a)=f(b)."
    elif theorem=="mvt":
        if derivative is None: raise ValueError("MVT requires a derivative function.")
        target=(fb-fa)/(b-a); d=np.array([evaluate(derivative,v) for v in x]); i=int(np.argmin(abs(d-target))); satisfied=abs(d[i]-target)<1e-3; message=f"Closest sampled c={x[i]:.6g}; derivative={d[i]:.6g}, secant={target:.6g}."
    else: raise ValueError("Unknown theorem.")
    return result(satisfied,[dict(x=xi,value=yi) for xi,yi in zip(x,values)],None,None,message,theorem=theorem,endpoint_values=(fa,fb))


@checked
def taylor_remainder_bound(derivative_bound, degree, center, x):
    """Return M*|x-c|^(n+1)/(n+1)! for Taylor's Lagrange remainder bound."""
    from math import factorial
    M=positive(derivative_bound,"Derivative bound"); n=int(degree); c=scalar(center); x=scalar(x)
    if n<0 or n>50: raise ValueError("Degree must be between 0 and 50.")
    bound=M*abs(x-c)**(n+1)/factorial(n+1)
    return result(bound,[dict(derivative_bound=M,degree=n,center=c,x=x,bound=bound)],bound,error_kind="Conditional Taylor remainder bound")

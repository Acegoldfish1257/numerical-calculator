import numpy as np
from core.validation import checked, interval, integer, evaluate
from core.results import result


@checked
def simpson_three_eighth(f, a, b, n):
    """Composite Simpson 3/8 rule with weights 1,3,3,2,...,3,3,1.

    ``n`` must be a positive multiple of three. The rule is fourth order for
    smooth integrands; the refinement difference is an estimate, not a proof.
    """
    a, b = interval(a, b); n = integer(n, "Subintervals", 3, 5000)
    if n % 3: raise ValueError("Simpson's 3/8 rule requires a multiple of 3 subintervals.")
    def compute(count):
        x=np.linspace(a,b,count+1); h=(b-a)/count; y=np.array([evaluate(f,v) for v in x])
        w=np.ones(count+1); w[1:-1:3]=3; w[2:-1:3]=3; w[3:-1:3]=2; w*=3*h/8
        return float(w@y),x,y,w
    q,x,y,w=compute(n); fine=compute(2*n)[0]
    return result(q,[dict(node=i,x=xi,value=yi,weight=wi,contribution=wi*yi) for i,(xi,yi,wi) in enumerate(zip(x,y,w))],abs(fine-q)*16/15,error_kind="Estimated refinement error (O(h⁴))",refined_integral=fine)

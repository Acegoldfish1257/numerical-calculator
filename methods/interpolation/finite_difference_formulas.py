import numpy as np
from core.validation import checked, array, scalar
from core.results import result


@checked
def difference_formula(x, y, point, formula="forward"):
    """Newton forward/backward and central Gauss-style first differences.

    Nodes must be equally spaced. Forward uses the first node, backward the
    last node, and Gauss forward uses the central node with a symmetric table.
    Returns the full difference table and the selected first-difference value.
    """
    x=array(x,"Nodes"); y=array(y,"Values"); t=scalar(point,"Evaluation point")
    if len(x)!=len(y) or len(x)<2: raise ValueError("Need at least two matching nodes.")
    h=np.diff(x)
    if not np.allclose(h,h[0]): raise ValueError("Difference formulas require equally spaced nodes.")
    if formula not in ("forward","backward","gauss_forward"): raise ValueError("Formula must be forward, backward, or gauss_forward.")
    table=[y.copy()]
    while len(table[-1])>1: table.append(np.diff(table[-1]))
    if formula=="forward":
        index=0; value=table[1][0]
    elif formula=="backward":
        index=len(y)-2; value=table[1][-1]
    else:
        index=(len(y)-1)//2; value=table[1][min(index,len(table[1])-1)]
    return result(float(value),[dict(order=j,values=row.tolist()) for j,row in enumerate(table)],None,formula=formula,step=float(h[0]),base_node=float(x[index]),point=t)

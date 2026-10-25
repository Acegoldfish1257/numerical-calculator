import math
import numpy as np
import pytest
from methods.integration.simpson_three_eighth import simpson_three_eighth
from methods.curve_fitting.model_fit import fit_model
from methods.theory.calculus_theorems import theorem_check, taylor_remainder_bound
from methods.interpolation.finite_difference_formulas import difference_formula
from methods.root_finding.fixed_point import fixed_point
from services.catalog import root_run

def test_simpson_three_eighth():
    assert simpson_three_eighth(lambda x:x**3,0,1,3)["result"]==pytest.approx(.25)
    with pytest.raises(ValueError): simpson_three_eighth(math.exp,0,1,4)

def test_model_fit_equation():
    x=np.arange(5.); y=3*2**x
    r=fit_model(x,y,"exponential")
    assert r["diagnostics"]["coefficients"]["a"]==pytest.approx(3)
    assert r["diagnostics"]["coefficients"]["b"]==pytest.approx(2)
    assert r["diagnostics"]["equation"].startswith("y =")

def test_theory_and_difference():
    assert theorem_check(lambda x:x*x-1,-2,0,"bolzano")["result"]
    assert theorem_check(lambda x:x*x,0,0.5,"continuity")["result"]
    assert taylor_remainder_bound(2,2,0,1)["result"]==pytest.approx(1/3)
    assert difference_formula([0,1,2],[0,1,4],1.5,"forward")["result"]==pytest.approx(1)

def test_root_validation():
    with pytest.raises(ValueError,match="opposite signs"):
        root_run({"f":"x^2+1","a":-1,"b":1,"tol":1e-8,"max_iter":10})
    with pytest.raises(ValueError,match="below 1"):
        root_run({"f":"x^2-2","g":"2*x","x0":1,"tol":1e-8,"max_iter":10}, __import__('methods.root_finding.fixed_point',fromlist=['fixed_point']).fixed_point)

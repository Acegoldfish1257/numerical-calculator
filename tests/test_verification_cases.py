import pytest
from examples.verification_cases import cases, evaluate_case


@pytest.mark.parametrize("case", cases(), ids=lambda c: c.method)
def test_demonstration_case(case):
    actual, error, passed = evaluate_case(case)
    assert (
        passed
    ), f"{case.method}: actual {actual}, error {error}, tolerance {case.tolerance}"

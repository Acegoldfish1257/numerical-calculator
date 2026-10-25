from methods.computer_arithmetic.significant_digits import significant_digits


def test_round_and_truncate():
    assert significant_digits("123.456", 4)["result"] == "123.5"
    assert significant_digits("-123.456", 4, mode="truncate")["result"] == "-123.4"
    assert significant_digits("0", 3)["error_estimate"] == 0

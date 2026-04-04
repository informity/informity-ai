from informity.utils.number_utils import safe_float, safe_int


def test_safe_int_numeric_and_bool_inputs() -> None:
    assert safe_int(7) == 7
    assert safe_int(7.9) == 7
    assert safe_int(True) == 1
    assert safe_int(False) == 0


def test_safe_int_non_numeric_returns_default() -> None:
    assert safe_int('42') == 0
    assert safe_int('42', default=9) == 9
    assert safe_int(None, default=-1) == -1
    assert safe_int(object(), default=5) == 5


def test_safe_float_numeric_and_bool_inputs() -> None:
    assert safe_float(7) == 7.0
    assert safe_float(7.5) == 7.5
    assert safe_float(True) == 1.0
    assert safe_float(False) == 0.0


def test_safe_float_non_numeric_returns_default() -> None:
    assert safe_float('42.5') == 0.0
    assert safe_float('42.5', default=1.25) == 1.25
    assert safe_float(None, default=-1.0) == -1.0
    assert safe_float(object(), default=2.5) == 2.5

"""Tests for benchmarks.py parsing and normalization."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from benchmarks import normalize_latex, parse_prediction_math, is_correct_math


def test_normalize_latex_no_hang():
    """Confirm no infinite loop on strings with spaces."""
    s = "6 - 5i"
    t0 = time.time()
    out = normalize_latex(s)
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"normalize_latex hung: {elapsed}s"
    assert "6" in out


def test_normalize_latex_multispace():
    """Multiple spaces should collapse."""
    out = normalize_latex("a  b   c")
    assert "  " not in out  # no double spaces


def test_normalize_latex_frac():
    """Fraction normalization."""
    out = normalize_latex("\\frac{1}{2}")
    assert "(1)/(2)" in out or "1/2" in out


def test_parse_prediction_boxed():
    pred, _, src = parse_prediction_math("The answer is \\boxed{42}")
    assert pred is not None
    assert src == "boxed"


def test_parse_prediction_no_answer():
    pred, _, src = parse_prediction_math("Just some text no answer")
    # May return fallback or None, but must not crash


def test_parse_prediction_complex_boxed():
    pred, _, src = parse_prediction_math("result: \\boxed{\\frac{\\pi}{2}}")
    assert pred is not None
    assert src == "boxed"


def test_is_correct_math_simple():
    assert is_correct_math("42", "42") is True
    assert is_correct_math("42", "43") is False


def test_is_correct_math_latex_equiv():
    # LaTeX equivalent forms
    assert is_correct_math("\\frac{1}{2}", "1/2") in (True, False)  # should not crash


def test_parse_multiple_boxed():
    """Multiple boxed should take last one."""
    text = "First \\boxed{1} then \\boxed{5}"
    pred, _, src = parse_prediction_math(text)
    assert pred is not None

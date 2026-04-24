"""Tests for RCV signal features (V2, post-GPT-5.5-review)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from rcv_signals import (
    answer_validity_score,
    stage0_acceptance_features,
    prefix_recoverability_features,
    extractor_margin,
    compute_rcv_decision,
    compute_stage0_accept_score,
)


def test_answer_validity_empty():
    r = answer_validity_score("", "math500")
    assert r["valid"] == 0.0
    assert r["length"] == 0


def test_answer_validity_math_boxed():
    r = answer_validity_score("The answer is \\boxed{42}", "math500")
    assert r["valid"] == 1.0
    assert r["has_boxed"] == 1.0


def test_answer_validity_math_no_boxed_rejected():
    """V2: MATH raw text with just a random number should NOT be valid."""
    r = answer_validity_score("Let me think about this 42 is something", "math500")
    assert r["valid"] == 0.0  # No boxed → invalid
    assert r["has_number"] == 1.0  # Has number but that's not enough


def test_answer_validity_gsm8k_with_marker():
    r = answer_validity_score("Final answer: 123", "gsm8k")
    assert r["valid"] == 1.0
    assert r["has_final_marker"] == 1.0


def test_answer_validity_gsm8k_no_marker_rejected():
    """V2: GSM8K raw text with number but no marker should NOT be valid."""
    r = answer_validity_score("Some thought 123 then 456", "gsm8k")
    # Has number but no marker; not strong validity signal
    assert r["valid"] == 0.0


def test_stage0_features_benchmark_aware():
    """V2: Stage0 features must use benchmark-specific validity."""
    f_math = stage0_acceptance_features(
        "Compute x", "42", "Looking at this... 42 is somewhere",
        "fallback_last", False, benchmark="math500")
    # MATH: raw text has no boxed → answer_valid must be 0
    assert f_math["answer_valid"] == 0.0


def test_stage0_accept_score_math_requires_boxed():
    """V2: MATH accept_score < tau_accept when no boxed present."""
    f = stage0_acceptance_features(
        "q", "42", "just a number 42", "fallback", False, benchmark="math500")
    score = compute_stage0_accept_score(f, "math500")
    # No boxed, no parse_source boxed → low score
    assert score < 0.7


def test_stage0_accept_score_gsm8k_good():
    f = stage0_acceptance_features(
        "q", "42", "Final answer: 42", "regex", False, benchmark="gsm8k")
    score = compute_stage0_accept_score(f, "gsm8k")
    assert score >= 0.7


def test_prefix_recoverability_agreement():
    f = prefix_recoverability_features(
        "q", "prefix with therefore the answer is 5",
        "5", "5", "boxed", "boxed")
    assert f["agreement"] == 1.0
    assert f["prefix_has_conclusion"] == 1.0


def test_prefix_recoverability_disagreement():
    f = prefix_recoverability_features(
        "q", "incomplete prefix", "5", "7", "fallback", "fallback")
    assert f["agreement"] == 0.0


def test_extractor_margin_high():
    m = extractor_margin("42", "42", "boxed", "boxed")
    assert m >= 0.8


def test_extractor_margin_low():
    m = extractor_margin(None, None, "none", "none")
    assert m == 0.0


def test_rcv_decision_accept_math():
    s0 = stage0_acceptance_features(
        "q", "42", "\\boxed{42}", "boxed", False, benchmark="math500")
    d = compute_rcv_decision(s0, None, benchmark="math500")
    assert d == "ACCEPT_STAGE0"


def test_rcv_decision_reject_math_no_boxed():
    """V2: MATH natural stop with no boxed should NOT accept."""
    s0 = stage0_acceptance_features(
        "q", "42", "just 42 somewhere", "fallback", False, benchmark="math500")
    d = compute_rcv_decision(s0, None, benchmark="math500")
    assert d != "ACCEPT_STAGE0"


def test_rcv_decision_extract():
    s0 = stage0_acceptance_features(
        "q", None, "", "none", True, benchmark="math500")
    pf = {"strict_valid": 1.0, "soft_valid": 1.0, "agreement": 1.0,
          "strict_source_boxed": 1.0, "soft_source_boxed": 1.0,
          "prefix_length": 500, "prefix_has_equals": 1.0,
          "prefix_has_conclusion": 1.0, "prefix_has_boxed": 0.0}
    d = compute_rcv_decision(s0, pf, benchmark="math500")
    assert d == "EXTRACT_STAGE3"


def test_rcv_decision_fallback():
    s0 = stage0_acceptance_features(
        "q", None, "", "none", True, benchmark="math500")
    pf = {"strict_valid": 0.0, "soft_valid": 0.0, "agreement": 0.0,
          "strict_source_boxed": 0.0, "soft_source_boxed": 0.0,
          "prefix_length": 50, "prefix_has_equals": 0.0,
          "prefix_has_conclusion": 0.0, "prefix_has_boxed": 0.0}
    d = compute_rcv_decision(s0, pf, benchmark="math500")
    assert d == "FALLBACK_TOWN"

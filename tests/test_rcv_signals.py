"""Tests for RCV signal features."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from rcv_signals import (
    answer_validity_score,
    stage0_acceptance_features,
    prefix_recoverability_features,
    extractor_margin,
    compute_rcv_decision,
)


def test_answer_validity_empty():
    r = answer_validity_score("", "math500")
    assert r["valid"] == 0.0
    assert r["length"] == 0


def test_answer_validity_boxed():
    r = answer_validity_score("The answer is \\boxed{42}", "math500")
    assert r["valid"] == 1.0
    assert r["has_boxed"] == 1.0
    assert r["has_number"] == 1.0


def test_answer_validity_number_gsm8k():
    r = answer_validity_score("Final answer: 123", "gsm8k")
    assert r["valid"] == 1.0
    assert r["has_number"] == 1.0


def test_stage0_features_natural_stop():
    f = stage0_acceptance_features("What is 2+2?", "4", "4", "regex", False)
    assert f["natural_stop"] == 1.0
    assert f["pred_is_none"] == 0.0


def test_stage0_features_hit_budget():
    f = stage0_acceptance_features("Hard question", None, "partial reasoning...", "none", True)
    assert f["natural_stop"] == 0.0
    assert f["pred_is_none"] == 1.0


def test_prefix_recoverability_agreement():
    f = prefix_recoverability_features(
        "q", "prefix with therefore the answer is 5",
        "5", "5", "boxed", "boxed"
    )
    assert f["agreement"] == 1.0
    assert f["strict_valid"] == 1.0
    assert f["prefix_has_conclusion"] == 1.0


def test_prefix_recoverability_disagreement():
    f = prefix_recoverability_features(
        "q", "incomplete prefix",
        "5", "7", "fallback", "fallback"
    )
    assert f["agreement"] == 0.0


def test_extractor_margin_high():
    m = extractor_margin("42", "42", "boxed", "boxed")
    assert m >= 0.8


def test_extractor_margin_low():
    m = extractor_margin(None, None, "none", "none")
    assert m == 0.0


def test_rcv_decision_accept():
    s0 = {"natural_stop": 1.0, "answer_valid": 1.0, "pred_is_none": 0.0,
           "parse_source_boxed": 1.0, "parse_source_fallback": 0.0,
           "answer_has_number": 1.0, "answer_length": 10,
           "question_length_chars": 20, "question_has_latex": 0.0}
    d = compute_rcv_decision(s0, None)
    assert d == "ACCEPT_STAGE0"


def test_rcv_decision_extract():
    s0 = {"natural_stop": 0.0, "answer_valid": 0.0, "pred_is_none": 1.0,
           "parse_source_boxed": 0.0, "parse_source_fallback": 0.0,
           "answer_has_number": 0.0, "answer_length": 0,
           "question_length_chars": 50, "question_has_latex": 1.0}
    pf = {"strict_valid": 1.0, "soft_valid": 1.0, "agreement": 1.0,
          "strict_source_boxed": 1.0, "soft_source_boxed": 1.0,
          "prefix_length": 500, "prefix_has_equals": 1.0,
          "prefix_has_conclusion": 1.0, "prefix_has_boxed": 0.0}
    d = compute_rcv_decision(s0, pf)
    assert d == "EXTRACT_STAGE3"


def test_rcv_decision_fallback():
    s0 = {"natural_stop": 0.0, "answer_valid": 0.0, "pred_is_none": 1.0,
           "parse_source_boxed": 0.0, "parse_source_fallback": 0.0,
           "answer_has_number": 0.0, "answer_length": 0,
           "question_length_chars": 50, "question_has_latex": 0.0}
    pf = {"strict_valid": 0.0, "soft_valid": 0.0, "agreement": 0.0,
          "strict_source_boxed": 0.0, "soft_source_boxed": 0.0,
          "prefix_length": 50, "prefix_has_equals": 0.0,
          "prefix_has_conclusion": 0.0, "prefix_has_boxed": 0.0}
    d = compute_rcv_decision(s0, pf)
    assert d == "FALLBACK_TOWN"

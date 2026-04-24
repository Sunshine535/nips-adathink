"""RCV signal features for Recoverability-Calibrated Verifier IRIS.

Pure feature computation — no model calls. Used by rcv_verifier.py.

V2 (GPT-5.5 review fixes):
- `stage0_acceptance_features` takes `benchmark` parameter (not hard-coded gsm8k)
- MATH validity requires parsed boxed / final marker, not just any number
- Adds tests for NO_OP_STAGE0_GATE detection
"""
import re
from typing import Dict, Optional


def answer_validity_score(text: str, benchmark: str,
                          parse_source: Optional[str] = None) -> Dict[str, float]:
    """Score answer validity based on format and parsing signals.

    For MATH: requires parsed boxed or explicit final marker — not just any digit.
    For GSM8K: requires at least one number in the trailing portion.
    """
    if not text or not text.strip():
        return {"valid": 0.0, "has_number": 0.0, "has_boxed": 0.0,
                "has_final_marker": 0.0, "length": 0}

    stripped = text.strip()
    has_boxed = 1.0 if ("\\boxed{" in stripped or "\\boxed " in stripped) else 0.0
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", stripped)
    has_number = 1.0 if nums else 0.0

    has_final_marker = 0.0
    low = stripped.lower()
    for marker in ["final answer", "the answer is", "answer:", "\\boxed", "####"]:
        if marker.lower() in low:
            has_final_marker = 1.0
            break

    if benchmark == "math500":
        # MATH requires explicit boxed OR parse_source == 'boxed'
        valid = 1.0 if (has_boxed or (parse_source == "boxed")) else 0.0
    else:
        # GSM8K requires number AND (final marker OR parse_source in boxed/regex_last)
        if has_number and (has_final_marker or parse_source in ("boxed", "regex")):
            valid = 1.0
        else:
            valid = 0.0

    return {
        "valid": valid,
        "has_number": has_number,
        "has_boxed": has_boxed,
        "has_final_marker": has_final_marker,
        "length": len(stripped),
    }


def stage0_acceptance_features(
    question: str,
    pred: Optional[str],
    raw_text: str,
    parse_source: str,
    hit_budget: bool,
    benchmark: str = "gsm8k",
) -> Dict[str, float]:
    """Features for Stage0 acceptance decision.

    V2: benchmark-aware. MATH requires boxed/marker, not just any digit.
    """
    validity = answer_validity_score(raw_text, benchmark, parse_source)

    return {
        "natural_stop": 0.0 if hit_budget else 1.0,
        "pred_is_none": 1.0 if pred is None else 0.0,
        "pred_is_empty": 1.0 if (pred is None or str(pred).strip() == "") else 0.0,
        "parse_source_boxed": 1.0 if parse_source == "boxed" else 0.0,
        "parse_source_fallback": 1.0 if "fallback" in str(parse_source) else 0.0,
        "answer_valid": validity["valid"],
        "answer_has_number": validity["has_number"],
        "answer_has_boxed": validity["has_boxed"],
        "answer_has_final_marker": validity["has_final_marker"],
        "answer_length": validity["length"],
        "question_length_chars": len(question),
        "question_has_latex": 1.0 if "\\" in question else 0.0,
    }


def prefix_recoverability_features(
    question: str,
    prefix: str,
    strict_pred: Optional[str],
    soft_pred: Optional[str],
    strict_source: str,
    soft_source: str,
) -> Dict[str, float]:
    """Features for prefix recoverability estimation."""
    strict_valid = strict_pred is not None and str(strict_pred).strip() != ""
    soft_valid = soft_pred is not None and str(soft_pred).strip() != ""

    agreement = 0.0
    if strict_valid and soft_valid:
        agreement = 1.0 if str(strict_pred).strip() == str(soft_pred).strip() else 0.0

    prefix_len = len(prefix.strip()) if prefix else 0
    has_equals = 1.0 if "=" in prefix else 0.0
    has_therefore = 1.0 if any(w in prefix.lower() for w in [
        "therefore", "thus", "so the answer", "final answer", "hence"]) else 0.0
    has_boxed = 1.0 if "\\boxed" in prefix else 0.0

    return {
        "strict_valid": 1.0 if strict_valid else 0.0,
        "soft_valid": 1.0 if soft_valid else 0.0,
        "agreement": agreement,
        "strict_source_boxed": 1.0 if strict_source == "boxed" else 0.0,
        "soft_source_boxed": 1.0 if soft_source == "boxed" else 0.0,
        "prefix_length": prefix_len,
        "prefix_has_equals": has_equals,
        "prefix_has_conclusion": has_therefore,
        "prefix_has_boxed": has_boxed,
    }


def extractor_margin(
    strict_pred: Optional[str],
    soft_pred: Optional[str],
    strict_source: str,
    soft_source: str,
) -> float:
    """Scalar margin: higher = more confident extraction will succeed."""
    score = 0.0
    if strict_pred is not None and str(strict_pred).strip() != "":
        score += 0.3
    if soft_pred is not None and str(soft_pred).strip() != "":
        score += 0.2
    if strict_source == "boxed":
        score += 0.2
    if soft_source == "boxed":
        score += 0.1
    if strict_pred and soft_pred:
        if str(strict_pred).strip() == str(soft_pred).strip():
            score += 0.2
    return min(score, 1.0)


def compute_stage0_accept_score(features: Dict[str, float], benchmark: str = "gsm8k") -> float:
    """Compute Stage0 acceptance score. Returns value in [0,1]."""
    # V2: MATH requires boxed; GSM8K requires marker + number
    if benchmark == "math500":
        return (features["answer_has_boxed"] * 0.5
                + features["parse_source_boxed"] * 0.3
                + (1.0 - features["pred_is_empty"]) * 0.2)
    else:
        return (features["answer_valid"] * 0.5
                + (1.0 - features["pred_is_empty"]) * 0.3
                + features["answer_has_final_marker"] * 0.2)


def compute_rcv_decision(
    stage0_features: Dict[str, float],
    prefix_features: Optional[Dict[str, float]],
    tau_accept: float = 0.7,
    tau_recover: float = 0.5,
    benchmark: str = "gsm8k",
) -> str:
    """Decide action based on RCV signals.

    Returns one of: ACCEPT_STAGE0, EXTRACT_STAGE3, FALLBACK_TOWN, ABSTAIN
    """
    if stage0_features["natural_stop"] > 0.5:
        accept_score = compute_stage0_accept_score(stage0_features, benchmark)
        if accept_score >= tau_accept:
            return "ACCEPT_STAGE0"

    if prefix_features is not None:
        recover_score = (
            prefix_features["strict_valid"] * 0.3
            + prefix_features["soft_valid"] * 0.2
            + prefix_features["agreement"] * 0.3
            + prefix_features["prefix_has_conclusion"] * 0.1
            + prefix_features["prefix_has_boxed"] * 0.1
        )
        if recover_score >= tau_recover:
            return "EXTRACT_STAGE3"

    return "FALLBACK_TOWN"

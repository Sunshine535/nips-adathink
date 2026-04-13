#!/usr/bin/env python3
"""Rescore MRSD MATH-500 pilot results using current parser logic.

Self-contained: copies only the math parsing/comparison functions from
benchmarks.py so no external dependencies are needed.

Usage:
    python scripts/rescore_mrsd_math500.py
"""

import json
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Parser functions (copied from benchmarks.py to avoid datasets import)
# ---------------------------------------------------------------------------

NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def _extract_boxed_brace_counting(text: str, start_idx: int) -> Optional[str]:
    open_pos = text.find("{", start_idx)
    if open_pos == -1:
        return None
    depth = 1
    i = open_pos + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[open_pos + 1 : i - 1].strip()
    return text[open_pos + 1 :].strip()


def extract_boxed(text: str) -> Optional[str]:
    if not text:
        return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed")
        if idx == -1:
            return None
    return _extract_boxed_brace_counting(text, idx)


def extract_last_number(text: str) -> Optional[str]:
    if not text:
        return None
    matches = NUM_RE.findall(text)
    return matches[-1] if matches else None


def to_float(num_str: Optional[str]) -> Optional[float]:
    if num_str is None:
        return None
    s = num_str.replace(",", "").strip()
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                denom = float(parts[1])
                if denom == 0:
                    return None
                return float(parts[0]) / denom
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def normalize_latex(s: str) -> str:
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("\\text{", "").replace("\\mathrm{", "").replace("\\mathbf{", "")
    s = s.replace("\\textbf{", "")
    s = s.replace("\\%", "%")
    s = s.replace("\\$", "$")
    while "  " in s:
        s = s.replace("  ", " ")

    def _replace_frac(text):
        for cmd in ("\\dfrac", "\\frac"):
            while cmd + "{" in text:
                idx = text.find(cmd + "{")
                after = idx + len(cmd)
                num = _extract_boxed_brace_counting(text, after)
                if num is None:
                    break
                open_pos = text.find("{", after)
                depth, i = 1, open_pos + 1
                while i < len(text) and depth > 0:
                    if text[i] == "{": depth += 1
                    elif text[i] == "}": depth -= 1
                    i += 1
                den = _extract_boxed_brace_counting(text, i)
                if den is None:
                    break
                open_pos2 = text.find("{", i)
                depth2, j = 1, open_pos2 + 1
                while j < len(text) and depth2 > 0:
                    if text[j] == "{": depth2 += 1
                    elif text[j] == "}": depth2 -= 1
                    j += 1
                text = text[:idx] + f"({num})/({den})" + text[j:]
        return text
    s = _replace_frac(s)

    def _replace_sqrt(text):
        cmd = "\\sqrt{"
        while cmd in text:
            idx = text.find(cmd)
            content = _extract_boxed_brace_counting(text, idx + len("\\sqrt"))
            if content is None:
                break
            open_pos = text.find("{", idx + len("\\sqrt"))
            depth, i = 1, open_pos + 1
            while i < len(text) and depth > 0:
                if text[i] == "{": depth += 1
                elif text[i] == "}": depth -= 1
                i += 1
            text = text[:idx] + f"sqrt({content})" + text[i:]
        text = re.sub(r"\\sqrt([a-zA-Z0-9])", r"sqrt(\1)", text)
        return text
    s = _replace_sqrt(s)

    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\pi", "pi").replace("\\infty", "inf")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    s = s.strip()
    return s


def _strip_variable_assignment(s: str) -> str:
    m = re.match(r"^[a-zA-Z]\s*=\s*(.+)$", s.strip())
    return m.group(1).strip() if m else s.strip()


def _try_eval_fraction(s: str) -> Optional[float]:
    s = s.strip().replace(" ", "")
    m = re.match(r"^\(?(-?[\d.]+)\)?\s*/\s*\(?(-?[\d.]+)\)?$", s)
    if m:
        try:
            denom = float(m.group(2))
            if denom == 0:
                return None
            return float(m.group(1)) / denom
        except ValueError:
            return None
    return to_float(s)


def math_answers_equiv(pred: str, gold: str) -> bool:
    pred_norm = normalize_latex(pred)
    gold_norm = normalize_latex(gold)
    if pred_norm == gold_norm:
        return True
    pred_val = _strip_variable_assignment(pred_norm)
    gold_val = _strip_variable_assignment(gold_norm)
    if pred_val == gold_val:
        return True
    for p, g in [(pred_norm, gold_norm), (pred_val, gold_val)]:
        pn = to_float(p)
        gn = to_float(g)
        if pn is not None and gn is not None:
            tol = 1e-4 * max(1.0, abs(gn))
            if abs(pn - gn) <= tol:
                return True
    for p, g in [(pred_val, gold_val), (pred_norm, gold_norm)]:
        pn = _try_eval_fraction(p)
        gn = _try_eval_fraction(g)
        if pn is not None and gn is not None:
            if abs(pn - gn) <= 1e-4 * max(1.0, abs(gn)):
                return True
    pred_clean = re.sub(r"\s+", "", pred_val).lower()
    gold_clean = re.sub(r"\s+", "", gold_val).lower()
    return pred_clean == gold_clean


def is_correct_math(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    return math_answers_equiv(pred, gold)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MRSD_PATH = "results/mrsd_pilot/mrsd_Qwen3_8B_math500_b1512_bt1024_ba256_r3_20260409_043123.json"
SPLIT_PATH = "results/split_budget/split_budget_Qwen3_8B_math500_n200_B512_1024_20260409_084433.json"
OUTPUT_PATH = "results/mrsd_pilot/mrsd_math500_rescored.json"


def main():
    with open(MRSD_PATH) as f:
        mrsd = json.load(f)
    with open(SPLIT_PATH) as f:
        split = json.load(f)

    # Build split-budget lookups
    split_nt512 = {}
    split_nt1024 = {}
    for s in split["per_sample"]:
        if s["method"] == "nothink" and s["B_total"] == 512:
            split_nt512[s["idx"]] = s
        elif s["method"] == "nothink" and s["B_total"] == 1024:
            split_nt1024[s["idx"]] = s

    n = len(mrsd["per_sample"])
    mrsd_orig_correct = 0
    mrsd_rescored_correct = 0
    s0_orig_correct = 0
    s0_rescored_correct = 0
    nothink_orig_correct = 0

    flipped_mrsd = []
    flipped_s0 = []

    for s in mrsd["per_sample"]:
        idx = s["idx"]
        gold = s["gold"]
        mrsd_orig_correct += s["mrsd_correct"]
        nothink_orig_correct += s["nothink_correct"]

        # Rescore MRSD
        mrsd_pred = s["mrsd_pred"]
        mrsd_new = 1 if is_correct_math(mrsd_pred, gold) else 0
        mrsd_rescored_correct += mrsd_new
        if mrsd_new != s["mrsd_correct"]:
            flipped_mrsd.append({
                "idx": idx, "gold": gold, "pred": mrsd_pred,
                "old": s["mrsd_correct"], "new": mrsd_new
            })

        # Rescore S0/nothink
        s0_pred = s["s0_pred"]
        s0_new = 1 if is_correct_math(s0_pred, gold) else 0
        s0_rescored_correct += s0_new
        if s0_new != s["nothink_correct"]:
            flipped_s0.append({
                "idx": idx, "gold": gold, "pred": s0_pred,
                "old": s["nothink_correct"], "new": s0_new
            })

    # Cross-experiment comparison
    cross_match = 0
    cross_mismatch = []
    for s in mrsd["per_sample"]:
        idx = s["idx"]
        if idx in split_nt512:
            sp = split_nt512[idx]
            s0_rescored = 1 if is_correct_math(s["s0_pred"], s["gold"]) else 0
            if s0_rescored == sp["correct"]:
                cross_match += 1
            else:
                cross_mismatch.append({
                    "idx": idx, "gold": s["gold"],
                    "mrsd_s0_pred": s["s0_pred"],
                    "mrsd_s0_rescored": s0_rescored,
                    "split_pred": sp["pred"],
                    "split_correct": sp["correct"]
                })

    results = {
        "original": {
            "mrsd_accuracy": round(mrsd_orig_correct / n, 4),
            "nothink_accuracy": round(nothink_orig_correct / n, 4),
            "n": n,
        },
        "rescored": {
            "mrsd_accuracy": round(mrsd_rescored_correct / n, 4),
            "s0_accuracy": round(s0_rescored_correct / n, 4),
            "n": n,
        },
        "delta": {
            "mrsd": round((mrsd_rescored_correct - mrsd_orig_correct) / n, 4),
            "nothink": round((s0_rescored_correct - nothink_orig_correct) / n, 4),
        },
        "flipped_mrsd": flipped_mrsd,
        "flipped_s0": flipped_s0,
        "cross_experiment": {
            "n_compared": len(split_nt512),
            "n_match": cross_match,
            "n_mismatch": len(cross_mismatch),
            "mismatches": cross_mismatch[:20],
        },
        "reference": {
            "split_nothink@512": split["aggregate"]["nothink@512"]["accuracy"],
            "split_nothink@1024": split["aggregate"]["nothink@1024"]["accuracy"],
            "split_think@512": split["aggregate"]["think@512"]["accuracy"],
            "split_think@1024": split["aggregate"]["think@1024"]["accuracy"],
            "split_town@1024": split["aggregate"]["town@1024"]["accuracy"],
            "split_best_split@1024": max(
                v["accuracy"]
                for k, v in split["aggregate"].items()
                if k.startswith("split_") and "@1024" in k
            ),
        },
    }

    print("=== MRSD MATH-500 Rescore Results ===")
    print(f"MRSD:    {mrsd_orig_correct/n:.1%} (orig) -> {mrsd_rescored_correct/n:.1%} (rescored)")
    gained = sum(1 for f in flipped_mrsd if f["new"] == 1)
    lost = sum(1 for f in flipped_mrsd if f["new"] == 0)
    print(f"  Flipped: {len(flipped_mrsd)} ({gained} gained, {lost} lost)")
    print(f"Nothink: {nothink_orig_correct/n:.1%} (orig) -> {s0_rescored_correct/n:.1%} (rescored)")
    gained_s0 = sum(1 for f in flipped_s0 if f["new"] == 1)
    lost_s0 = sum(1 for f in flipped_s0 if f["new"] == 0)
    print(f"  Flipped: {len(flipped_s0)} ({gained_s0} gained, {lost_s0} lost)")
    print(f"\nCross-experiment (MRSD s0 vs split nothink@512):")
    print(f"  Match: {cross_match}/{len(split_nt512)}, Mismatch: {len(cross_mismatch)}")
    print(f"\nSplit-budget reference:")
    for k, v in results["reference"].items():
        print(f"  {k}: {v:.1%}")

    if cross_mismatch:
        print(f"\nMismatches (pred text differs between experiments, same sample):")
        for m in cross_mismatch[:5]:
            print(f"  idx={m['idx']}: gold={m['gold'][:50]}")
            print(f"    MRSD s0: '{m['mrsd_s0_pred'][:50]}' -> {m['mrsd_s0_rescored']}")
            print(f"    Split:   '{m['split_pred'][:50]}' -> {m['split_correct']}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

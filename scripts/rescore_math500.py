#!/usr/bin/env python3
"""Re-score MATH-500 checkpoint files using the fixed brace-counting extract_boxed.

Since existing checkpoints don't store raw model output, this script:
1. Reloads the MATH-500 dataset to get correct gold answers (with fixed extract_boxed)
2. Re-evaluates correctness using fixed is_correct_math (better LaTeX normalization)
3. Reports old vs new scores and identifies samples affected by the fix

Usage:
    python scripts/rescore_math500.py --checkpoint results/mrsd_pilot/checkpoint_100.json
    python scripts/rescore_math500.py --checkpoint results/mrsd_pilot/checkpoint_100.json --save
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from benchmarks import load_math500, extract_boxed, is_correct_math, normalize_latex

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_gold_answers(n_samples: int = 500, seed: int = 42) -> dict:
    """Load MATH-500 gold answers with FIXED extract_boxed."""
    samples = load_math500()
    # Return map: idx -> gold answer (using fixed parser)
    golds = {}
    for i, s in enumerate(samples[:n_samples]):
        golds[i] = s.gold
    return golds


def rescore_checkpoint(path: str, gold_answers: dict, save: bool = False) -> dict:
    """Re-score a checkpoint with corrected gold answers and comparison."""
    log.info(f"Loading {path}...")
    with open(path) as f:
        data = json.load(f)

    # Handle checkpoint format: {"n_done": N, "per_sample": [...]}
    if isinstance(data, dict):
        samples = data.get("per_sample", data.get("samples", data.get("results", [])))
        metadata = {k: v for k, v in data.items() if k != "per_sample"}
    elif isinstance(data, list):
        samples = data
        metadata = {}
    else:
        log.error(f"Unknown format: {type(data)}")
        return {}

    if not samples:
        log.error("No samples found!")
        return {}

    n = len(samples)
    log.info(f"Re-scoring {n} samples...")

    # Track changes per field
    fields = [
        ("mrsd", "mrsd_pred", "mrsd_correct"),
        ("nothink", "s0_pred", "nothink_correct"),
        ("town", None, "town_correct"),  # no stored pred
        ("iris", None, "iris_correct"),    # no stored pred
    ]

    results = {f: {"old_correct": 0, "new_correct": 0, "flipped_up": [], "flipped_down": []}
               for f, _, _ in fields}

    rescored_samples = []

    for s in samples:
        idx = s.get("idx", 0)
        old_gold = s.get("gold", "")
        new_gold = gold_answers.get(idx, old_gold)

        rs = dict(s)
        rs["gold_fixed"] = new_gold
        rs["gold_changed"] = old_gold != new_gold

        for fname, pred_key, correct_key in fields:
            old_ok = bool(s.get(correct_key, 0))
            results[fname]["old_correct"] += int(old_ok)

            if pred_key and pred_key in s:
                pred = s[pred_key]
                new_ok = is_correct_math(str(pred), new_gold)
                results[fname]["new_correct"] += int(new_ok)
                rs[f"{fname}_correct_fixed"] = int(new_ok)

                if new_ok and not old_ok:
                    results[fname]["flipped_up"].append({
                        "idx": idx, "gold_old": old_gold, "gold_new": new_gold,
                        "pred": str(pred),
                    })
                elif old_ok and not new_ok:
                    results[fname]["flipped_down"].append({
                        "idx": idx, "gold_old": old_gold, "gold_new": new_gold,
                        "pred": str(pred),
                    })
            else:
                # Can't re-evaluate without stored prediction
                results[fname]["new_correct"] += int(old_ok)
                rs[f"{fname}_correct_fixed"] = int(old_ok)

        rescored_samples.append(rs)

    # Report
    log.info(f"\n{'='*60}")
    log.info(f"Re-scoring Results: {os.path.basename(path)}")
    log.info(f"{'='*60}")
    log.info(f"Samples: {n}")

    # Gold answer changes
    gold_changed = sum(1 for s in rescored_samples if s["gold_changed"])
    log.info(f"\nGold answers changed by fix: {gold_changed}/{n}")
    for s in rescored_samples:
        if s["gold_changed"]:
            log.info(f"  idx={s['idx']}: '{s['gold']}' → '{s['gold_fixed']}'")

    log.info("")
    summary = {"file": path, "n_samples": n, "gold_changed": gold_changed, "per_field": {}}

    for fname, _, correct_key in fields:
        r = results[fname]
        old_acc = r["old_correct"] / n
        new_acc = r["new_correct"] / n
        delta = r["new_correct"] - r["old_correct"]
        log.info(f"{fname:>8}: {r['old_correct']}/{n} ({100*old_acc:.1f}%) → "
                 f"{r['new_correct']}/{n} ({100*new_acc:.1f}%)  Δ={delta:+d}")

        if r["flipped_up"]:
            log.info(f"         ↑ Newly correct: {[f['idx'] for f in r['flipped_up']]}")
            for f in r["flipped_up"]:
                log.info(f"           idx={f['idx']}: pred='{f['pred']}' "
                         f"gold: '{f['gold_old']}' → '{f['gold_new']}'")
        if r["flipped_down"]:
            log.info(f"         ↓ REGRESSION:    {[f['idx'] for f in r['flipped_down']]}")
            for f in r["flipped_down"]:
                log.info(f"           idx={f['idx']}: pred='{f['pred']}' "
                         f"gold: '{f['gold_old']}' → '{f['gold_new']}'")

        summary["per_field"][fname] = {
            "old_correct": r["old_correct"], "new_correct": r["new_correct"],
            "old_accuracy": old_acc, "new_accuracy": new_acc,
            "delta": delta,
            "flipped_up": r["flipped_up"], "flipped_down": r["flipped_down"],
        }

    if save:
        out_path = path.replace(".json", "_rescored.json")
        output = {"metadata": {**metadata, "rescoring": summary}, "per_sample": rescored_samples}
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        log.info(f"\nSaved to {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Re-score MATH-500 checkpoints with fixed extract_boxed")
    parser.add_argument("--checkpoint", required=True, nargs="+", help="Checkpoint JSON file(s)")
    parser.add_argument("--save", action="store_true", help="Save rescored checkpoint")
    parser.add_argument("--n_math500", type=int, default=500, help="Number of MATH-500 samples")
    args = parser.parse_args()

    log.info("Loading MATH-500 gold answers with FIXED parser...")
    gold_answers = load_gold_answers(args.n_math500)
    log.info(f"Loaded {len(gold_answers)} gold answers")

    for cp in args.checkpoint:
        if not os.path.exists(cp):
            log.warning(f"File not found: {cp}")
            continue
        rescore_checkpoint(cp, gold_answers, save=args.save)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Split-Budget Experiment — Decoupling reasoning from answer generation.

Hypothesis ("Coupling Tax"):
    Under fixed output-token budgets, forcing reasoning AND answer into one
    output channel causes truncation waste. Thinking is 98.27% accurate when
    chains complete (α_c), but 98.6% of chains are truncated at tight budgets.

    By *splitting* a total budget B into Br (thinking) + Ba (answering),
    the model can reason partially and then extract an answer without
    truncation, recovering much of the lost accuracy.

Methods compared (all under the same total budget B):
    1. nothink@B     — direct answer, all tokens for output
    2. think@B       — all tokens for think+answer (often truncated)
    3. split@Br+Ba   — think@Br then nothink(trace)@Ba (KEY EXPERIMENT)
    4. town@B        — Stage 1 nothink@B, Stage 2 think@B on failures

For total budgets B ∈ {256, 512, 768, 1024, 1536, 2048}, sweep split ratios:
    Ba ∈ {64, 128, 256, B/4, B/2}  (answer budget)
    Br = B - Ba                      (thinking budget)

Supports: gsm8k, math500

Usage:
    # Quick pilot (50 samples, 2 budgets)
    python scripts/pilot_split_budget.py \
        --model Qwen/Qwen3-8B --benchmark gsm8k \
        --n_samples 50 --budgets 256 512 --seed 42

    # Full sweep (200 samples, all budgets)
    python scripts/pilot_split_budget.py \
        --model Qwen/Qwen3-8B --benchmark gsm8k \
        --n_samples 200 --seed 42

    # MATH-500
    python scripts/pilot_split_budget.py \
        --model Qwen/Qwen3-8B --benchmark math500 \
        --n_samples 200 --seed 42
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Reuse core utilities from run_iris.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_iris import (
    load_model_and_tokenizer,
    generate_simple,
    generate_decoupled_answer,
    build_prompt,
    parse_prediction_dispatch,
    is_correct_dispatch,
    load_benchmark_data,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BUDGETS = [256, 512, 768, 1024, 1536, 2048]
DEFAULT_BA_VALUES = [64, 128, 256]  # fixed answer budgets
FRACTIONAL_BA = [0.25, 0.5]         # Ba = fraction * B

THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_thinking_trace(text: str) -> str:
    """Extract reasoning between <think> and </think>.

    If </think> is missing (truncated), returns everything after <think>.
    """
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip()
    if "<think>" in text:
        return text.split("<think>", 1)[1].strip()
    return ""


def compute_split_configs(total_budget: int, ba_fixed: List[int],
                          ba_fractions: List[float]) -> List[Tuple[int, int]]:
    """Compute (Br, Ba) pairs for a given total budget B.

    Returns deduplicated list of (Br, Ba) where Br + Ba == B,
    Br >= 32, Ba >= 32.
    """
    seen = set()
    configs = []
    # Fixed answer budgets
    for ba in ba_fixed:
        br = total_budget - ba
        if br >= 32 and ba >= 32 and (br, ba) not in seen:
            seen.add((br, ba))
            configs.append((br, ba))
    # Fractional answer budgets
    for frac in ba_fractions:
        ba = int(total_budget * frac)
        br = total_budget - ba
        if br >= 32 and ba >= 32 and (br, ba) not in seen:
            seen.add((br, ba))
            configs.append((br, ba))
    # Sort by Br descending (more thinking first)
    configs.sort(key=lambda x: -x[0])
    return configs


# ---------------------------------------------------------------------------
# Method runners (single sample)
# ---------------------------------------------------------------------------
def run_nothink(model, tokenizer, question: str, budget: int,
                benchmark: str) -> Dict:
    """Method 1: nothink@B — direct answer generation."""
    prompt = build_prompt(question, tokenizer, enable_thinking=False,
                          benchmark=benchmark)
    text, n_tok, elapsed, hit = generate_simple(
        model, tokenizer, prompt, max_new_tokens=budget, temperature=0.0)
    pred, src = parse_prediction_dispatch(text, benchmark)
    return {
        "method": "nothink",
        "pred": pred, "pred_source": src,
        "tokens_used": n_tok, "hit_budget": hit,
        "elapsed": round(elapsed, 4),
    }


def run_think(model, tokenizer, question: str, budget: int,
              benchmark: str) -> Dict:
    """Method 2: think@B — thinking mode, entire budget for think+answer."""
    prompt = build_prompt(question, tokenizer, enable_thinking=True,
                          benchmark=benchmark)
    text, n_tok, elapsed, hit = generate_simple(
        model, tokenizer, prompt, max_new_tokens=budget, temperature=0.0)
    # Try to parse answer from after </think> first
    pred = None
    src = "none"
    if "</think>" in text:
        after = text.split("</think>", 1)[1]
        pred, src = parse_prediction_dispatch(after, benchmark)
    if pred is None:
        pred, src = parse_prediction_dispatch(text, benchmark)
    trace = extract_thinking_trace(text)
    truncated = "</think>" not in text
    return {
        "method": "think",
        "pred": pred, "pred_source": src,
        "tokens_used": n_tok, "hit_budget": hit,
        "elapsed": round(elapsed, 4),
        "truncated": truncated,
        "trace_len_chars": len(trace),
    }


def run_split(model, tokenizer, question: str, br: int, ba: int,
              benchmark: str) -> Dict:
    """Method 3: split@Br+Ba — think for Br tokens, then nothink with trace for Ba tokens."""
    # Stage A: thinking with budget Br
    prompt_think = build_prompt(question, tokenizer, enable_thinking=True,
                                benchmark=benchmark)
    text_think, tok_think, elapsed_think, hit_think = generate_simple(
        model, tokenizer, prompt_think, max_new_tokens=br, temperature=0.0)

    trace = extract_thinking_trace(text_think)
    truncated = "</think>" not in text_think

    # Check if thinking already produced a complete answer
    complete_pred = None
    if not truncated and "</think>" in text_think:
        after = text_think.split("</think>", 1)[1]
        complete_pred, _ = parse_prediction_dispatch(after, benchmark)

    # Stage B: decoupled answer with budget Ba
    ans_text, tok_ans, elapsed_ans = generate_decoupled_answer(
        model, tokenizer, question, trace,
        answer_budget=ba, benchmark=benchmark)
    pred, src = parse_prediction_dispatch(ans_text, benchmark)

    # If decoupled answer failed but thinking had complete answer, use that
    if pred is None and complete_pred is not None:
        pred = complete_pred
        src = "think_complete_fallback"

    return {
        "method": "split",
        "Br": br, "Ba": ba,
        "pred": pred, "pred_source": src,
        "tokens_think": tok_think, "tokens_answer": tok_ans,
        "tokens_used": tok_think + tok_ans,
        "hit_budget_think": hit_think,
        "elapsed": round(elapsed_think + elapsed_ans, 4),
        "truncated": truncated,
        "trace_len_chars": len(trace),
        "think_complete": not truncated,
    }


def run_town(model, tokenizer, question: str, budget: int,
             benchmark: str) -> Dict:
    """Method 4: TOWN — nothink@B first, think@B on failures."""
    # Stage 1: nothink
    prompt_s1 = build_prompt(question, tokenizer, enable_thinking=False,
                             benchmark=benchmark)
    text_s1, tok_s1, elapsed_s1, hit_s1 = generate_simple(
        model, tokenizer, prompt_s1, max_new_tokens=budget, temperature=0.0)
    pred_s1, src_s1 = parse_prediction_dispatch(text_s1, benchmark)

    if not hit_s1:
        # Easy — accepted at Stage 1
        return {
            "method": "town",
            "stage": 1,
            "pred": pred_s1, "pred_source": src_s1,
            "tokens_used": tok_s1, "hit_budget": False,
            "elapsed": round(elapsed_s1, 4),
        }

    # Stage 2: think
    prompt_s2 = build_prompt(question, tokenizer, enable_thinking=True,
                             benchmark=benchmark)
    text_s2, tok_s2, elapsed_s2, hit_s2 = generate_simple(
        model, tokenizer, prompt_s2, max_new_tokens=budget, temperature=0.0)
    pred_s2 = None
    src_s2 = "none"
    if "</think>" in text_s2:
        after = text_s2.split("</think>", 1)[1]
        pred_s2, src_s2 = parse_prediction_dispatch(after, benchmark)
    if pred_s2 is None:
        pred_s2, src_s2 = parse_prediction_dispatch(text_s2, benchmark)

    return {
        "method": "town",
        "stage": 2,
        "pred": pred_s2, "pred_source": src_s2,
        "tokens_s1": tok_s1, "tokens_s2": tok_s2,
        "tokens_used": tok_s1 + tok_s2,
        "hit_budget": hit_s2,
        "elapsed": round(elapsed_s1 + elapsed_s2, 4),
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run_experiment(
    model, tokenizer, items: List[Dict], benchmark: str,
    budgets: List[int], ba_fixed: List[int], ba_fractions: List[float],
    run_baselines: bool = True,
) -> Dict:
    """Run all methods across all budgets for all samples.

    Returns structured results dict with per-sample and aggregate data.
    """
    n = len(items)
    all_results = []        # flat list: one entry per (sample, method, budget)
    summary_table = {}      # {(method_key, B): {correct, total, tokens}}

    def update_summary(key: str, B: int, correct: bool, tokens: int):
        k = (key, B)
        if k not in summary_table:
            summary_table[k] = {"correct": 0, "total": 0, "tokens_sum": 0}
        summary_table[k]["total"] += 1
        summary_table[k]["correct"] += int(correct)
        summary_table[k]["tokens_sum"] += tokens

    total_configs = 0
    for B in budgets:
        splits = compute_split_configs(B, ba_fixed, ba_fractions)
        total_configs += len(splits)
        if run_baselines:
            total_configs += 3  # nothink, think, town
    log.info(f"Total budget levels: {len(budgets)}, "
             f"configs per budget (incl. baselines): ~{total_configs // len(budgets)}, "
             f"samples: {n}")

    t_start = time.perf_counter()
    samples_done = 0

    for i, item in enumerate(items):
        question, gold = item["question"], item["gold"]

        for B in budgets:
            # --- Baselines ---
            if run_baselines:
                # 1. nothink@B
                res_nt = run_nothink(model, tokenizer, question, B, benchmark)
                ok_nt = is_correct_dispatch(res_nt["pred"], gold, benchmark)
                all_results.append({
                    "idx": i, "gold": gold, "B_total": B,
                    "method": "nothink", "Br": 0, "Ba": B,
                    "pred": res_nt["pred"], "correct": int(ok_nt),
                    "tokens_used": res_nt["tokens_used"],
                    "elapsed": res_nt["elapsed"],
                    "hit_budget": res_nt.get("hit_budget", False),
                })
                update_summary("nothink", B, ok_nt, res_nt["tokens_used"])

                # 2. think@B
                res_tk = run_think(model, tokenizer, question, B, benchmark)
                ok_tk = is_correct_dispatch(res_tk["pred"], gold, benchmark)
                all_results.append({
                    "idx": i, "gold": gold, "B_total": B,
                    "method": "think", "Br": B, "Ba": 0,
                    "pred": res_tk["pred"], "correct": int(ok_tk),
                    "tokens_used": res_tk["tokens_used"],
                    "elapsed": res_tk["elapsed"],
                    "truncated": res_tk.get("truncated", False),
                    "hit_budget": res_tk.get("hit_budget", False),
                })
                update_summary("think", B, ok_tk, res_tk["tokens_used"])

                # 4. town@B
                res_tw = run_town(model, tokenizer, question, B, benchmark)
                ok_tw = is_correct_dispatch(res_tw["pred"], gold, benchmark)
                all_results.append({
                    "idx": i, "gold": gold, "B_total": B,
                    "method": "town", "Br": B, "Ba": B,
                    "pred": res_tw["pred"], "correct": int(ok_tw),
                    "tokens_used": res_tw["tokens_used"],
                    "elapsed": res_tw["elapsed"],
                    "stage": res_tw.get("stage", 0),
                })
                update_summary("town", B, ok_tw, res_tw["tokens_used"])

            # --- Split-budget configurations ---
            splits = compute_split_configs(B, ba_fixed, ba_fractions)
            for br, ba in splits:
                res_sp = run_split(model, tokenizer, question, br, ba, benchmark)
                ok_sp = is_correct_dispatch(res_sp["pred"], gold, benchmark)
                split_key = f"split_Br{br}_Ba{ba}"
                all_results.append({
                    "idx": i, "gold": gold, "B_total": B,
                    "method": "split", "Br": br, "Ba": ba,
                    "pred": res_sp["pred"], "correct": int(ok_sp),
                    "tokens_used": res_sp["tokens_used"],
                    "elapsed": res_sp["elapsed"],
                    "truncated": res_sp.get("truncated", False),
                    "think_complete": res_sp.get("think_complete", False),
                    "trace_len_chars": res_sp.get("trace_len_chars", 0),
                })
                update_summary(split_key, B, ok_sp, res_sp["tokens_used"])

        samples_done += 1

        # Progress
        if samples_done % 10 == 0 or samples_done == n:
            elapsed = time.perf_counter() - t_start
            rate = samples_done / (elapsed / 60) if elapsed > 0 else 0
            eta = (n - samples_done) / rate if rate > 0 else 0
            log.info(
                f"  [{samples_done}/{n}]  "
                f"{rate:.1f} samp/min  ETA={eta:.0f}min  "
                f"elapsed={elapsed/60:.1f}min"
            )

    # --- Build aggregate summary ---
    agg = {}
    for (method_key, B), stats in sorted(summary_table.items()):
        t = stats["total"]
        c = stats["correct"]
        acc = c / t if t > 0 else 0.0
        avg_tok = stats["tokens_sum"] / t if t > 0 else 0.0
        agg[f"{method_key}@{B}"] = {
            "accuracy": round(acc, 4),
            "n_correct": c,
            "n_total": t,
            "avg_tokens": round(avg_tok, 2),
        }

    total_elapsed = time.perf_counter() - t_start
    return {
        "per_sample": all_results,
        "aggregate": agg,
        "total_elapsed_s": round(total_elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Pretty-print summary table
# ---------------------------------------------------------------------------
def print_summary_table(agg: Dict, budgets: List[int]):
    """Print a readable comparison table grouped by total budget."""
    log.info("")
    log.info("=" * 80)
    log.info("  Split-Budget Experiment — Summary")
    log.info("=" * 80)

    for B in budgets:
        log.info("")
        log.info(f"  Total Budget B = {B}")
        log.info(f"  {'Method':<30} {'Acc':>8} {'Avg Tok':>10} {'n':>6}")
        log.info("  " + "-" * 56)

        # Collect entries for this budget
        entries = []
        for key, stats in agg.items():
            if key.endswith(f"@{B}"):
                method = key.rsplit("@", 1)[0]
                entries.append((method, stats))

        # Sort: baselines first, then splits by accuracy descending
        baseline_order = {"nothink": 0, "think": 1, "town": 2}
        baselines = [(m, s) for m, s in entries if m in baseline_order]
        splits = [(m, s) for m, s in entries if m not in baseline_order]
        baselines.sort(key=lambda x: baseline_order.get(x[0], 99))
        splits.sort(key=lambda x: -x[1]["accuracy"])

        for method, stats in baselines + splits:
            marker = " *" if splits and stats["accuracy"] == splits[0][1]["accuracy"] and method not in baseline_order else ""
            log.info(
                f"  {method:<30} {stats['accuracy']:>7.1%} "
                f"{stats['avg_tokens']:>10.0f} {stats['n_total']:>6}"
                f"{marker}"
            )

        # Best split vs baselines
        if splits:
            best_split = splits[0]
            nt_key = f"nothink@{B}"
            tk_key = f"think@{B}"
            if nt_key in agg and tk_key in agg:
                nt_acc = agg[nt_key]["accuracy"]
                tk_acc = agg[tk_key]["accuracy"]
                sp_acc = best_split[1]["accuracy"]
                log.info(f"  → Best split vs nothink: "
                         f"{(sp_acc - nt_acc)*100:+.1f}pp  "
                         f"vs think: {(sp_acc - tk_acc)*100:+.1f}pp")

    log.info("")
    log.info("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Split-Budget Experiment: decoupling reasoning from answer generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model & data
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                        choices=["gsm8k", "math500"])
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # Budget configuration
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=DEFAULT_BUDGETS,
                        help="Total budgets B to sweep")
    parser.add_argument("--ba_fixed", type=int, nargs="+",
                        default=DEFAULT_BA_VALUES,
                        help="Fixed answer budgets Ba to try")
    parser.add_argument("--ba_fractions", type=float, nargs="+",
                        default=FRACTIONAL_BA,
                        help="Fractional answer budgets (Ba = frac * B)")

    # Ablation flags
    parser.add_argument("--no_baselines", action="store_true", default=False,
                        help="Skip nothink/think/town baselines (only run splits)")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="results/pilot_split_budget")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Checkpoint every N samples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Reproducibility ---
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.split("/")[-1].replace(".", "_").replace("-", "_")

    # --- Log experiment plan ---
    log.info("=" * 70)
    log.info("  Split-Budget Pilot Experiment")
    log.info("=" * 70)
    log.info(f"  Model:      {args.model}")
    log.info(f"  Benchmark:  {args.benchmark}")
    log.info(f"  Samples:    {args.n_samples}")
    log.info(f"  Seed:       {args.seed}")
    log.info(f"  Budgets:    {args.budgets}")
    log.info(f"  Ba fixed:   {args.ba_fixed}")
    log.info(f"  Ba fracs:   {args.ba_fractions}")

    # Preview configurations
    total_model_calls = 0
    for B in args.budgets:
        splits = compute_split_configs(B, args.ba_fixed, args.ba_fractions)
        n_methods = len(splits) + (3 if not args.no_baselines else 0)
        # Each split = 2 calls (think + nothink), think = 1, nothink = 1, town = 1-2
        n_calls = len(splits) * 2 + (4 if not args.no_baselines else 0)
        total_model_calls += n_calls
        log.info(f"  B={B:>5}: {len(splits)} splits + "
                 f"{'3 baselines' if not args.no_baselines else 'no baselines'} "
                 f"= {n_methods} methods ({n_calls} model calls)")
    total_model_calls *= args.n_samples
    log.info(f"  Total model calls: ~{total_model_calls}")
    log.info("=" * 70)

    # --- Load model ---
    model, tokenizer = load_model_and_tokenizer(args.model)

    # --- Load data ---
    items = load_benchmark_data(args.benchmark, args.n_samples, args.seed)
    n = len(items)
    log.info(f"Loaded {n} {args.benchmark} samples (seed={args.seed})")

    # --- Run experiment ---
    results = run_experiment(
        model, tokenizer, items, args.benchmark,
        budgets=args.budgets,
        ba_fixed=args.ba_fixed,
        ba_fractions=args.ba_fractions,
        run_baselines=not args.no_baselines,
    )

    # --- Save outputs ---
    budget_tag = "_".join(str(b) for b in args.budgets)
    out_fname = (
        f"split_budget_{model_tag}_{args.benchmark}"
        f"_n{n}_B{budget_tag}_{timestamp}.json"
    )
    out_path = os.path.join(args.output_dir, out_fname)

    output = {
        "meta": {
            "script": "pilot_split_budget.py",
            "timestamp_utc": timestamp,
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": n,
            "seed": args.seed,
            "elapsed_s": results["total_elapsed_s"],
        },
        "config": {
            "budgets": args.budgets,
            "ba_fixed": args.ba_fixed,
            "ba_fractions": args.ba_fractions,
            "baselines": not args.no_baselines,
        },
        "aggregate": results["aggregate"],
        "per_sample": results["per_sample"],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"\nSaved JSON: {out_path}")

    # Also save a lightweight CSV of per-sample results
    csv_path = out_path.replace(".json", ".csv")
    _save_csv(results["per_sample"], csv_path)
    log.info(f"Saved CSV:  {csv_path}")

    # --- Print summary table ---
    print_summary_table(results["aggregate"], args.budgets)


def _save_csv(rows: List[Dict], path: str):
    """Save per-sample results to CSV."""
    import csv
    if not rows:
        return
    fieldnames = [
        "idx", "gold", "B_total", "method", "Br", "Ba",
        "pred", "correct", "tokens_used", "elapsed",
        "truncated", "think_complete", "trace_len_chars",
        "hit_budget", "stage",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

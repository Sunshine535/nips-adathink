#!/usr/bin/env python3
"""BBH (BIG-Bench Hard) thinking-tax experiment.

Demonstrates that the "thinking tax" phenomenon generalises beyond math
benchmarks (GSM8K / MATH-500) to diverse reasoning tasks.  Runs both
*thinking* and *nothink* modes at multiple token budgets on a curated set
of BBH subtasks, measuring accuracy, token usage, and overthinking rate.

Usage:
    # Quick sanity check (10 samples per task, 2 budgets)
    python scripts/run_bbh_experiment.py \
        --model Qwen/Qwen3-8B \
        --n_samples 10 \
        --budgets 256 512 \
        --seed 42

    # Full run (all samples, 4 budgets, both modes)
    python scripts/run_bbh_experiment.py \
        --model Qwen/Qwen3-8B \
        --n_samples 0 \
        --budgets 256 512 1024 2048 \
        --seed 42

    # Single subtask only
    python scripts/run_bbh_experiment.py \
        --model Qwen/Qwen3-8B \
        --tasks boolean_expressions \
        --n_samples 50

    # Multi-GPU with device_map=auto (for large models)
    python scripts/run_bbh_experiment.py \
        --model Qwen/Qwen3.5-27B \
        --device_map auto \
        --n_samples 0
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Append script directory to sys.path so we can import benchmarks.py
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from benchmarks import (
    BenchmarkSample,
    build_prompt,
    is_correct,
    load_benchmark,
    parse_prediction,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-8B"

# Representative BBH subtasks covering diverse reasoning skills:
#   boolean_expressions          - symbolic / logical
#   causal_judgement             - causal reasoning
#   date_understanding           - temporal reasoning
#   logical_deduction_five_objects - constraint satisfaction
#   tracking_shuffled_objects_three_objects - state tracking
DEFAULT_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "logical_deduction_five_objects",
    "tracking_shuffled_objects_three_objects",
]

DEFAULT_BUDGETS = [256, 512, 1024, 2048]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model helpers (same pattern as run_nothink_baseline.py / run_experiment.py)
# ---------------------------------------------------------------------------

def model_input_device(model) -> torch.device:
    """Resolve the device a model expects its inputs on."""
    if hasattr(model, "device"):
        dev = model.device
        if isinstance(dev, torch.device):
            return dev
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        for _, device in hf_map.items():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
        for _, device in hf_map.items():
            if isinstance(device, str) and device == "cpu":
                return torch.device("cpu")
    return next(model.parameters()).device


def load_model_and_tokenizer(
    model_id: str,
    device_map: str = "auto",
    allow_cpu: bool = False,
) -> Tuple:
    """Load HF model + tokenizer with bf16 on CUDA (or fp32 on CPU)."""
    use_cuda = torch.cuda.is_available()
    if not use_cuda and not allow_cpu:
        raise RuntimeError(
            "CUDA required but unavailable. Pass --allow_cpu to override."
        )

    log.info(f"Loading model: {model_id}  (cuda={use_cuda}, device_map={device_map})")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_dtype = torch.bfloat16 if use_cuda else torch.float32
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": model_dtype,
    }
    if use_cuda:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    if not use_cuda:
        model.to(torch.device("cpu"))

    # Force deterministic greedy defaults
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        for k in ("top_p", "top_k", "temperature"):
            if hasattr(model.generation_config, k):
                setattr(model.generation_config, k, None)
    model.eval()

    return model, tokenizer


def generate_once(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> Tuple[str, int, float, bool]:
    """Single greedy generation pass. Returns (text, n_tokens, elapsed_s, hit_budget)."""
    inputs = tokenizer(prompt, return_tensors="pt")
    target_device = model_input_device(model)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    elapsed = time.perf_counter() - start

    gen_ids = out[0][in_len:]
    n_tokens = int(gen_ids.shape[0])
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    hit_budget = n_tokens >= int(max_new_tokens * 0.95)
    return text, n_tokens, elapsed, hit_budget


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bbh_tasks(
    tasks: List[str],
    n_samples: int,
    seed: int,
) -> Tuple[List[BenchmarkSample], Dict[str, int]]:
    """Load BBH samples from specified subtasks.

    Args:
        tasks: List of BBH subtask names.
        n_samples: Max samples per subtask. 0 means use all.
        seed: Random seed for reproducible sub-sampling.

    Returns:
        (all_samples, task_counts) where task_counts maps task->N.
    """
    all_samples: List[BenchmarkSample] = []
    task_counts: Dict[str, int] = {}
    rng = random.Random(seed)

    for task_name in tasks:
        log.info(f"Loading BBH subtask: {task_name}")
        try:
            samples = load_benchmark("bbh", split="test", task=task_name)
        except Exception:
            log.warning(f"Could not load BBH task '{task_name}', skipping.")
            continue

        rng.shuffle(samples)
        if n_samples > 0:
            samples = samples[:n_samples]

        task_counts[task_name] = len(samples)
        all_samples.extend(samples)
        log.info(f"  -> {len(samples)} samples loaded")

    return all_samples, task_counts


# ---------------------------------------------------------------------------
# Summarization helpers (same schema as run_experiment.py)
# ---------------------------------------------------------------------------

def summarize(records: List[Dict], key_prefix: str) -> Dict[str, float]:
    if not records:
        return {"accuracy": 0.0, "avg_tokens": 0.0, "avg_latency_s": 0.0}
    n = len(records)
    return {
        "accuracy": sum(r[f"{key_prefix}_correct"] for r in records) / n,
        "avg_tokens": sum(r[f"{key_prefix}_tokens"] for r in records) / n,
        "avg_latency_s": sum(r.get(f"{key_prefix}_latency_s", 0) for r in records) / n,
    }


def compute_oer(records: List[Dict], short_key: str, long_key: str) -> float:
    """Overthinking error rate: fraction where short is correct but long is wrong."""
    if not records:
        return 0.0
    bad = sum(
        1 for r in records
        if r[f"{short_key}_correct"] and not r[f"{long_key}_correct"]
    )
    return bad / len(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BBH thinking-tax experiment: thinking vs nothink at multiple budgets"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=DEFAULT_TASKS,
        help="BBH subtask names to evaluate",
    )
    parser.add_argument(
        "--n_samples", type=int, default=0,
        help="Max samples per subtask (0 = all)",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
        help="Token budgets to sweep",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_seed", type=int, default=None,
        help="Dataset shuffle seed. Defaults to --seed.",
    )
    parser.add_argument(
        "--modes", type=str, nargs="+", default=["nothink", "thinking"],
        choices=["nothink", "thinking"],
        help="Which generation modes to run",
    )
    parser.add_argument("--results_dir", type=str, default="results/bbh")
    parser.add_argument(
        "--device_map", type=str, default="auto",
        help="HF device_map for model loading",
    )
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument(
        "--strict_final_only", action="store_true",
        help="Only accept explicit answer markers as valid predictions.",
    )
    args = parser.parse_args()

    if args.data_seed is None:
        args.data_seed = args.seed

    # --- Reproducibility ---
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.split("/")[-1].replace("-", "_").replace(".", "_")

    # --- Load model ---
    model, tokenizer = load_model_and_tokenizer(
        args.model, device_map=args.device_map, allow_cpu=args.allow_cpu,
    )

    # --- Load data ---
    samples, task_counts = load_bbh_tasks(args.tasks, args.n_samples, args.data_seed)
    total_n = len(samples)
    log.info(f"Total samples: {total_n}  Tasks: {task_counts}")

    if total_n == 0:
        log.error("No samples loaded. Exiting.")
        return

    # --- Run experiments ---
    # records: one row per (sample, mode, budget) — but we store all budgets
    # in one row per (sample, mode) to match the per_sample CSV format.
    records: List[Dict] = []
    total_runs = len(args.modes) * total_n
    run_count = 0

    for mode in args.modes:
        enable_thinking = (mode == "thinking")
        log.info(f"\n{'='*70}")
        log.info(f"Mode: {mode}  (enable_thinking={enable_thinking})")
        log.info(f"{'='*70}")

        for sample_idx, sample in enumerate(samples):
            question = sample.question
            gold = sample.gold
            is_mc = sample.meta.get("is_mc", True)
            task_name = sample.meta.get("task", "unknown")

            row: Dict = {
                "idx": sample_idx,
                "mode": mode,
                "enable_thinking": int(enable_thinking),
                "question": question,
                "gold": gold,
                "benchmark": "bbh",
                "bbh_task": task_name,
                "is_mc": int(is_mc),
            }

            prompt = build_prompt(
                question,
                benchmark="bbh",
                tokenizer=tokenizer,
                prompt_format="chat",
                direct_answer=False,
                enable_thinking=enable_thinking,
            )

            for budget in args.budgets:
                text, n_tokens, latency, hit_budget = generate_once(
                    model, tokenizer, prompt, max_new_tokens=budget, temperature=0.0,
                )

                pred, has_final, pred_source = parse_prediction(
                    text, benchmark="bbh",
                    strict_final_only=args.strict_final_only,
                    is_mc=is_mc,
                )
                correct = int(is_correct(pred, gold, "bbh", is_mc=is_mc))

                row[f"fixed_{budget}_pred"] = pred
                row[f"fixed_{budget}_tokens"] = n_tokens
                row[f"fixed_{budget}_latency_s"] = round(latency, 4)
                row[f"fixed_{budget}_correct"] = correct
                row[f"fixed_{budget}_has_final"] = int(has_final)
                row[f"fixed_{budget}_pred_source"] = pred_source
                row[f"fixed_{budget}_hit_budget"] = int(hit_budget)
                row[f"fixed_{budget}_raw"] = text

            records.append(row)
            run_count += 1

            if (run_count % 20 == 0) or (run_count == total_runs):
                # Quick progress with latest budget accuracy
                last_b = args.budgets[-1]
                mode_recs = [r for r in records if r["mode"] == mode]
                if mode_recs:
                    acc = sum(r[f"fixed_{last_b}_correct"] for r in mode_recs) / len(mode_recs)
                    log.info(
                        f"  [{run_count}/{total_runs}] {mode} "
                        f"acc@{last_b}={acc:.3f}  (n={len(mode_recs)})"
                    )

    # --- Build summary ---
    log.info(f"\n{'='*70}")
    log.info("Building summary...")

    summary: Dict = {
        "meta": {
            "timestamp_utc": timestamp,
            "benchmark": "bbh",
            "tasks": args.tasks,
            "task_counts": task_counts,
            "model": args.model,
            "n_total_samples": total_n,
            "n_samples_per_task": args.n_samples,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "budgets": args.budgets,
            "modes": args.modes,
            "strict_final_only": bool(args.strict_final_only),
            "engine": "hf_transformers",
        },
        "per_mode": {},
        "overthinking": {},
        "per_task": {},
    }

    # Per-mode, per-budget summary
    for mode in args.modes:
        mode_recs = [r for r in records if r["mode"] == mode]
        summary["per_mode"][mode] = {}
        for budget in args.budgets:
            s = summarize(mode_recs, f"fixed_{budget}")
            s["final_rate"] = sum(
                r.get(f"fixed_{budget}_has_final", 0) for r in mode_recs
            ) / max(1, len(mode_recs))
            s["hit_budget_rate"] = sum(
                r.get(f"fixed_{budget}_hit_budget", 0) for r in mode_recs
            ) / max(1, len(mode_recs))
            summary["per_mode"][mode][str(budget)] = s

    # Overthinking rates (for each mode, short budget correct → long budget wrong)
    for mode in args.modes:
        mode_recs = [r for r in records if r["mode"] == mode]
        if len(args.budgets) >= 2:
            short_b = min(args.budgets)
            long_b = max(args.budgets)
            oer = compute_oer(mode_recs, f"fixed_{short_b}", f"fixed_{long_b}")
            summary["overthinking"][f"{mode}_fixed_{short_b}_vs_fixed_{long_b}"] = oer

    # Thinking tax: accuracy drop from nothink → thinking at each budget
    if "nothink" in args.modes and "thinking" in args.modes:
        nothink_recs = [r for r in records if r["mode"] == "nothink"]
        thinking_recs = [r for r in records if r["mode"] == "thinking"]
        thinking_tax = {}
        for budget in args.budgets:
            nt_acc = summarize(nothink_recs, f"fixed_{budget}")["accuracy"]
            th_acc = summarize(thinking_recs, f"fixed_{budget}")["accuracy"]
            thinking_tax[str(budget)] = {
                "nothink_acc": round(nt_acc, 4),
                "thinking_acc": round(th_acc, 4),
                "delta": round(th_acc - nt_acc, 4),
                "thinking_tax": round(nt_acc - th_acc, 4),
            }
        summary["thinking_tax"] = thinking_tax

    # Per-task breakdown
    task_groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        task_groups[r["bbh_task"]].append(r)

    for task_name, task_recs in sorted(task_groups.items()):
        task_summary: Dict = {"n": len(task_recs) // len(args.modes)}
        for mode in args.modes:
            mode_task_recs = [r for r in task_recs if r["mode"] == mode]
            task_summary[mode] = {}
            for budget in args.budgets:
                task_summary[mode][str(budget)] = summarize(mode_task_recs, f"fixed_{budget}")
        # Per-task thinking tax
        if "nothink" in args.modes and "thinking" in args.modes:
            nt_task = [r for r in task_recs if r["mode"] == "nothink"]
            th_task = [r for r in task_recs if r["mode"] == "thinking"]
            tax = {}
            for budget in args.budgets:
                nt_a = summarize(nt_task, f"fixed_{budget}")["accuracy"]
                th_a = summarize(th_task, f"fixed_{budget}")["accuracy"]
                tax[str(budget)] = round(nt_a - th_a, 4)
            task_summary["thinking_tax"] = tax
        summary["per_task"][task_name] = task_summary

    # --- Print summary table ---
    log.info("")
    log.info(f"{'Mode':<12s} {'Budget':>8s} {'Accuracy':>10s} {'AvgTok':>8s} {'HitBudget':>10s} {'HasFinal':>10s}")
    log.info("-" * 64)
    for mode in args.modes:
        for budget in args.budgets:
            s = summary["per_mode"][mode][str(budget)]
            log.info(
                f"{mode:<12s} {budget:>8d} {s['accuracy']:>9.1%} "
                f"{s['avg_tokens']:>8.0f} {s['hit_budget_rate']:>9.1%} "
                f"{s['final_rate']:>9.1%}"
            )
        log.info("-" * 64)

    if "thinking_tax" in summary:
        log.info("")
        log.info("Thinking Tax (nothink_acc - thinking_acc; positive = thinking hurts):")
        for budget_str, tax_info in summary["thinking_tax"].items():
            sign = "+" if tax_info["thinking_tax"] >= 0 else ""
            log.info(
                f"  Budget {budget_str:>5s}: nothink={tax_info['nothink_acc']:.3f}  "
                f"thinking={tax_info['thinking_acc']:.3f}  "
                f"tax={sign}{tax_info['thinking_tax']:.3f}"
            )

    if summary["overthinking"]:
        log.info("")
        log.info("Overthinking Error Rate (correct@short -> wrong@long):")
        for k, v in summary["overthinking"].items():
            log.info(f"  {k}: {v:.1%}")

    # --- Save JSON summary ---
    json_path = os.path.join(
        args.results_dir, f"summary_bbh_{model_tag}_{timestamp}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"\nSaved summary: {json_path}")

    # --- Save per-sample CSV ---
    csv_path = os.path.join(
        args.results_dir, f"per_sample_bbh_{model_tag}_{timestamp}.csv"
    )
    if records:
        # Deterministic column order: fixed columns first, then budget columns sorted
        fixed_cols = [
            "idx", "mode", "enable_thinking", "benchmark", "bbh_task",
            "is_mc", "question", "gold",
        ]
        budget_cols = sorted(
            [k for k in records[0].keys() if k not in fixed_cols],
        )
        fieldnames = fixed_cols + budget_cols

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    log.info(f"Saved per-sample: {csv_path}")

    # --- Save per-sample JSONL (for incremental recovery) ---
    jsonl_path = os.path.join(
        args.results_dir, f"incremental_bbh_{model_tag}_{timestamp}.jsonl"
    )
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Saved incremental: {jsonl_path}")

    log.info("\nDone.")


if __name__ == "__main__":
    main()

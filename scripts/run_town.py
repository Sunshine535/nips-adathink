#!/usr/bin/env python3
"""TOWN (Think Only When Needed) — two-stage cascade for adaptive inference.

Algorithm:
    For each question q:
      Stage 1: Generate nothink response with budget B1 (default 256)
        - If tokens_used < B1 (natural stop): ACCEPT → move to next question
        - If tokens_used >= B1 (hit budget):   REJECT → go to Stage 2
      Stage 2: Generate thinking response with budget B2 (default 512)
        - Use this answer regardless

The intuition: easy questions get answered quickly in nothink mode; hard
questions that exhaust the nothink budget are escalated to full thinking mode.

Usage:
    python scripts/run_town.py \
        --model Qwen/Qwen3-8B \
        --benchmark gsm8k \
        --b1 256 --b2 512 \
        --n_samples 99999 \
        --seed 42

    # Dry-run with fewer samples
    python scripts/run_town.py --n_samples 50 --b1 64 --b2 128
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer\s*[:：]|the answer is\s*)([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
HASH_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)")


def extract_last_number(text: str) -> Optional[str]:
    """Return the last number found in *text*, or None."""
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = s.replace(",", "").strip()
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                d = float(parts[1])
                return float(parts[0]) / d if d != 0 else None
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def is_correct(pred: Optional[str], gold: Optional[str], tol: float = 1e-6) -> bool:
    p, g = to_float(pred), to_float(gold)
    if p is None or g is None:
        return False
    return abs(p - g) <= tol * max(1.0, abs(g))


def get_gold_from_gsm8k(answer_field: str) -> Optional[str]:
    """Extract gold number from GSM8K answer field (#### pattern)."""
    if "####" in answer_field:
        after = answer_field.split("####")[-1]
        match = NUM_RE.search(after)
        if match:
            return match.group(0)
    return extract_last_number(answer_field)


def parse_prediction(text: str) -> Tuple[Optional[str], str]:
    """Extract predicted number from model output.

    Returns (pred_str, source) where source ∈
    {"boxed", "hash", "final_answer", "last_number", "none"}.
    """
    # 1. \boxed{...}
    m = BOXED_RE.search(text)
    if m:
        inner = m.group(1).replace(",", "")
        num = NUM_RE.search(inner)
        if num:
            return num.group(0), "boxed"

    # 2. #### pattern (sometimes models mimic GSM8K format)
    m = HASH_RE.search(text)
    if m:
        return m.group(1).replace(",", ""), "hash"

    # 3. "Final answer: <number>" or "the answer is <number>"
    m = FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", ""), "final_answer"

    # 4. Fallback: last number in text
    last = extract_last_number(text)
    if last:
        return last.replace(",", ""), "last_number"

    return None, "none"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(
    model_id: str, device_map: str = "auto"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    log.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def model_input_device(model) -> torch.device:
    """Resolve the device that model inputs should be placed on."""
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
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_once(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> Tuple[str, int, float, bool]:
    """Generate a single response.

    Returns:
        text:       decoded generated text
        n_tokens:   number of new tokens generated
        elapsed:    wall-clock seconds
        hit_budget: True if generation consumed ≥95% of budget (likely truncated)
    """
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
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(
    question: str,
    tokenizer,
    enable_thinking: bool,
) -> str:
    """Build chat prompt with enable_thinking control."""
    system_text = (
        "You are a careful math solver. Solve the problem step by step briefly. "
        "End with a single line: Final answer: <number>."
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": question},
    ]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    chat_kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **chat_kwargs)
    except TypeError:
        # Fallback if tokenizer doesn't support enable_thinking
        chat_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **chat_kwargs)
    except Exception:
        return f"{system_text}\n\nQuestion: {question}\nSolution:\n"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_gsm8k(n_samples: int, seed: int) -> List[Dict]:
    """Load GSM8K test set, shuffle deterministically, return list of dicts."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idxs = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(idxs)
    selected = [ds[i] for i in idxs[:n_samples]]
    items = []
    for raw in selected:
        gold = get_gold_from_gsm8k(raw["answer"])
        items.append({"question": raw["question"], "gold": gold})
    return items


# ---------------------------------------------------------------------------
# TOWN cascade
# ---------------------------------------------------------------------------
def run_town_sample(
    model,
    tokenizer,
    question: str,
    b1: int,
    b2: int,
) -> Dict:
    """Run the TOWN two-stage cascade for a single question.

    Stage 1: nothink mode with budget B1.
      - Accept if natural stop (tokens < budget).
      - Reject (escalate) if budget exhausted.
    Stage 2: thinking mode with budget B2.
      - Always accept.
    """
    # --- Stage 1: nothink ---
    prompt_s1 = build_prompt(question, tokenizer, enable_thinking=False)
    text_s1, tokens_s1, elapsed_s1, hit_budget_s1 = generate_once(
        model, tokenizer, prompt_s1, max_new_tokens=b1, temperature=0.0,
    )
    pred_s1, source_s1 = parse_prediction(text_s1)

    result = {
        "text_s1": text_s1,
        "tokens_s1": tokens_s1,
        "elapsed_s1": elapsed_s1,
        "hit_budget_s1": hit_budget_s1,
        "pred_s1": pred_s1,
        "source_s1": source_s1,
    }

    if not hit_budget_s1:
        # Natural stop → accept Stage 1 answer
        result["stage"] = 1
        result["pred"] = pred_s1
        result["pred_source"] = source_s1
        result["text_s2"] = None
        result["tokens_s2"] = 0
        result["elapsed_s2"] = 0.0
        result["pred_s2"] = None
        result["source_s2"] = "n/a"
    else:
        # Hit budget → escalate to Stage 2 (thinking mode)
        prompt_s2 = build_prompt(question, tokenizer, enable_thinking=True)
        text_s2, tokens_s2, elapsed_s2, hit_budget_s2 = generate_once(
            model, tokenizer, prompt_s2, max_new_tokens=b2, temperature=0.0,
        )
        pred_s2, source_s2 = parse_prediction(text_s2)

        result["stage"] = 2
        result["pred"] = pred_s2
        result["pred_source"] = source_s2
        result["text_s2"] = text_s2
        result["tokens_s2"] = tokens_s2
        result["elapsed_s2"] = elapsed_s2
        result["pred_s2"] = pred_s2
        result["source_s2"] = source_s2

    result["tokens_total"] = result["tokens_s1"] + result["tokens_s2"]
    result["elapsed_total"] = result["elapsed_s1"] + result["elapsed_s2"]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TOWN: Think Only When Needed — two-stage nothink→think cascade",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model ID")
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                        choices=["gsm8k"],
                        help="Benchmark dataset")
    parser.add_argument("--n_samples", type=int, default=99999,
                        help="Max samples to evaluate (99999 = full test set)")
    parser.add_argument("--b1", type=int, default=256,
                        help="Stage 1 (nothink) token budget")
    parser.add_argument("--b2", type=int, default=512,
                        help="Stage 2 (thinking) token budget")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/town)")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="Device map for model loading")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "results/town"

    # --- Reproducibility ---
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.split("/")[-1].replace(".", "_").replace("-", "_")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    model, tokenizer = load_model_and_tokenizer(args.model, args.device_map)

    # --- Load data ---
    log.info(f"Loading {args.benchmark} (n_samples={args.n_samples}, seed={args.seed})")
    items = load_gsm8k(args.n_samples, args.seed)
    n = len(items)
    log.info(f"Loaded {n} samples")

    # --- Run TOWN cascade ---
    log.info(f"TOWN cascade: B1={args.b1} (nothink) → B2={args.b2} (thinking)")
    per_sample: List[Dict] = []
    n_correct = 0
    n_stage1 = 0
    n_stage2 = 0
    total_tokens = 0
    total_elapsed = 0.0

    for i, item in enumerate(items):
        result = run_town_sample(
            model, tokenizer, item["question"], args.b1, args.b2,
        )

        correct = is_correct(result["pred"], item["gold"])
        if correct:
            n_correct += 1
        if result["stage"] == 1:
            n_stage1 += 1
        else:
            n_stage2 += 1
        total_tokens += result["tokens_total"]
        total_elapsed += result["elapsed_total"]

        row = {
            "idx": i,
            "gold": item["gold"],
            "stage": result["stage"],
            "pred": result["pred"],
            "correct": int(correct),
            "pred_source": result["pred_source"],
            "tokens_used_stage1": result["tokens_s1"],
            "tokens_used_stage2": result["tokens_s2"],
            "tokens_total": result["tokens_total"],
            "hit_budget_s1": int(result["hit_budget_s1"]),
            "elapsed_s1": round(result["elapsed_s1"], 4),
            "elapsed_s2": round(result["elapsed_s2"], 4),
            "elapsed_total": round(result["elapsed_total"], 4),
            "pred_s1": result["pred_s1"],
            "source_s1": result["source_s1"],
            "pred_s2": result["pred_s2"],
            "source_s2": result["source_s2"],
        }
        per_sample.append(row)

        # --- Progress logging every 20 samples ---
        if (i + 1) % 20 == 0 or i == n - 1:
            done = i + 1
            acc = n_correct / done
            s1_rate = n_stage1 / done
            avg_tok = total_tokens / done
            log.info(
                f"  [{done}/{n}] acc={acc:.3f}  "
                f"stage1_rate={s1_rate:.1%}  "
                f"avg_tokens={avg_tok:.0f}  "
                f"stage1={n_stage1} stage2={n_stage2}"
            )

    # --- Compute summary statistics ---
    acc = n_correct / n if n > 0 else 0.0
    stage1_rate = n_stage1 / n if n > 0 else 0.0
    stage2_rate = n_stage2 / n if n > 0 else 0.0
    avg_tokens = total_tokens / n if n > 0 else 0.0
    avg_elapsed = total_elapsed / n if n > 0 else 0.0

    # Accuracy breakdown by stage
    stage1_samples = [r for r in per_sample if r["stage"] == 1]
    stage2_samples = [r for r in per_sample if r["stage"] == 2]
    stage1_acc = (
        sum(r["correct"] for r in stage1_samples) / len(stage1_samples)
        if stage1_samples else 0.0
    )
    stage2_acc = (
        sum(r["correct"] for r in stage2_samples) / len(stage2_samples)
        if stage2_samples else 0.0
    )
    avg_tokens_s1_only = (
        sum(r["tokens_total"] for r in stage1_samples) / len(stage1_samples)
        if stage1_samples else 0.0
    )
    avg_tokens_s2_escalated = (
        sum(r["tokens_total"] for r in stage2_samples) / len(stage2_samples)
        if stage2_samples else 0.0
    )

    summary = {
        "meta": {
            "method": "town",
            "description": "TOWN: Think Only When Needed — two-stage cascade",
            "timestamp_utc": timestamp,
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": n,
            "b1_nothink": args.b1,
            "b2_thinking": args.b2,
            "seed": args.seed,
        },
        "results": {
            "accuracy": round(acc, 4),
            "stage1_accept_rate": round(stage1_rate, 4),
            "stage2_escalation_rate": round(stage2_rate, 4),
            "stage1_accuracy": round(stage1_acc, 4),
            "stage2_accuracy": round(stage2_acc, 4),
            "avg_tokens_total": round(avg_tokens, 2),
            "avg_tokens_stage1_accepted": round(avg_tokens_s1_only, 2),
            "avg_tokens_stage2_escalated": round(avg_tokens_s2_escalated, 2),
            "avg_elapsed_s": round(avg_elapsed, 4),
            "total_elapsed_s": round(total_elapsed, 2),
            "n_stage1": n_stage1,
            "n_stage2": n_stage2,
        },
        "per_sample": per_sample,
    }

    # --- Print summary ---
    log.info("")
    log.info("=" * 70)
    log.info("TOWN Results Summary")
    log.info("=" * 70)
    log.info(f"  Model:               {args.model}")
    log.info(f"  Benchmark:           {args.benchmark} (n={n})")
    log.info(f"  Budgets:             B1={args.b1} (nothink), B2={args.b2} (thinking)")
    log.info(f"  Overall accuracy:    {acc:.1%}")
    log.info(f"  Stage 1 accept rate: {stage1_rate:.1%} ({n_stage1}/{n})")
    log.info(f"  Stage 1 accuracy:    {stage1_acc:.1%}")
    log.info(f"  Stage 2 accuracy:    {stage2_acc:.1%}")
    log.info(f"  Avg tokens (all):    {avg_tokens:.0f}")
    log.info(f"  Avg tokens (S1):     {avg_tokens_s1_only:.0f}")
    log.info(f"  Avg tokens (S2):     {avg_tokens_s2_escalated:.0f}")
    log.info(f"  Total wall time:     {total_elapsed:.1f}s")
    log.info("=" * 70)

    # --- Save JSON ---
    json_fname = f"town_{model_tag}_b1{args.b1}_b2{args.b2}_{timestamp}.json"
    json_path = os.path.join(args.output_dir, json_fname)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Saved JSON: {json_path}")

    # --- Save CSV (per-sample, without raw text) ---
    csv_fname = f"per_sample_town_{model_tag}_b1{args.b1}_b2{args.b2}_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_fname)
    if per_sample:
        fieldnames = list(per_sample[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sample)
    log.info(f"Saved CSV:  {csv_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()

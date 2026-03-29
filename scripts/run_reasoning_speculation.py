#!/usr/bin/env python3
"""Reasoning Speculation: Parallel Exploration with Cross-Path Fusion
for Adaptive Test-Time Compute.

Core idea: Instead of allocating a single budget and hoping it's right,
generate K SHORT reasoning paths in parallel, measure their consensus,
and adaptively decide how much more compute to invest.

Three-phase architecture:
  Phase 1 (Explore):  K parallel paths × B_probe tokens each
  Phase 2 (Decide):   Compute consensus signal → route to Easy/Medium/Hard
  Phase 3 (Resolve):  Easy=stop, Medium=extend best, Hard=full deliberation

Usage:
  python scripts/run_reasoning_speculation.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 4 \
    --probe_budget 128 \
    --medium_budget 256 \
    --hard_budget 512 \
    --seed 42
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer extraction (from gsm8k_utils.py)
# ---------------------------------------------------------------------------
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer\s*[:：]|the answer is\s*)([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_last_number(text):
    if not text:
        return None
    matches = NUM_RE.findall(text)
    return matches[-1] if matches else None


def extract_final_number(text):
    if not text:
        return None
    matches = list(FINAL_ANSWER_RE.finditer(text))
    return matches[-1].group(1) if matches else None


def extract_boxed_number(text):
    if not text:
        return None
    for m in reversed(list(BOXED_RE.finditer(text))):
        val = extract_last_number(m.group(1))
        if val is not None:
            return val
    return None


def parse_prediction(text, strict_final_only=False):
    final = extract_final_number(text)
    if final is not None:
        return final, True, "final_marker"
    boxed = extract_boxed_number(text)
    if boxed is not None:
        return boxed, False, "boxed"
    if strict_final_only:
        return None, False, "none"
    tail = extract_last_number(text)
    if tail is not None:
        return tail, False, "fallback_last"
    return None, False, "none"


def to_float(s):
    if s is None:
        return None
    s = str(s).replace(",", "").strip()
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


def is_correct(pred, gold, tol=1e-6):
    p, g = to_float(pred), to_float(gold)
    if p is None or g is None:
        return False
    return abs(p - g) <= tol * max(1.0, abs(g))


def get_gold_from_gsm8k(answer_field):
    if "####" in answer_field:
        after = answer_field.split("####")[-1]
        match = NUM_RE.search(after)
        if match:
            return match.group(0)
    return extract_last_number(answer_field)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_id, device_map="auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

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


def model_input_device(model):
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


def generate_once(model, tokenizer, prompt, max_new_tokens, temperature=0.0, top_p=0.95):
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
        gen_kwargs["top_p"] = top_p

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    elapsed = time.perf_counter() - start

    gen_ids = out[0][in_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, int(gen_ids.shape[0]), elapsed


def build_prompt(question, tokenizer, enable_thinking=True):
    system_text = (
        "You are a careful math solver. Solve the problem step by step briefly. "
        "End with a single line: Final answer: <number>."
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": question},
    ]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        chat_kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **chat_kwargs)
    except TypeError:
        chat_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **chat_kwargs)
    except Exception:
        return f"{system_text}\n\nQuestion: {question}\nSolution:\n"


# ---------------------------------------------------------------------------
# Core: Reasoning Speculation
# ---------------------------------------------------------------------------
@dataclass
class PathResult:
    """Result from a single exploration path."""
    text: str
    tokens: int
    latency_s: float
    pred: Optional[str]
    has_final: bool
    pred_source: str


@dataclass
class ConsensusSignal:
    """Consensus signal from K paths."""
    agreement_ratio: float       # fraction of paths agreeing on majority answer
    majority_answer: Optional[str]
    n_valid_preds: int
    n_paths: int
    answer_distribution: Dict[str, int]  # answer → count
    confidence: float            # 0-1 confidence score
    route: str                   # "easy" / "medium" / "hard"


@dataclass
class SpeculationResult:
    """Final result from Reasoning Speculation."""
    pred: Optional[str]
    correct: bool
    total_tokens: int
    total_latency_s: float
    route: str                   # easy/medium/hard
    consensus: ConsensusSignal
    n_explore_paths: int
    explore_tokens: int
    resolve_tokens: int
    final_source: str            # "consensus_vote" / "extended" / "deliberation"


def compute_consensus(paths: List[PathResult]) -> ConsensusSignal:
    """Compute consensus signal from K exploration paths."""
    valid_preds = []
    for p in paths:
        v = to_float(p.pred)
        if v is not None:
            valid_preds.append((round(v, 6), p.pred))

    if not valid_preds:
        return ConsensusSignal(
            agreement_ratio=0.0,
            majority_answer=None,
            n_valid_preds=0,
            n_paths=len(paths),
            answer_distribution={},
            confidence=0.0,
            route="hard",
        )

    # Count answers
    answer_counts = Counter(v for v, _ in valid_preds)
    majority_val, majority_count = answer_counts.most_common(1)[0]
    agreement = majority_count / len(valid_preds)

    # Find the string representation of majority answer
    majority_str = None
    for v, s in valid_preds:
        if v == majority_val:
            majority_str = s
            break

    # Confidence: combination of agreement ratio and prediction validity
    pred_validity = len(valid_preds) / len(paths)
    # Also consider if paths produced "Final answer:" markers
    final_rate = sum(1 for p in paths if p.has_final) / len(paths)
    confidence = 0.5 * agreement + 0.3 * pred_validity + 0.2 * final_rate

    # Route decision
    if agreement >= 0.75 and confidence >= 0.6:
        route = "easy"
    elif agreement >= 0.5 or confidence >= 0.4:
        route = "medium"
    else:
        route = "hard"

    return ConsensusSignal(
        agreement_ratio=agreement,
        majority_answer=majority_str,
        n_valid_preds=len(valid_preds),
        n_paths=len(paths),
        answer_distribution={str(k): v for k, v in answer_counts.items()},
        confidence=confidence,
        route=route,
    )


def reasoning_speculation(
    model,
    tokenizer,
    question: str,
    gold: str,
    k_paths: int = 4,
    probe_budget: int = 128,
    medium_budget: int = 256,
    hard_budget: int = 512,
    temperature: float = 0.7,
    enable_thinking: bool = True,
    strict_final_only: bool = True,
    projection_on_missing: bool = True,
    easy_threshold: float = 0.75,
    medium_threshold: float = 0.5,
) -> SpeculationResult:
    """Execute the three-phase Reasoning Speculation pipeline.

    Phase 1 (Explore):  Generate K paths at probe_budget tokens each.
    Phase 2 (Decide):   Compute consensus → route.
    Phase 3 (Resolve):  Execute resolution strategy based on route.
    """
    prompt = build_prompt(question, tokenizer, enable_thinking=enable_thinking)

    # ---- Phase 1: Parallel Exploration ----
    explore_paths = []
    explore_tokens = 0
    explore_latency = 0.0

    for k in range(k_paths):
        # Use temperature > 0 for diversity (except first path uses greedy)
        temp = 0.0 if k == 0 else temperature

        text, tokens, latency = generate_once(
            model, tokenizer, prompt,
            max_new_tokens=probe_budget,
            temperature=temp,
        )

        pred, has_final, pred_source = parse_prediction(text, strict_final_only=strict_final_only)

        # If no answer found and projection enabled, try projection
        if pred is None and projection_on_missing:
            proj_prompt = (
                "Read the question and draft solution. "
                "Output exactly one line: Final answer: <number>\n\n"
                f"Question: {question}\n\n"
                f"Draft solution:\n{text}\n\n"
                "Final answer:"
            )
            proj_text, proj_tokens, proj_latency = generate_once(
                model, tokenizer, proj_prompt,
                max_new_tokens=16,
                temperature=0.0,
            )
            pred_proj, has_final_proj, _ = parse_prediction(proj_text, strict_final_only=False)
            if pred_proj is not None:
                pred = pred_proj
                has_final = has_final_proj
                pred_source = "projection"
                tokens += proj_tokens
                latency += proj_latency

        path = PathResult(
            text=text, tokens=tokens, latency_s=latency,
            pred=pred, has_final=has_final, pred_source=pred_source,
        )
        explore_paths.append(path)
        explore_tokens += tokens
        explore_latency += latency

    # ---- Phase 2: Compute Consensus ----
    consensus = compute_consensus(explore_paths)

    # ---- Phase 3: Resolve based on route ----
    resolve_tokens = 0
    resolve_latency = 0.0
    final_pred = None
    final_source = ""

    if consensus.route == "easy":
        # High consensus → just use majority vote answer
        final_pred = consensus.majority_answer
        final_source = "consensus_vote"

    elif consensus.route == "medium":
        # Partial consensus → extend the best path to medium_budget
        # Find the greedy path (k=0, most reliable) or the path with majority answer
        best_path = explore_paths[0]  # greedy path
        for p in explore_paths:
            if p.pred and to_float(p.pred) is not None:
                v = round(to_float(p.pred), 6)
                if str(v) in consensus.answer_distribution:
                    if consensus.answer_distribution[str(v)] > consensus.answer_distribution.get(
                        str(round(to_float(best_path.pred), 6)) if to_float(best_path.pred) is not None else "None", 0
                    ):
                        best_path = p

        # Extend: continue from the best path's text
        remaining_budget = medium_budget - best_path.tokens
        if remaining_budget > 32:
            ext_text, ext_tokens, ext_latency = generate_once(
                model, tokenizer,
                prompt + best_path.text,
                max_new_tokens=remaining_budget,
                temperature=0.0,  # greedy for extension
            )
            combined_text = best_path.text + ext_text
            resolve_tokens = ext_tokens
            resolve_latency = ext_latency

            pred, has_final, pred_source = parse_prediction(
                combined_text, strict_final_only=strict_final_only
            )
            if pred is not None:
                final_pred = pred
                final_source = "extended"
            else:
                # Projection fallback
                proj_prompt = (
                    "Read the question and draft solution. "
                    "Output exactly one line: Final answer: <number>\n\n"
                    f"Question: {question}\n\n"
                    f"Draft solution:\n{combined_text}\n\n"
                    "Final answer:"
                )
                proj_text, proj_tok, proj_lat = generate_once(
                    model, tokenizer, proj_prompt, max_new_tokens=16, temperature=0.0,
                )
                pred_p, _, _ = parse_prediction(proj_text, strict_final_only=False)
                if pred_p is not None:
                    final_pred = pred_p
                    final_source = "extended_projection"
                resolve_tokens += proj_tok
                resolve_latency += proj_lat
        else:
            # Not enough room to extend, use consensus vote
            final_pred = consensus.majority_answer
            final_source = "consensus_vote_medium"

    else:  # hard
        # Low consensus → full deliberation with path summaries as context
        # Construct a "deliberation" prompt that includes insights from all paths
        path_summaries = []
        for i, p in enumerate(explore_paths):
            pred_str = f"Answer: {p.pred}" if p.pred else "No clear answer"
            # Truncate path text to avoid exceeding context
            excerpt = p.text[:500] if len(p.text) > 500 else p.text
            path_summaries.append(f"Approach {i+1}: {excerpt}\n{pred_str}")

        deliberation_context = "\n\n".join(path_summaries)

        deliberation_prompt = build_prompt(
            f"{question}\n\n"
            f"I explored {k_paths} different reasoning approaches:\n\n"
            f"{deliberation_context}\n\n"
            f"Now, carefully analyzing all approaches, let me find the correct answer:",
            tokenizer,
            enable_thinking=enable_thinking,
        )

        delib_text, delib_tokens, delib_latency = generate_once(
            model, tokenizer, deliberation_prompt,
            max_new_tokens=hard_budget,
            temperature=0.0,
        )
        resolve_tokens = delib_tokens
        resolve_latency = delib_latency

        pred, has_final, pred_source = parse_prediction(
            delib_text, strict_final_only=strict_final_only
        )
        if pred is not None:
            final_pred = pred
            final_source = "deliberation"
        else:
            # Projection fallback
            proj_prompt = (
                "Read the question and the reasoning. "
                "Output exactly one line: Final answer: <number>\n\n"
                f"Question: {question}\n\n"
                f"Reasoning:\n{delib_text}\n\n"
                "Final answer:"
            )
            proj_text, proj_tok, proj_lat = generate_once(
                model, tokenizer, proj_prompt, max_new_tokens=16, temperature=0.0,
            )
            pred_p, _, _ = parse_prediction(proj_text, strict_final_only=False)
            if pred_p is not None:
                final_pred = pred_p
                final_source = "deliberation_projection"
            resolve_tokens += proj_tok
            resolve_latency += proj_lat

    # If still no answer, fall back to consensus majority
    if final_pred is None:
        final_pred = consensus.majority_answer
        final_source = "fallback_consensus"

    correct = is_correct(final_pred, gold)

    return SpeculationResult(
        pred=final_pred,
        correct=correct,
        total_tokens=explore_tokens + resolve_tokens,
        total_latency_s=explore_latency + resolve_latency,
        route=consensus.route,
        consensus=consensus,
        n_explore_paths=k_paths,
        explore_tokens=explore_tokens,
        resolve_tokens=resolve_tokens,
        final_source=final_source,
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def run_fixed_baseline(model, tokenizer, question, gold, budget, enable_thinking=True,
                       strict_final_only=True, projection_on_missing=True):
    prompt = build_prompt(question, tokenizer, enable_thinking=enable_thinking)
    text, tokens, latency = generate_once(model, tokenizer, prompt, max_new_tokens=budget, temperature=0.0)

    pred, has_final, pred_source = parse_prediction(text, strict_final_only=strict_final_only)
    if pred is None and projection_on_missing:
        proj_prompt = (
            "Read the question and draft solution. "
            "Output exactly one line: Final answer: <number>\n\n"
            f"Question: {question}\n\nDraft solution:\n{text}\n\nFinal answer:"
        )
        proj_text, proj_tok, proj_lat = generate_once(
            model, tokenizer, proj_prompt, max_new_tokens=16, temperature=0.0
        )
        pred_p, _, _ = parse_prediction(proj_text, strict_final_only=False)
        if pred_p is not None:
            pred = pred_p
            tokens += proj_tok
            latency += proj_lat

    return pred, is_correct(pred, gold), tokens, latency


def run_sc_baseline(model, tokenizer, question, gold, budget, k_paths,
                    temperature=0.7, enable_thinking=True,
                    strict_final_only=True, projection_on_missing=True):
    """Self-consistency baseline: K samples + majority vote."""
    prompt = build_prompt(question, tokenizer, enable_thinking=enable_thinking)
    preds = []
    total_tokens = 0
    total_latency = 0.0

    for k in range(k_paths):
        temp = 0.0 if k == 0 else temperature
        text, tokens, latency = generate_once(
            model, tokenizer, prompt, max_new_tokens=budget, temperature=temp
        )
        total_tokens += tokens
        total_latency += latency

        pred, has_final, _ = parse_prediction(text, strict_final_only=strict_final_only)
        if pred is None and projection_on_missing:
            proj_prompt = (
                "Read the question and draft solution. "
                "Output exactly one line: Final answer: <number>\n\n"
                f"Question: {question}\n\nDraft solution:\n{text}\n\nFinal answer:"
            )
            proj_text, proj_tok, proj_lat = generate_once(
                model, tokenizer, proj_prompt, max_new_tokens=16, temperature=0.0
            )
            pred_p, _, _ = parse_prediction(proj_text, strict_final_only=False)
            if pred_p is not None:
                pred = pred_p
                total_tokens += proj_tok
                total_latency += proj_lat

        if pred is not None:
            preds.append(pred)

    # Majority vote
    if not preds:
        return None, False, total_tokens, total_latency

    float_preds = [(round(to_float(p), 6) if to_float(p) is not None else None, p) for p in preds]
    float_preds = [(v, s) for v, s in float_preds if v is not None]
    if not float_preds:
        return None, False, total_tokens, total_latency

    ctr = Counter(v for v, _ in float_preds)
    majority_val = ctr.most_common(1)[0][0]
    majority_str = next(s for v, s in float_preds if v == majority_val)

    return majority_str, is_correct(majority_str, gold), total_tokens, total_latency


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_gsm8k(n_samples=None, seed=42):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for row in ds:
        gold = get_gold_from_gsm8k(row["answer"])
        items.append({"question": row["question"], "gold": gold})
    if n_samples and n_samples < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n_samples)
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Reasoning Speculation Experiment")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--k_paths", type=int, default=4)
    parser.add_argument("--probe_budget", type=int, default=128)
    parser.add_argument("--medium_budget", type=int, default=256)
    parser.add_argument("--hard_budget", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", action="store_true")
    parser.add_argument("--strict_final_only", action="store_true", default=True)
    parser.add_argument("--projection", action="store_true", default=True)
    parser.add_argument("--easy_threshold", type=float, default=0.75)
    parser.add_argument("--medium_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--run_baselines", action="store_true", default=True,
                        help="Also run fixed-budget and SC baselines for comparison")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    if args.no_thinking:
        args.enable_thinking = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.split("/")[-1].replace(".", "_")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.device_map)

    # Load data
    log.info(f"Loading {args.benchmark} (n={args.n_samples})...")
    if args.benchmark == "gsm8k":
        items = load_gsm8k(args.n_samples, args.seed)
    log.info(f"Loaded {len(items)} items")

    # ---- Run Reasoning Speculation ----
    spec_results = []
    route_counts = Counter()
    correct_by_route = Counter()
    total_by_route = Counter()
    total_correct = 0
    total_tokens = 0

    log.info(f"Running Reasoning Speculation: K={args.k_paths}, probe={args.probe_budget}, "
             f"medium={args.medium_budget}, hard={args.hard_budget}")

    for i, item in enumerate(items):
        q, gold = item["question"], item["gold"]

        result = reasoning_speculation(
            model, tokenizer, q, gold,
            k_paths=args.k_paths,
            probe_budget=args.probe_budget,
            medium_budget=args.medium_budget,
            hard_budget=args.hard_budget,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            strict_final_only=args.strict_final_only,
            projection_on_missing=args.projection,
            easy_threshold=args.easy_threshold,
            medium_threshold=args.medium_threshold,
        )

        spec_results.append(result)
        route_counts[result.route] += 1
        total_by_route[result.route] += 1
        if result.correct:
            total_correct += 1
            correct_by_route[result.route] += 1
        total_tokens += result.total_tokens

        if (i + 1) % 10 == 0 or i == 0:
            acc = total_correct / (i + 1)
            avg_tok = total_tokens / (i + 1)
            log.info(
                f"[{i+1}/{len(items)}] acc={acc:.4f} avg_tok={avg_tok:.1f} "
                f"routes: easy={route_counts['easy']} med={route_counts['medium']} "
                f"hard={route_counts['hard']}"
            )

    spec_acc = total_correct / len(items)
    spec_avg_tokens = total_tokens / len(items)

    log.info(f"\n{'='*60}")
    log.info(f"Reasoning Speculation: acc={spec_acc:.4f}, avg_tokens={spec_avg_tokens:.1f}")
    log.info(f"Route distribution: {dict(route_counts)}")
    for route in ["easy", "medium", "hard"]:
        if total_by_route[route] > 0:
            r_acc = correct_by_route[route] / total_by_route[route]
            log.info(f"  {route}: {total_by_route[route]} ({total_by_route[route]/len(items):.1%}), "
                     f"acc={r_acc:.4f}")

    # ---- Run baselines ----
    baseline_results = {}

    if args.run_baselines:
        for budget in [args.probe_budget, args.medium_budget, args.hard_budget]:
            log.info(f"Running Fixed-{budget} baseline...")
            b_correct = 0
            b_tokens = 0
            for i, item in enumerate(items):
                pred, correct, tokens, _ = run_fixed_baseline(
                    model, tokenizer, item["question"], item["gold"],
                    budget=budget, enable_thinking=args.enable_thinking,
                    strict_final_only=args.strict_final_only,
                    projection_on_missing=args.projection,
                )
                b_correct += int(correct)
                b_tokens += tokens
            baseline_results[f"fixed_{budget}"] = {
                "accuracy": b_correct / len(items),
                "avg_tokens": b_tokens / len(items),
            }
            log.info(f"  Fixed-{budget}: acc={b_correct/len(items):.4f}, "
                     f"avg_tok={b_tokens/len(items):.1f}")

        # SC baseline (same K, same probe budget)
        log.info(f"Running SC@{args.k_paths}×{args.probe_budget} baseline...")
        sc_correct = 0
        sc_tokens = 0
        for i, item in enumerate(items):
            pred, correct, tokens, _ = run_sc_baseline(
                model, tokenizer, item["question"], item["gold"],
                budget=args.probe_budget, k_paths=args.k_paths,
                temperature=args.temperature,
                enable_thinking=args.enable_thinking,
                strict_final_only=args.strict_final_only,
                projection_on_missing=args.projection,
            )
            sc_correct += int(correct)
            sc_tokens += tokens
        baseline_results[f"sc_{args.k_paths}x{args.probe_budget}"] = {
            "accuracy": sc_correct / len(items),
            "avg_tokens": sc_tokens / len(items),
        }
        log.info(f"  SC@{args.k_paths}×{args.probe_budget}: acc={sc_correct/len(items):.4f}, "
                 f"avg_tok={sc_tokens/len(items):.1f}")

    # ---- Save results ----
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"reasoning_speculation_{model_tag}_{args.benchmark}_{timestamp}.json"
    )

    summary = {
        "meta": {
            "timestamp": timestamp,
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": len(items),
            "k_paths": args.k_paths,
            "probe_budget": args.probe_budget,
            "medium_budget": args.medium_budget,
            "hard_budget": args.hard_budget,
            "temperature": args.temperature,
            "enable_thinking": args.enable_thinking,
            "seed": args.seed,
            "easy_threshold": args.easy_threshold,
            "medium_threshold": args.medium_threshold,
        },
        "reasoning_speculation": {
            "accuracy": spec_acc,
            "avg_tokens": spec_avg_tokens,
            "route_distribution": dict(route_counts),
            "accuracy_by_route": {
                route: correct_by_route[route] / total_by_route[route]
                if total_by_route[route] > 0 else 0
                for route in ["easy", "medium", "hard"]
            },
            "fraction_by_route": {
                route: total_by_route[route] / len(items)
                for route in ["easy", "medium", "hard"]
            },
        },
        "baselines": baseline_results,
        "per_sample": [
            {
                "idx": i,
                "pred": r.pred,
                "correct": r.correct,
                "total_tokens": r.total_tokens,
                "route": r.route,
                "consensus_agreement": r.consensus.agreement_ratio,
                "consensus_confidence": r.consensus.confidence,
                "explore_tokens": r.explore_tokens,
                "resolve_tokens": r.resolve_tokens,
                "final_source": r.final_source,
            }
            for i, r in enumerate(spec_results)
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Results saved to {output_path}")

    # ---- Print comparison table ----
    print(f"\n{'='*70}")
    print(f"REASONING SPECULATION vs BASELINES")
    print(f"Model: {args.model} | Benchmark: {args.benchmark} | n={len(items)}")
    print(f"{'='*70}")
    print(f"{'Method':<35} {'Accuracy':>10} {'Avg Tokens':>12} {'vs Fixed-max':>12}")
    print(f"{'-'*35} {'-'*10} {'-'*12} {'-'*12}")

    # Baselines
    for name, b in sorted(baseline_results.items()):
        delta = ""
        print(f"{name:<35} {b['accuracy']:>10.4f} {b['avg_tokens']:>12.1f} {delta:>12}")

    # Reasoning Speculation
    max_fixed = max(
        (b["accuracy"] for b in baseline_results.values()),
        default=0
    )
    delta_str = f"{spec_acc - max_fixed:+.4f}" if max_fixed > 0 else ""
    print(f"{'Reasoning Speculation':<35} {spec_acc:>10.4f} {spec_avg_tokens:>12.1f} {delta_str:>12}")

    # Token efficiency
    if baseline_results:
        best_match = None
        for name, b in baseline_results.items():
            if b["accuracy"] <= spec_acc:
                if best_match is None or b["accuracy"] > best_match[1]["accuracy"]:
                    best_match = (name, b)
        if best_match:
            token_savings = 1.0 - spec_avg_tokens / best_match[1]["avg_tokens"]
            print(f"\nToken savings vs {best_match[0]} (matched accuracy): {token_savings:.1%}")

    print(f"\nRoute distribution: easy={route_counts['easy']/len(items):.1%}, "
          f"medium={route_counts['medium']/len(items):.1%}, "
          f"hard={route_counts['hard']/len(items):.1%}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

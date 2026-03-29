#!/usr/bin/env python3
"""Reasoning Speculation V3: Key optimizations over V1/V2.

Improvements:
  1. No-think probes: Disable thinking mode for exploration (saves tokens for actual reasoning)
  2. Smarter Medium route: Re-generate with consensus hints instead of blind extension
  3. Quality-aware routing: If too many probes need projection, route to hard
  4. Better Hard deliberation prompt with structured path comparison

Usage:
  python scripts/run_reasoning_speculation_v3.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 2 \
    --probe_budget 256 \
    --medium_budget 512 \
    --hard_budget 1024 \
    --seed 42 \
    --v3  # Enable V3 optimizations
"""

import argparse
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict
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
# Answer extraction (same as v1)
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
        model_id, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True,
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
        max_new_tokens=max_new_tokens, do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
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
# Core data classes
# ---------------------------------------------------------------------------
@dataclass
class PathResult:
    text: str
    tokens: int
    latency_s: float
    pred: Optional[str]
    has_final: bool
    pred_source: str
    used_projection: bool = False


@dataclass
class ConsensusSignal:
    agreement_ratio: float
    majority_answer: Optional[str]
    n_valid_preds: int
    n_paths: int
    answer_distribution: Dict[str, int]
    confidence: float
    route: str
    projection_rate: float  # V3: track how many paths needed projection


@dataclass
class SpeculationResult:
    pred: Optional[str]
    correct: bool
    total_tokens: int
    total_latency_s: float
    route: str
    consensus: ConsensusSignal
    n_explore_paths: int
    explore_tokens: int
    resolve_tokens: int
    final_source: str


# ---------------------------------------------------------------------------
# V3 Consensus with quality awareness
# ---------------------------------------------------------------------------
def compute_consensus_v3(paths: List[PathResult],
                         easy_threshold: float = 0.75,
                         medium_threshold: float = 0.5) -> ConsensusSignal:
    """V3 consensus: accounts for projection quality and path reliability."""
    valid_preds = []
    for p in paths:
        v = to_float(p.pred)
        if v is not None:
            valid_preds.append((round(v, 6), p.pred))

    projection_rate = sum(1 for p in paths if p.used_projection) / max(len(paths), 1)

    if not valid_preds:
        return ConsensusSignal(
            agreement_ratio=0.0, majority_answer=None, n_valid_preds=0,
            n_paths=len(paths), answer_distribution={}, confidence=0.0,
            route="hard", projection_rate=projection_rate,
        )

    answer_counts = Counter(v for v, _ in valid_preds)
    majority_val, majority_count = answer_counts.most_common(1)[0]
    agreement = majority_count / len(valid_preds)

    majority_str = None
    for v, s in valid_preds:
        if v == majority_val:
            majority_str = s
            break

    pred_validity = len(valid_preds) / len(paths)
    final_rate = sum(1 for p in paths if p.has_final) / len(paths)
    confidence = 0.5 * agreement + 0.3 * pred_validity + 0.2 * final_rate

    # V3 KEY IMPROVEMENT: Penalize confidence when projection rate is high
    # If most paths needed projection, the consensus is less reliable
    confidence *= (1.0 - 0.3 * projection_rate)

    # V3: Stricter routing when projection rate > 50%
    if projection_rate > 0.5:
        # Downgrade: easy→medium, medium→hard
        if agreement >= easy_threshold and confidence >= 0.6:
            route = "medium"  # Was easy, but projections make it uncertain
        elif agreement >= medium_threshold or confidence >= 0.4:
            route = "hard"  # Was medium, but projections → hard
        else:
            route = "hard"
    else:
        # Standard routing
        if agreement >= easy_threshold and confidence >= 0.6:
            route = "easy"
        elif agreement >= medium_threshold or confidence >= 0.4:
            route = "medium"
        else:
            route = "hard"

    return ConsensusSignal(
        agreement_ratio=agreement, majority_answer=majority_str,
        n_valid_preds=len(valid_preds), n_paths=len(paths),
        answer_distribution={str(k): v for k, v in answer_counts.items()},
        confidence=confidence, route=route, projection_rate=projection_rate,
    )


# ---------------------------------------------------------------------------
# V3 Reasoning Speculation
# ---------------------------------------------------------------------------
def reasoning_speculation_v3(
    model, tokenizer, question: str, gold: str,
    k_paths: int = 2, probe_budget: int = 256,
    medium_budget: int = 512, hard_budget: int = 1024,
    temperature: float = 0.7,
    enable_thinking: bool = True,
    strict_final_only: bool = True,
    projection_on_missing: bool = True,
    easy_threshold: float = 0.75,
    medium_threshold: float = 0.5,
    no_think_probes: bool = True,  # V3: disable thinking for probes
) -> SpeculationResult:
    """V3 Reasoning Speculation with optimized exploration and routing.

    Key improvements over V1/V2:
    1. No-think probes: exploration paths use non-thinking mode (more content per token)
    2. Quality-aware routing: high projection rate → downgrade confidence
    3. Smarter medium: re-generate with consensus hints
    4. Better hard deliberation with structured comparison
    """

    # V3: Probes use no-think mode to maximize reasoning content per token
    probe_thinking = not no_think_probes  # False if no_think_probes=True
    probe_prompt = build_prompt(question, tokenizer, enable_thinking=probe_thinking)

    # Resolution always uses thinking mode for deeper reasoning
    resolve_prompt = build_prompt(question, tokenizer, enable_thinking=enable_thinking)

    # ---- Phase 1: Parallel Exploration ----
    explore_paths = []
    explore_tokens = 0
    explore_latency = 0.0

    for k in range(k_paths):
        temp = 0.0 if k == 0 else temperature

        text, tokens, latency = generate_once(
            model, tokenizer, probe_prompt,
            max_new_tokens=probe_budget, temperature=temp,
        )

        pred, has_final, pred_source = parse_prediction(text, strict_final_only=strict_final_only)
        used_projection = False

        if pred is None and projection_on_missing:
            proj_prompt = (
                "Read the question and draft solution. "
                "Output exactly one line: Final answer: <number>\n\n"
                f"Question: {question}\n\n"
                f"Draft solution:\n{text}\n\n"
                "Final answer:"
            )
            proj_text, proj_tokens, proj_latency = generate_once(
                model, tokenizer, proj_prompt, max_new_tokens=16, temperature=0.0,
            )
            pred_proj, has_final_proj, _ = parse_prediction(proj_text, strict_final_only=False)
            if pred_proj is not None:
                pred = pred_proj
                has_final = has_final_proj
                pred_source = "projection"
                tokens += proj_tokens
                latency += proj_latency
                used_projection = True

        path = PathResult(
            text=text, tokens=tokens, latency_s=latency,
            pred=pred, has_final=has_final, pred_source=pred_source,
            used_projection=used_projection,
        )
        explore_paths.append(path)
        explore_tokens += tokens
        explore_latency += latency

    # ---- Phase 2: Quality-Aware Consensus ----
    consensus = compute_consensus_v3(explore_paths, easy_threshold, medium_threshold)

    # ---- Phase 3: Resolve ----
    resolve_tokens = 0
    resolve_latency = 0.0
    final_pred = None
    final_source = ""

    if consensus.route == "easy":
        final_pred = consensus.majority_answer
        final_source = "consensus_vote"

    elif consensus.route == "medium":
        # V3 IMPROVEMENT: Instead of blind extension, re-generate with hints
        # Provide the consensus answer and ask model to verify/correct
        hint_answers = []
        for p in explore_paths:
            if p.pred:
                hint_answers.append(p.pred)

        if hint_answers:
            unique_answers = list(set(str(to_float(a)) for a in hint_answers if to_float(a) is not None))
            hint_text = ", ".join(unique_answers[:3])
            medium_prompt = build_prompt(
                f"{question}\n\n"
                f"[Hint: Quick calculations suggest the answer might be around {hint_text}. "
                f"Please solve carefully and verify.]",
                tokenizer, enable_thinking=enable_thinking,
            )
        else:
            medium_prompt = resolve_prompt

        med_text, med_tokens, med_latency = generate_once(
            model, tokenizer, medium_prompt,
            max_new_tokens=medium_budget, temperature=0.0,
        )
        resolve_tokens = med_tokens
        resolve_latency = med_latency

        pred, has_final, pred_source = parse_prediction(med_text, strict_final_only=strict_final_only)
        if pred is not None:
            final_pred = pred
            final_source = "medium_guided"
        else:
            # Projection fallback
            proj_prompt = (
                "Read the question and solution. "
                "Output exactly one line: Final answer: <number>\n\n"
                f"Question: {question}\n\nSolution:\n{med_text}\n\nFinal answer:"
            )
            proj_text, proj_tok, proj_lat = generate_once(
                model, tokenizer, proj_prompt, max_new_tokens=16, temperature=0.0,
            )
            pred_p, _, _ = parse_prediction(proj_text, strict_final_only=False)
            if pred_p is not None:
                final_pred = pred_p
                final_source = "medium_projection"
            resolve_tokens += proj_tok
            resolve_latency += proj_lat

    else:  # hard
        # V3 IMPROVEMENT: Structured comparison prompt
        path_summaries = []
        for i, p in enumerate(explore_paths):
            pred_str = f"→ Got answer: {p.pred}" if p.pred else "→ No clear answer"
            # Include more context from each path
            excerpt = p.text[:800] if len(p.text) > 800 else p.text
            reliability = "reliable" if p.has_final and not p.used_projection else "uncertain"
            path_summaries.append(
                f"Approach {i+1} ({reliability}):\n{excerpt}\n{pred_str}"
            )

        deliberation_context = "\n\n---\n\n".join(path_summaries)

        # Add consensus info to deliberation
        consensus_info = ""
        if consensus.majority_answer:
            consensus_info = (
                f"\n\nNote: {consensus.n_valid_preds}/{consensus.n_paths} approaches "
                f"produced answers. The most common answer was {consensus.majority_answer} "
                f"(agreement: {consensus.agreement_ratio:.0%}). "
                f"Please carefully verify whether this is correct."
            )

        deliberation_prompt = build_prompt(
            f"{question}\n\n"
            f"I tried {k_paths} different approaches:\n\n"
            f"{deliberation_context}"
            f"{consensus_info}\n\n"
            f"Now let me carefully work through this problem step by step to find the correct answer:",
            tokenizer, enable_thinking=enable_thinking,
        )

        delib_text, delib_tokens, delib_latency = generate_once(
            model, tokenizer, deliberation_prompt,
            max_new_tokens=hard_budget, temperature=0.0,
        )
        resolve_tokens = delib_tokens
        resolve_latency = delib_latency

        pred, has_final, pred_source = parse_prediction(delib_text, strict_final_only=strict_final_only)
        if pred is not None:
            final_pred = pred
            final_source = "deliberation"
        else:
            proj_prompt = (
                "Read the question and the reasoning. "
                "Output exactly one line: Final answer: <number>\n\n"
                f"Question: {question}\n\nReasoning:\n{delib_text}\n\nFinal answer:"
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

    if final_pred is None:
        final_pred = consensus.majority_answer
        final_source = "fallback_consensus"

    correct = is_correct(final_pred, gold)

    return SpeculationResult(
        pred=final_pred, correct=correct,
        total_tokens=explore_tokens + resolve_tokens,
        total_latency_s=explore_latency + resolve_latency,
        route=consensus.route, consensus=consensus,
        n_explore_paths=k_paths, explore_tokens=explore_tokens,
        resolve_tokens=resolve_tokens, final_source=final_source,
    )


# ---------------------------------------------------------------------------
# Baselines (same as v1)
# ---------------------------------------------------------------------------
def run_fixed_baseline(model, tokenizer, question, gold, budget,
                       enable_thinking=True, strict_final_only=True,
                       projection_on_missing=True):
    prompt = build_prompt(question, tokenizer, enable_thinking=enable_thinking)
    text, tokens, latency = generate_once(model, tokenizer, prompt,
                                          max_new_tokens=budget, temperature=0.0)
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
    parser = argparse.ArgumentParser(description="Reasoning Speculation V3")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--k_paths", type=int, default=2)
    parser.add_argument("--probe_budget", type=int, default=256)
    parser.add_argument("--medium_budget", type=int, default=512)
    parser.add_argument("--hard_budget", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", action="store_true")
    parser.add_argument("--strict_final_only", action="store_true", default=True)
    parser.add_argument("--projection", action="store_true", default=True)
    parser.add_argument("--easy_threshold", type=float, default=0.75)
    parser.add_argument("--medium_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/speculation")
    parser.add_argument("--run_baselines", action="store_true")
    parser.add_argument("--device_map", default="auto")
    # V3 flags
    parser.add_argument("--v3", action="store_true", default=True,
                        help="Enable V3 optimizations")
    parser.add_argument("--no_think_probes", action="store_true", default=True,
                        help="Disable thinking mode for probe paths")
    parser.add_argument("--think_probes", action="store_true",
                        help="Enable thinking mode for probes (override no_think_probes)")
    args = parser.parse_args()

    if args.no_thinking:
        args.enable_thinking = False
    if args.think_probes:
        args.no_think_probes = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.split("/")[-1].replace(".", "_")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device_map)

    log.info(f"Loading {args.benchmark} (n={args.n_samples})...")
    if args.benchmark == "gsm8k":
        items = load_gsm8k(args.n_samples, args.seed)
    log.info(f"Loaded {len(items)} items")

    # ---- Run V3 Reasoning Speculation ----
    spec_results = []
    route_counts = Counter()
    correct_by_route = Counter()
    total_by_route = Counter()
    total_correct = 0
    total_tokens = 0

    version = "v3" if args.v3 else "v2"
    ntp = "no-think" if args.no_think_probes else "think"
    log.info(f"Running ReasonSpec {version} ({ntp} probes): K={args.k_paths}, "
             f"probe={args.probe_budget}, medium={args.medium_budget}, hard={args.hard_budget}")

    for i, item in enumerate(items):
        result = reasoning_speculation_v3(
            model, tokenizer, item["question"], item["gold"],
            k_paths=args.k_paths, probe_budget=args.probe_budget,
            medium_budget=args.medium_budget, hard_budget=args.hard_budget,
            temperature=args.temperature, enable_thinking=args.enable_thinking,
            strict_final_only=args.strict_final_only,
            projection_on_missing=args.projection,
            easy_threshold=args.easy_threshold,
            medium_threshold=args.medium_threshold,
            no_think_probes=args.no_think_probes,
        )
        spec_results.append(result)
        route_counts[result.route] += 1
        total_by_route[result.route] += 1
        if result.correct:
            total_correct += 1
            correct_by_route[result.route] += 1
        total_tokens += result.total_tokens

        if (i + 1) % 10 == 0 or (i + 1) == len(items):
            acc = total_correct / (i + 1)
            avg_tok = total_tokens / (i + 1)
            routes_str = " ".join(f"{r}={c}" for r, c in sorted(route_counts.items()))
            log.info(f"[{i+1}/{len(items)}] acc={acc:.4f} avg_tok={avg_tok:.1f} routes: {routes_str}")

    # Summary
    n = len(items)
    spec_acc = total_correct / n
    spec_avg_tok = total_tokens / n

    log.info("\n" + "=" * 60)
    log.info(f"ReasonSpec {version}: acc={spec_acc:.4f}, avg_tokens={spec_avg_tok:.1f}")
    log.info(f"Route distribution: {dict(route_counts)}")
    for route in ["easy", "medium", "hard"]:
        if total_by_route[route] > 0:
            r_acc = correct_by_route[route] / total_by_route[route]
            r_pct = total_by_route[route] / n * 100
            log.info(f"  {route}: {total_by_route[route]} ({r_pct:.1f}%), acc={r_acc:.4f}")

    # ---- Baselines ----
    baselines = {}
    if args.run_baselines:
        for budget in [args.probe_budget, args.medium_budget, args.hard_budget]:
            bname = f"fixed_{budget}"
            log.info(f"Running {bname} baseline...")
            b_correct = 0
            b_tokens = 0
            for item in items:
                pred, correct, tokens, _ = run_fixed_baseline(
                    model, tokenizer, item["question"], item["gold"],
                    budget, args.enable_thinking, args.strict_final_only, args.projection)
                b_correct += correct
                b_tokens += tokens
            b_acc = b_correct / n
            b_avg = b_tokens / n
            baselines[bname] = {"accuracy": b_acc, "avg_tokens": b_avg}
            log.info(f"  {bname}: acc={b_acc:.4f}, avg_tok={b_avg:.1f}")

        # SC baseline with same K and probe budget
        sc_name = f"sc_{args.k_paths}x{args.probe_budget}"
        log.info(f"Running {sc_name} baseline...")
        sc_correct = 0
        sc_tokens = 0
        for item in items:
            pred, correct, tokens, _ = run_sc_baseline(
                model, tokenizer, item["question"], item["gold"],
                args.probe_budget, args.k_paths, args.temperature,
                args.enable_thinking, args.strict_final_only, args.projection)
            sc_correct += correct
            sc_tokens += tokens
        sc_acc = sc_correct / n
        sc_avg = sc_tokens / n
        baselines[sc_name] = {"accuracy": sc_acc, "avg_tokens": sc_avg}
        log.info(f"  {sc_name}: acc={sc_acc:.4f}, avg_tok={sc_avg:.1f}")

    # ---- Save ----
    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "meta": {
            "version": version,
            "timestamp": timestamp,
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": n,
            "k_paths": args.k_paths,
            "probe_budget": args.probe_budget,
            "medium_budget": args.medium_budget,
            "hard_budget": args.hard_budget,
            "temperature": args.temperature,
            "enable_thinking": args.enable_thinking,
            "no_think_probes": args.no_think_probes,
            "seed": args.seed,
            "easy_threshold": args.easy_threshold,
            "medium_threshold": args.medium_threshold,
        },
        "reasoning_speculation": {
            "accuracy": spec_acc,
            "avg_tokens": spec_avg_tok,
            "route_distribution": {r: c/n for r, c in route_counts.items()},
            "route_accuracy": {
                r: {"count": total_by_route[r],
                    "accuracy": correct_by_route[r] / total_by_route[r] if total_by_route[r] > 0 else 0}
                for r in route_counts
            },
        },
        "baselines": baselines,
        "per_sample": [
            {
                "idx": i,
                "pred": r.pred,
                "correct": r.correct,
                "total_tokens": r.total_tokens,
                "route": r.route,
                "consensus_agreement": r.consensus.agreement_ratio,
                "consensus_confidence": r.consensus.confidence,
                "projection_rate": r.consensus.projection_rate,
                "explore_tokens": r.explore_tokens,
                "resolve_tokens": r.resolve_tokens,
                "final_source": r.final_source,
            }
            for i, r in enumerate(spec_results)
        ],
    }

    fname = f"reasonspec_{version}_{model_tag}_{args.benchmark}_{timestamp}.json"
    fpath = os.path.join(args.output_dir, fname)
    with open(fpath, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"Results saved to {fpath}")

    # Print comparison
    print("\n" + "=" * 70)
    print(f"REASONSPEC {version.upper()} vs BASELINES")
    print(f"Model: {args.model} | Benchmark: {args.benchmark} | n={n}")
    print(f"V3 features: no_think_probes={args.no_think_probes}")
    print("=" * 70)
    all_methods = {f"ReasonSpec-{version}": {"accuracy": spec_acc, "avg_tokens": spec_avg_tok}}
    all_methods.update(baselines)
    print(f"{'Method':<35} {'Accuracy':>10} {'Avg Tokens':>12}")
    print("-" * 60)
    for m, d in sorted(all_methods.items(), key=lambda x: -x[1]["accuracy"]):
        print(f"{m:<35} {d['accuracy']:>10.4f} {d['avg_tokens']:>12.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

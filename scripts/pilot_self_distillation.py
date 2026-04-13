#!/usr/bin/env python3
"""MRSD (Multi-Round Self-Distillation) — Iterative reasoning refinement.

Per-sample pipeline:
    Stage 0: nothink@B1 -> natural stop = ACCEPT (easy), else escalate
    Round 1: think@B_think -> trace R1 -> nothink(R1)@B_answer -> A1
    Round 2: think_with_hint(Q, A1)@B_think -> R2 -> nothink(R2)@B_answer -> A2
             If A1==A2: CONVERGED, accept A2
    Round 3: think_with_hint(Q, A2)@B_think -> R3 -> nothink(R3)@B_answer -> A3
             Majority vote {A1, A2, A3}

Baselines (same samples, no extra inference for nothink/iris):
    nothink_only  -- nothink@B1 (= Stage 0)
    town          -- nothink@B1 -> think@B_think (parse answer from think text)
    iris_single   -- nothink@B1 -> think@B_think -> nothink(trace)@B_answer  (= Round 1)
    mrsd          -- full multi-round

Usage:
    python scripts/pilot_self_distillation.py \
        --model Qwen/Qwen3-8B --benchmark gsm8k \
        --n_samples 200 --b1 256 --b_think 512 --b_answer 128 \
        --max_rounds 3 --seed 42
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Reuse from run_iris.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_iris import (
    load_model_and_tokenizer,
    generate_simple,
    model_input_device,
    build_prompt,
    parse_prediction_dispatch,
    is_correct_dispatch,
    load_benchmark_data,
    to_float,
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
# Thinking-trace extraction
# ---------------------------------------------------------------------------
THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)


def extract_thinking_trace(text: str) -> str:
    """Extract reasoning between <think> and </think>.

    If </think> is missing (truncated), returns everything after <think>.
    Returns empty string if no <think> tag found.
    """
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip()
    if "<think>" in text:
        return text.split("<think>", 1)[1].strip()
    return ""


def has_complete_answer_after_think(text: str, benchmark: str) -> Tuple[bool, Optional[str]]:
    """Check if thinking output contains </think> followed by a parseable answer."""
    if "</think>" not in text:
        return False, None
    after = text.split("</think>", 1)[1]
    pred, _src = parse_prediction_dispatch(after, benchmark)
    return pred is not None, pred


# ---------------------------------------------------------------------------
# Answer agreement & majority vote
# ---------------------------------------------------------------------------

def normalize_math_answer(s: str) -> str:
    """Light normalization for math500 answer comparison."""
    s = s.strip()
    for wrapper in [r"\boxed{", r"\text{"]:
        if s.startswith(wrapper) and s.endswith("}"):
            s = s[len(wrapper):-1]
    s = s.replace("$", "").replace(" ", "").replace(",", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    return s.lower().strip()


def answers_agree(a: Optional[str], b: Optional[str], benchmark: str) -> bool:
    """Check if two answers are equivalent (benchmark-aware)."""
    if a is None or b is None:
        return False
    if benchmark == "math500":
        return normalize_math_answer(a) == normalize_math_answer(b)
    else:  # gsm8k
        fa, fb = to_float(a), to_float(b)
        if fa is not None and fb is not None:
            return abs(fa - fb) <= 1e-6 * max(1.0, abs(fb))
        return a.strip() == b.strip()


def majority_vote(answers: List[Optional[str]], benchmark: str) -> Optional[str]:
    """Majority vote with benchmark-aware equivalence.

    Groups answers by equivalence, returns the representative of the largest
    group. Ties broken by first occurrence.
    """
    valid = [(i, a) for i, a in enumerate(answers) if a is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0][1]

    groups: List[List[Tuple[int, str]]] = []
    for idx, ans in valid:
        placed = False
        for g in groups:
            if answers_agree(ans, g[0][1], benchmark):
                g.append((idx, ans))
                placed = True
                break
        if not placed:
            groups.append([(idx, ans)])

    groups.sort(key=lambda g: (-len(g), g[0][0]))
    return groups[0][0][1]


# ---------------------------------------------------------------------------
# Hint-based thinking prompt (Round 2+)
# ---------------------------------------------------------------------------

def generate_with_hint(
    model, tokenizer, question: str, hint_answer: str,
    max_new_tokens: int, benchmark: str,
) -> Tuple[str, int, float, bool]:
    """Think mode with a hint from the previous round's answer.

    Returns (text, n_tokens, elapsed, hit_budget).
    """
    hint_system = {
        "gsm8k": (
            f"You are a careful math solver. A previous attempt got the answer {hint_answer}. "
            f"Think carefully about whether this is correct. If wrong, find the right answer. "
            f"End with: Final answer: <number>."
        ),
        "math500": (
            f"You are an expert mathematician. A previous attempt got \\boxed{{{hint_answer}}}. "
            f"Verify this answer. If wrong, solve again. Put your final answer inside \\boxed{{}}."
        ),
    }
    system_text = hint_system.get(benchmark, hint_system["gsm8k"])

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": question},
    ]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True}
    try:
        prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
    except TypeError:
        chat_kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
    except Exception:
        prompt = f"{system_text}\n\nQuestion: {question}\nSolution:\n"

    return generate_simple(model, tokenizer, prompt, max_new_tokens, temperature=0.0)


# ---------------------------------------------------------------------------
# Decoupled answer generation from thinking trace
# ---------------------------------------------------------------------------

def generate_decoupled_from_trace(
    model, tokenizer, question: str, trace: str,
    max_new_tokens: int, benchmark: str,
) -> Tuple[str, int, float, bool]:
    """Nothink mode with reasoning trace as context.

    Returns (text, n_tokens, elapsed, hit_budget).
    """
    base_nothink_prompt = build_prompt(
        question, tokenizer, enable_thinking=False, benchmark=benchmark,
    )

    clean_trace = trace.replace("<think>", "").replace("</think>", "").strip()
    if not clean_trace:
        return generate_simple(
            model, tokenizer, base_nothink_prompt, max_new_tokens, temperature=0.0,
        )

    if benchmark == "math500":
        suffix = f"Based on my reasoning: {clean_trace}\n\nThe final answer is \\boxed{{"
    else:
        suffix = f"Based on my reasoning: {clean_trace}\n\nFinal answer: "
    prompt = base_nothink_prompt + suffix

    return generate_simple(model, tokenizer, prompt, max_new_tokens, temperature=0.0)


# ---------------------------------------------------------------------------
# MRSD per-sample pipeline
# ---------------------------------------------------------------------------

def run_mrsd_sample(
    model, tokenizer, question: str, gold: str,
    benchmark: str, b1: int, b_think: int, b_answer: int, max_rounds: int,
) -> Dict:
    """Full MRSD cascade for one sample.

    Returns dict with per-stage details, final prediction, and token counts.
    Also stores intermediate data needed for baseline computation.
    """
    result: Dict = {"rounds": [], "stage0": {}}
    total_tokens = 0

    # ==================== Stage 0: nothink@B1 ====================
    prompt_s0 = build_prompt(question, tokenizer, enable_thinking=False, benchmark=benchmark)
    text_s0, tok_s0, elapsed_s0, hit_s0 = generate_simple(
        model, tokenizer, prompt_s0, max_new_tokens=b1, temperature=0.0,
    )
    pred_s0, src_s0 = parse_prediction_dispatch(text_s0, benchmark)
    total_tokens += tok_s0

    result["stage0"] = {
        "pred": pred_s0, "tokens": tok_s0,
        "elapsed": round(elapsed_s0, 4), "hit_budget": hit_s0,
        "raw_text": text_s0,
    }

    if not hit_s0:
        result.update(
            final_pred=pred_s0, final_method="stage0_accept",
            final_round=0, tokens_total=total_tokens, converged=True,
        )
        return result

    # ==================== Round 1: think -> trace -> nothink ====================
    prompt_r1 = build_prompt(question, tokenizer, enable_thinking=True, benchmark=benchmark)
    text_r1, tok_r1, elapsed_r1, hit_r1 = generate_simple(
        model, tokenizer, prompt_r1, max_new_tokens=b_think, temperature=0.0,
    )
    total_tokens += tok_r1

    complete_r1, direct_pred_r1 = has_complete_answer_after_think(text_r1, benchmark)
    trace_r1 = extract_thinking_trace(text_r1)

    # Decoupled answering
    ans_r1, tok_a1, elapsed_a1, _ = generate_decoupled_from_trace(
        model, tokenizer, question, trace_r1, b_answer, benchmark,
    )
    total_tokens += tok_a1
    pred_a1, _ = parse_prediction_dispatch(ans_r1, benchmark)

    if pred_a1 is None and direct_pred_r1 is not None:
        pred_a1 = direct_pred_r1

    # Store Round 1 think text for TOWN baseline extraction
    result["_r1_think_text"] = text_r1
    result["_r1_think_complete"] = complete_r1

    result["rounds"].append({
        "round": 1,
        "think_tokens": tok_r1, "think_hit_budget": hit_r1,
        "think_natural_stop": complete_r1,
        "trace_len_chars": len(trace_r1),
        "answer_tokens": tok_a1, "pred": pred_a1,
        "elapsed": round(elapsed_r1 + elapsed_a1, 4),
        "raw_think": text_r1, "raw_answer": ans_r1,
    })
    answers = [pred_a1]
    prev_answer = pred_a1

    # ==================== Rounds 2+ with hints ====================
    converged = False
    for rnd in range(2, max_rounds + 1):
        if prev_answer is None:
            break

        text_rk, tok_rk, elapsed_rk, hit_rk = generate_with_hint(
            model, tokenizer, question, prev_answer, b_think, benchmark,
        )
        total_tokens += tok_rk

        trace_rk = extract_thinking_trace(text_rk)
        complete_k, direct_pred_k = has_complete_answer_after_think(text_rk, benchmark)

        ans_rk, tok_ak, elapsed_ak, _ = generate_decoupled_from_trace(
            model, tokenizer, question, trace_rk, b_answer, benchmark,
        )
        total_tokens += tok_ak
        pred_ak, _ = parse_prediction_dispatch(ans_rk, benchmark)

        if pred_ak is None and direct_pred_k is not None:
            pred_ak = direct_pred_k

        result["rounds"].append({
            "round": rnd, "hint_answer": prev_answer,
            "think_tokens": tok_rk, "think_hit_budget": hit_rk,
            "think_natural_stop": complete_k,
            "trace_len_chars": len(trace_rk),
            "answer_tokens": tok_ak, "pred": pred_ak,
            "elapsed": round(elapsed_rk + elapsed_ak, 4),
            "raw_think": text_rk, "raw_answer": ans_rk,
        })
        answers.append(pred_ak)

        if answers_agree(pred_ak, prev_answer, benchmark):
            converged = True
            result.update(
                final_pred=pred_ak, final_method=f"converged_round{rnd}",
                final_round=rnd, tokens_total=total_tokens, converged=True,
            )
            return result

        prev_answer = pred_ak

    # ==================== No convergence -> majority vote ====================
    voted = majority_vote(answers, benchmark)
    result.update(
        final_pred=voted,
        final_method=f"majority_vote_rounds{len(answers)}",
        final_round=len(answers),
        tokens_total=total_tokens, converged=False,
        vote_answers=[str(a) for a in answers],
    )
    return result


# ---------------------------------------------------------------------------
# Baseline computation (extracted from MRSD per-sample data, no extra calls)
# ---------------------------------------------------------------------------

def compute_baselines(per_sample: List[Dict], benchmark: str) -> Dict[str, Dict]:
    """Compute nothink_only / town / iris_single metrics from stored MRSD data."""
    n = len(per_sample)
    if n == 0:
        empty = {"accuracy": 0.0, "avg_tokens": 0.0}
        return {"nothink_only": empty, "town": empty, "iris_single": empty}

    nothink_c = town_c = iris_c = 0
    nothink_t = town_t = iris_t = 0

    for s in per_sample:
        # nothink_only: Stage 0 prediction regardless
        nothink_c += s["nothink_correct"]
        nothink_t += s["s0_tokens"]

        if not s["s0_hit_budget"]:
            # Easy sample -- all methods use Stage 0
            town_c += s["nothink_correct"]
            town_t += s["s0_tokens"]
            iris_c += s["nothink_correct"]
            iris_t += s["s0_tokens"]
        else:
            # TOWN: parse answer from Round 1 thinking text directly
            town_c += s["town_correct"]
            town_t += s["s0_tokens"] + s["r1_think_tokens"]
            # IRIS single: Round 1 decoupled answer
            iris_c += s["iris_correct"]
            iris_t += s["s0_tokens"] + s["r1_think_tokens"] + s["r1_ans_tokens"]

    return {
        "nothink_only": {
            "accuracy": round(nothink_c / n, 4),
            "avg_tokens": round(nothink_t / n, 2),
        },
        "town": {
            "accuracy": round(town_c / n, 4),
            "avg_tokens": round(town_t / n, 2),
        },
        "iris_single": {
            "accuracy": round(iris_c / n, 4),
            "avg_tokens": round(iris_t / n, 2),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MRSD: Multi-Round Self-Distillation pilot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                        choices=["gsm8k", "math500"])
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--b1", type=int, default=256,
                        help="Stage 0 nothink budget")
    parser.add_argument("--b_think", type=int, default=512,
                        help="Per-round thinking budget")
    parser.add_argument("--b_answer", type=int, default=128,
                        help="Per-round decoupled answer budget")
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--no_baselines", action="store_true", default=False,
                        help="Skip baseline computation")
    parser.add_argument("--output_dir", type=str,
                        default="results/pilot_self_distillation")
    parser.add_argument("--checkpoint_every", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.split("/")[-1].replace(".", "_").replace("-", "_")

    model, tokenizer = load_model_and_tokenizer(args.model)

    items = load_benchmark_data(args.benchmark, args.n_samples, args.seed)
    n = len(items)
    log.info(f"Loaded {n} {args.benchmark} samples (seed={args.seed})")
    log.info(f"Config: B1={args.b1}  B_think={args.b_think}  "
             f"B_answer={args.b_answer}  max_rounds={args.max_rounds}")

    # ====================== Run MRSD ======================
    per_sample: List[Dict] = []
    mrsd_correct = 0
    mrsd_total_tokens = 0
    mrsd_total_rounds = 0
    converged_count = 0
    round_correct: Dict[int, int] = {}
    round_total: Dict[int, int] = {}
    conv_correct = conv_total = unconv_correct = unconv_total = 0

    t_start = time.perf_counter()

    for i, item in enumerate(items):
        question, gold = item["question"], item["gold"]

        out = run_mrsd_sample(
            model, tokenizer, question, gold, args.benchmark,
            b1=args.b1, b_think=args.b_think, b_answer=args.b_answer,
            max_rounds=args.max_rounds,
        )
        pred = out["final_pred"]
        ok = is_correct_dispatch(pred, gold, args.benchmark)
        if ok:
            mrsd_correct += 1
        mrsd_total_tokens += out["tokens_total"]
        fr = out["final_round"]
        mrsd_total_rounds += fr

        round_total[fr] = round_total.get(fr, 0) + 1
        if ok:
            round_correct[fr] = round_correct.get(fr, 0) + 1

        if out["converged"]:
            converged_count += 1
            conv_total += 1
            conv_correct += int(ok)
        else:
            unconv_total += 1
            unconv_correct += int(ok)

        # --- Extract baseline info from MRSD run ---
        s0 = out["stage0"]
        nothink_ok = is_correct_dispatch(s0["pred"], gold, args.benchmark)

        town_ok = False
        r1_think_tokens = 0
        r1_ans_tokens = 0
        iris_ok = False

        if not s0["hit_budget"]:
            town_ok = nothink_ok
            iris_ok = nothink_ok
        elif out["rounds"]:
            r1 = out["rounds"][0]
            r1_think_tokens = r1["think_tokens"]
            r1_ans_tokens = r1["answer_tokens"]

            # TOWN: parse from think text (answer only if thinking completed)
            r1_text = out.get("_r1_think_text", "")
            if out.get("_r1_think_complete") and "</think>" in r1_text:
                after = r1_text.split("</think>", 1)[1]
                town_pred, _ = parse_prediction_dispatch(after, args.benchmark)
            else:
                town_pred, _ = parse_prediction_dispatch(r1_text, args.benchmark)
            town_ok = is_correct_dispatch(town_pred, gold, args.benchmark)

            # IRIS single = Round 1 decoupled answer
            iris_ok = is_correct_dispatch(r1["pred"], gold, args.benchmark)

        # Collect raw outputs for re-scoring capability
        s0_raw = s0.get("raw_text", "")
        r1_raw_think = ""
        r1_raw_answer = ""
        if out["rounds"]:
            r1_raw_think = out["rounds"][0].get("raw_think", "")
            r1_raw_answer = out["rounds"][0].get("raw_answer", "")

        row = {
            "idx": i, "gold": gold,
            "mrsd_pred": str(pred), "mrsd_correct": int(ok),
            "mrsd_tokens": out["tokens_total"],
            "mrsd_round": fr, "mrsd_method": out["final_method"],
            "mrsd_converged": out["converged"],
            "s0_pred": str(s0["pred"]), "s0_tokens": s0["tokens"],
            "s0_hit_budget": s0["hit_budget"],
            "nothink_correct": int(nothink_ok),
            "town_correct": int(town_ok),
            "iris_correct": int(iris_ok),
            "r1_think_tokens": r1_think_tokens,
            "r1_ans_tokens": r1_ans_tokens,
            # Raw outputs for re-scoring
            "s0_raw": s0_raw,
            "r1_raw_think": r1_raw_think,
            "r1_raw_answer": r1_raw_answer,
        }
        per_sample.append(row)

        # Progress
        if (i + 1) % 20 == 0 or i == n - 1:
            done = i + 1
            elapsed = time.perf_counter() - t_start
            rate = done / (elapsed / 60) if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else 0
            log.info(
                f"  [{done}/{n}] acc={mrsd_correct/done:.3f}  "
                f"avg_tok={mrsd_total_tokens/done:.0f}  "
                f"avg_rnd={mrsd_total_rounds/done:.1f}  "
                f"conv={converged_count}/{done}  "
                f"{rate:.1f} samp/min  ETA={eta:.0f}min"
            )

        # Checkpoint
        if (i + 1) % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{i+1}.json")
            with open(ckpt_path, "w") as f:
                json.dump({"n_done": i + 1, "per_sample": per_sample},
                          f, indent=2, default=str)
            log.info(f"  Checkpoint: {ckpt_path}")

    # ====================== Summaries ======================
    mrsd_acc = mrsd_correct / n if n else 0.0
    mrsd_avg_tok = mrsd_total_tokens / n if n else 0.0
    mrsd_avg_rnd = mrsd_total_rounds / n if n else 0.0
    conv_rate = converged_count / n if n else 0.0

    acc_by_round = {}
    for rnd in sorted(round_total):
        t = round_total[rnd]
        c = round_correct.get(rnd, 0)
        acc_by_round[rnd] = round(c / t, 4) if t else 0.0

    mrsd_summary = {
        "accuracy": round(mrsd_acc, 4),
        "avg_tokens": round(mrsd_avg_tok, 2),
        "avg_rounds": round(mrsd_avg_rnd, 2),
        "convergence_rate": round(conv_rate, 4),
        "accuracy_by_round": acc_by_round,
        "accuracy_converged": round(conv_correct / conv_total, 4) if conv_total else 0.0,
        "accuracy_unconverged": round(unconv_correct / unconv_total, 4) if unconv_total else 0.0,
    }

    baselines = {}
    if not args.no_baselines:
        baselines = compute_baselines(per_sample, args.benchmark)

    # ====================== Output JSON ======================
    output = {
        "meta": {
            "script": "pilot_self_distillation.py",
            "timestamp_utc": timestamp,
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": n,
            "seed": args.seed,
            "elapsed_s": round(time.perf_counter() - t_start, 1),
        },
        "config": {
            "b1": args.b1, "b_think": args.b_think,
            "b_answer": args.b_answer, "max_rounds": args.max_rounds,
        },
        "mrsd_summary": mrsd_summary,
        "baselines": baselines,
        "per_sample": per_sample,
    }

    out_fname = (
        f"mrsd_{model_tag}_{args.benchmark}_b1{args.b1}_bt{args.b_think}"
        f"_ba{args.b_answer}_r{args.max_rounds}_{timestamp}.json"
    )
    out_path = os.path.join(args.output_dir, out_fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"\nSaved: {out_path}")

    # ====================== Print comparison table ======================
    log.info("")
    log.info("=" * 72)
    log.info("  MRSD Pilot Results")
    log.info("=" * 72)
    log.info(f"  Model:       {args.model}")
    log.info(f"  Benchmark:   {args.benchmark} (n={n})")
    log.info(f"  Budgets:     B1={args.b1}  B_think={args.b_think}  B_answer={args.b_answer}")
    log.info(f"  Max rounds:  {args.max_rounds}")
    log.info("-" * 72)
    log.info(f"  {'Method':<20} {'Accuracy':>10} {'Avg Tokens':>12}")
    log.info("-" * 72)

    if not args.no_baselines:
        for name in ["nothink_only", "town", "iris_single"]:
            bl = baselines[name]
            log.info(f"  {name:<20} {bl['accuracy']:>10.1%} {bl['avg_tokens']:>12.0f}")

    log.info(f"  {'mrsd':<20} {mrsd_acc:>10.1%} {mrsd_avg_tok:>12.0f}")
    log.info("-" * 72)
    log.info(f"  Avg rounds:        {mrsd_avg_rnd:.2f}")
    log.info(f"  Convergence rate:  {conv_rate:.1%}")
    log.info(f"  Acc (converged):   {mrsd_summary['accuracy_converged']:.1%}")
    log.info(f"  Acc (unconverged): {mrsd_summary['accuracy_unconverged']:.1%}")
    log.info(f"  Round distrib:     {dict(round_total)}")
    log.info(f"  Acc by round:      {acc_by_round}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()

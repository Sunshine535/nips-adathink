#!/usr/bin/env python3
"""RCV-IRIS: Recoverability-Calibrated Verifier IRIS.

Upgrades IRIS cascade with:
1. Stage0 acceptance verifier (validates natural-stop answers)
2. Prefix recoverability gate (estimates Stage3 extraction success)
3. Full audit logging per sample

Three variants for A/B/C comparison:
  --variant existing_fragment  (A: original IRIS, no gates)
  --variant rcv_no_gate        (B: online IRIS, no verifier gates)
  --variant full_rcv           (C: full RCV-IRIS with gates)

Usage:
    python scripts/run_rcv_iris.py \
        --model Qwen/Qwen3-8B --benchmark math500 \
        --n_samples 200 --b1 512 --b2_max 4096 --b_answer 512 \
        --variant full_rcv --seed 42
"""
import argparse, json, logging, os, random, re, sys, time
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import parse_prediction_math, is_correct_math
from rcv_signals import (
    answer_validity_score, stage0_acceptance_features,
    prefix_recoverability_features, extractor_margin, compute_rcv_decision,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_benchmark(benchmark, n, seed):
    from datasets import load_dataset
    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        idxs = list(range(len(ds)))
        random.seed(seed); random.shuffle(idxs)
        return [{"q": ds[i]["question"],
                 "gold": ds[i]["answer"].split("####")[-1].strip().replace(",", ""),
                 "idx": idxs[i]} for i in range(min(n, len(idxs)))]
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = list(ds)
        random.seed(seed); random.shuffle(items)
        return [{"q": s["problem"], "gold": s["answer"], "idx": i}
                for i, s in enumerate(items[:n])]


def generate_simple(model, tok, question, budget, enable_thinking, benchmark="gsm8k"):
    system = "You are a careful math solver."
    if enable_thinking:
        system += " Think step by step."
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=enable_thinking)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    dev = model_input_device(model)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    elapsed = time.perf_counter() - t0
    gen = out[0][in_len:]
    text = tok.decode(gen, skip_special_tokens=True)
    return text, len(gen), len(gen) < budget, elapsed


def generate_extraction(model, tok, question, prefix, budget, benchmark, prompt_type="strict"):
    clean = prefix
    for tag in ["<think>", "</think>", "<|think|>", "<|/think|>"]:
        clean = clean.replace(tag, "")
    clean = clean.strip()

    if prompt_type == "strict":
        if benchmark == "math500":
            sys_text = ("You are an expert mathematician. I have done most of the reasoning already. "
                        "Your job is ONLY to extract or compute the final answer and output it in "
                        "\\boxed{ANSWER}. Do not re-solve.")
        else:
            sys_text = ("You are a careful math solver. Extract the final numerical answer. "
                        "End with: Final answer: <number>.")
    else:
        sys_text = "You are a careful math solver."

    messages = [{"role": "system", "content": sys_text},
                {"role": "user", "content": question}]
    try:
        base = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        base = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)

    if clean:
        if benchmark == "math500":
            prompt = base + f"Based on my reasoning: {clean}\n\nThe final answer is \\boxed{{"
        else:
            prompt = base + f"Based on my reasoning: {clean}\n\nFinal answer: "
    else:
        prompt = base

    dev = model_input_device(model)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    elapsed = time.perf_counter() - t0
    gen = out[0][in_len:]
    text = tok.decode(gen, skip_special_tokens=True)
    return text, len(gen), elapsed


def parse_answer(text, benchmark):
    if benchmark == "math500":
        pred, _, src = parse_prediction_math(text)
        return pred, src
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return (nums[-1].replace(",", "") if nums else None), "regex"


def check_correct(pred, gold, benchmark):
    if benchmark == "math500":
        return is_correct_math(pred, gold)
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-4
    except (ValueError, TypeError):
        return False


def run_rcv_sample(model, tok, item, args):
    """Run one sample through RCV-IRIS pipeline."""
    r = {"idx": item["idx"], "gold": item["gold"]}
    bm = args.benchmark
    variant = args.variant

    # === Stage 0: Nothink probe ===
    s0_text, s0_tok, s0_natural, s0_elapsed = generate_simple(
        model, tok, item["q"], args.b1, False, bm)
    s0_pred, s0_src = parse_answer(s0_text, bm)
    s0_correct = check_correct(s0_pred, item["gold"], bm)

    r["stage0"] = {"text_len": len(s0_text), "tokens": s0_tok,
                   "natural_stop": s0_natural, "pred": s0_pred,
                   "pred_source": s0_src, "elapsed": round(s0_elapsed, 3)}

    # === Stage 0 acceptance decision ===
    use_stage0_gate = variant in ("full_rcv", "stage0_only")
    if s0_natural:
        s0_features = stage0_acceptance_features(
            item["q"], s0_pred, s0_text, s0_src, not s0_natural)
        r["stage0"]["features"] = s0_features

        if use_stage0_gate:
            accept_score = (s0_features["answer_valid"] * 0.5
                            + (1.0 - s0_features["pred_is_none"]) * 0.3
                            + s0_features["parse_source_boxed"] * 0.2)
            r["stage0"]["accept_score"] = round(accept_score, 3)

            if accept_score < args.tau_accept:
                r["stage0"]["decision"] = "REJECT_ESCALATE"
            else:
                r["stage0"]["decision"] = "ACCEPT"
                r.update({"final_stage": 0, "pred": s0_pred, "correct": s0_correct,
                          "tokens_total": s0_tok, "decision": "ACCEPT_STAGE0"})
                return r
        else:
            r["stage0"]["decision"] = "ACCEPT"
            r.update({"final_stage": 0, "pred": s0_pred, "correct": s0_correct,
                      "tokens_total": s0_tok, "decision": "ACCEPT_STAGE0"})
            return r

    # === Stage 2: Thinking ===
    s2_text, s2_tok, s2_natural, s2_elapsed = generate_simple(
        model, tok, item["q"], args.b2_max, True, bm)
    r["stage2"] = {"tokens": s2_tok, "natural_stop": s2_natural,
                   "elapsed": round(s2_elapsed, 3)}

    if s2_natural:
        s2_pred, s2_src = parse_answer(s2_text, bm)
        s2_correct = check_correct(s2_pred, item["gold"], bm)
        r.update({"final_stage": 2, "pred": s2_pred, "correct": s2_correct,
                  "tokens_total": s0_tok + s2_tok, "decision": "STAGE2_COMPLETE"})
        return r

    # === Stage 3: Extraction decision ===
    # Run strict extraction probe
    strict_text, strict_tok, strict_elapsed = generate_extraction(
        model, tok, item["q"], s2_text, args.b_answer, bm, "strict")
    strict_pred, strict_src = parse_answer(strict_text, bm)

    # Run soft extraction probe (same prompt, no scaffold)
    soft_text, soft_tok, soft_elapsed = generate_extraction(
        model, tok, item["q"], s2_text, args.b_answer, bm, "soft")
    soft_pred, soft_src = parse_answer(soft_text, bm)

    # Compute recoverability features
    pf = prefix_recoverability_features(
        item["q"], s2_text, strict_pred, soft_pred, strict_src, soft_src)
    margin = extractor_margin(strict_pred, soft_pred, strict_src, soft_src)

    r["stage3"] = {
        "strict_pred": strict_pred, "strict_source": strict_src, "strict_tokens": strict_tok,
        "soft_pred": soft_pred, "soft_source": soft_src, "soft_tokens": soft_tok,
        "recoverability_features": pf, "extractor_margin": round(margin, 3),
    }

    use_recover_gate = variant in ("full_rcv", "recover_only")
    if use_recover_gate:
        decision = compute_rcv_decision(
            stage0_acceptance_features(item["q"], s0_pred, s0_text, s0_src, True),
            pf, args.tau_accept, args.tau_recover)
        r["stage3"]["decision"] = decision

        if decision == "EXTRACT_STAGE3":
            final_pred = strict_pred
            final_src = strict_src
            total_tok = s0_tok + s2_tok + strict_tok
        else:
            # FALLBACK_TOWN: parse from truncated thinking
            town_pred, town_src = parse_answer(s2_text, bm)
            final_pred = town_pred
            final_src = f"town_{town_src}"
            total_tok = s0_tok + s2_tok
    else:
        # No gate: always extract
        final_pred = strict_pred
        final_src = strict_src
        total_tok = s0_tok + s2_tok + strict_tok
        r["stage3"]["decision"] = "EXTRACT_ALWAYS"

    final_correct = check_correct(final_pred, item["gold"], bm)
    r.update({"final_stage": 3, "pred": final_pred, "pred_source": final_src,
              "correct": final_correct, "tokens_total": total_tok,
              "decision": r["stage3"]["decision"]})
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--benchmark", default="math500", choices=["math500", "gsm8k"])
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--b1", type=int, default=512)
    p.add_argument("--b2_max", type=int, default=4096)
    p.add_argument("--b_answer", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variant", default="full_rcv",
                   choices=["existing_fragment", "rcv_no_gate", "full_rcv",
                            "stage0_only", "recover_only"])
    p.add_argument("--tau_accept", type=float, default=0.7)
    p.add_argument("--tau_recover", type=float, default=0.5)
    p.add_argument("--output_dir", default="results/rcv_iris")
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    log.info(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    items = load_benchmark(args.benchmark, args.n_samples, args.seed)
    log.info(f"Loaded {len(items)} items, variant={args.variant}")
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results = []
    n_correct = 0
    total_tokens = 0
    decisions = {}

    for i, item in enumerate(items):
        r = run_rcv_sample(model, tok, item, args)
        results.append(r)
        if r["correct"]: n_correct += 1
        total_tokens += r["tokens_total"]
        d = r.get("decision", "unknown")
        decisions[d] = decisions.get(d, 0) + 1

        if (i + 1) % 20 == 0:
            acc = n_correct / (i + 1)
            avg_tok = total_tokens / (i + 1)
            log.info(f"[{i+1}/{len(items)}] acc={acc*100:.1f}% avg_tok={avg_tok:.0f} decisions={decisions}")

    n = len(results)
    summary = {
        "meta": {"model": args.model, "benchmark": args.benchmark,
                 "n": n, "seed": args.seed, "variant": args.variant,
                 "b1": args.b1, "b2_max": args.b2_max, "b_answer": args.b_answer,
                 "tau_accept": args.tau_accept, "tau_recover": args.tau_recover,
                 "timestamp": ts},
        "accuracy": n_correct / n,
        "avg_tokens": total_tokens / n,
        "n_correct": n_correct,
        "decisions": decisions,
        "per_sample": results,
    }

    fname = f"rcv_{args.variant}_{args.benchmark}_{ts}.json"
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\n=== RCV-IRIS ({args.variant}) ===")
    log.info(f"Accuracy: {n_correct}/{n} = {n_correct/n*100:.1f}%")
    log.info(f"Avg tokens: {total_tokens/n:.0f}")
    log.info(f"Decisions: {decisions}")


if __name__ == "__main__":
    main()

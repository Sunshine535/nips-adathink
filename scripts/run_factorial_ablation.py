#!/usr/bin/env python3
"""2×2 Factorial ablation: mode (think/nothink) × prompt (neutral/extraction).

On Stage-2-exhausted samples (same truncated prefix), run all 4 conditions:
  (A) think + neutral prompt
  (B) think + extraction prompt
  (C) nothink + neutral prompt
  (D) nothink + extraction prompt  ← our Stage-3

Reports main effects, interaction, and McNemar for all pairs.
"""
import argparse, json, logging, os, random, re, sys, time
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import parse_prediction_math, is_correct_math

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
                 "gold": ds[i]["answer"].split("####")[-1].strip().replace(",", "")}
                for i in idxs[:n]]
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = list(ds)
        random.seed(seed); random.shuffle(items)
        return [{"q": s["problem"], "gold": s["answer"]} for s in items[:n]]


def generate_simple(model, tok, question, budget, enable_thinking):
    messages = [{"role": "system", "content": "You are a careful math solver."},
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
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][in_len:]
    return tok.decode(gen, skip_special_tokens=True), len(gen), len(gen) < budget


EXTRACTION_PROMPT = {
    "gsm8k": ("You are a careful math solver. I have done most of the reasoning already. "
              "Your job is ONLY to extract the final numerical answer. "
              "End with: Final answer: <number>."),
    "math500": ("You are an expert mathematician. I have done most of the reasoning already. "
                "Your job is ONLY to extract or compute the final answer and output it in the form "
                "\\boxed{ANSWER}. Do not re-solve or re-explain."),
}

NEUTRAL_PROMPT = "You are a careful math solver."


def extract_with_condition(model, tok, question, thinking_prefix, answer_budget,
                           benchmark, enable_thinking, use_extraction_prompt):
    clean_trace = thinking_prefix
    for tag in ["<think>", "</think>", "<|think|>", "<|/think|>"]:
        clean_trace = clean_trace.replace(tag, "")
    clean_trace = clean_trace.strip()

    if use_extraction_prompt:
        system_text = EXTRACTION_PROMPT[benchmark]
        if benchmark == "math500":
            user_text = question + "\n\nHere is my partial reasoning:\n" + clean_trace + "\n\nThe final answer is \\boxed{"
        else:
            user_text = question + "\n\nHere is my partial reasoning:\n" + clean_trace + "\n\nFinal answer: "
    else:
        system_text = NEUTRAL_PROMPT
        user_text = question + "\n\nHere is my partial reasoning:\n" + clean_trace + "\n\nPlease give the final answer."

    messages = [{"role": "system", "content": system_text},
                {"role": "user", "content": user_text}]
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
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=answer_budget, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][in_len:]
    return tok.decode(gen, skip_special_tokens=True), len(gen)


def parse_answer(text, benchmark):
    if benchmark == "math500":
        pred, _, src = parse_prediction_math(text)
        return pred
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def check_correct(pred, gold, benchmark):
    if benchmark == "math500":
        return is_correct_math(pred, gold)
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-4
    except (ValueError, TypeError):
        return False


CONDITIONS = [
    ("think_neutral",     True,  False),
    ("think_extraction",  True,  True),
    ("nothink_neutral",   False, False),
    ("nothink_extraction", False, True),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--benchmark", default="gsm8k", choices=["math500", "gsm8k"])
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--b1", type=int, default=256)
    p.add_argument("--b2_max", type=int, default=512)
    p.add_argument("--b_answer", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/factorial_ablation")
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    log.info(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    items = load_benchmark(args.benchmark, args.n_samples, args.seed)
    log.info(f"Loaded {len(items)} items")
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results = []
    correct = {name: 0 for name, _, _ in CONDITIONS}
    correct["town"] = 0
    counts = {"stage1": 0, "stage2_nat": 0, "stage2_budget": 0}

    for i, item in enumerate(items):
        r = {"idx": i, "gold": item["gold"]}

        # Stage 1: nothink triage
        s1_text, s1_tok, s1_natural = generate_simple(model, tok, item["q"], args.b1, False)
        if s1_natural:
            pred = parse_answer(s1_text, args.benchmark)
            c = check_correct(pred, item["gold"], args.benchmark)
            counts["stage1"] += 1
            for name, _, _ in CONDITIONS:
                r[name] = c
                if c: correct[name] += 1
            r["town"] = c
            if c: correct["town"] += 1
            r["stage"] = 1
            results.append(r)
            log.info(f"[{i+1}/{len(items)}] S1 pass: {c}")
            continue

        # Stage 2: thinking
        s2_text, s2_tok, s2_natural = generate_simple(model, tok, item["q"], args.b2_max, True)
        if s2_natural:
            pred = parse_answer(s2_text, args.benchmark)
            c = check_correct(pred, item["gold"], args.benchmark)
            counts["stage2_nat"] += 1
            for name, _, _ in CONDITIONS:
                r[name] = c
                if c: correct[name] += 1
            r["town"] = c
            if c: correct["town"] += 1
            r["stage"] = 2
            results.append(r)
            log.info(f"[{i+1}/{len(items)}] S2 natural: {c}")
            continue

        # Stage 2 budget exhausted — run all 4 conditions
        counts["stage2_budget"] += 1
        r["stage"] = 3

        # TOWN baseline
        pred_town = parse_answer(s2_text, args.benchmark)
        c_town = check_correct(pred_town, item["gold"], args.benchmark)
        r["town"] = c_town
        if c_town: correct["town"] += 1

        # All 4 factorial conditions on SAME truncated prefix
        for name, enable_thinking, use_extraction in CONDITIONS:
            text, _ = extract_with_condition(
                model, tok, item["q"], s2_text, args.b_answer,
                args.benchmark, enable_thinking, use_extraction)
            pred = parse_answer(text, args.benchmark)
            c = check_correct(pred, item["gold"], args.benchmark)
            r[name] = c
            if c: correct[name] += 1

        results.append(r)
        log.info(f"[{i+1}/{len(items)}] S3: TN={r['think_neutral']} TE={r['think_extraction']} "
                 f"NN={r['nothink_neutral']} NE={r['nothink_extraction']}")

        if (i + 1) % 25 == 0:
            n = i + 1
            log.info(f"--- Checkpoint {n} ---")
            for name, _, _ in CONDITIONS:
                log.info(f"  {name}: {correct[name]}/{n} = {correct[name]/n*100:.1f}%")

    n = len(results)
    stage3 = [r for r in results if r["stage"] == 3]

    # Compute interaction effect on stage-3 samples
    interaction = {}
    if stage3:
        for name, _, _ in CONDITIONS:
            interaction[name] = sum(1 for r in stage3 if r[name]) / len(stage3) * 100

        # McNemar for all 6 pairs
        mcnemar_pairs = {}
        cond_names = [name for name, _, _ in CONDITIONS]
        for i_c in range(len(cond_names)):
            for j_c in range(i_c + 1, len(cond_names)):
                a, b_name = cond_names[i_c], cond_names[j_c]
                disc = sum(1 for r in stage3 if r[a] != r[b_name])
                a_only = sum(1 for r in stage3 if r[a] and not r[b_name])
                b_only = sum(1 for r in stage3 if not r[a] and r[b_name])
                mcnemar_pairs[f"{a}_vs_{b_name}"] = {
                    "discordant": disc, f"{a}_only": a_only, f"{b_name}_only": b_only}

    summary = {
        "meta": {"model": args.model, "benchmark": args.benchmark,
                 "n": n, "seed": args.seed, "ts": ts,
                 "b1": args.b1, "b2_max": args.b2_max, "b_answer": args.b_answer},
        "stages": counts,
        "accuracy_overall": {name: correct[name]/n for name, _, _ in CONDITIONS},
        "accuracy_overall_town": correct["town"]/n,
        "accuracy_stage3_only": interaction,
        "mcnemar_stage3": mcnemar_pairs if stage3 else {},
        "n_stage3": len(stage3),
        "per_sample": results,
    }

    fname = f"factorial_{args.benchmark}_{ts}.json"
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\n{'='*60}")
    log.info(f"2×2 FACTORIAL ABLATION (n={n}, stage3={len(stage3)})")
    log.info(f"{'='*60}")
    log.info(f"Stages: {counts}")
    log.info(f"TOWN baseline: {correct['town']}/{n} = {correct['town']/n*100:.1f}%")
    log.info(f"")
    log.info(f"{'Condition':<25} {'Overall':>10} {'Stage-3 only':>12}")
    log.info(f"{'-'*25} {'-'*10} {'-'*12}")
    for name, _, _ in CONDITIONS:
        ov = correct[name]/n*100
        s3 = interaction.get(name, 0)
        log.info(f"{name:<25} {ov:>9.1f}% {s3:>11.1f}%")
    log.info(f"")
    log.info(f"INTERACTION (stage-3 only):")
    log.info(f"  Mode effect (nothink - think):")
    log.info(f"    w/ neutral:    {interaction.get('nothink_neutral',0) - interaction.get('think_neutral',0):+.1f}pp")
    log.info(f"    w/ extraction: {interaction.get('nothink_extraction',0) - interaction.get('think_extraction',0):+.1f}pp")
    log.info(f"  Prompt effect (extraction - neutral):")
    log.info(f"    w/ think:      {interaction.get('think_extraction',0) - interaction.get('think_neutral',0):+.1f}pp")
    log.info(f"    w/ nothink:    {interaction.get('nothink_extraction',0) - interaction.get('nothink_neutral',0):+.1f}pp")
    log.info(f"  Interaction: {(interaction.get('nothink_extraction',0) - interaction.get('nothink_neutral',0)) - (interaction.get('think_extraction',0) - interaction.get('think_neutral',0)):+.1f}pp")


if __name__ == "__main__":
    main()

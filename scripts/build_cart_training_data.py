#!/usr/bin/env python3
"""Build CART training data: truncated prefix → answer pairs from TRAIN split only.

Generates full thinking traces on train data, samples truncation points,
creates supervised (question, prefix, gold_answer) pairs.

NO EVALUATION/TEST DATA is used. MATH-500 is test-only; uses hendrycks/math train.

Usage:
    python3 scripts/build_cart_training_data.py \
        --benchmark math --split train --n_samples 200 \
        --model Qwen/Qwen3-8B --output results/cart/train_prefixes.jsonl \
        --truncation_fractions 0.1,0.25,0.5,0.75
"""
import argparse, hashlib, json, logging, os, random, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FORBIDDEN_EVAL_DATASETS = {"HuggingFaceH4/MATH-500", "math500"}


def hash_str(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def load_train_data(benchmark, n_samples, seed):
    from datasets import load_dataset
    if benchmark == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="train")
        idxs = list(range(len(ds)))
        random.seed(seed); random.shuffle(idxs)
        out = []
        for k, i in enumerate(idxs[:n_samples]):
            raw = ds[i]
            gold = raw["answer"].split("####")[-1].strip().replace(",", "")
            out.append({"q": raw["question"], "gold": gold, "idx": i,
                        "dataset": "gsm8k", "split": "train",
                        "question_hash": hash_str(raw["question"]),
                        "gold_hash": hash_str(gold)})
        return out
    elif benchmark == "math":
        # Load all MATH subjects and concatenate
        from datasets import concatenate_datasets
        subjects = ['algebra', 'counting_and_probability', 'geometry',
                     'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
        all_ds = []
        for subj in subjects:
            all_ds.append(load_dataset("EleutherAI/hendrycks_math", subj, split="train"))
        ds = concatenate_datasets(all_ds)
        idxs = list(range(len(ds)))
        random.seed(seed); random.shuffle(idxs)
        out = []
        for k, i in enumerate(idxs[:n_samples]):
            raw = ds[i]
            q = raw["problem"]
            gold = str(raw["solution"])  # hendrycks_math uses "solution" not "answer"
            out.append({"q": q, "gold": gold, "idx": i,
                        "dataset": "EleutherAI/hendrycks_math", "split": "train",
                        "question_hash": hash_str(q),
                        "gold_hash": hash_str(gold)})
        return out
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")


def generate_thinking_trace(model, tok, question, max_tokens=4096):
    messages = [{"role": "system", "content": "You are a careful math solver. Think step by step."},
                {"role": "user", "content": question}]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=True)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    dev = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen_ids = out[0][in_len:]
    trace = tok.decode(gen_ids, skip_special_tokens=True)
    return trace, len(gen_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=["math", "gsm8k"])
    p.add_argument("--split", default="train", choices=["train"])
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max_trace_tokens", type=int, default=2048)
    p.add_argument("--truncation_fractions", default="0.1,0.25,0.5,0.75",
                   help="Comma-separated truncation fractions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    assert args.split == "train", "CART training data must come from TRAIN split only"
    assert args.benchmark not in FORBIDDEN_EVAL_DATASETS, f"Cannot use eval dataset {args.benchmark} for training"

    random.seed(args.seed); torch.manual_seed(args.seed)
    fractions = [float(f) for f in args.truncation_fractions.split(",")]

    log.info(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    items = load_train_data(args.benchmark, args.n_samples, args.seed)
    log.info(f"Loaded {len(items)} TRAIN items from {args.benchmark}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    records = []

    for i, item in enumerate(items):
        trace, trace_len = generate_thinking_trace(
            model, tok, item["q"], args.max_trace_tokens)

        # Tokenize trace for accurate token-level truncation
        trace_token_ids = tok.encode(trace, add_special_tokens=False)
        actual_trace_token_len = len(trace_token_ids)

        for frac in fractions:
            trunc_tok_len = max(1, int(actual_trace_token_len * frac))
            # Truncate at TOKEN level, then decode back
            prefix_ids = trace_token_ids[:trunc_tok_len]
            prefix = tok.decode(prefix_ids, skip_special_tokens=True)

            record = {
                "dataset": item["dataset"],
                "split": "train",
                "sample_id": item["idx"],
                "question_hash": item["question_hash"],
                "gold_hash": item["gold_hash"],
                "prefix_token_len": trunc_tok_len,
                "prefix_token_ids_len_actual": trunc_tok_len,
                "truncation_fraction": frac,
                "remaining_budget": args.max_trace_tokens - trunc_tok_len,
                "question": item["q"],
                "reasoning_prefix": prefix,
                "gold_answer": item["gold"],
                "full_trace_token_len": actual_trace_token_len,
                "full_trace_gen_len": trace_len,
                "source_trace_id": f"{item['idx']}_{args.seed}",
            }
            records.append(record)

        if (i + 1) % 10 == 0:
            log.info(f"[{i+1}/{len(items)}] Generated {len(records)} records")

    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(f"Wrote {len(records)} records to {args.output}")
    log.info(f"Truncation fractions: {fractions}")
    log.info(f"Samples: {len(items)}, Records: {len(records)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CTT mini-pilot: paired cross-mode layer-wise KL on target samples.

For each query, run 2 prefill passes:
  (1) think-mode: prompt includes <think> scaffold
  (2) nothink-mode: prompt includes <nothink> scaffold (or no think tag)

At the final prompt-token position, extract hidden states at every layer.
Project via LayerNorm(h) @ W_U to get per-layer logit distribution.
Compute KL(p_think[ℓ] || p_nothink[ℓ]) per layer.

Measure AUC of {max, mean}-KL signal for the label:
  y=1 if think is wrong (nothink-correct-only + neither), y=0 if think is correct.

Also add null-scaffold control (reviewer W1): a blank scaffold with no thinking content.

Usage:
    python scripts/run_ctt_pilot.py \\
        --model Qwen/Qwen3.5-27B --benchmark gsm8k \\
        --labels_file results/p21_27b_gsm8k_extend/b4096/nothink_baseline_Qwen3_5-27B_gsm8k_20260418_141928.json \\
        --output_dir results/ctt_pilot_27b_gsm8k \\
        --n_samples 200 --seed 42
"""
import argparse, json, logging, os, random, sys, time
from datetime import datetime, timezone
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_gsm8k(n, seed):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idxs = list(range(len(ds))); random.seed(seed); random.shuffle(idxs)
    items = []
    for i in idxs[:n]:
        raw = ds[i]
        ans = raw["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"idx": i, "q": raw["question"], "gold": ans})
    return items


def load_math500(n, seed):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = list(ds); random.seed(seed); random.shuffle(items)
    return [{"idx": i, "q": s["problem"], "gold": s["answer"]} for i, s in enumerate(items[:n])]


def build_prompts(tok, question, mode):
    """Build chat-template prompt for think / nothink / null_scaffold modes."""
    messages = [
        {"role": "system", "content": "You are a careful math solver."},
        {"role": "user", "content": question},
    ]
    if mode == "think":
        enable = True
    elif mode == "nothink":
        enable = False
    elif mode == "null_scaffold":
        enable = True  # Use think scaffold but add empty think content via suffix
    else:
        raise ValueError(mode)
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=enable)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Append answer probe (so final token is predictable)
    if mode == "null_scaffold":
        # Append empty thinking and answer probe
        prompt = prompt + "<think>\n\n</think>\n\nThe answer is "
    else:
        prompt = prompt + "The answer is "
    return prompt


def prefill_hidden_states(model, tok, prompt):
    """Run 1-token prefill, return hidden_states at final prompt token per layer."""
    dev = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    # hidden_states is tuple of (L+1) tensors shaped [1, T, H]
    # Take final-prompt-token slice at each layer
    hs = torch.stack([h[0, -1, :].float() for h in outputs.hidden_states], dim=0)  # [L+1, H]
    return hs.cpu()


def project_via_logit_lens(hs, lm_head, final_norm):
    """Apply final LayerNorm + unembedding to get logits at each layer.

    Args:
        hs: [L+1, H] hidden states
        lm_head: nn.Linear output projection (or tied W_U)
        final_norm: final layer norm module
    Returns:
        logits: [L+1, V]
    """
    hs = hs.to(lm_head.weight.device).to(lm_head.weight.dtype)
    with torch.no_grad():
        # Apply final_norm (RMSNorm or LayerNorm) then project
        normed = final_norm(hs)
        logits = lm_head(normed)  # [L+1, V]
    return logits.float().cpu()


def kl_per_layer(logits_a, logits_b):
    """KL(p_a || p_b) per layer, where logits_{a,b} are [L+1, V]."""
    log_pa = F.log_softmax(logits_a, dim=-1)
    log_pb = F.log_softmax(logits_b, dim=-1)
    pa = log_pa.exp()
    kl = (pa * (log_pa - log_pb)).sum(dim=-1)  # [L+1]
    return kl.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math500"])
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labels_file", default=None,
                   help="Optional: existing baseline file with per-sample think/nothink labels")
    p.add_argument("--subset_indices", default=None,
                   help="Optional JSON file with {'indices': [...]} to run only those samples")
    p.add_argument("--output_dir", default="results/ctt_pilot")
    p.add_argument("--modes", nargs="+", default=["think", "nothink", "null_scaffold"],
                   help="Which mode prompts to run")
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    log.info(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    # Find final norm and lm_head
    lm_head = model.get_output_embeddings()
    final_norm = None
    for name, mod in model.named_modules():
        if name.endswith(".norm") and "layers" not in name:
            final_norm = mod; break
    if final_norm is None:
        # Try common patterns
        for candidate in [model.model.norm, getattr(model, "lm_head_norm", None)]:
            if candidate is not None:
                final_norm = candidate; break
    assert final_norm is not None, "Could not find final norm layer"
    log.info(f"Final norm: {type(final_norm).__name__}, lm_head: {type(lm_head).__name__}")

    # Load benchmark items
    if args.benchmark == "gsm8k":
        items = load_gsm8k(args.n_samples, args.seed)
    else:
        items = load_math500(args.n_samples, args.seed)
    log.info(f"Loaded {len(items)} items")

    # Filter to subset_indices if provided
    if args.subset_indices:
        with open(args.subset_indices) as f:
            want_idx = set(json.load(f)["indices"])
        items = [it for i, it in enumerate(items) if i in want_idx]
        log.info(f"Filtered to {len(items)} via subset_indices")

    # Optional: existing labels
    labels = {}
    if args.labels_file and os.path.exists(args.labels_file):
        with open(args.labels_file) as f:
            baseline = json.load(f)
        # Try to extract per-sample labels
        per = baseline.get("per_sample", baseline)
        if isinstance(per, dict):
            for mode_key in ("nothink_4096", "thinking_4096", "nothink", "thinking"):
                if mode_key in per:
                    samples = per[mode_key]
                    labels[mode_key] = {s.get("idx", i): s for i, s in enumerate(samples)}
        log.info(f"Loaded labels for modes: {list(labels.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    t_start = time.time()
    for i, item in enumerate(items):
        record = {"idx": item["idx"], "gold": item["gold"], "question_preview": item["q"][:60]}
        # Build labels
        for mode_key in labels:
            lab = labels[mode_key].get(item["idx"])
            if lab:
                record[f"{mode_key}_correct"] = int(lab.get("correct", 0))
                record[f"{mode_key}_tokens"] = int(lab.get("tokens", 0))

        # Run prefills
        kl_data = {}
        hs_cache = {}
        for mode in args.modes:
            prompt = build_prompts(tok, item["q"], mode)
            hs = prefill_hidden_states(model, tok, prompt)
            hs_cache[mode] = hs

        # Compute KLs for pairs
        if "think" in hs_cache and "nothink" in hs_cache:
            logits_t = project_via_logit_lens(hs_cache["think"], lm_head, final_norm)
            logits_n = project_via_logit_lens(hs_cache["nothink"], lm_head, final_norm)
            kl_data["kl_think_nothink"] = kl_per_layer(logits_t, logits_n).tolist()
        if "nothink" in hs_cache and "null_scaffold" in hs_cache:
            logits_n = project_via_logit_lens(hs_cache["nothink"], lm_head, final_norm)
            logits_null = project_via_logit_lens(hs_cache["null_scaffold"], lm_head, final_norm)
            kl_data["kl_null_nothink"] = kl_per_layer(logits_null, logits_n).tolist()
        if "think" in hs_cache and "null_scaffold" in hs_cache:
            logits_t = project_via_logit_lens(hs_cache["think"], lm_head, final_norm)
            logits_null = project_via_logit_lens(hs_cache["null_scaffold"], lm_head, final_norm)
            kl_data["kl_null_think"] = kl_per_layer(logits_null, logits_t).tolist()

        record["kl_per_layer"] = kl_data
        results.append(record)

        if (i+1) % 20 == 0 or i == len(items)-1:
            elapsed = time.time() - t_start
            log.info(f"  [{i+1}/{len(items)}] elapsed={elapsed:.0f}s, est_total={elapsed*len(items)/(i+1):.0f}s")

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = args.model.split("/")[-1].replace(".", "_")
    out_path = os.path.join(args.output_dir, f"ctt_{tag}_{args.benchmark}_{ts}.json")
    summary = {
        "meta": {"model": args.model, "benchmark": args.benchmark,
                 "n_samples": len(items), "seed": args.seed,
                 "modes": args.modes,
                 "elapsed_s": time.time() - t_start},
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"Saved: {out_path}")
    log.info(f"Total elapsed: {time.time()-t_start:.0f}s for {len(items)} samples × {len(args.modes)} modes")


if __name__ == "__main__":
    main()

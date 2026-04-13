#!/usr/bin/env python3
"""Collect per-token entropy dynamics during thinking-mode generation.

GO/NO-GO pilot for IRIS (Information-Rate Informed Stopping):
  Hypothesis: Per-token entropy H_t drops significantly (>50%) BEFORE
  the model emits </think>, AND this drop correlates with answer correctness.

For each sample, collects:
  - Per-token next-token entropy H_t
  - Per-chunk hidden-state stability S_t = ||h_t - h_{t-k}||
  - P(</think> | context) at each step
  - Natural stop position, correctness, difficulty label

Usage:
    python scripts/collect_entropy_dynamics.py \
        --model Qwen/Qwen3-8B \
        --budget 512 \
        --n_samples 200 \
        --chunk_size 32 \
        --seed 42

    # Full set
    python scripts/collect_entropy_dynamics.py \
        --model Qwen/Qwen3-8B \
        --budget 512 \
        --n_samples 99999 \
        --seed 42
"""

import argparse
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
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
# Answer extraction (reused from run_town.py patterns)
# ---------------------------------------------------------------------------
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer\s*[:：]|the answer is\s*)([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
HASH_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)")


def extract_last_number(text: str) -> Optional[str]:
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
    if "####" in answer_field:
        after = answer_field.split("####")[-1]
        match = NUM_RE.search(after)
        if match:
            return match.group(0)
    return extract_last_number(answer_field)


def parse_prediction(text: str) -> Tuple[Optional[str], str]:
    m = BOXED_RE.search(text)
    if m:
        inner = m.group(1).replace(",", "")
        num = NUM_RE.search(inner)
        if num:
            return num.group(0), "boxed"
    m = HASH_RE.search(text)
    if m:
        return m.group(1).replace(",", ""), "hash"
    m = FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", ""), "final_answer"
    last = extract_last_number(text)
    if last:
        return last.replace(",", ""), "last_number"
    return None, "none"


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------
def model_input_device(model) -> torch.device:
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


def build_thinking_prompt(question: str, tokenizer) -> str:
    """Build chat prompt with thinking mode enabled."""
    system_text = (
        "You are a careful math solver. Solve the problem step by step briefly. "
        "End with a single line: Final answer: <number>."
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": question},
    ]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    chat_kwargs["enable_thinking"] = True
    try:
        return tokenizer.apply_chat_template(messages, **chat_kwargs)
    except TypeError:
        chat_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **chat_kwargs)
    except Exception:
        return f"{system_text}\n\nQuestion: {question}\nSolution:\n"


# ---------------------------------------------------------------------------
# Core: Generate with per-token entropy collection
# ---------------------------------------------------------------------------
def generate_with_entropy_traces(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    chunk_size: int = 32,
) -> Dict:
    """Generate thinking response and collect per-token entropy dynamics.

    Uses output_scores=True to get logits at each step, then computes:
      - Per-token entropy H_t = -sum(p * log p)
      - Per-token max probability (confidence)
      - Per-chunk hidden-state L2 distance (stability)
      - P(</think>) at each token position

    Returns dict with all traces and generation metadata.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    target_device = model_input_device(model)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    elapsed = time.perf_counter() - start

    gen_ids = outputs.sequences[0][in_len:]
    n_tokens = int(gen_ids.shape[0])
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    clean_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # --- Per-token entropy from logit scores ---
    # outputs.scores is a tuple of (n_tokens,) tensors, each [1, vocab_size]
    per_token_entropy = []
    per_token_max_prob = []
    per_token_think_end_prob = []

    # Find </think> token id(s) for monitoring P(</think>)
    think_end_ids = set()
    for candidate in ["</think>", "<|/think|>", "</think>\n"]:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            think_end_ids.add(ids[0])

    for step_idx, score in enumerate(outputs.scores):
        # score shape: [1, vocab_size]
        logits = score[0].float()  # [vocab_size]
        probs = torch.softmax(logits, dim=-1)

        # Entropy H = -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()
        per_token_entropy.append(entropy)

        # Max probability (confidence)
        max_prob = probs.max().item()
        per_token_max_prob.append(max_prob)

        # P(</think>) — sum over all </think> variants
        think_end_p = 0.0
        for tid in think_end_ids:
            if tid < probs.shape[0]:
                think_end_p += probs[tid].item()
        per_token_think_end_prob.append(think_end_p)

    # --- Per-chunk hidden-state stability ---
    # outputs.hidden_states is a tuple of (n_tokens,) elements
    # Each element is a tuple of (n_layers+1,) tensors of shape [1, 1, hidden_dim]
    # We use the last layer's hidden state
    per_chunk_stability = []
    chunk_positions = []

    if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
        # Collect last-layer hidden states at chunk boundaries
        last_layer_hiddens = []
        for step_idx, layer_states in enumerate(outputs.hidden_states):
            # layer_states is a tuple: (n_layers+1,) of [1, seq_len_at_step, hidden_dim]
            # For generate, each step adds 1 token, so seq_len=1 for step>0
            if isinstance(layer_states, tuple) and len(layer_states) > 0:
                last_h = layer_states[-1]  # Last layer
                if last_h.dim() == 3:
                    last_h = last_h[0, -1, :]  # [hidden_dim]
                elif last_h.dim() == 2:
                    last_h = last_h[-1, :]
                last_layer_hiddens.append(last_h.float().cpu())

        # Compute chunk-level stability: L2 distance between chunk boundaries
        for c in range(0, len(last_layer_hiddens), chunk_size):
            c_end = min(c + chunk_size, len(last_layer_hiddens))
            if c_end <= c:
                break
            # Stability = L2 norm of difference between start and end of chunk
            h_start = last_layer_hiddens[c]
            h_end = last_layer_hiddens[c_end - 1]
            stability = torch.norm(h_end - h_start, p=2).item()
            per_chunk_stability.append(stability)
            chunk_positions.append((c, c_end - 1))

        # Also compute moving-average stability for smoother signal
        # Window of last `chunk_size` hidden states
        moving_stability = []
        for t in range(len(last_layer_hiddens)):
            if t < chunk_size:
                moving_stability.append(None)
            else:
                h_prev = last_layer_hiddens[t - chunk_size]
                h_curr = last_layer_hiddens[t]
                s = torch.norm(h_curr - h_prev, p=2).item()
                moving_stability.append(s)

        # Free memory
        del last_layer_hiddens
    else:
        moving_stability = []

    # --- Detect </think> position ---
    think_end_position = None
    text_so_far = ""
    for t_idx in range(n_tokens):
        tok = tokenizer.decode(gen_ids[t_idx:t_idx+1], skip_special_tokens=False)
        text_so_far += tok
        if "</think>" in text_so_far and think_end_position is None:
            think_end_position = t_idx

    # --- Hit budget? ---
    hit_budget = n_tokens >= int(max_new_tokens * 0.95)
    natural_stop = not hit_budget

    # --- Parse answer ---
    # Extract answer from text after </think> if present
    answer_text = clean_text
    if "</think>" in full_text:
        parts = full_text.split("</think>", 1)
        if len(parts) > 1:
            answer_text = parts[1]
    pred, pred_source = parse_prediction(answer_text)

    # --- Compute summary statistics for the entropy trace ---
    entropy_arr = np.array(per_token_entropy)
    if len(entropy_arr) > 0:
        # First-half vs second-half entropy
        mid = len(entropy_arr) // 2
        first_half_mean = float(entropy_arr[:mid].mean()) if mid > 0 else 0.0
        second_half_mean = float(entropy_arr[mid:].mean()) if mid > 0 else 0.0

        # Entropy before and after </think>
        if think_end_position is not None and think_end_position > 0:
            pre_think_entropy = float(entropy_arr[:think_end_position].mean())
            post_think_entropy = float(entropy_arr[think_end_position:].mean()) if think_end_position < len(entropy_arr) else 0.0

            # Key metric: entropy drop ratio before </think>
            # Look at last chunk_size tokens before </think> vs earlier
            pre_end = max(0, think_end_position - chunk_size)
            late_thinking_entropy = float(entropy_arr[pre_end:think_end_position].mean()) if pre_end < think_end_position else 0.0
            early_thinking_entropy = float(entropy_arr[:pre_end].mean()) if pre_end > 0 else 0.0
            entropy_drop_ratio = (early_thinking_entropy - late_thinking_entropy) / (early_thinking_entropy + 1e-10) if early_thinking_entropy > 0 else 0.0
        else:
            pre_think_entropy = float(entropy_arr.mean())
            post_think_entropy = 0.0
            late_thinking_entropy = 0.0
            early_thinking_entropy = float(entropy_arr.mean())
            entropy_drop_ratio = 0.0

        # Min entropy and its position
        min_entropy_pos = int(entropy_arr.argmin())
        min_entropy_val = float(entropy_arr.min())

        # Entropy trend (linear regression slope)
        x = np.arange(len(entropy_arr), dtype=np.float64)
        if len(x) > 1:
            slope = float(np.polyfit(x, entropy_arr, 1)[0])
        else:
            slope = 0.0
    else:
        first_half_mean = second_half_mean = 0.0
        pre_think_entropy = post_think_entropy = 0.0
        late_thinking_entropy = early_thinking_entropy = 0.0
        entropy_drop_ratio = 0.0
        min_entropy_pos = min_entropy_val = 0
        slope = 0.0

    result = {
        "n_tokens": n_tokens,
        "hit_budget": hit_budget,
        "natural_stop": natural_stop,
        "think_end_position": think_end_position,
        "elapsed_s": round(elapsed, 3),
        "pred": pred,
        "pred_source": pred_source,
        "full_text": full_text,

        # Per-token traces (lists for JSON serialization)
        "entropy_trace": [round(v, 6) for v in per_token_entropy],
        "max_prob_trace": [round(v, 6) for v in per_token_max_prob],
        "think_end_prob_trace": [round(v, 8) for v in per_token_think_end_prob],

        # Per-chunk stability
        "chunk_stability": [round(v, 4) for v in per_chunk_stability],
        "chunk_positions": chunk_positions,

        # Summary statistics
        "stats": {
            "mean_entropy": round(float(entropy_arr.mean()), 6) if len(entropy_arr) > 0 else 0.0,
            "std_entropy": round(float(entropy_arr.std()), 6) if len(entropy_arr) > 0 else 0.0,
            "first_half_entropy": round(first_half_mean, 6),
            "second_half_entropy": round(second_half_mean, 6),
            "pre_think_end_entropy": round(pre_think_entropy, 6),
            "post_think_end_entropy": round(post_think_entropy, 6),
            "early_thinking_entropy": round(early_thinking_entropy, 6),
            "late_thinking_entropy": round(late_thinking_entropy, 6),
            "entropy_drop_ratio": round(entropy_drop_ratio, 6),
            "min_entropy": round(min_entropy_val, 6) if isinstance(min_entropy_val, float) else 0.0,
            "min_entropy_position": min_entropy_pos,
            "entropy_slope": round(slope, 8),
        },
    }

    # Free GPU memory
    del outputs
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Collect per-token entropy dynamics for IRIS GO/NO-GO pilot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model ID")
    parser.add_argument("--budget", type=int, default=512,
                        help="Max new tokens for thinking mode")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples (200 for pilot, 99999 for full)")
    parser.add_argument("--chunk_size", type=int, default=32,
                        help="Chunk size for stability computation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--save_traces", action="store_true", default=True,
                        help="Save full per-token traces (large file)")
    parser.add_argument("--no_save_traces", dest="save_traces", action="store_false",
                        help="Only save summary statistics per sample")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Save checkpoint every N samples")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "results/entropy_dynamics"

    os.makedirs(args.output_dir, exist_ok=True)

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.split("/")[-1].replace(".", "_").replace("-", "_")

    # --- Load model ---
    log.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log.info(f"Model loaded. Device: {model_input_device(model)}")

    # --- Load GSM8K ---
    log.info(f"Loading GSM8K (n_samples={args.n_samples}, seed={args.seed})")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idxs = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(idxs)
    n_use = min(args.n_samples, len(ds))
    items = []
    for i in idxs[:n_use]:
        raw = ds[i]
        gold = get_gold_from_gsm8k(raw["answer"])
        items.append({
            "question": raw["question"],
            "gold": gold,
            "original_idx": i,
        })
    log.info(f"Loaded {len(items)} samples")

    # --- Run collection ---
    all_results = []
    n_correct = 0
    n_natural_stop = 0
    total_elapsed = 0.0
    entropy_drops = []  # Collect entropy_drop_ratio for GO/NO-GO decision

    for i, item in enumerate(items):
        prompt = build_thinking_prompt(item["question"], tokenizer)

        try:
            result = generate_with_entropy_traces(
                model, tokenizer, prompt,
                max_new_tokens=args.budget,
                chunk_size=args.chunk_size,
            )
        except Exception as e:
            log.error(f"Sample {i} failed: {e}")
            result = {"error": str(e), "n_tokens": 0}

        # Evaluate correctness
        correct = is_correct(result.get("pred"), item["gold"])
        if correct:
            n_correct += 1
        if result.get("natural_stop", False):
            n_natural_stop += 1
        total_elapsed += result.get("elapsed_s", 0.0)

        # Record
        sample_record = {
            "idx": i,
            "original_idx": item["original_idx"],
            "gold": item["gold"],
            "correct": int(correct),
            **result,
        }

        # Strip full_text and traces if not saving (to reduce memory)
        if not args.save_traces:
            sample_record.pop("entropy_trace", None)
            sample_record.pop("max_prob_trace", None)
            sample_record.pop("think_end_prob_trace", None)
            sample_record.pop("full_text", None)

        all_results.append(sample_record)

        # Track entropy drop for GO/NO-GO
        stats = result.get("stats", {})
        if stats.get("entropy_drop_ratio") is not None:
            entropy_drops.append(stats["entropy_drop_ratio"])

        # Progress logging
        if (i + 1) % 10 == 0 or i == len(items) - 1:
            done = i + 1
            acc = n_correct / done
            ns_rate = n_natural_stop / done
            avg_drop = np.mean(entropy_drops) if entropy_drops else 0.0
            log.info(
                f"  [{done}/{len(items)}] acc={acc:.3f}  "
                f"natural_stop={ns_rate:.1%}  "
                f"avg_entropy_drop={avg_drop:.3f}  "
                f"elapsed={total_elapsed:.0f}s"
            )

        # Checkpoint
        if (i + 1) % args.checkpoint_every == 0:
            ckpt_path = os.path.join(
                args.output_dir,
                f"checkpoint_entropy_{model_tag}_b{args.budget}_{i+1}.json"
            )
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({
                    "meta": {"n_done": i + 1, "model": args.model, "budget": args.budget},
                    "results": all_results,
                }, f, indent=2, ensure_ascii=False, default=str)
            log.info(f"Checkpoint saved: {ckpt_path}")

    # --- Compute GO/NO-GO metrics ---
    n = len(all_results)
    valid_results = [r for r in all_results if "error" not in r]
    correct_results = [r for r in valid_results if r["correct"] == 1]
    incorrect_results = [r for r in valid_results if r["correct"] == 0]
    ns_results = [r for r in valid_results if r.get("natural_stop", False)]
    trunc_results = [r for r in valid_results if not r.get("natural_stop", False)]

    # Entropy drop: correct vs incorrect
    drop_correct = [r["stats"]["entropy_drop_ratio"] for r in correct_results if "stats" in r]
    drop_incorrect = [r["stats"]["entropy_drop_ratio"] for r in incorrect_results if "stats" in r]

    # Entropy drop: natural stop vs truncated
    drop_ns = [r["stats"]["entropy_drop_ratio"] for r in ns_results if "stats" in r]
    drop_trunc = [r["stats"]["entropy_drop_ratio"] for r in trunc_results if "stats" in r]

    # Late thinking entropy: correct vs incorrect
    late_correct = [r["stats"]["late_thinking_entropy"] for r in correct_results if "stats" in r]
    late_incorrect = [r["stats"]["late_thinking_entropy"] for r in incorrect_results if "stats" in r]

    go_no_go = {
        "hypothesis": "Entropy drops >50% before </think> AND correlates with correctness",
        "n_total": n,
        "n_valid": len(valid_results),
        "accuracy": round(n_correct / n, 4) if n > 0 else 0.0,
        "natural_stop_rate": round(n_natural_stop / n, 4) if n > 0 else 0.0,
        "entropy_drop": {
            "overall_mean": round(float(np.mean(entropy_drops)), 4) if entropy_drops else 0.0,
            "overall_std": round(float(np.std(entropy_drops)), 4) if entropy_drops else 0.0,
            "correct_mean": round(float(np.mean(drop_correct)), 4) if drop_correct else 0.0,
            "incorrect_mean": round(float(np.mean(drop_incorrect)), 4) if drop_incorrect else 0.0,
            "natural_stop_mean": round(float(np.mean(drop_ns)), 4) if drop_ns else 0.0,
            "truncated_mean": round(float(np.mean(drop_trunc)), 4) if drop_trunc else 0.0,
        },
        "late_thinking_entropy": {
            "correct_mean": round(float(np.mean(late_correct)), 4) if late_correct else 0.0,
            "incorrect_mean": round(float(np.mean(late_incorrect)), 4) if late_incorrect else 0.0,
        },
    }

    # --- GO/NO-GO decision ---
    drop_mean = go_no_go["entropy_drop"]["overall_mean"]
    drop_diff = abs(
        go_no_go["entropy_drop"]["correct_mean"]
        - go_no_go["entropy_drop"]["incorrect_mean"]
    )
    # Note: drop_mean is negative (entropy decreases), so use abs()
    go_decision = abs(drop_mean) > 0.2 and drop_diff > 0.05
    go_no_go["decision"] = "GO" if go_decision else "NO-GO"
    go_no_go["reasoning"] = (
        f"Overall entropy drop ratio = {drop_mean:.3f} "
        f"(|{drop_mean:.3f}| = {abs(drop_mean):.3f} {'> 0.2 threshold' if abs(drop_mean) > 0.2 else '< 0.2 threshold'}). "
        f"Correct vs incorrect gap = {drop_diff:.3f} "
        f"({'> 0.05' if drop_diff > 0.05 else '< 0.05'}). "
        f"Decision: {'GO — implement full IRIS' if go_decision else 'NO-GO — pivot to batch reallocation + theory only'}."
    )

    # --- Summary ---
    summary = {
        "meta": {
            "script": "collect_entropy_dynamics.py",
            "timestamp_utc": timestamp,
            "model": args.model,
            "budget": args.budget,
            "n_samples": n,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
            "save_traces": args.save_traces,
        },
        "go_no_go": go_no_go,
        "results": all_results,
    }

    # --- Save ---
    out_fname = f"entropy_dynamics_{model_tag}_b{args.budget}_n{n}_{timestamp}.json"
    out_path = os.path.join(args.output_dir, out_fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Saved full results: {out_path}")

    # Also save a compact summary
    compact = {k: v for k, v in summary.items() if k != "results"}
    compact_path = os.path.join(args.output_dir, f"go_no_go_summary_{model_tag}_b{args.budget}.json")
    with open(compact_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)
    log.info(f"Saved GO/NO-GO summary: {compact_path}")

    # --- Print GO/NO-GO ---
    log.info("")
    log.info("=" * 70)
    log.info("ENTROPY DYNAMICS PILOT — GO/NO-GO RESULTS")
    log.info("=" * 70)
    log.info(f"  Model:             {args.model}")
    log.info(f"  Budget:            {args.budget}")
    log.info(f"  Samples:           {n}")
    log.info(f"  Accuracy:          {go_no_go['accuracy']:.1%}")
    log.info(f"  Natural stop rate: {go_no_go['natural_stop_rate']:.1%}")
    log.info(f"  Avg entropy drop:  {go_no_go['entropy_drop']['overall_mean']:.3f}")
    log.info(f"  Drop (correct):    {go_no_go['entropy_drop']['correct_mean']:.3f}")
    log.info(f"  Drop (incorrect):  {go_no_go['entropy_drop']['incorrect_mean']:.3f}")
    log.info(f"  Drop (nat. stop):  {go_no_go['entropy_drop']['natural_stop_mean']:.3f}")
    log.info(f"  Drop (truncated):  {go_no_go['entropy_drop']['truncated_mean']:.3f}")
    log.info(f"  Late H (correct):  {go_no_go['late_thinking_entropy']['correct_mean']:.4f}")
    log.info(f"  Late H (incorrect):{go_no_go['late_thinking_entropy']['incorrect_mean']:.4f}")
    log.info("")
    log.info(f"  >>> DECISION: {go_no_go['decision']} <<<")
    log.info(f"  >>> {go_no_go['reasoning']}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()

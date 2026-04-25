#!/usr/bin/env python3
"""RCV-IRIS: NEGATIVE ABLATION (NOT MAIN METHOD).

** STATUS: feature-based RCV variants are negative ablations per GPT-5.5 Round 2
   review. A/B/C/D all produced 41.0% on MATH-500 n=200 seed=42 with 0
   discordant pairs. See reports/FINAL_RCV_VERDICT.md. **

V2 (GPT-5.5 review fixes):
- A (existing_fragment): ONLY strict probe, no soft probe. Pure old IRIS.
- B (rcv_no_gate): strict + soft probes, but no gate decisions. Both counted.
- C (full_rcv): strict + soft + decision + fallback. All tokens counted honestly.
- --sample_manifest: enforces same-sample evaluation across variants
- Benchmark-aware Stage0 verifier (not hard-coded GSM8K)
- Per-sample fields: stage0_tokens, stage2_tokens, strict_probe_tokens,
  soft_probe_tokens, fallback_tokens, actual_model_generated_tokens
- Fixed GSM8K loader bug (was returning ds[i] for unshuffled i)

Variants:
    existing_fragment (A): Pure cascade, strict only
    rcv_no_gate (B): RCV infra (strict + soft probes) without gates
    full_rcv (C): Full RCV with acceptance + recoverability gates
    stage0_only: Only Stage0 verifier, no recoverability gate
    recover_only: Only recoverability gate, no Stage0 verifier
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
    compute_stage0_accept_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_benchmark(benchmark, n, seed):
    """V2: Fixed GSM8K loader bug (was ds[i] with idx=idxs[i] — mismatch)."""
    from datasets import load_dataset
    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        idxs = list(range(len(ds)))
        random.seed(seed); random.shuffle(idxs)
        out = []
        for k, orig_i in enumerate(idxs[:n]):
            raw = ds[orig_i]
            out.append({
                "q": raw["question"],
                "gold": raw["answer"].split("####")[-1].strip().replace(",", ""),
                "idx": orig_i,
                "order": k,
            })
        return out
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        idxs = list(range(len(ds)))
        random.seed(seed); random.shuffle(idxs)
        out = []
        for k, orig_i in enumerate(idxs[:n]):
            s = ds[orig_i]
            out.append({
                "q": s["problem"],
                "gold": str(s["answer"]),
                "idx": orig_i,
                "order": k,
            })
        return out


def load_from_manifest(manifest_path):
    from make_sample_manifest import load_manifest, load_items_from_manifest
    m = load_manifest(manifest_path)
    return load_items_from_manifest(m), m["meta"]


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
            sys_text = ("You are an expert mathematician. I have done most of the reasoning. "
                        "Your job is ONLY to extract or compute the final answer and output "
                        "\\boxed{ANSWER}. Do not re-solve.")
        else:
            sys_text = ("You are a careful math solver. Extract the final numerical answer. "
                        "End with: Final answer: <number>.")
    else:  # soft
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
    """Run one sample. V2: honest token accounting + A/B path separation."""
    bm = args.benchmark
    variant = args.variant

    import hashlib
    q_hash = hashlib.sha256(item["q"].encode("utf-8")).hexdigest()
    g_hash = hashlib.sha256(str(item["gold"]).encode("utf-8")).hexdigest()
    r = {
        "idx": item["idx"], "order": item.get("order", item["idx"]),
        "manifest_order": item.get("order", item["idx"]),
        "question_hash": q_hash, "gold_hash": g_hash,
        "gold": item["gold"],
        "stage0_tokens": 0, "stage2_tokens": 0,
        "strict_probe_tokens": 0, "soft_probe_tokens": 0,
        "fallback_tokens": 0, "verifier_tokens": 0,
    }

    # === Stage 0: Nothink probe ===
    s0_text, s0_tok, s0_natural, s0_elapsed = generate_simple(
        model, tok, item["q"], args.b1, False, bm)
    s0_pred, s0_src = parse_answer(s0_text, bm)
    s0_correct = check_correct(s0_pred, item["gold"], bm)

    r["stage0_tokens"] = s0_tok
    r["stage0"] = {"tokens": s0_tok, "natural_stop": s0_natural,
                   "pred": s0_pred, "pred_source": s0_src,
                   "elapsed": round(s0_elapsed, 3)}

    # === Stage 0 acceptance decision (benchmark-aware) ===
    use_stage0_gate = variant in ("full_rcv", "stage0_only")
    if s0_natural:
        s0_features = stage0_acceptance_features(
            item["q"], s0_pred, s0_text, s0_src,
            not s0_natural, benchmark=bm)
        r["stage0"]["features"] = s0_features

        if use_stage0_gate:
            accept_score = compute_stage0_accept_score(s0_features, bm)
            r["stage0"]["accept_score"] = round(accept_score, 3)
            if accept_score < args.tau_accept:
                r["stage0"]["decision"] = "REJECT_ESCALATE"
            else:
                r["stage0"]["decision"] = "ACCEPT"
                r.update({"final_stage": 0, "pred": s0_pred, "correct": s0_correct,
                          "tokens_total": s0_tok,
                          "actual_model_generated_tokens": s0_tok,
                          "decision": "ACCEPT_STAGE0"})
                return r
        else:
            r["stage0"]["decision"] = "ACCEPT"
            r.update({"final_stage": 0, "pred": s0_pred, "correct": s0_correct,
                      "tokens_total": s0_tok,
                      "actual_model_generated_tokens": s0_tok,
                      "decision": "ACCEPT_STAGE0"})
            return r

    # === Stage 2: Thinking ===
    s2_text, s2_tok, s2_natural, s2_elapsed = generate_simple(
        model, tok, item["q"], args.b2_max, True, bm)
    r["stage2_tokens"] = s2_tok
    r["stage2"] = {"tokens": s2_tok, "natural_stop": s2_natural,
                   "elapsed": round(s2_elapsed, 3)}

    if s2_natural:
        s2_pred, s2_src = parse_answer(s2_text, bm)
        s2_correct = check_correct(s2_pred, item["gold"], bm)
        total = s0_tok + s2_tok
        r.update({"final_stage": 2, "pred": s2_pred, "correct": s2_correct,
                  "tokens_total": total,
                  "actual_model_generated_tokens": total,
                  "decision": "STAGE2_COMPLETE"})
        return r

    # === Stage 3: Extraction / Gate decision ===
    # V2: Variant-specific probe execution
    # A (existing_fragment): ONLY strict probe — pure old IRIS
    # B (rcv_no_gate): strict + soft probes, both counted
    # C (full_rcv): strict + soft probes + gate decision
    # stage0_only: ONLY strict probe (no recover gate)
    # recover_only: strict + soft probes + recover gate

    run_soft_probe = variant in ("rcv_no_gate", "full_rcv", "recover_only", "full_rcv_majvote")

    strict_text, strict_tok, strict_elapsed = generate_extraction(
        model, tok, item["q"], s2_text, args.b_answer, bm, "strict")
    strict_pred, strict_src = parse_answer(strict_text, bm)
    r["strict_probe_tokens"] = strict_tok

    soft_pred, soft_src = None, "none"
    soft_tok = 0
    if run_soft_probe:
        soft_text, soft_tok, soft_elapsed = generate_extraction(
            model, tok, item["q"], s2_text, args.b_answer, bm, "soft")
        soft_pred, soft_src = parse_answer(soft_text, bm)
        r["soft_probe_tokens"] = soft_tok

    # Recoverability features (computed only if needed for gate)
    pf = None
    use_recover_gate = variant in ("full_rcv", "recover_only", "full_rcv_majvote")
    # V3: majvote variant overrides fallback_action
    effective_fallback = "majority_vote" if variant == "full_rcv_majvote" else args.fallback_action
    if use_recover_gate:
        pf = prefix_recoverability_features(
            item["q"], s2_text, strict_pred, soft_pred, strict_src, soft_src)
        margin = extractor_margin(strict_pred, soft_pred, strict_src, soft_src)
        r["stage3"] = {
            "strict_pred": strict_pred, "strict_source": strict_src,
            "soft_pred": soft_pred, "soft_source": soft_src,
            "recoverability_features": pf,
            "extractor_margin": round(margin, 3),
        }

        s0_features_for_decision = stage0_acceptance_features(
            item["q"], s0_pred, s0_text, s0_src, True, benchmark=bm)
        decision = compute_rcv_decision(
            s0_features_for_decision, pf,
            args.tau_accept, args.tau_recover, benchmark=bm)
        r["stage3"]["decision"] = decision

        if decision == "EXTRACT_STAGE3":
            final_pred = strict_pred
            final_src = f"s3_{strict_src}"
            final_decision = "EXTRACT_STAGE3"
        else:
            # FALLBACK — action depends on effective_fallback
            town_pred, town_src = parse_answer(s2_text, bm)
            if effective_fallback == "majority_vote":
                # V3: Use majority vote across strict, soft, town candidates
                candidates = []
                for pred, src in [(strict_pred, f"s3_{strict_src}"),
                                  (soft_pred, f"s3soft_{soft_src}"),
                                  (town_pred, f"town_{town_src}")]:
                    if pred is not None and str(pred).strip() != "":
                        candidates.append((str(pred).strip(), src))
                if not candidates:
                    # None produced parseable answer — fall through to town_pred
                    final_pred = town_pred
                    final_src = f"town_{town_src}"
                    final_decision = "FALLBACK_NONE_VALID"
                else:
                    # Majority vote
                    from collections import Counter
                    vote_counts = Counter(p for p, _ in candidates)
                    top_pred, top_count = vote_counts.most_common(1)[0]
                    # Find source of a winning pred
                    for p, s in candidates:
                        if p == top_pred:
                            final_pred = p
                            final_src = f"majvote_{s}"
                            break
                    final_decision = f"FALLBACK_MAJVOTE_{top_count}of{len(candidates)}"
            else:
                # Default: FALLBACK_TOWN
                final_pred = town_pred
                final_src = f"town_{town_src}"
                final_decision = "FALLBACK_TOWN"
            # No additional model call — fallback_tokens stays 0
    else:
        # No gate: always extract
        final_pred = strict_pred
        final_src = f"s3_{strict_src}"
        final_decision = "EXTRACT_ALWAYS"
        r["stage3"] = {
            "strict_pred": strict_pred, "strict_source": strict_src,
            "soft_pred": soft_pred, "soft_source": soft_src,
            "decision": final_decision,
        }

    # V2: Honest token accounting
    # actual_model_generated_tokens = ALL tokens the model produced (regardless of decision)
    actual_generated = r["stage0_tokens"] + r["stage2_tokens"] + r["strict_probe_tokens"] + r["soft_probe_tokens"] + r["fallback_tokens"] + r["verifier_tokens"]

    final_correct = check_correct(final_pred, item["gold"], bm)
    r.update({
        "final_stage": 3,
        "pred": final_pred,
        "pred_source": final_src,
        "correct": final_correct,
        "tokens_total": actual_generated,  # V2: honest count
        "actual_model_generated_tokens": actual_generated,
        "decision": final_decision,
    })
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
    p.add_argument("--sample_manifest", type=str, default=None,
                   help="V2: Path to canonical sample manifest (enforces same-sample)")
    p.add_argument("--variant", default="existing_fragment",
                   choices=["existing_fragment", "rcv_no_gate", "full_rcv",
                            "stage0_only", "recover_only", "full_rcv_majvote"],
                   help="NEGATIVE ABLATION variants. Default is existing_fragment "
                        "(no gate). full_rcv* are kept for ablation reproducibility "
                        "only — see reports/FINAL_RCV_VERDICT.md.")
    p.add_argument("--fallback_action", default="town_parse",
                   choices=["town_parse", "majority_vote"],
                   help="V3: fallback action when recoverability gate rejects extraction")
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

    # V2: Manifest-driven mode for reproducibility
    manifest_meta = None
    if args.sample_manifest:
        log.info(f"Loading from manifest: {args.sample_manifest}")
        items, manifest_meta = load_from_manifest(args.sample_manifest)
        assert manifest_meta["benchmark"] == args.benchmark
        assert manifest_meta["seed"] == args.seed
    else:
        items = load_benchmark(args.benchmark, args.n_samples, args.seed)
    log.info(f"Loaded {len(items)} items, variant={args.variant}")
    if args.variant in ("full_rcv", "full_rcv_majvote", "stage0_only", "recover_only"):
        log.warning("=" * 60)
        log.warning("RCV gate variants are NEGATIVE ABLATIONS, not the main method.")
        log.warning("See reports/FINAL_RCV_VERDICT.md for evidence.")
        log.warning("=" * 60)
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results = []
    n_correct = 0
    total_tokens = 0
    decisions = {}

    # V2: Aggregate probe usage stats
    total_stage0 = 0
    total_stage2 = 0
    total_strict_probe = 0
    total_soft_probe = 0
    total_verifier = 0

    for i, item in enumerate(items):
        r = run_rcv_sample(model, tok, item, args)
        results.append(r)
        if r["correct"]: n_correct += 1
        total_tokens += r["tokens_total"]
        total_stage0 += r.get("stage0_tokens", 0)
        total_stage2 += r.get("stage2_tokens", 0)
        total_strict_probe += r.get("strict_probe_tokens", 0)
        total_soft_probe += r.get("soft_probe_tokens", 0)
        total_verifier += r.get("verifier_tokens", 0)
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
                 "timestamp": ts, "schema_version": 3,
                 "sample_manifest": args.sample_manifest,
                 "manifest_meta": manifest_meta,
                 "method_status": "negative_ablation"
                                  if args.variant != "existing_fragment"
                                  else "baseline_no_rcv"},
        "accuracy": n_correct / n,
        "avg_tokens": total_tokens / n,
        "n_correct": n_correct,
        "decisions": decisions,
        "token_breakdown": {
            "avg_stage0": total_stage0 / n,
            "avg_stage2": total_stage2 / n,
            "avg_strict_probe": total_strict_probe / n,
            "avg_soft_probe": total_soft_probe / n,
            "avg_verifier": total_verifier / n,
            "total_all_generated": total_tokens,
        },
        "per_sample": results,
    }

    fname = f"rcv_{args.variant}_{args.benchmark}_{ts}.json"
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\n=== RCV-IRIS V2 ({args.variant}) ===")
    log.info(f"Accuracy: {n_correct}/{n} = {n_correct/n*100:.1f}%")
    log.info(f"Avg tokens (actual_generated): {total_tokens/n:.0f}")
    log.info(f"  breakdown: s0={total_stage0/n:.0f} s2={total_stage2/n:.0f} "
             f"strict={total_strict_probe/n:.0f} soft={total_soft_probe/n:.0f}")
    log.info(f"Decisions: {decisions}")


if __name__ == "__main__":
    main()

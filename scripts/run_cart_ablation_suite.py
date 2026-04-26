#!/usr/bin/env python3
"""CART A/B/C/D ablation suite — corrected version with 4 real distinct arms.

Arms:
  A (existing_fragment): old IRIS Stage-3 prompt extraction via base model nothink
  B (cart_question_only): CART transducer question-only, separate checkpoint
  C (cart_prefix_conditioned): CART transducer with prefix, separate checkpoint
  D (full_cart): prefix-conditioned transducer + online readiness (future)

Usage:
    python3 scripts/run_cart_ablation_suite.py \
        --model Qwen/Qwen3-8B --benchmark math500 --n_samples 50 --seed 42 \
        --prefix_checkpoint checkpoints/cart/math_prefix_conditioned_dev \
        --question_only_checkpoint checkpoints/cart/math_question_only_dev \
        --b2_max 512 --b_answer 128 \
        --output_dir results/cart/corrected_gate_n50
"""
import argparse, glob, json, math, os, subprocess, sys


def paired_analysis(all_results):
    arms = sorted(all_results.keys())
    has_hash = all(
        all_results[a]["per_sample"] and "question_hash" in all_results[a]["per_sample"][0]
        for a in arms)
    if has_hash:
        idxs = {a: {r["question_hash"]: r for r in all_results[a]["per_sample"]} for a in arms}
        common = sorted(set.intersection(*[set(idxs[a].keys()) for a in arms]))
        print(f"[hash-join] {len(common)} common samples across {len(arms)} arms")
    else:
        idxs = {a: {r["idx"]: r for r in all_results[a]["per_sample"]} for a in arms}
        common = sorted(set.intersection(*[set(idxs[a].keys()) for a in arms]))
        print(f"[idx-join] {len(common)} common samples")

    report = {"n_common": len(common), "arms": arms, "accuracy": {}, "avg_tokens": {}, "pairwise": {}}

    for a in arms:
        correct = sum(1 for i in common if idxs[a][i]["correct"])
        tokens = sum(idxs[a][i].get("total_tokens", 0) for i in common)
        report["accuracy"][a] = correct / len(common) if common else 0
        report["avg_tokens"][a] = tokens / len(common) if common else 0

    for i_a, a in enumerate(arms):
        for b in arms[i_a + 1:]:
            a_only = sum(1 for i in common if idxs[a][i]["correct"] and not idxs[b][i]["correct"])
            b_only = sum(1 for i in common if not idxs[a][i]["correct"] and idxs[b][i]["correct"])
            disc = a_only + b_only
            if disc > 0:
                k = min(a_only, b_only)
                p_val = min(sum(math.comb(disc, j) for j in range(k + 1)) / (2 ** (disc - 1)), 1.0)
            else:
                p_val = 1.0
            report["pairwise"][f"{a}_vs_{b}"] = {
                f"{a}_only": a_only, f"{b}_only": b_only,
                "discordant": disc, "mcnemar_p": round(p_val, 4)}

    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--benchmark", default="math500", choices=["math500", "gsm8k"])
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix_checkpoint", default=None,
                   help="LoRA checkpoint for prefix-conditioned (C/D)")
    p.add_argument("--question_only_checkpoint", default=None,
                   help="Separate LoRA checkpoint for question-only (B)")
    p.add_argument("--b2_max", type=int, default=512)
    p.add_argument("--b_answer", type=int, default=128)
    p.add_argument("--output_dir", default="results/cart/ablation")
    p.add_argument("--from_results", nargs="+", help="Analyze existing files")
    args = p.parse_args()

    if args.from_results:
        all_results = {}
        for f in args.from_results:
            with open(f) as fh:
                d = json.load(fh)
            mode = d["meta"].get("mode", "unknown")
            if mode in all_results:
                print(f"WARNING: duplicate mode {mode}, skipping {f}")
                continue
            all_results[mode] = d
        report = paired_analysis(all_results)
        out = os.path.join(os.path.dirname(args.from_results[0]), "cart_ablation_report.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    eval_script = os.path.join(os.path.dirname(__file__), "eval_cart_transducer.py")
    base_args = [
        "--model", args.model, "--benchmark", args.benchmark,
        "--n_samples", str(args.n_samples), "--seed", str(args.seed),
        "--b2_max", str(args.b2_max), "--b_answer", str(args.b_answer),
        "--output_dir", args.output_dir,
    ]

    # A: existing_fragment — base model, trace-conditioned, NO LoRA (old Stage-3 style)
    print("\n=== ARM A: existing_fragment (base model + prefix, no LoRA) ===")
    cmd_a = ["python3", eval_script] + base_args + ["--trace_conditioned"]
    print(f"CMD: {' '.join(cmd_a)}")
    subprocess.run(cmd_a, check=True)

    # B: cart_question_only — question-only checkpoint (separate from prefix checkpoint)
    if args.question_only_checkpoint:
        print("\n=== ARM B: cart_question_only (LoRA, no prefix) ===")
        cmd_b = ["python3", eval_script] + base_args + [
            "--checkpoint", args.question_only_checkpoint]
        print(f"CMD: {' '.join(cmd_b)}")
        subprocess.run(cmd_b, check=True)
    else:
        print("\n=== ARM B: SKIPPED (no --question_only_checkpoint) ===")

    # C: cart_prefix_conditioned — prefix checkpoint + trace
    if args.prefix_checkpoint:
        print("\n=== ARM C: cart_prefix_conditioned (LoRA + prefix) ===")
        cmd_c = ["python3", eval_script] + base_args + [
            "--checkpoint", args.prefix_checkpoint, "--trace_conditioned"]
        print(f"CMD: {' '.join(cmd_c)}")
        subprocess.run(cmd_c, check=True)
    else:
        print("\n=== ARM C: SKIPPED (no --prefix_checkpoint) ===")

    # D: full_cart — TODO: online readiness + answer reservation
    print("\n=== ARM D: full_cart — NOT YET IMPLEMENTED ===")

    # Analyze
    result_files = sorted(glob.glob(os.path.join(args.output_dir, "cart_*.json")))
    if result_files:
        all_results = {}
        for f in result_files:
            with open(f) as fh:
                d = json.load(fh)
            mode = d["meta"].get("mode", "unknown")
            all_results[mode] = d
        report = paired_analysis(all_results)
        out_path = os.path.join(args.output_dir, "cart_ablation_report.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*60}")
        print("CART ABLATION REPORT")
        print(f"{'='*60}")
        for a in report["arms"]:
            print(f"  {a}: acc={report['accuracy'][a]*100:.1f}% tok={report['avg_tokens'][a]:.0f}")
        for k, v in report["pairwise"].items():
            print(f"  {k}: disc={v['discordant']}, p={v['mcnemar_p']}")
        print(f"Arms present: {len(report['arms'])} (need 4 for full gate)")


if __name__ == "__main__":
    main()

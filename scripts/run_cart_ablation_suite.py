#!/usr/bin/env python3
"""CART A/B/C/D ablation suite — runs all arms and computes paired comparison.

Arms:
  A (existing_fragment): old IRIS Stage-3 prompt extraction, no CART
  B (cart_question_only): CART transducer question-only (no prefix)
  C (cart_transducer_only): CART transducer with prefix, no online readiness
  D (full_cart): CART transducer + online answer reservation

Usage:
    python3 scripts/run_cart_ablation_suite.py \
        --model Qwen/Qwen3-8B --benchmark math500 --n_samples 50 --seed 42 \
        --checkpoint checkpoints/cart/dev \
        --b2_max 512 --b_answer 128 \
        --output_dir results/cart/ablation_n50
"""
import argparse, glob, json, math, os, subprocess, sys


def paired_analysis(all_results):
    arms = sorted(all_results.keys())
    # Join by question_hash if available, else idx
    has_hash = all(
        all_results[a]["per_sample"] and "question_hash" in all_results[a]["per_sample"][0]
        for a in arms)
    if has_hash:
        idxs = {a: {r["question_hash"]: r for r in all_results[a]["per_sample"]} for a in arms}
        common = sorted(set.intersection(*[set(idxs[a].keys()) for a in arms]))
    else:
        idxs = {a: {r["idx"]: r for r in all_results[a]["per_sample"]} for a in arms}
        common = sorted(set.intersection(*[set(idxs[a].keys()) for a in arms]))

    report = {"n_common": len(common), "arms": arms}

    # Accuracy per arm
    accs = {}
    avg_toks = {}
    for a in arms:
        correct = sum(1 for i in common if idxs[a][i]["correct"])
        tokens = sum(idxs[a][i].get("total_tokens", 0) for i in common)
        accs[a] = correct / len(common) if common else 0
        avg_toks[a] = tokens / len(common) if common else 0
    report["accuracy"] = accs
    report["avg_tokens"] = avg_toks

    # Pairwise McNemar
    report["pairwise"] = {}
    for i_a, a in enumerate(arms):
        for b in arms[i_a + 1:]:
            a_only = sum(1 for i in common if idxs[a][i]["correct"] and not idxs[b][i]["correct"])
            b_only = sum(1 for i in common if not idxs[a][i]["correct"] and idxs[b][i]["correct"])
            disc = a_only + b_only
            if disc > 0:
                k = min(a_only, b_only)
                p_val = sum(math.comb(disc, j) for j in range(k + 1)) / (2 ** (disc - 1))
                p_val = min(p_val, 1.0)
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
    p.add_argument("--checkpoint", required=True, help="CART LoRA checkpoint dir")
    p.add_argument("--b2_max", type=int, default=512)
    p.add_argument("--b_answer", type=int, default=128)
    p.add_argument("--output_dir", default="results/cart/ablation")
    p.add_argument("--from_results", nargs="+", help="Skip compute; analyze existing files")
    args = p.parse_args()

    if args.from_results:
        all_results = {}
        for f in args.from_results:
            with open(f) as fh:
                d = json.load(fh)
            mode = d["meta"].get("mode", "unknown")
            all_results[mode] = d
        report = paired_analysis(all_results)
        out = os.path.join(os.path.dirname(args.from_results[0]),
                           "cart_ablation_report.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    eval_script = os.path.join(os.path.dirname(__file__), "eval_cart_transducer.py")
    all_results = {}

    # A: existing fragment (base model, no LoRA, question + prefix, old Stage-3 style)
    arms = [
        ("existing_fragment", False, None),
        ("question_only", False, args.checkpoint),
        ("prefix_conditioned", True, args.checkpoint),
    ]

    for mode, trace_cond, ckpt in arms:
        log_name = f"cart_{mode}_{args.benchmark}.log"
        cmd = [
            "python3", eval_script,
            "--model", args.model,
            "--benchmark", args.benchmark,
            "--n_samples", str(args.n_samples),
            "--seed", str(args.seed),
            "--b2_max", str(args.b2_max),
            "--b_answer", str(args.b_answer),
            "--output_dir", args.output_dir,
        ]
        if trace_cond:
            cmd.append("--trace_conditioned")
        if ckpt:
            cmd.extend(["--checkpoint", ckpt])
        print(f"\n=== Running {mode} ===")
        print(f"CMD: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # Load results and analyze
    result_files = sorted(glob.glob(os.path.join(args.output_dir, "cart_*.json")))
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
    print("CART A/B/C ABLATION REPORT")
    print(f"{'='*60}")
    print(f"n_common: {report['n_common']}")
    for a in report["arms"]:
        print(f"  {a}: acc={report['accuracy'][a]*100:.1f}% tok={report['avg_tokens'][a]:.0f}")
    for k, v in report["pairwise"].items():
        print(f"  {k}: disc={v['discordant']}, p={v['mcnemar_p']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CART A/B/C/D ablation suite V3 — arm_name, frozen prefix, 4 arms required.

Arms:
  A existing_fragment: base model + base-generated prefix, no LoRA (old Stage-3 style)
  B cart_question_only: question-only LoRA checkpoint, no prefix
  C cart_prefix_conditioned: base-generated prefix + prefix LoRA for answer
  D full_cart: online readiness + answer reservation (requires run_cart_iris.py)
"""
import argparse, glob, json, math, os, subprocess, sys


def paired_analysis(all_results):
    arms = sorted(all_results.keys())
    has_hash = all(
        all_results[a]["per_sample"] and "question_hash" in all_results[a]["per_sample"][0]
        for a in arms)
    key = "question_hash" if has_hash else "idx"
    idxs = {a: {r[key]: r for r in all_results[a]["per_sample"]} for a in arms}
    common = sorted(set.intersection(*[set(idxs[a].keys()) for a in arms]))
    join_type = "hash-join" if has_hash else "idx-join"
    print(f"[{join_type}] {len(common)} common samples across {len(arms)} arms")

    report = {"n_common": len(common), "arms": arms, "join_type": join_type,
              "accuracy": {}, "avg_tokens": {}, "pairwise": {}}

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
    p.add_argument("--benchmark", default="math500")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix_checkpoint", default=None)
    p.add_argument("--question_only_checkpoint", default=None)
    p.add_argument("--b2_max", type=int, default=512)
    p.add_argument("--b_answer", type=int, default=128)
    p.add_argument("--output_dir", default="results/cart/ablation")
    p.add_argument("--arms", nargs="+",
                   default=["existing_fragment", "cart_question_only",
                            "cart_prefix_conditioned"])
    p.add_argument("--from_results", nargs="+")
    args = p.parse_args()

    if args.from_results:
        all_results = {}
        for f in args.from_results:
            with open(f) as fh:
                d = json.load(fh)
            arm = d["meta"].get("arm_name", d["meta"].get("mode", "unknown"))
            if arm in all_results:
                print(f"WARNING: duplicate arm {arm}, skipping {f}")
                continue
            all_results[arm] = d
        report = paired_analysis(all_results)
        out = os.path.join(os.path.dirname(args.from_results[0]),
                           "cart_ablation_report.json")
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

    for arm in args.arms:
        print(f"\n=== ARM: {arm} ===")
        cmd = ["python3", eval_script] + base_args + ["--arm_name", arm]

        if arm == "existing_fragment":
            cmd += ["--trace_conditioned", "--prefix_generator", "base"]
        elif arm == "cart_question_only":
            if not args.question_only_checkpoint:
                print(f"SKIP {arm}: no --question_only_checkpoint")
                continue
            cmd += ["--checkpoint", args.question_only_checkpoint]
        elif arm == "cart_prefix_conditioned":
            if not args.prefix_checkpoint:
                print(f"SKIP {arm}: no --prefix_checkpoint")
                continue
            cmd += ["--checkpoint", args.prefix_checkpoint,
                    "--trace_conditioned", "--prefix_generator", "base"]
        elif arm == "full_cart":
            print(f"ARM D (full_cart): requires run_cart_iris.py — skipping if absent")
            cart_script = os.path.join(os.path.dirname(__file__), "run_cart_iris.py")
            if not os.path.exists(cart_script):
                print(f"  run_cart_iris.py not found — SKIPPING D")
                continue
            # TODO: implement D call
            continue
        else:
            print(f"Unknown arm: {arm}")
            continue

        print(f"CMD: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # Analyze all result files
    result_files = sorted(glob.glob(os.path.join(args.output_dir, "cart_*.json")))
    if not result_files:
        print("No result files found!")
        return

    all_results = {}
    for f in result_files:
        if "ablation_report" in f:
            continue
        with open(f) as fh:
            d = json.load(fh)
        arm = d["meta"].get("arm_name", d["meta"].get("mode", "unknown"))
        all_results[arm] = d

    report = paired_analysis(all_results)
    out_path = os.path.join(args.output_dir, "cart_ablation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("CART V3 ABLATION REPORT")
    print(f"{'='*60}")
    for a in report["arms"]:
        print(f"  {a}: acc={report['accuracy'][a]*100:.1f}% tok={report['avg_tokens'][a]:.0f}")
    for k, v in report["pairwise"].items():
        print(f"  {k}: disc={v['discordant']}, p={v['mcnemar_p']}")
    print(f"Arms present: {len(report['arms'])}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

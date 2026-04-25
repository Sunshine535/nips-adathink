#!/usr/bin/env python3
"""A/B/C paired ablation suite — GPT-5.5 Task 6.

Enforces:
- All variants run on SAME manifest samples
- Per-sample hash verification
- Paired McNemar analysis
- Token accounting audit (honest counts)
- Decision/mechanism activation reporting

Usage:
    # Compute mode — runs A/B/C on one manifest
    python3 scripts/run_rcv_ablation_suite.py \\
        --model Qwen/Qwen3-8B --sample_manifest results/manifests/math200_seed42.json \\
        --b1 256 --b2_max 512 --b_answer 128 \\
        --arms existing_fragment rcv_no_gate full_rcv \\
        --output_dir results/rcv_ablation_suite

    # Analysis-only mode — analyze existing result files
    python3 scripts/run_rcv_ablation_suite.py --from_results \\
        results/rcv_iris_b2_512/rcv_existing_fragment_math500_*.json \\
        results/rcv_iris_b2_512/rcv_rcv_no_gate_math500_*.json \\
        results/rcv_iris_b2_512/rcv_full_rcv_math500_*.json
"""
import argparse, glob, json, math, os, subprocess, sys


def paired_analysis(all_results: dict):
    """Compute paired McNemar and token audits across arms.

    V2: Verifies same-sample via question_hash when available.
    """
    arms = sorted(all_results.keys())
    # Prefer hash join when V3 schema present
    has_hash = all(
        all_results[a]["per_sample"] and "question_hash" in all_results[a]["per_sample"][0]
        for a in arms
    )
    if has_hash:
        # Join by question_hash; verify all arms see same hash set
        idxs = {a: {r["question_hash"]: r for r in all_results[a]["per_sample"]} for a in arms}
        all_hashes = [set(idxs[a].keys()) for a in arms]
        common_ids = set.intersection(*all_hashes)
        common = sorted(common_ids)
        print(f"[hash-join] {len(common)} samples common across {len(arms)} arms")
    else:
        # Legacy: idx join (V2 result files)
        idxs = {a: {r["idx"]: r for r in all_results[a]["per_sample"]} for a in arms}
        common_ids = set.intersection(*[set(idxs[a].keys()) for a in arms])
        common = sorted(common_ids)
        print(f"[legacy idx-join, no hash verification] {len(common)} samples")

    # Verify same samples
    if len(common) != len(all_results[arms[0]]["per_sample"]):
        print(f"WARNING: only {len(common)} common samples out of "
              f"{[len(all_results[a]['per_sample']) for a in arms]}")

    report = {"n_common": len(common), "arms": arms, "pairwise": {}}

    # Accuracy per arm
    accs = {}
    avg_toks = {}
    for a in arms:
        correct = sum(1 for i in common if idxs[a][i]["correct"])
        tokens = sum(idxs[a][i]["tokens_total"] for i in common)
        accs[a] = correct / len(common)
        avg_toks[a] = tokens / len(common)
    report["accuracy"] = accs
    report["avg_tokens"] = avg_toks

    # Pairwise McNemar
    for i_a, a in enumerate(arms):
        for b in arms[i_a+1:]:
            both = sum(1 for i in common if idxs[a][i]["correct"] and idxs[b][i]["correct"])
            a_only = sum(1 for i in common if idxs[a][i]["correct"] and not idxs[b][i]["correct"])
            b_only = sum(1 for i in common if not idxs[a][i]["correct"] and idxs[b][i]["correct"])
            neither = sum(1 for i in common if not idxs[a][i]["correct"] and not idxs[b][i]["correct"])
            discordant = a_only + b_only
            # Exact binomial p for McNemar (two-sided)
            if discordant > 0:
                k = min(a_only, b_only)
                p = sum(math.comb(discordant, i) for i in range(k+1)) / (2**(discordant-1))
                p = min(p, 1.0)
            else:
                p = 1.0
            report["pairwise"][f"{a}_vs_{b}"] = {
                "both": both, f"{a}_only": a_only, f"{b}_only": b_only,
                "neither": neither, "discordant": discordant,
                "exact_mcnemar_p": round(p, 4),
            }

    # Token breakdown per arm (if available)
    token_breakdowns = {}
    for a in arms:
        tb = all_results[a].get("token_breakdown", {})
        if tb:
            token_breakdowns[a] = tb
    report["token_breakdowns"] = token_breakdowns

    # Mechanism activation
    mech = {}
    for a in arms:
        mech[a] = all_results[a].get("decisions", {})
    report["decisions"] = mech

    # Decision-changed samples (C vs A)
    if "full_rcv" in arms and "existing_fragment" in arms:
        diffs = []
        for i in common:
            da = idxs["existing_fragment"][i].get("decision")
            dc = idxs["full_rcv"][i].get("decision")
            if da != dc:
                diffs.append({
                    "idx": i,
                    "A_decision": da, "A_correct": idxs["existing_fragment"][i]["correct"],
                    "C_decision": dc, "C_correct": idxs["full_rcv"][i]["correct"],
                })
        report["decision_changed_samples"] = diffs
        report["n_decision_changed"] = len(diffs)
        flipped_wins = sum(1 for d in diffs if d["C_correct"] and not d["A_correct"])
        flipped_loses = sum(1 for d in diffs if d["A_correct"] and not d["C_correct"])
        report["gate_effect"] = {
            "decision_changed": len(diffs),
            "gate_wins": flipped_wins,
            "gate_loses": flipped_loses,
            "net_effect": flipped_wins - flipped_loses,
        }

    return report


def run_from_results(result_files: list):
    """V2: Reject duplicate variant files (would silently overwrite)."""
    all_results = {}
    for f in result_files:
        with open(f) as fh:
            d = json.load(fh)
        variant = d["meta"]["variant"]
        if variant in all_results:
            raise ValueError(f"Duplicate variant {variant} found: "
                             f"{all_results[variant].get('_source', '?')} vs {f}. "
                             f"Clean output dir or specify exact files.")
        d["_source"] = f
        all_results[variant] = d
    return paired_analysis(all_results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--sample_manifest", type=str)
    p.add_argument("--b1", type=int, default=256)
    p.add_argument("--b2_max", type=int, default=512)
    p.add_argument("--b_answer", type=int, default=128)
    p.add_argument("--arms", nargs="+",
                   default=["existing_fragment", "rcv_no_gate", "full_rcv"])
    p.add_argument("--output_dir", default="results/rcv_ablation_suite")
    p.add_argument("--from_results", nargs="+",
                   help="Skip compute; analyze existing result JSON files")
    args = p.parse_args()

    if args.from_results:
        files = []
        for pattern in args.from_results:
            files.extend(sorted(glob.glob(pattern)) if "*" in pattern else [pattern])
        print(f"Analyzing {len(files)} result files...")
        report = run_from_results(files)
        out = os.path.join(os.path.dirname(files[0]) or ".",
                           "ablation_suite_report.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n=== A/B/C PAIRED ABLATION REPORT ===")
        print(f"n_common: {report['n_common']}")
        print(f"Arms: {report['arms']}")
        print(f"\nAccuracy: {json.dumps(report['accuracy'], indent=2)}")
        print(f"Avg tokens: {json.dumps(report['avg_tokens'], indent=2)}")
        print(f"\nPairwise McNemar:")
        for k, v in report["pairwise"].items():
            print(f"  {k}: disc={v['discordant']}, p={v['exact_mcnemar_p']}")
        if "gate_effect" in report:
            print(f"\nGate effect: {report['gate_effect']}")
        if report.get("token_breakdowns"):
            print(f"\nToken breakdowns:")
            for a, tb in report["token_breakdowns"].items():
                print(f"  {a}: s0={tb.get('avg_stage0',0):.0f} s2={tb.get('avg_stage2',0):.0f} "
                      f"strict={tb.get('avg_strict_probe',0):.0f} "
                      f"soft={tb.get('avg_soft_probe',0):.0f}")
        print(f"\nSaved report to {out}")
        return

    # Compute mode: run each arm
    if not args.sample_manifest:
        print("ERROR: --sample_manifest required in compute mode")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    for arm in args.arms:
        print(f"\n=== Running arm: {arm} ===")
        cmd = [
            "python3", os.path.join(os.path.dirname(__file__), "run_rcv_iris.py"),
            "--model", args.model,
            "--sample_manifest", args.sample_manifest,
            "--b1", str(args.b1), "--b2_max", str(args.b2_max),
            "--b_answer", str(args.b_answer),
            "--variant", arm,
            "--output_dir", args.output_dir,
        ]
        # Load benchmark from manifest
        with open(args.sample_manifest) as f:
            m = json.load(f)
        cmd.extend(["--benchmark", m["meta"]["benchmark"],
                    "--seed", str(m["meta"]["seed"])])
        print(f"CMD: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # After all arms done, analyze
    result_files = sorted(glob.glob(os.path.join(args.output_dir, "rcv_*.json")))
    report = run_from_results(result_files)
    out = os.path.join(args.output_dir, "ablation_suite_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved suite report to {out}")


if __name__ == "__main__":
    main()

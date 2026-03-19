#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Sweep parametric budget controller configs and compute paired significance"
    )
    ap.add_argument("--manifest_json", required=True)
    ap.add_argument("--train_lambdas", type=str, default="0.30,0.40,0.60,0.80")
    ap.add_argument("--target_budgets", type=str, default="256,128")
    ap.add_argument("--eval_lambda", type=float, default=0.15)
    ap.add_argument("--norm_tokens", type=float, default=512.0)
    ap.add_argument("--epochs_grid", type=str, default="30,50,80,120")
    ap.add_argument("--lr_grid", type=str, default="0.03,0.05,0.1,0.2")
    ap.add_argument("--l2_grid", type=str, default="1e-6,1e-5,1e-4,5e-4")
    ap.add_argument(
        "--cost_weight_grid",
        type=str,
        default="0.0,2.0,4.0,8.0,16.0,32.0,64.0,128.0,256.0",
    )
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap_seed", type=int, default=20260303)
    ap.add_argument(
        "--result_dir", type=str, default="methods/01_adathink/results"
    )
    ap.add_argument("--tag_prefix", type=str, default="")
    ap.add_argument("--output_csv", type=str, default="")
    ap.add_argument("--output_json", type=str, default="")
    args = ap.parse_args()

    train_lams = parse_float_list(args.train_lambdas)
    targets = parse_int_list(args.target_budgets)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = args.tag_prefix if args.tag_prefix else f"param_sweep_{ts}"

    rows = []
    for train_lam in train_lams:
        lam_tag = str(train_lam).replace(".", "p")
        for target_budget in targets:
            tag = f"{prefix}_lamtrain{lam_tag}_t{target_budget}"
            out_json = result_dir / f"param_controller_{tag}.json"
            out_csv = result_dir / f"param_controller_rows_{tag}.csv"
            sig_json = (
                result_dir
                / f"param_controller_significance_{tag}_vs_fixed256_evalLam{str(args.eval_lambda).replace('.', 'p')}.json"
            )

            cmd_train = [
                sys.executable,
                "methods/01_adathink/scripts/run_parametric_from_manifest.py",
                "--manifest_json",
                args.manifest_json,
                "--lambda_cost",
                str(train_lam),
                "--norm_tokens",
                str(args.norm_tokens),
                "--target_budget",
                str(target_budget),
                "--epochs_grid",
                args.epochs_grid,
                "--lr_grid",
                args.lr_grid,
                "--l2_grid",
                args.l2_grid,
                "--cost_weight_grid",
                args.cost_weight_grid,
                "--seed",
                str(args.seed),
                "--output_json",
                str(out_json),
                "--output_csv",
                str(out_csv),
            ]
            print(f"[RUN] {tag}")
            subprocess.run(cmd_train, check=True)

            cmd_sig = [
                sys.executable,
                "methods/01_adathink/scripts/run_template_controller_significance.py",
                "--rows_csv",
                str(out_csv),
                "--compare_budget",
                "256",
                "--lambda_cost",
                str(args.eval_lambda),
                "--norm_tokens",
                str(args.norm_tokens),
                "--n_bootstrap",
                str(args.n_bootstrap),
                "--bootstrap_seed",
                str(args.bootstrap_seed),
                "--output_json",
                str(sig_json),
            ]
            subprocess.run(cmd_sig, check=True)

            sig = json.loads(sig_json.read_text(encoding="utf-8"))
            rec = {
                "tag": tag,
                "train_lambda": train_lam,
                "target_budget": target_budget,
                "delta_acc": sig["paired_delta_learned_minus_fixed"]["accuracy"]["mean"],
                "delta_tokens": sig["paired_delta_learned_minus_fixed"]["avg_tokens"]["mean"],
                "delta_utility": sig["paired_delta_learned_minus_fixed"]["avg_utility"]["mean"],
                "learned_acc": sig["learned_mean"]["accuracy"],
                "learned_tokens": sig["learned_mean"]["avg_tokens"],
                "learned_utility": sig["learned_mean"]["avg_utility"],
                "sig_json": str(sig_json),
                "controller_json": str(out_json),
                "controller_rows_csv": str(out_csv),
            }
            rows.append(rec)
            print(
                f"[DONE] {tag} | dAcc={rec['delta_acc']:+.4f} "
                f"| dTok={rec['delta_tokens']:+.2f} | dUtil={rec['delta_utility']:+.5f}"
            )

    ranked = sorted(rows, key=lambda r: (-r["delta_utility"], abs(r["delta_tokens"])))

    out_csv_path = (
        Path(args.output_csv)
        if args.output_csv
        else result_dir / f"{prefix}_scoreboard.csv"
    )
    out_json_path = (
        Path(args.output_json)
        if args.output_json
        else result_dir / f"{prefix}_scoreboard.json"
    )

    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "train_lambda",
                "target_budget",
                "delta_acc",
                "delta_tokens",
                "delta_utility",
                "learned_acc",
                "learned_tokens",
                "learned_utility",
                "sig_json",
                "controller_json",
                "controller_rows_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(ranked)

    out_json_path.write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_csv_path}")
    print(f"Saved: {out_json_path}")


if __name__ == "__main__":
    main()

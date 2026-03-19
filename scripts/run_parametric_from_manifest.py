#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Run parametric budget controller from a manifest json")
    ap.add_argument("--manifest_json", required=True)
    ap.add_argument("--lambda_cost", type=float, default=0.15)
    ap.add_argument("--norm_tokens", type=float, default=512.0)
    ap.add_argument("--target_budget", type=int, default=0)
    ap.add_argument("--epochs_grid", type=str, default="30,50")
    ap.add_argument("--lr_grid", type=str, default="0.1,0.2")
    ap.add_argument("--l2_grid", type=str, default="1e-4,5e-4")
    ap.add_argument("--cost_weight_grid", type=str, default="0.0,0.5")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    csvs = [r["per_sample_csv"] for r in manifest["runs"]]

    cmd = [
        sys.executable,
        "methods/01_adathink/scripts/run_parametric_budget_controller.py",
        "--input_csvs",
        *csvs,
        "--lambda_cost",
        str(args.lambda_cost),
        "--norm_tokens",
        str(args.norm_tokens),
        "--target_budget",
        str(args.target_budget),
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
        args.output_json,
        "--output_csv",
        args.output_csv,
    ]
    print("Running:", " ".join(cmd[:8]), "...", f"(args={len(cmd)})")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

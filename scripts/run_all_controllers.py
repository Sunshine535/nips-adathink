#!/usr/bin/env python3
"""Run all missing controller evaluations and analyses on new benchmarks.

Fills the gaps identified by comparing with GitHub repo design:
1. Value Controller on MATH-500/BBH
2. Parametric Controller on MATH-500/BBH
3. Overthinking aggregate analysis
4. Template Controller significance tests
5. Self-Consistency baseline analysis
"""

import csv
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def find_csvs(pattern, budget_filter):
    """Find per_sample CSVs matching budget requirements."""
    csvs = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        sf = f.replace("per_sample_", "summary_").replace(".csv", ".json")
        if os.path.exists(sf):
            with open(sf) as fh:
                s = json.load(fh)
            if all(str(b) in s.get("fixed", {}) for b in budget_filter):
                csvs.append(f)
    return csvs


def run_cmd(cmd, label, timeout=1800):
    log(f"  Running: {label}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - start
    if result.returncode == 0:
        log(f"    OK ({elapsed:.0f}s)")
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-5:]:
                log(f"    {line}")
    else:
        log(f"    FAILED ({elapsed:.0f}s): {result.stderr[-300:]}")
    return result.returncode == 0


def main():
    log("=" * 60)
    log("CONTROLLER & ANALYSIS GAP-FILL PIPELINE")
    log("=" * 60)
    total_start = time.time()

    # CSV inventories
    math500_27b = find_csvs("per_sample_math500_Qwen3.5_27B_*.csv", [2048, 4096, 8192])
    bbh_27b = find_csvs("per_sample_bbh_Qwen3.5_27B_*.csv", [1024, 2048, 4096])
    math500_8b = find_csvs("per_sample_math500_Qwen3_8B_*.csv", [512, 1024, 2048])
    bbh_8b = find_csvs("per_sample_bbh_Qwen3_8B_*.csv", [256, 512, 1024])

    log(f"\nCSV Inventory: MATH500-27B={len(math500_27b)}, BBH-27B={len(bbh_27b)}, "
        f"MATH500-8B={len(math500_8b)}, BBH-8B={len(bbh_8b)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ==========================================================
    # 1. Value Controller on all new benchmarks
    # ==========================================================
    log("\n>>> Phase 1: Value Controller")
    value_script = os.path.join(SCRIPT_DIR, "run_value_budget_controller.py")

    configs = [
        ("math500_27b_v2", math500_27b, 4096, 8192),
        ("bbh_27b_v2", bbh_27b, 2048, 4096),
        ("math500_8b", math500_8b, 1024, 2048),
        ("bbh_8b", bbh_8b, 512, 1024),
    ]

    for tag, csvs, target_budget, norm_tok in configs:
        if not csvs:
            log(f"  Skipping {tag}: no CSVs")
            continue
        for penalty in [0.0, 0.4, 0.8, 1.2]:
            pen_tag = str(penalty).replace(".", "p")
            out_json = os.path.join(RESULTS_DIR, f"value_controller_{tag}_pen{pen_tag}_{ts}.json")
            out_csv = os.path.join(RESULTS_DIR, f"value_controller_rows_{tag}_pen{pen_tag}_{ts}.csv")
            run_cmd([
                sys.executable, value_script,
                "--input_csvs", *csvs,
                "--target_budget", str(target_budget),
                "--budget_penalty", str(penalty),
                "--norm_tokens", str(norm_tok),
                "--output_json", out_json,
                "--output_csv", out_csv,
            ], f"Value Controller {tag} penalty={penalty}")

    # ==========================================================
    # 2. Parametric Controller on all new benchmarks
    # ==========================================================
    log("\n>>> Phase 2: Parametric Controller")
    param_script = os.path.join(SCRIPT_DIR, "run_parametric_budget_controller.py")

    for tag, csvs, norm_tok in [
        ("math500_27b_v2", math500_27b, 8192),
        ("bbh_27b_v2", bbh_27b, 4096),
        ("math500_8b", math500_8b, 2048),
        ("bbh_8b", bbh_8b, 1024),
    ]:
        if not csvs:
            log(f"  Skipping {tag}: no CSVs")
            continue
        out_json = os.path.join(RESULTS_DIR, f"param_controller_{tag}_{ts}.json")
        out_csv = os.path.join(RESULTS_DIR, f"param_controller_rows_{tag}_{ts}.csv")
        run_cmd([
            sys.executable, param_script,
            "--input_csvs", *csvs,
            "--lambda_cost", "0.15",
            "--norm_tokens", str(norm_tok),
            "--output_json", out_json,
            "--output_csv", out_csv,
        ], f"Parametric Controller {tag}")

    # ==========================================================
    # 3. Overthinking aggregate analysis
    # ==========================================================
    log("\n>>> Phase 3: Overthinking Aggregate")
    ot_script = os.path.join(SCRIPT_DIR, "run_overthinking_aggregate.py")

    for tag, csvs in [
        ("math500_27b_v2", math500_27b),
        ("bbh_27b_v2", bbh_27b),
        ("math500_8b", math500_8b),
        ("bbh_8b", bbh_8b),
    ]:
        if not csvs:
            continue
        out = os.path.join(RESULTS_DIR, f"overthinking_{tag}_{len(csvs)}seed_{ts}.json")
        run_cmd([
            sys.executable, ot_script,
            "--input_csvs", *csvs,
            "--output_json", out,
        ], f"Overthinking {tag}")

    # ==========================================================
    # 4. Template Controller significance tests
    # ==========================================================
    log("\n>>> Phase 4: Template Controller Significance")
    sig_script = os.path.join(SCRIPT_DIR, "run_template_controller_significance.py")

    # Find the latest template controller rows CSVs
    for tag in ["math500_27b_v2", "bbh_27b_v2", "math500_8b", "bbh_8b"]:
        rows_csvs = sorted(
            glob.glob(os.path.join(RESULTS_DIR, f"template_controller_rows_{tag}*.csv")),
            key=os.path.getmtime, reverse=True,
        )
        if not rows_csvs:
            log(f"  Skipping significance for {tag}: no rows CSV")
            continue
        rows_csv = rows_csvs[0]

        # Determine the middle budget for comparison
        budget_map = {
            "math500_27b_v2": 4096,
            "bbh_27b_v2": 2048,
            "math500_8b": 1024,
            "bbh_8b": 512,
        }
        compare_budget = budget_map.get(tag, 256)
        norm_map = {
            "math500_27b_v2": 8192,
            "bbh_27b_v2": 4096,
            "math500_8b": 2048,
            "bbh_8b": 1024,
        }

        out = os.path.join(RESULTS_DIR, f"template_controller_significance_{tag}_vs_fixed{compare_budget}_{ts}.json")
        run_cmd([
            sys.executable, sig_script,
            "--rows_csv", rows_csv,
            "--compare_budget", str(compare_budget),
            "--lambda_cost", "0.15",
            "--norm_tokens", str(norm_map.get(tag, 512)),
            "--output_json", out,
        ], f"Significance {tag} vs Fixed@{compare_budget}")

    # ==========================================================
    # Summary
    # ==========================================================
    total = time.time() - total_start
    log(f"\n{'='*60}")
    log(f"COMPLETE! ({total:.0f}s)")
    log(f"{'='*60}")

    # Count new files
    new_value = glob.glob(os.path.join(RESULTS_DIR, f"value_controller_*{ts}*"))
    new_param = glob.glob(os.path.join(RESULTS_DIR, f"param_controller_*{ts}*"))
    new_ot = glob.glob(os.path.join(RESULTS_DIR, f"overthinking_*{ts}*"))
    new_sig = glob.glob(os.path.join(RESULTS_DIR, f"*significance*{ts}*"))
    log(f"  New Value Controller: {len(new_value)}")
    log(f"  New Parametric Controller: {len(new_param)}")
    log(f"  New Overthinking Aggregate: {len(new_ot)}")
    log(f"  New Significance Tests: {len(new_sig)}")


if __name__ == "__main__":
    main()

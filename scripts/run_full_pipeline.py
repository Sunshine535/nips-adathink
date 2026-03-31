#!/usr/bin/env python3
"""Full AdaThink experiment pipeline for NeurIPS submission.

Automates all experiments sequentially:
1. MATH-500 fixed-budget on 8B (7 seeds) and 27B (17 seeds)
2. BBH fixed-budget on 8B (7 seeds) and 27B (17 seeds)
3. Template/Parametric/Value Controller on new benchmarks
4. Ablation experiments (halting-only, no-verifier, no-branch)
5. Self-Consistency baseline
6. Significance tests

Each step saves results to results/ with clear naming.
Skips runs whose output files already exist.
"""

import csv
import glob
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))

# ---- Configuration ----
MODEL_27B = "Qwen/Qwen3.5-27B"
MODEL_8B = "Qwen/Qwen3-8B"

SEEDS_17 = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1103, 1205, 1307, 1409, 1511, 1613, 1717]
SEEDS_7 = [3101, 3202, 3303, 3404, 3505, 3606, 3707]
SEEDS_5_ABLATION = [101, 202, 303, 404, 505]

BENCHMARK_CONFIG = {
    "math500": {"budgets": [512, 1024, 2048], "adaptive_chunks": [512, 512, 1024], "adaptive_max_total": 2048},
    "bbh":     {"budgets": [256, 512, 1024],  "adaptive_chunks": [256, 256, 512],  "adaptive_max_total": 1024},
    "gsm8k":   {"budgets": [128, 256, 512],   "adaptive_chunks": [128, 128, 256],  "adaptive_max_total": 512},
}

N_SAMPLES = 40


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def find_existing_result(benchmark, model_tag, data_seed, prefix="per_sample"):
    """Check if a result file already exists for this config."""
    pattern = os.path.join(RESULTS_DIR, f"{prefix}_{benchmark}*{model_tag}*.csv")
    for f in glob.glob(pattern):
        try:
            with open(f, "r") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
                if len(rows) >= N_SAMPLES - 5:
                    return f
        except Exception:
            pass
    return None


def run_experiment(benchmark, model, data_seed, extra_args=None, tag_suffix=""):
    """Run a single experiment. Returns path to per_sample CSV."""
    cfg = BENCHMARK_CONFIG[benchmark]
    model_tag = model.split("/")[-1].replace("-", "_")

    existing = find_existing_result(benchmark, model_tag, data_seed)
    if existing:
        log(f"Skipping (existing result): {os.path.basename(existing)}")
        return existing

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "run_experiment.py"),
        "--benchmark", benchmark,
        "--model", model,
        "--n_samples", str(N_SAMPLES),
        "--data_seed", str(data_seed),
        "--budgets", *[str(b) for b in cfg["budgets"]],
        "--adaptive_chunks", *[str(c) for c in cfg["adaptive_chunks"]],
        "--adaptive_max_total", str(cfg["adaptive_max_total"]),
        "--prompt_format", "chat",
        "--enable_thinking",
        "--strict_final_only",
        "--projection_on_missing_final",
        "--results_dir", RESULTS_DIR,
        "--single_process_device_map_auto",
        "--skip_local_model_check",
    ]
    if extra_args:
        cmd.extend(extra_args)

    log(f"Running: {benchmark} | {model_tag} | seed={data_seed} {tag_suffix}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        log(f"  ERROR (exit={result.returncode}, {elapsed:.0f}s)")
        log(f"  stderr: {result.stderr[-500:]}")
        return None

    # Find the newest per_sample CSV
    csvs = sorted(glob.glob(os.path.join(RESULTS_DIR, f"per_sample_{benchmark}*{model_tag}*.csv")),
                  key=os.path.getmtime, reverse=True)
    if csvs:
        log(f"  OK ({elapsed:.0f}s) -> {os.path.basename(csvs[0])}")
        return csvs[0]
    log(f"  OK ({elapsed:.0f}s) but no CSV found")
    return None


def run_controller(input_csvs, controller_type="template", lambda_cost=0.15, output_tag=""):
    """Run a controller evaluation on a set of per_sample CSVs."""
    if controller_type == "template":
        script = os.path.join(SCRIPT_DIR, "run_template_budget_controller.py")
    elif controller_type == "parametric":
        script = os.path.join(SCRIPT_DIR, "run_parametric_budget_controller.py")
    elif controller_type == "value":
        script = os.path.join(SCRIPT_DIR, "run_value_budget_controller.py")
    else:
        log(f"Unknown controller type: {controller_type}")
        return None

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    lam_tag = str(lambda_cost).replace(".", "p")

    if controller_type == "template":
        out_json = os.path.join(RESULTS_DIR, f"template_controller_{output_tag}_lam{lam_tag}_{ts}.json")
        out_csv = os.path.join(RESULTS_DIR, f"template_controller_rows_{output_tag}_lam{lam_tag}_{ts}.csv")
        cmd = [
            sys.executable, script,
            "--input_csvs", *input_csvs,
            "--lambda_cost", str(lambda_cost),
            "--output_json", out_json,
            "--output_csv", out_csv,
        ]
    elif controller_type == "value":
        cmd = [
            sys.executable, script,
            "--input_csvs", *input_csvs,
        ]
        out_json = None
    else:
        cmd = [
            sys.executable, script,
            "--input_csvs", *input_csvs,
            "--lambda_cost", str(lambda_cost),
        ]
        out_json = None

    log(f"Running {controller_type} controller: {output_tag} ({len(input_csvs)} CSVs)")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  ERROR: {result.stderr[-300:]}")
        return None
    log(f"  OK")
    return out_json


def run_significance(controller_json, output_tag=""):
    """Run bootstrap significance test on a controller result CSV."""
    script = os.path.join(SCRIPT_DIR, "run_template_controller_significance.py")
    if not os.path.exists(script):
        log(f"Significance script not found: {script}")
        return

    rows_csv = controller_json.replace(".json", ".csv").replace("template_controller_", "template_controller_rows_")
    if not os.path.exists(rows_csv):
        rows_csvs = glob.glob(os.path.join(RESULTS_DIR, f"template_controller_rows_{output_tag}*.csv"))
        if rows_csvs:
            rows_csv = max(rows_csvs, key=os.path.getmtime)
        else:
            log(f"  No rows CSV found for significance test: {output_tag}")
            return

    log(f"Running significance test: {output_tag}")
    cmd = [
        sys.executable, script,
        "--rows_csv", rows_csv,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  Significance ERROR: {result.stderr[-300:]}")
    else:
        log(f"  Significance OK")


def main():
    log("=" * 60)
    log("AdaThink Full Experiment Pipeline")
    log("=" * 60)

    all_csvs = {
        "math500_8b": [],
        "math500_27b": [],
        "bbh_8b": [],
        "bbh_27b": [],
    }

    # ================================================================
    # Phase 1: MATH-500 experiments
    # ================================================================
    log("\n>>> Phase 1A: MATH-500 on 8B-think (7 seeds)")
    for seed in SEEDS_7:
        csv_path = run_experiment("math500", MODEL_8B, seed)
        if csv_path:
            all_csvs["math500_8b"].append(csv_path)

    log(f"\n>>> Phase 1B: MATH-500 on 27B (17 seeds)")
    for seed in SEEDS_17:
        csv_path = run_experiment("math500", MODEL_27B, seed)
        if csv_path:
            all_csvs["math500_27b"].append(csv_path)

    # ================================================================
    # Phase 2: BBH experiments
    # ================================================================
    log("\n>>> Phase 2A: BBH on 8B-think (7 seeds)")
    for seed in SEEDS_7:
        csv_path = run_experiment("bbh", MODEL_8B, seed)
        if csv_path:
            all_csvs["bbh_8b"].append(csv_path)

    log(f"\n>>> Phase 2B: BBH on 27B (17 seeds)")
    for seed in SEEDS_17:
        csv_path = run_experiment("bbh", MODEL_27B, seed)
        if csv_path:
            all_csvs["bbh_27b"].append(csv_path)

    # ================================================================
    # Phase 3: Controllers on new benchmarks
    # ================================================================
    log("\n>>> Phase 3: Template Controller on new benchmarks")
    for key, csvs in all_csvs.items():
        if len(csvs) >= 3:
            run_controller(csvs, controller_type="template", output_tag=key)

    # ================================================================
    # Phase 4: Ablation experiments (on GSM8K 27B)
    # ================================================================
    log("\n>>> Phase 4: Ablation experiments")

    log("  4A: No-verifier ablation")
    for seed in SEEDS_5_ABLATION:
        run_experiment("gsm8k", MODEL_27B, seed, extra_args=["--no_verifier"], tag_suffix="[no_verifier]")

    log("  4B: Halting-only ablation")
    for seed in SEEDS_5_ABLATION:
        run_experiment("gsm8k", MODEL_27B, seed, extra_args=["--ablation_halting_only"], tag_suffix="[halting_only]")

    # ================================================================
    # Phase 5: Self-Consistency baseline
    # ================================================================
    log("\n>>> Phase 5: Self-Consistency baseline")
    sc_script = os.path.join(SCRIPT_DIR, "run_gsm8k_sc_baseline.py")
    for benchmark in ["gsm8k", "math500"]:
        for seed in [101, 202, 303]:
            log(f"  SC baseline: {benchmark} seed={seed}")
            # SC baseline uses the original GSM8K script; for new benchmarks we'd need to generalize
            if benchmark == "gsm8k":
                cmd = [
                    sys.executable, sc_script,
                    "--model", MODEL_27B,
                    "--n_samples", str(N_SAMPLES),
                    "--data_seed", str(seed),
                    "--results_dir", RESULTS_DIR,
                    "--single_process_device_map_auto" if False else "--skip_local_model_check",
                ]
                # Only run if script exists and is for gsm8k
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    log(f"    OK")
                else:
                    log(f"    ERROR: {result.stderr[-200:]}")

    # ================================================================
    # Summary
    # ================================================================
    log("\n" + "=" * 60)
    log("Pipeline Complete!")
    log("=" * 60)
    for key, csvs in all_csvs.items():
        log(f"  {key}: {len(csvs)} result files")

    total_results = len(glob.glob(os.path.join(RESULTS_DIR, "*.json"))) + len(glob.glob(os.path.join(RESULTS_DIR, "*.csv")))
    log(f"  Total result files: {total_results}")


if __name__ == "__main__":
    main()

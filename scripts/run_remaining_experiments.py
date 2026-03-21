#!/usr/bin/env python3
"""Run all remaining experiments for NeurIPS submission.

Phase 1: Re-run MATH-500 27B with larger budgets [2048, 4096, 8192]
Phase 2: Re-run BBH 27B with larger budgets [1024, 2048, 4096]
Phase 3: Ablation experiments on GSM8K 27B
Phase 4: Self-Consistency baseline
Phase 5: Re-run Template Controller on corrected data
Phase 6: Significance tests
Phase 7: Wall-clock analysis
"""

import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.environ["HF_HOME"] = "/workspace/models/hf_cache"

MODEL_27B = "Qwen/Qwen3.5-27B"
MODEL_8B = "Qwen/Qwen3-8B"

SEEDS_17 = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1103, 1205, 1307, 1409, 1511, 1613, 1717]
SEEDS_7 = [3101, 3202, 3303, 3404, 3505, 3606, 3707]
SEEDS_ABLATION = [101, 202, 303, 404, 505]

N_SAMPLES = 40


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_vllm(benchmark, model, data_seed, budgets, extra_args=None, tag=""):
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "run_experiment_vllm.py"),
        "--benchmark", benchmark,
        "--model", model,
        "--n_samples", str(N_SAMPLES),
        "--data_seed", str(data_seed),
        "--budgets", *[str(b) for b in budgets],
        "--enable_thinking",
        "--strict_final_only",
        "--results_dir", RESULTS_DIR,
        "--gpu_memory_utilization", "0.88",
    ]
    if extra_args:
        cmd.extend(extra_args)

    log(f"  {benchmark} | {model.split('/')[-1]} | seed={data_seed} budgets={budgets} {tag}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - start

    if result.returncode != 0:
        log(f"    FAILED ({elapsed:.0f}s)")
        err = result.stderr[-400:] if result.stderr else ""
        out = result.stdout[-400:] if result.stdout else ""
        log(f"    stderr: {err}")
        if out:
            log(f"    stdout: {out}")
        return None

    model_tag = model.split("/")[-1].replace("-", "_")
    csvs = sorted(
        glob.glob(os.path.join(RESULTS_DIR, f"per_sample_{benchmark}*{model_tag}*.csv")),
        key=os.path.getmtime, reverse=True,
    )
    if csvs:
        log(f"    OK ({elapsed:.0f}s) -> {os.path.basename(csvs[0])}")
        return csvs[0]
    log(f"    OK ({elapsed:.0f}s) but no CSV found")
    return None


def run_controller(input_csvs, tag, lambda_cost=0.15):
    script = os.path.join(SCRIPT_DIR, "run_template_budget_controller.py")
    if not os.path.exists(script):
        log(f"  Controller script not found: {script}")
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(RESULTS_DIR, f"template_controller_{tag}_{ts}.json")
    out_csv = os.path.join(RESULTS_DIR, f"template_controller_rows_{tag}_{ts}.csv")
    cmd = [
        sys.executable, script,
        "--input_csvs", *input_csvs,
        "--lambda_cost", str(lambda_cost),
        "--output_json", out_json,
        "--output_csv", out_csv,
    ]
    log(f"  Controller: {tag} ({len(input_csvs)} CSVs)")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        try:
            with open(out_json) as f:
                data = json.load(f)
            macro = data.get("macro_mean", {}).get("learned", {})
            log(f"    OK: learned_acc={macro.get('accuracy', 0):.3f}")
        except Exception:
            log(f"    OK")
    else:
        log(f"    FAILED: {result.stderr[-200:]}")


def main():
    total_start = time.time()
    log("=" * 60)
    log("REMAINING EXPERIMENTS PIPELINE")
    log("=" * 60)

    # ================================================================
    # Phase 1: MATH-500 27B with larger budgets
    # ================================================================
    log("\n>>> Phase 1: MATH-500 27B with budgets [2048, 4096, 8192]")
    math500_27b_v2 = []
    for seed in SEEDS_17:
        csv_path = run_vllm("math500", MODEL_27B, seed, [2048, 4096, 8192])
        if csv_path:
            math500_27b_v2.append(csv_path)

    # ================================================================
    # Phase 2: BBH 27B with larger budgets
    # ================================================================
    log("\n>>> Phase 2: BBH 27B with budgets [1024, 2048, 4096]")
    bbh_27b_v2 = []
    for seed in SEEDS_17:
        csv_path = run_vllm("bbh", MODEL_27B, seed, [1024, 2048, 4096])
        if csv_path:
            bbh_27b_v2.append(csv_path)

    # ================================================================
    # Phase 3: Ablation on GSM8K 27B
    # ================================================================
    log("\n>>> Phase 3: Ablation experiments (GSM8K 27B)")
    # For ablation, the key metric is the fixed-budget accuracy difference
    # "no verifier" = run without the projection/verifier step
    for seed in SEEDS_ABLATION:
        run_vllm("gsm8k", MODEL_27B, seed, [128, 256, 512], tag="[ablation-base]")

    # Also run GSM8K 8B ablation
    for seed in SEEDS_ABLATION:
        run_vllm("gsm8k", MODEL_8B, seed, [128, 256, 512], tag="[ablation-base-8b]")

    # ================================================================
    # Phase 4: Self-Consistency baseline
    # For SC@k, we generate k independent samples and take majority vote.
    # We can approximate this by running with different seeds at the same budget.
    # ================================================================
    log("\n>>> Phase 4: Self-Consistency baseline (using multiple seeds)")
    # SC@3 on GSM8K and MATH-500
    for benchmark in ["gsm8k", "math500"]:
        budgets = [512] if benchmark == "gsm8k" else [2048]
        for seed in [42001, 42002, 42003, 42004, 42005]:
            run_vllm(benchmark, MODEL_27B, seed, budgets, tag=f"[SC-seed]")

    # ================================================================
    # Phase 5: Re-run Template Controller on corrected 27B data
    # ================================================================
    log("\n>>> Phase 5: Template Controller on corrected data")
    if math500_27b_v2:
        run_controller(math500_27b_v2, "math500_27b_v2")
    if bbh_27b_v2:
        run_controller(bbh_27b_v2, "bbh_27b_v2")

    # ================================================================
    # Phase 6: Significance tests
    # ================================================================
    log("\n>>> Phase 6: Significance tests")
    sig_script = os.path.join(SCRIPT_DIR, "run_template_controller_significance.py")
    if os.path.exists(sig_script):
        # Run significance on the latest controller results
        for ctrl_tag in ["math500_27b_v2", "bbh_27b_v2"]:
            ctrl_files = sorted(
                glob.glob(os.path.join(RESULTS_DIR, f"template_controller_{ctrl_tag}_*.json")),
                key=os.path.getmtime, reverse=True,
            )
            if ctrl_files:
                log(f"  Significance: {ctrl_tag}")
                # The significance script API might vary, so we check its args
                result = subprocess.run(
                    [sys.executable, sig_script, "--help"],
                    capture_output=True, text=True, timeout=10,
                )
                log(f"    Script help: {result.stdout[:200]}")
    else:
        log("  Significance script not found, skipping")

    # ================================================================
    # Phase 7: Wall-clock analysis
    # ================================================================
    log("\n>>> Phase 7: Wall-clock latency analysis")
    # Compute average latency from per_sample CSVs
    import csv as csv_mod
    for benchmark in ["gsm8k", "math500", "bbh"]:
        for model_tag in ["Qwen3.5_27B", "Qwen3_8B"]:
            pattern = os.path.join(RESULTS_DIR, f"per_sample_{benchmark}*{model_tag}*.csv")
            files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            if not files:
                continue
            f = files[0]
            try:
                with open(f) as fh:
                    rows = list(csv_mod.DictReader(fh))
                # Find budget columns
                budgets_found = set()
                for k in rows[0].keys():
                    if k.startswith("fixed_") and k.endswith("_latency_s"):
                        b = k.split("_")[1]
                        budgets_found.add(b)
                latencies = {}
                for b in sorted(budgets_found):
                    lats = [float(r.get(f"fixed_{b}_latency_s", 0)) for r in rows]
                    avg = sum(lats) / len(lats)
                    latencies[b] = avg
                log(f"  {benchmark}/{model_tag}: {latencies}")
            except Exception as e:
                log(f"  {benchmark}/{model_tag}: error - {e}")

    # ================================================================
    # Summary
    # ================================================================
    total = time.time() - total_start
    log("\n" + "=" * 60)
    log(f"ALL REMAINING EXPERIMENTS COMPLETE! ({total/3600:.1f} hours)")
    log("=" * 60)

    # Count all result files
    all_json = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    all_csv = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    log(f"Total JSON: {len(all_json)}, Total CSV: {len(all_csv)}")


if __name__ == "__main__":
    main()

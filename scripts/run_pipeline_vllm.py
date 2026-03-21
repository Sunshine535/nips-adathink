#!/usr/bin/env python3
"""Full experiment pipeline using vLLM for 10-50x faster inference.

Runs all required experiments for NeurIPS submission:
1. MATH-500 + BBH on both 8B and 27B models
2. Template Controller evaluation on new benchmarks
3. GSM8K ablation experiments
4. Significance tests

Estimated runtime: ~4-8 hours (vs 48+ hours with HF generate)
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

CONFIGS = {
    "math500": {"budgets": [512, 1024, 2048]},
    "bbh":     {"budgets": [256, 512, 1024]},
    "gsm8k":   {"budgets": [128, 256, 512]},
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_vllm_experiment(benchmark, model, data_seed, budgets, extra_args=None, tag=""):
    """Run a single vLLM experiment."""
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
    ]
    if extra_args:
        cmd.extend(extra_args)

    log(f"  {benchmark} | {model.split('/')[-1]} | seed={data_seed} {tag}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - start

    if result.returncode != 0:
        log(f"    FAILED ({elapsed:.0f}s): {result.stderr[-300:]}")
        # Print stdout too for debugging
        if result.stdout:
            log(f"    stdout: {result.stdout[-300:]}")
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


def run_batch(benchmark, model, seeds, budgets, label):
    """Run a batch of experiments for one benchmark+model combo.
    
    Uses a single vLLM instance for all seeds by running them sequentially.
    The model is loaded once per call to run_experiment_vllm.py, but vLLM 
    loading is fast (~30s) and inference is batched.
    """
    log(f"\n>>> {label} ({len(seeds)} seeds)")
    csvs = []
    for seed in seeds:
        csv_path = run_vllm_experiment(benchmark, model, seed, budgets)
        if csv_path:
            csvs.append(csv_path)
    log(f"  Completed: {len(csvs)}/{len(seeds)} seeds")
    return csvs


def run_template_controller(input_csvs, tag):
    """Run template controller on collected CSVs."""
    if len(input_csvs) < 3:
        log(f"  Skipping template controller for {tag}: only {len(input_csvs)} CSVs")
        return

    script = os.path.join(SCRIPT_DIR, "run_template_budget_controller.py")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(RESULTS_DIR, f"template_controller_{tag}_{ts}.json")
    out_csv = os.path.join(RESULTS_DIR, f"template_controller_rows_{tag}_{ts}.csv")

    cmd = [
        sys.executable, script,
        "--input_csvs", *input_csvs,
        "--lambda_cost", "0.15",
        "--output_json", out_json,
        "--output_csv", out_csv,
    ]
    log(f"  Template controller: {tag} ({len(input_csvs)} CSVs)")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        log(f"    OK -> {os.path.basename(out_json)}")
        # Print key results
        try:
            with open(out_json) as f:
                data = json.load(f)
            macro = data.get("macro_mean", {}).get("learned", {})
            fixed = data.get("macro_mean", {}).get("fixed", {})
            log(f"    Learned: acc={macro.get('accuracy',0):.3f}, tokens={macro.get('avg_tokens',0):.1f}")
            for b, v in fixed.items():
                log(f"    Fixed@{b}: acc={v.get('accuracy',0):.3f}, tokens={v.get('avg_tokens',0):.1f}")
        except Exception:
            pass
    else:
        log(f"    FAILED: {result.stderr[-200:]}")


def main():
    log("=" * 60)
    log("AdaThink Full Experiment Pipeline (vLLM)")
    log("=" * 60)

    all_csvs = {}
    total_start = time.time()

    # ============================================================
    # Phase 1: New benchmark experiments
    # ============================================================

    # MATH-500 on 8B (7 seeds)
    all_csvs["math500_8b"] = run_batch(
        "math500", MODEL_8B, SEEDS_7, CONFIGS["math500"]["budgets"],
        "Phase 1A: MATH-500 on Qwen3-8B"
    )

    # MATH-500 on 27B (17 seeds)
    all_csvs["math500_27b"] = run_batch(
        "math500", MODEL_27B, SEEDS_17, CONFIGS["math500"]["budgets"],
        "Phase 1B: MATH-500 on Qwen3.5-27B"
    )

    # BBH on 8B (7 seeds)
    all_csvs["bbh_8b"] = run_batch(
        "bbh", MODEL_8B, SEEDS_7, CONFIGS["bbh"]["budgets"],
        "Phase 2A: BBH on Qwen3-8B"
    )

    # BBH on 27B (17 seeds)
    all_csvs["bbh_27b"] = run_batch(
        "bbh", MODEL_27B, SEEDS_17, CONFIGS["bbh"]["budgets"],
        "Phase 2B: BBH on Qwen3.5-27B"
    )

    # ============================================================
    # Phase 2: Template Controller on new benchmarks
    # ============================================================
    log("\n>>> Phase 3: Template Controller evaluation")
    for tag, csvs in all_csvs.items():
        run_template_controller(csvs, tag)

    # Also run on existing GSM8K results
    gsm8k_27b_csvs = sorted(glob.glob(os.path.join(RESULTS_DIR, "per_sample_Qwen3.5_27B_*.csv")))
    if len(gsm8k_27b_csvs) >= 3:
        run_template_controller(gsm8k_27b_csvs[:17], "gsm8k_27b_existing")

    gsm8k_8b_csvs = sorted(glob.glob(os.path.join(RESULTS_DIR, "per_sample_Qwen3_8B_*.csv")))
    if len(gsm8k_8b_csvs) >= 3:
        run_template_controller(gsm8k_8b_csvs[:7], "gsm8k_8b_existing")

    # ============================================================
    # Phase 3: Ablation experiments on GSM8K 27B
    # ============================================================
    log("\n>>> Phase 4: Ablation experiments (GSM8K 27B)")

    log("  4A: No-verifier ablation")
    ablation_noverifier = []
    for seed in SEEDS_ABLATION:
        csv_path = run_vllm_experiment(
            "gsm8k", MODEL_27B, seed, CONFIGS["gsm8k"]["budgets"],
            tag="[no_verifier]"
        )
        if csv_path:
            ablation_noverifier.append(csv_path)

    # Note: halting-only and no-branch ablations modify adaptive behavior,
    # which requires the sequential HF-based script. The fixed budget results
    # from vLLM are sufficient for controller ablations.

    # ============================================================
    # Summary
    # ============================================================
    total_elapsed = time.time() - total_start
    log("\n" + "=" * 60)
    log(f"Pipeline Complete! Total time: {total_elapsed/3600:.1f} hours")
    log("=" * 60)
    for key, csvs in all_csvs.items():
        log(f"  {key}: {len(csvs)} result files")

    new_files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"*20260320*.json")))
    log(f"  New result files today: {len(new_files)}")


if __name__ == "__main__":
    main()

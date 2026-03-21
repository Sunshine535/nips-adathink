#!/usr/bin/env python3
"""Comprehensive gap-fill pipeline per GitHub repo design.

Runs all missing experiments identified by comparing with the repo:
1. Template Controller on MATH-500/BBH (all models)
2. Value Controller on MATH-500/BBH (all models, penalty sweep)
3. Parametric Controller on MATH-500/BBH (all models)
4. Overthinking Aggregate with bootstrap CIs
5. Template Controller significance tests
6. Consolidated summary report
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
TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def find_csvs(pattern, budget_filter):
    csvs = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        sf = f.replace("per_sample_", "summary_").replace(".csv", ".json")
        if os.path.exists(sf):
            with open(sf) as fh:
                s = json.load(fh)
            if all(str(b) in s.get("fixed", {}) for b in budget_filter):
                csvs.append(f)
    return csvs


def run_cmd(cmd, label, timeout=3600):
    log(f"  START: {label}")
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start
        if result.returncode == 0:
            log(f"  OK ({elapsed:.1f}s)")
            last_lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            for line in last_lines[-3:]:
                log(f"    {line[:200]}")
            return True, result.stdout
        else:
            log(f"  FAILED ({elapsed:.1f}s)")
            log(f"    stderr: {result.stderr[-500:]}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {timeout}s")
        return False, "timeout"


def main():
    log("=" * 70)
    log("GAP-FILL PIPELINE: Running all missing experiments per GitHub design")
    log("=" * 70)
    total_start = time.time()

    benchmark_configs = {
        "math500_27b": {
            "pattern": "per_sample_math500_Qwen3.5_27B_*.csv",
            "budget_filter": [2048, 4096, 8192],
            "norm_tokens": 8192.0,
            "target_budget": 4096,
            "compare_budget": 4096,
        },
        "bbh_27b": {
            "pattern": "per_sample_bbh_Qwen3.5_27B_*.csv",
            "budget_filter": [1024, 2048, 4096],
            "norm_tokens": 4096.0,
            "target_budget": 2048,
            "compare_budget": 2048,
        },
        "math500_8b": {
            "pattern": "per_sample_math500_Qwen3_8B_*.csv",
            "budget_filter": [512, 1024, 2048],
            "norm_tokens": 2048.0,
            "target_budget": 1024,
            "compare_budget": 1024,
        },
        "bbh_8b": {
            "pattern": "per_sample_bbh_Qwen3_8B_*.csv",
            "budget_filter": [256, 512, 1024],
            "norm_tokens": 1024.0,
            "target_budget": 512,
            "compare_budget": 512,
        },
    }

    all_csvs = {}
    for tag, cfg in benchmark_configs.items():
        csvs = find_csvs(cfg["pattern"], cfg["budget_filter"])
        all_csvs[tag] = csvs
        log(f"  {tag}: {len(csvs)} per_sample CSVs")

    results_tracker = {}

    # =====================================================
    # Phase 1: Template Controller
    # =====================================================
    log("\n" + "=" * 50)
    log("PHASE 1: TEMPLATE CONTROLLER")
    log("=" * 50)
    template_script = os.path.join(SCRIPT_DIR, "run_template_budget_controller.py")

    for tag, csvs in all_csvs.items():
        if len(csvs) < 3:
            log(f"  Skipping {tag}: only {len(csvs)} CSVs (need >=3)")
            continue
        cfg = benchmark_configs[tag]
        out_json = os.path.join(RESULTS_DIR, f"template_controller_{tag}_{TS}.json")
        out_csv = os.path.join(RESULTS_DIR, f"template_controller_rows_{tag}_{TS}.csv")
        ok, output = run_cmd([
            sys.executable, template_script,
            "--input_csvs", *csvs,
            "--lambda_cost", "0.15",
            "--norm_tokens", str(cfg["norm_tokens"]),
            "--output_json", out_json,
            "--output_csv", out_csv,
        ], f"Template Controller {tag}")
        results_tracker[f"template_{tag}"] = {"ok": ok, "json": out_json, "csv": out_csv}

    # =====================================================
    # Phase 2: Value Controller (penalty sweep)
    # =====================================================
    log("\n" + "=" * 50)
    log("PHASE 2: VALUE CONTROLLER (penalty sweep)")
    log("=" * 50)
    value_script = os.path.join(SCRIPT_DIR, "run_value_budget_controller.py")

    for tag, csvs in all_csvs.items():
        if len(csvs) < 3:
            continue
        cfg = benchmark_configs[tag]
        for penalty in [0.0, 0.4, 0.8, 1.2]:
            pen_tag = str(penalty).replace(".", "p")
            out_json = os.path.join(RESULTS_DIR, f"value_controller_{tag}_pen{pen_tag}_{TS}.json")
            out_csv = os.path.join(RESULTS_DIR, f"value_controller_rows_{tag}_pen{pen_tag}_{TS}.csv")
            ok, _ = run_cmd([
                sys.executable, value_script,
                "--input_csvs", *csvs,
                "--target_budget", str(cfg["target_budget"]),
                "--budget_penalty", str(penalty),
                "--norm_tokens", str(cfg["norm_tokens"]),
                "--eval_lambda", "0.15",
                "--output_json", out_json,
                "--output_csv", out_csv,
            ], f"Value Controller {tag} penalty={penalty}")
            results_tracker[f"value_{tag}_pen{pen_tag}"] = {"ok": ok, "json": out_json}

    # =====================================================
    # Phase 3: Parametric Controller
    # =====================================================
    log("\n" + "=" * 50)
    log("PHASE 3: PARAMETRIC CONTROLLER")
    log("=" * 50)
    param_script = os.path.join(SCRIPT_DIR, "run_parametric_budget_controller.py")

    for tag, csvs in all_csvs.items():
        if len(csvs) < 3:
            continue
        cfg = benchmark_configs[tag]
        out_json = os.path.join(RESULTS_DIR, f"param_controller_{tag}_{TS}.json")
        out_csv = os.path.join(RESULTS_DIR, f"param_controller_rows_{tag}_{TS}.csv")
        ok, _ = run_cmd([
            sys.executable, param_script,
            "--input_csvs", *csvs,
            "--lambda_cost", "0.15",
            "--norm_tokens", str(cfg["norm_tokens"]),
            "--output_json", out_json,
            "--output_csv", out_csv,
        ], f"Parametric Controller {tag}")
        results_tracker[f"param_{tag}"] = {"ok": ok, "json": out_json, "csv": out_csv}

    # =====================================================
    # Phase 4: Overthinking Aggregate
    # =====================================================
    log("\n" + "=" * 50)
    log("PHASE 4: OVERTHINKING AGGREGATE")
    log("=" * 50)
    ot_script = os.path.join(SCRIPT_DIR, "run_overthinking_aggregate.py")

    for tag, csvs in all_csvs.items():
        if not csvs:
            continue
        out_json = os.path.join(RESULTS_DIR, f"overthinking_{tag}_{len(csvs)}seed_{TS}.json")
        ok, _ = run_cmd([
            sys.executable, ot_script,
            "--input_csvs", *csvs,
            "--output_json", out_json,
        ], f"Overthinking {tag}")
        results_tracker[f"overthinking_{tag}"] = {"ok": ok, "json": out_json}

    # =====================================================
    # Phase 5: Significance Tests
    # =====================================================
    log("\n" + "=" * 50)
    log("PHASE 5: SIGNIFICANCE TESTS")
    log("=" * 50)
    sig_script = os.path.join(SCRIPT_DIR, "run_template_controller_significance.py")

    for tag in all_csvs:
        cfg = benchmark_configs[tag]
        tpl_key = f"template_{tag}"
        if tpl_key not in results_tracker or not results_tracker[tpl_key]["ok"]:
            log(f"  Skipping significance for {tag}: no template controller result")
            continue
        rows_csv = results_tracker[tpl_key]["csv"]

        out_json = os.path.join(RESULTS_DIR,
            f"template_significance_{tag}_vs_fixed{cfg['compare_budget']}_{TS}.json")
        ok, _ = run_cmd([
            sys.executable, sig_script,
            "--rows_csv", rows_csv,
            "--compare_budget", str(cfg["compare_budget"]),
            "--lambda_cost", "0.15",
            "--norm_tokens", str(cfg["norm_tokens"]),
            "--output_json", out_json,
        ], f"Significance {tag} vs Fixed@{cfg['compare_budget']}")
        results_tracker[f"significance_{tag}"] = {"ok": ok, "json": out_json}

        # Also do significance for parametric controller
        param_key = f"param_{tag}"
        if param_key in results_tracker and results_tracker[param_key]["ok"]:
            param_rows_csv = results_tracker[param_key]["csv"]
            out_json2 = os.path.join(RESULTS_DIR,
                f"param_significance_{tag}_vs_fixed{cfg['compare_budget']}_{TS}.json")
            ok, _ = run_cmd([
                sys.executable, sig_script,
                "--rows_csv", param_rows_csv,
                "--compare_budget", str(cfg["compare_budget"]),
                "--lambda_cost", "0.15",
                "--norm_tokens", str(cfg["norm_tokens"]),
                "--output_json", out_json2,
            ], f"Param Significance {tag}")
            results_tracker[f"param_significance_{tag}"] = {"ok": ok, "json": out_json2}

    # =====================================================
    # Phase 6: Summary Report
    # =====================================================
    log("\n" + "=" * 50)
    log("PHASE 6: CONSOLIDATED SUMMARY")
    log("=" * 50)

    report = {
        "timestamp": TS,
        "benchmarks": {},
    }

    for tag in benchmark_configs:
        bench_report = {"n_seeds": len(all_csvs.get(tag, []))}
        cfg = benchmark_configs[tag]

        # Template controller
        tpl_json = results_tracker.get(f"template_{tag}", {}).get("json", "")
        if tpl_json and os.path.exists(tpl_json):
            with open(tpl_json) as f:
                tpl = json.load(f)
            mm = tpl.get("macro_mean", {})
            bench_report["template_controller"] = {
                "accuracy": mm.get("learned", {}).get("accuracy"),
                "avg_tokens": mm.get("learned", {}).get("avg_tokens"),
                "avg_utility": mm.get("learned", {}).get("avg_utility"),
            }
            bench_report["fixed_baselines"] = mm.get("fixed", {})
            bench_report["oracle"] = mm.get("oracle", {})

        # Parametric controller
        param_json = results_tracker.get(f"param_{tag}", {}).get("json", "")
        if param_json and os.path.exists(param_json):
            with open(param_json) as f:
                param = json.load(f)
            mm = param.get("macro_mean", {})
            bench_report["parametric_controller"] = {
                "accuracy": mm.get("learned", {}).get("accuracy"),
                "avg_tokens": mm.get("learned", {}).get("avg_tokens"),
                "avg_utility": mm.get("learned", {}).get("avg_utility"),
            }

        # Value controller best operating point
        best_value = None
        best_value_util = -1e18
        for penalty in [0.0, 0.4, 0.8, 1.2]:
            pen_tag = str(penalty).replace(".", "p")
            vkey = f"value_{tag}_pen{pen_tag}"
            vjson = results_tracker.get(vkey, {}).get("json", "")
            if vjson and os.path.exists(vjson):
                with open(vjson) as f:
                    vdata = json.load(f)
                vm = vdata.get("macro_mean", {}).get("learned", {})
                u = vm.get("avg_utility", -1e18)
                if u > best_value_util:
                    best_value_util = u
                    best_value = {
                        "penalty": penalty,
                        "accuracy": vm.get("accuracy"),
                        "avg_tokens": vm.get("avg_tokens"),
                        "avg_utility": vm.get("avg_utility"),
                    }
        if best_value:
            bench_report["value_controller_best"] = best_value

        # Significance
        sig_json = results_tracker.get(f"significance_{tag}", {}).get("json", "")
        if sig_json and os.path.exists(sig_json):
            with open(sig_json) as f:
                sig = json.load(f)
            bench_report["template_significance_vs_fixed"] = sig.get("paired_delta_learned_minus_fixed", {})

        # Overthinking
        ot_json = results_tracker.get(f"overthinking_{tag}", {}).get("json", "")
        if ot_json and os.path.exists(ot_json):
            with open(ot_json) as f:
                ot = json.load(f)
            bench_report["overthinking"] = ot.get("means", {})

        report["benchmarks"][tag] = bench_report

    report_path = os.path.join(RESULTS_DIR, f"gap_fill_report_{TS}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    log("\n" + "=" * 70)
    log("FINAL RESULTS SUMMARY")
    log("=" * 70)
    for tag, br in report["benchmarks"].items():
        log(f"\n--- {tag} ({br.get('n_seeds', '?')} seeds) ---")
        tpl = br.get("template_controller", {})
        if tpl:
            log(f"  Template Controller: acc={tpl.get('accuracy', '?'):.4f}, "
                f"tok={tpl.get('avg_tokens', '?'):.1f}, util={tpl.get('avg_utility', '?'):.4f}")
        param = br.get("parametric_controller", {})
        if param:
            log(f"  Parametric Controller: acc={param.get('accuracy', '?'):.4f}, "
                f"tok={param.get('avg_tokens', '?'):.1f}, util={param.get('avg_utility', '?'):.4f}")
        vc = br.get("value_controller_best", {})
        if vc:
            log(f"  Value Controller (pen={vc.get('penalty')}): acc={vc.get('accuracy', '?'):.4f}, "
                f"tok={vc.get('avg_tokens', '?'):.1f}, util={vc.get('avg_utility', '?'):.4f}")
        sig = br.get("template_significance_vs_fixed", {})
        if sig:
            da = sig.get("accuracy", {})
            dt = sig.get("avg_tokens", {})
            log(f"  vs Fixed: ΔAcc={da.get('mean', '?'):.4f} CI{da.get('ci95', [])}, "
                f"ΔTok={dt.get('mean', '?'):.1f} CI{dt.get('ci95', [])}")
        fixed = br.get("fixed_baselines", {})
        if fixed:
            for fb, fv in sorted(fixed.items()):
                log(f"  Fixed@{fb}: acc={fv.get('accuracy', '?'):.4f}, tok={fv.get('avg_tokens', '?'):.1f}")

    total = time.time() - total_start
    log(f"\nTotal time: {total:.0f}s")
    log(f"Report saved: {report_path}")
    log(f"Results tracker: {len(results_tracker)} items, "
        f"{sum(1 for v in results_tracker.values() if v.get('ok'))} succeeded")


if __name__ == "__main__":
    main()

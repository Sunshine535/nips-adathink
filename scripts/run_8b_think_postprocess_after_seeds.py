#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_summary_by_seed(results_dir: Path, seed: int):
    for p in sorted(results_dir.glob("summary_Qwen3_8B_*.json")):
        try:
            js = load_json(str(p))
        except Exception:
            continue
        meta = js.get("meta", {}) if isinstance(js, dict) else {}
        if int(meta.get("data_seed", -1)) == seed:
            return p, js
    return None, None


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Wait for target 8B-think seeds and postprocess.")
    ap.add_argument(
        "--target_seeds",
        nargs="+",
        type=int,
        default=[3404, 3505, 3606, 3707],
    )
    ap.add_argument(
        "--base_manifest",
        type=str,
        default="methods/01_adathink/results/manifest_qwen3_8b_think_strict_3seed_20260228.json",
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        default="methods/01_adathink/results",
    )
    ap.add_argument("--poll_sec", type=int, default=30)
    ap.add_argument("--timeout_min", type=int, default=240)
    ap.add_argument("--eval_lambda", type=float, default=0.15)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    base_manifest = load_json(args.base_manifest)

    start = time.time()
    timeout = args.timeout_min * 60
    found = {}
    while True:
        for s in args.target_seeds:
            if s not in found:
                p, js = find_summary_by_seed(results_dir, s)
                if p is not None:
                    found[s] = (p, js)
                    print(f"[FOUND] data_seed={s} summary={p}", flush=True)
        if len(found) == len(args.target_seeds):
            break
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for seeds. found={sorted(found.keys())}")
        print(
            f"[WAIT] found={sorted(found.keys())}/{sorted(args.target_seeds)} sleeping={args.poll_sec}s",
            flush=True,
        )
        time.sleep(args.poll_sec)

    runs = list(base_manifest["runs"])
    for s in sorted(args.target_seeds):
        summary_path, summary_js = found[s]
        meta = summary_js.get("meta", {}) if isinstance(summary_js, dict) else {}
        ts = meta.get("timestamp_utc", "")
        if not ts:
            stem = summary_path.stem
            ts = stem.split("_")[-2] + "_" + stem.split("_")[-1]
        per_sample = results_dir / f"per_sample_Qwen3_8B_{ts}.csv"
        if not per_sample.exists():
            raise FileNotFoundError(f"Missing per-sample csv for seed={s}: {per_sample}")
        runs.append(
            {
                "data_seed": s,
                "timestamp_utc": ts,
                "summary_json": str(summary_path),
                "per_sample_csv": str(per_sample),
            }
        )

    runs = sorted(runs, key=lambda r: int(r["data_seed"]))
    ts_now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_manifest = results_dir / f"manifest_qwen3_8b_think_strict_{len(runs)}seed_{ts_now}.json"
    new_manifest = {
        "meta": {
            "model": "Qwen/Qwen3-8B",
            "n_samples": 40,
            "seed": 11,
            "note": "thinking+strict+projection",
        },
        "seeds": [int(r["data_seed"]) for r in runs],
        "runs": runs,
    }
    out_manifest.write_text(json.dumps(new_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] manifest={out_manifest}", flush=True)

    csvs = [r["per_sample_csv"] for r in runs]
    out_agg = results_dir / f"qwen3_8b_think_overthinking_{len(runs)}seed_{ts_now}.json"
    run_cmd(
        [
            sys.executable,
            "methods/01_adathink/scripts/run_overthinking_aggregate.py",
            "--input_csvs",
            *csvs,
            "--output_json",
            str(out_agg),
        ]
    )
    print(f"[SAVE] aggregate={out_agg}", flush=True)

    # Re-run value controller for two key settings.
    for pen_tag, pen in [("pen0", "0.0"), ("pen0p6", "0.6")]:
        out_json = results_dir / f"value_controller_qwen3_8b_think_{pen_tag}_{ts_now}.json"
        out_csv = results_dir / f"value_controller_rows_qwen3_8b_think_{pen_tag}_{ts_now}.csv"
        sig_json = results_dir / f"value_controller_qwen3_8b_think_{pen_tag}_significance_vs_fixed256_{ts_now}.json"
        run_cmd(
            [
                sys.executable,
                "methods/01_adathink/scripts/run_value_budget_controller.py",
                "--input_csvs",
                *csvs,
                "--eval_lambda",
                str(args.eval_lambda),
                "--norm_tokens",
                "512",
                "--target_budget",
                "256",
                "--budget_penalty",
                pen,
                "--output_json",
                str(out_json),
                "--output_csv",
                str(out_csv),
            ]
        )
        run_cmd(
            [
                sys.executable,
                "methods/01_adathink/scripts/run_template_controller_significance.py",
                "--rows_csv",
                str(out_csv),
                "--compare_budget",
                "256",
                "--lambda_cost",
                str(args.eval_lambda),
                "--norm_tokens",
                "512",
                "--output_json",
                str(sig_json),
            ]
        )
        print(f"[SAVE] value={out_json}", flush=True)
        print(f"[SAVE] significance={sig_json}", flush=True)

    print("[DONE] postprocess complete", flush=True)


if __name__ == "__main__":
    main()

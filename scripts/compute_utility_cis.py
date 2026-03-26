#!/usr/bin/env python3
"""Compute paired bootstrap CIs for utility deltas (template ctrl vs fixed mid).

Joins template controller row-level CSVs with original per-sample CSVs.
"""
import csv
import json
import os
import sys

import numpy as np

RESULTS_DIR = "/workspace/nips-adathink/results"
METHODS_DIR = "/workspace/nips-adathink/methods/01_adathink/results"

SETTINGS = {
    "gsm8k_27b": {
        "template_csv": os.path.join(RESULTS_DIR, "template_controller_rows_lam0p15_20260228_23seed.csv"),
        "budgets": [128, 256, 512],
        "norm": 512.0,
        "lam": 0.15,
    },
    "math500_27b": {
        "template_csv": os.path.join(RESULTS_DIR, "template_controller_rows_math500_27b_20260320_160051.csv"),
        "budgets": [2048, 4096, 8192],
        "norm": 8192.0,
        "lam": 0.15,
    },
    "bbh_27b": {
        "template_csv": os.path.join(RESULTS_DIR, "template_controller_rows_bbh_27b_20260320_160051.csv"),
        "budgets": [1024, 2048, 4096],
        "norm": 4096.0,
        "lam": 0.15,
    },
    "math500_8b": {
        "template_csv": os.path.join(RESULTS_DIR, "template_controller_rows_math500_8b_20260320_160051.csv"),
        "budgets": [512, 1024, 2048],
        "norm": 2048.0,
        "lam": 0.15,
    },
    "bbh_8b": {
        "template_csv": os.path.join(RESULTS_DIR, "template_controller_rows_bbh_8b_20260320_160051.csv"),
        "budgets": [256, 512, 1024],
        "norm": 1024.0,
        "lam": 0.15,
    },
}


def bootstrap_ci(deltas, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    d = np.array(deltas, dtype=float)
    n = len(d)
    boot = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))


def load_per_sample_index(csv_path, budgets):
    idx_map = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            idx = r.get("idx", "")
            rec = {"idx": idx}
            for b in budgets:
                rec[f"fixed_{b}_correct"] = int(float(r.get(f"fixed_{b}_correct", 0)))
                rec[f"fixed_{b}_tokens"] = float(r.get(f"fixed_{b}_tokens", 0))
                rec[f"fixed_{b}_has_final"] = int(float(r.get(f"fixed_{b}_has_final", 0)))
            idx_map[idx] = rec
    return idx_map


def find_per_sample_csv(test_csv_ref):
    """Try multiple paths to locate the per-sample CSV."""
    candidates = [
        test_csv_ref,
        os.path.join(METHODS_DIR, os.path.basename(test_csv_ref)),
        os.path.join(RESULTS_DIR, os.path.basename(test_csv_ref)),
    ]
    base = os.path.basename(test_csv_ref)
    if base.startswith("per_sample_"):
        for d in [METHODS_DIR, RESULTS_DIR, "/workspace/nips-adathink/results"]:
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn == base:
                        candidates.append(os.path.join(d, fn))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    all_results = {}

    for name, cfg in SETTINGS.items():
        tcp = cfg["template_csv"]
        if not os.path.exists(tcp):
            print(f"SKIP {name}: {tcp} not found")
            continue

        with open(tcp) as f:
            trows = list(csv.DictReader(f))
        print(f"\n=== {name}: {len(trows)} rows ===")

        budgets = cfg["budgets"]
        b1, b_mid, b_max = budgets[0], budgets[1], budgets[-1]
        lam, norm = cfg["lam"], cfg["norm"]

        ps_cache = {}
        matched = 0
        ctrl_utils, fixed_mid_utils = [], []
        ctrl_accs, fixed_mid_accs = [], []
        ctrl_toks_e2e, fixed_mid_toks = [], []
        fixed_max_accs, fixed_max_utils = [], []

        for tr in trows:
            test_csv = tr.get("test_csv", "")
            idx = tr.get("idx", "")

            if test_csv not in ps_cache:
                path = find_per_sample_csv(test_csv)
                ps_cache[test_csv] = load_per_sample_index(path, budgets) if path else None

            ps_data = ps_cache.get(test_csv)
            if ps_data is None or idx not in ps_data:
                continue

            matched += 1
            orig = ps_data[idx]

            chosen = int(tr["chosen_budget"])
            ctrl_correct = int(float(tr["correct"]))
            ctrl_second_pass_tok = float(tr["tokens"])

            probe_tok = float(orig.get(f"fixed_{b1}_tokens", b1))
            e2e_tok = ctrl_second_pass_tok + (probe_tok if chosen > b1 else 0)

            u_ctrl = ctrl_correct - lam * (e2e_tok / norm)
            ctrl_utils.append(u_ctrl)
            ctrl_accs.append(ctrl_correct)
            ctrl_toks_e2e.append(e2e_tok)

            fixed_mid_c = orig[f"fixed_{b_mid}_correct"]
            fixed_mid_t = orig[f"fixed_{b_mid}_tokens"]
            u_fixed_mid = fixed_mid_c - lam * (fixed_mid_t / norm)
            fixed_mid_utils.append(u_fixed_mid)
            fixed_mid_accs.append(fixed_mid_c)
            fixed_mid_toks.append(fixed_mid_t)

            fixed_max_c = orig[f"fixed_{b_max}_correct"]
            fixed_max_t = orig[f"fixed_{b_max}_tokens"]
            u_fixed_max = fixed_max_c - lam * (fixed_max_t / norm)
            fixed_max_accs.append(fixed_max_c)
            fixed_max_utils.append(u_fixed_max)

        n = matched
        print(f"  Matched: {n}/{len(trows)}")

        if n == 0:
            print(f"  ERROR: no matches")
            continue

        print(f"  Template ctrl: acc={np.mean(ctrl_accs):.4f}, tok_e2e={np.mean(ctrl_toks_e2e):.1f}, util={np.mean(ctrl_utils):.4f}")
        print(f"  Fixed-{b_mid}: acc={np.mean(fixed_mid_accs):.4f}, tok={np.mean(fixed_mid_toks):.1f}, util={np.mean(fixed_mid_utils):.4f}")
        print(f"  Fixed-{b_max} (Max): acc={np.mean(fixed_max_accs):.4f}, util={np.mean(fixed_max_utils):.4f}")

        delta_acc = [ctrl_accs[i] - fixed_mid_accs[i] for i in range(n)]
        delta_tok = [ctrl_toks_e2e[i] - fixed_mid_toks[i] for i in range(n)]
        delta_util = [ctrl_utils[i] - fixed_mid_utils[i] for i in range(n)]

        mean_da, lo_da, hi_da = bootstrap_ci(delta_acc)
        mean_dt, lo_dt, hi_dt = bootstrap_ci(delta_tok)
        mean_du, lo_du, hi_du = bootstrap_ci(delta_util)

        print(f"  vs Fixed-{b_mid}:")
        print(f"    ΔAcc:  {mean_da:+.4f} [{lo_da:+.4f}, {hi_da:+.4f}] ({mean_da*100:+.1f} pp)")
        print(f"    ΔTok:  {mean_dt:+.1f} [{lo_dt:+.1f}, {hi_dt:+.1f}]")
        print(f"    ΔUtil: {mean_du:+.4f} [{lo_du:+.4f}, {hi_du:+.4f}] ({mean_du*100:+.1f} pp)")

        delta_acc_max = [ctrl_accs[i] - fixed_max_accs[i] for i in range(n)]
        delta_util_max = [ctrl_utils[i] - fixed_max_utils[i] for i in range(n)]
        mean_dam, lo_dam, hi_dam = bootstrap_ci(delta_acc_max)
        mean_dum, lo_dum, hi_dum = bootstrap_ci(delta_util_max)

        print(f"  vs Fixed-{b_max} (Max):")
        print(f"    ΔAcc:  {mean_dam:+.4f} ({mean_dam*100:+.1f} pp)")
        print(f"    ΔUtil: {mean_dum:+.4f} ({mean_dum*100:+.1f} pp)")

        all_results[name] = {
            "n": n,
            "budgets": budgets,
            "ctrl": {"acc": float(np.mean(ctrl_accs)), "tok_e2e": float(np.mean(ctrl_toks_e2e)), "util": float(np.mean(ctrl_utils))},
            "fixed_mid": {"b": b_mid, "acc": float(np.mean(fixed_mid_accs)), "tok": float(np.mean(fixed_mid_toks)), "util": float(np.mean(fixed_mid_utils))},
            "fixed_max": {"b": b_max, "acc": float(np.mean(fixed_max_accs)), "util": float(np.mean(fixed_max_utils))},
            "vs_mid": {
                "delta_acc": {"mean": mean_da, "lo": lo_da, "hi": hi_da, "pp": mean_da*100},
                "delta_tok": {"mean": mean_dt, "lo": lo_dt, "hi": hi_dt},
                "delta_util": {"mean": mean_du, "lo": lo_du, "hi": hi_du, "pp": mean_du*100},
            },
            "vs_max": {
                "delta_acc": {"mean": mean_dam, "pp": mean_dam*100},
                "delta_util": {"mean": mean_dum, "pp": mean_dum*100},
            },
        }

    out_path = os.path.join(RESULTS_DIR, "utility_cis_v2.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

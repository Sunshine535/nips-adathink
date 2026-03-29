#!/usr/bin/env python3
"""
TOWN (Think Only When Needed) Simulation
=========================================
Stage 1: Run nothink@256. If early stop (hit_budget=False), accept result.
Stage 2: For samples that hit budget in nothink@256, route to thinking@512.

Uses:
  - 8B nothink JSON (200 samples): per-sample nothink@256 results
  - 8B fulltest CSV (1319 samples): per-sample thinking@512 results

Outputs:
  Part A: Exact TOWN simulation on 200-sample overlap
  Part B: Estimated TOWN on full 1319 samples (using known aggregate stats)
  Part C: Sensitivity analysis
"""

import json
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
NOTHINK_JSON = REPO / "results_kun" / "nothink_baseline_Qwen3-8B_gsm8k_fullset.json"
FULLTEST_CSV = REPO / "results_kun" / "fulltest" / "per_sample_gsm8k_Qwen3_8B_20260324_120316.csv"
OUTPUT_DIR   = REPO / "results" / "town_simulation"


def load_data():
    """Load nothink JSON and fulltest CSV."""
    log.info("Loading nothink JSON: %s", NOTHINK_JSON)
    with open(NOTHINK_JSON) as f:
        nothink_data = json.load(f)

    log.info("Loading fulltest CSV: %s", FULLTEST_CSV)
    df_full = pd.read_csv(FULLTEST_CSV)
    log.info("  CSV shape: %s, idx range: %d-%d", df_full.shape, df_full["idx"].min(), df_full["idx"].max())

    return nothink_data, df_full


def exact_town_200(nothink_data, df_full):
    """
    Part A: Exact TOWN simulation on the 200-sample overlap.
    For each sample idx in [0, 199]:
      - If nothink@256 stopped early (hit_budget=False): use nothink result
      - Otherwise: use thinking@512 result from fulltest CSV
    """
    print("\n" + "=" * 80)
    print("Part A: EXACT TOWN Simulation (200-sample overlap, idx 0-199)")
    print("=" * 80)

    nt256 = {s["idx"]: s for s in nothink_data["per_sample"]["nothink_256"]}
    df_sub = df_full[df_full["idx"].isin(nt256.keys())].set_index("idx")

    results = []
    for idx in sorted(nt256.keys()):
        nt = nt256[idx]
        early_stop = not nt["hit_budget"]

        if early_stop:
            # Stage 1: accept nothink result
            correct = nt["correct"]
            tokens = nt["tokens"]
            source = "nothink@256"
        else:
            # Stage 2: fall back to thinking@512
            row = df_sub.loc[idx]
            correct = bool(row["fixed_512_correct"])
            tokens = int(row["fixed_512_tokens"])
            source = "thinking@512"

        results.append({
            "idx": idx,
            "correct": correct,
            "tokens": tokens,
            "source": source,
            "nothink_correct": nt["correct"],
            "nothink_tokens": nt["tokens"],
            "nothink_early_stop": early_stop,
            "thinking_correct": bool(df_sub.loc[idx]["fixed_512_correct"]),
            "thinking_tokens": int(df_sub.loc[idx]["fixed_512_tokens"]),
        })

    df_res = pd.DataFrame(results)
    n = len(df_res)

    # ── TOWN stats ──
    town_acc   = df_res["correct"].mean()
    town_tokens = df_res["tokens"].mean()
    n_nothink  = (df_res["source"] == "nothink@256").sum()
    n_thinking = (df_res["source"] == "thinking@512").sum()

    # ── Baseline stats (on same 200 samples) ──
    nt_acc  = df_res["nothink_correct"].mean()
    nt_tok  = df_res["nothink_tokens"].mean()
    th_acc  = df_res["thinking_correct"].mean()
    th_tok  = df_res["thinking_tokens"].mean()

    # ── Accuracy breakdown by routing ──
    nothink_routed = df_res[df_res["source"] == "nothink@256"]
    thinking_routed = df_res[df_res["source"] == "thinking@512"]

    print(f"\n{'Metric':<40} {'TOWN':>10} {'nothink@256':>12} {'thinking@512':>13}")
    print("-" * 80)
    print(f"{'Accuracy':<40} {town_acc:>10.1%} {nt_acc:>12.1%} {th_acc:>13.1%}")
    print(f"{'Avg tokens':<40} {town_tokens:>10.1f} {nt_tok:>12.1f} {th_tok:>13.1f}")
    print()

    print(f"Routing breakdown (n={n}):")
    print(f"  Kept by nothink@256 (early stop):  {n_nothink:>4d} ({n_nothink/n:.1%})")
    print(f"  Routed to thinking@512 (hit budget): {n_thinking:>4d} ({n_thinking/n:.1%})")
    print()

    print(f"Accuracy by route:")
    if len(nothink_routed) > 0:
        print(f"  nothink-kept samples:    {nothink_routed['correct'].mean():.1%}  (n={len(nothink_routed)})")
    if len(thinking_routed) > 0:
        print(f"  thinking-routed samples: {thinking_routed['correct'].mean():.1%}  (n={len(thinking_routed)})")
    print()

    # ── Token budget analysis ──
    nothink_token_cost = nothink_routed["tokens"].sum()
    thinking_token_cost = thinking_routed["tokens"].sum()
    total_tokens = nothink_token_cost + thinking_token_cost
    print(f"Token budget analysis:")
    print(f"  Nothink path total tokens:  {nothink_token_cost:>8,}")
    print(f"  Thinking path total tokens: {thinking_token_cost:>8,}")
    print(f"  TOWN total tokens:          {total_tokens:>8,}")
    print(f"  Pure nothink@256 total:     {int(nt_tok * n):>8,}")
    print(f"  Pure thinking@512 total:    {int(th_tok * n):>8,}")
    print(f"  TOWN savings vs think@512:  {1 - total_tokens / (th_tok * n):.1%}")
    print()

    # ── Error analysis ──
    # Cases where TOWN gets it right but nothink alone doesn't
    town_wins = df_res[(df_res["correct"]) & (~df_res["nothink_correct"])].shape[0]
    # Cases where nothink was right but TOWN routing changed to wrong thinking answer
    # (shouldn't happen since we only route when nothink hit budget, but let's check)
    town_losses = df_res[(~df_res["correct"]) & (df_res["nothink_correct"])].shape[0]
    print(f"Error analysis vs nothink@256:")
    print(f"  TOWN wins  (nothink wrong -> thinking right): {town_wins}")
    print(f"  TOWN losses (nothink right -> thinking wrong, should be 0 by design): {town_losses}")

    # Actually, a "loss" can only happen if nothink was correct but hit_budget=True,
    # and then thinking@512 was wrong. Let's check:
    corner_cases = df_res[(df_res["nothink_early_stop"] == False) & (df_res["nothink_correct"]) & (~df_res["thinking_correct"])]
    print(f"  Corner case (nothink hit budget & correct, but thinking wrong): {len(corner_cases)}")
    if len(corner_cases) > 0:
        print(f"    These are samples where nothink got lucky despite hitting budget.")
        for _, row in corner_cases.iterrows():
            print(f"      idx={row['idx']}: nothink_tokens={row['nothink_tokens']}, thinking_tokens={row['thinking_tokens']}")
    print()

    # ── Compute effective efficiency: acc per 100 tokens ──
    print(f"Efficiency (accuracy per 100 tokens):")
    print(f"  TOWN:          {town_acc / (town_tokens / 100):.3f}")
    print(f"  nothink@256:   {nt_acc / (nt_tok / 100):.3f}")
    print(f"  thinking@512:  {th_acc / (th_tok / 100):.3f}")

    return df_res


def estimate_town_full(nothink_data, df_full):
    """
    Part B: Estimate TOWN on full 1319 samples using aggregate stats
    and per-sample data from the fulltest CSV.
    """
    print("\n" + "=" * 80)
    print("Part B: Estimated TOWN on FULL 1319 samples")
    print("=" * 80)

    N = 1319

    # ── Known fullset nothink@256 stats ──
    nt_acc        = 0.875   # from fulltest run
    nt_early_stop = 0.888   # from fulltest run
    nt_avg_tokens = 146.0   # from fulltest run

    # ── Known fullset thinking@512 stats ──
    th_acc        = df_full["fixed_512_correct"].mean()
    th_avg_tokens = df_full["fixed_512_tokens"].mean()
    th_has_final  = df_full["fixed_512_has_final"].mean()

    n_early = int(round(nt_early_stop * N))  # samples kept by nothink
    n_route = N - n_early                     # samples routed to thinking

    print(f"\nKnown aggregate stats:")
    print(f"  nothink@256: acc={nt_acc:.1%}, early_stop={nt_early_stop:.1%}, avg_tok={nt_avg_tokens:.0f}")
    print(f"  thinking@512: acc={th_acc:.1%}, avg_tok={th_avg_tokens:.1f}, has_final={th_has_final:.1%}")
    print(f"\nRouting split: {n_early} kept ({n_early/N:.1%}) + {n_route} routed ({n_route/N:.1%})")

    # ── Estimate accuracy for each group ──
    # Group A (early stop): These are "easy" samples. nothink solved them quickly.
    # For the 200-sample overlap, we can compute the exact accuracy of early-stop samples.
    nt256 = {s["idx"]: s for s in nothink_data["per_sample"]["nothink_256"]}
    early_samples_200 = [s for s in nt256.values() if not s["hit_budget"]]
    late_samples_200  = [s for s in nt256.values() if s["hit_budget"]]

    early_acc_200 = np.mean([s["correct"] for s in early_samples_200]) if early_samples_200 else 0
    late_acc_200  = np.mean([s["correct"] for s in late_samples_200]) if late_samples_200 else 0

    print(f"\nFrom 200-sample pilot:")
    print(f"  Early-stop samples: n={len(early_samples_200)}, nothink acc={early_acc_200:.1%}")
    print(f"  Hit-budget samples: n={len(late_samples_200)}, nothink acc={late_acc_200:.1%}")

    # For routed (hit-budget) samples, what's thinking@512 accuracy?
    late_idxs = [s["idx"] for s in late_samples_200]
    df_late = df_full[df_full["idx"].isin(late_idxs)]
    thinking_on_late_acc_200 = df_late["fixed_512_correct"].mean()
    thinking_on_late_tok_200 = df_late["fixed_512_tokens"].mean()
    print(f"  thinking@512 on hit-budget samples: acc={thinking_on_late_acc_200:.1%}, avg_tok={thinking_on_late_tok_200:.1f}")

    # ── Method 1: Direct extrapolation from 200-sample ratios ──
    print(f"\n--- Method 1: Extrapolation from 200-sample pilot ---")
    # Assume the 200-sample ratios hold for the full set
    est_early_acc = early_acc_200  # accuracy on early-stop samples
    est_late_thinking_acc = thinking_on_late_acc_200  # thinking@512 on routed samples

    # But we need to reconcile: the fullset nothink accuracy is 87.5%, not 89%.
    # And fullset early_stop is 88.8%, not 92%.
    # The fullset has harder samples (idx 200+), so routed samples might have lower thinking acc.

    # Estimate: overall nothink accuracy = early_stop_rate * early_acc + (1-early_stop_rate) * late_acc
    # 0.875 = 0.888 * early_acc_full + 0.112 * late_acc_full
    # We know early_acc should be very high (nothink solved it within budget)
    # From 200-sample: early_acc = 95.1%, late_acc = 25.0%
    # Let's solve:
    early_acc_full_est = (nt_acc - (1 - nt_early_stop) * late_acc_200) / nt_early_stop
    print(f"  Estimated early-stop accuracy (full): {early_acc_full_est:.1%}")
    print(f"  (Using 200-sample late_acc={late_acc_200:.1%} as proxy)")

    # For TOWN accuracy:
    town_acc_m1 = nt_early_stop * early_acc_full_est + (1 - nt_early_stop) * est_late_thinking_acc
    print(f"  TOWN accuracy estimate: {town_acc_m1:.1%}")

    # Token estimate
    # Early-stop tokens: nothink tokens for samples that stopped early
    early_tokens_200 = np.mean([s["tokens"] for s in early_samples_200])
    town_tokens_m1 = nt_early_stop * early_tokens_200 + (1 - nt_early_stop) * thinking_on_late_tok_200
    print(f"  Early-stop avg tokens: {early_tokens_200:.1f}")
    print(f"  TOWN avg tokens estimate: {town_tokens_m1:.1f}")

    # ── Method 2: Use fulltest CSV thinking@512 for bottom-N samples ──
    print(f"\n--- Method 2: Per-sample thinking@512 from fulltest CSV ---")
    # We know the overall stats. We can estimate which samples would be "hard" (hit budget)
    # by using the fulltest thinking@512 correctness as a difficulty proxy.

    # Sort by thinking@512 correctness (wrong first = hard) then by token usage (high first)
    df_sorted = df_full.sort_values(
        by=["fixed_512_correct", "fixed_512_tokens"],
        ascending=[True, False]
    ).reset_index(drop=True)

    # The n_route hardest samples (those that nothink would route to thinking)
    # Assumption: samples that are hard for nothink are also hard for thinking
    routed_samples = df_sorted.head(n_route)
    kept_samples = df_sorted.tail(n_early)

    thinking_acc_on_routed = routed_samples["fixed_512_correct"].mean()
    thinking_tok_on_routed = routed_samples["fixed_512_tokens"].mean()
    print(f"  Bottom {n_route} samples (proxy for routed):")
    print(f"    thinking@512 acc: {thinking_acc_on_routed:.1%}")
    print(f"    thinking@512 avg tokens: {thinking_tok_on_routed:.1f}")

    town_acc_m2 = nt_early_stop * early_acc_full_est + (1 - nt_early_stop) * thinking_acc_on_routed
    town_tokens_m2 = nt_early_stop * early_tokens_200 + (1 - nt_early_stop) * thinking_tok_on_routed
    print(f"  TOWN accuracy estimate: {town_acc_m2:.1%}")
    print(f"  TOWN avg tokens estimate: {town_tokens_m2:.1f}")

    # ── Method 3: Conservative bound ──
    print(f"\n--- Method 3: Conservative / optimistic bounds ---")
    # Conservative: assume thinking@512 does 0% on routed samples
    town_acc_low = nt_early_stop * early_acc_full_est + (1 - nt_early_stop) * 0.0
    # Optimistic: assume thinking@512 does its average on routed samples
    town_acc_high = nt_early_stop * early_acc_full_est + (1 - nt_early_stop) * th_acc
    # Best case: thinking does 100% on routed
    town_acc_best = nt_early_stop * early_acc_full_est + (1 - nt_early_stop) * 1.0
    print(f"  Conservative (thinking=0% on routed):   {town_acc_low:.1%}")
    print(f"  Using average thinking acc on routed:    {town_acc_high:.1%}")
    print(f"  Best case (thinking=100% on routed):     {town_acc_best:.1%}")
    print(f"  Method 1 estimate (200-sample proxy):    {town_acc_m1:.1%}")
    print(f"  Method 2 estimate (difficulty proxy):    {town_acc_m2:.1%}")

    # ── Summary table ──
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: Estimated TOWN performance on full GSM8K (n={N})")
    print(f"{'=' * 80}")
    print(f"\n{'Method':<45} {'Acc':>8} {'Avg Tok':>10}")
    print("-" * 65)
    print(f"{'nothink@256 (baseline)':<45} {nt_acc:>8.1%} {nt_avg_tokens:>10.0f}")
    print(f"{'thinking@512 (baseline)':<45} {th_acc:>8.1%} {th_avg_tokens:>10.1f}")
    print(f"{'TOWN M1 (200-sample extrapolation)':<45} {town_acc_m1:>8.1%} {town_tokens_m1:>10.1f}")
    print(f"{'TOWN M2 (difficulty-proxy)':<45} {town_acc_m2:>8.1%} {town_tokens_m2:>10.1f}")
    print(f"{'TOWN lower bound':<45} {town_acc_low:>8.1%} {'—':>10}")
    print(f"{'TOWN upper bound (avg thinking)':<45} {town_acc_high:>8.1%} {'—':>10}")

    return {
        "m1_acc": town_acc_m1, "m1_tokens": town_tokens_m1,
        "m2_acc": town_acc_m2, "m2_tokens": town_tokens_m2,
        "lower": town_acc_low, "upper": town_acc_high,
    }


def sensitivity_analysis(nothink_data, df_full):
    """
    Part C: Sensitivity — vary the nothink budget and see how TOWN changes.
    Use the 200-sample overlap where we have nothink@128/256/512 data.
    """
    print("\n" + "=" * 80)
    print("Part C: Sensitivity Analysis — TOWN with different nothink budgets (200 samples)")
    print("=" * 80)

    df_sub = df_full[df_full["idx"] < 200].set_index("idx")

    configs = [
        ("nothink@128 -> thinking@256", "nothink_128", "fixed_256"),
        ("nothink@128 -> thinking@512", "nothink_128", "fixed_512"),
        ("nothink@256 -> thinking@512", "nothink_256", "fixed_512"),
        ("nothink@512 -> thinking@512", "nothink_512", "fixed_512"),
    ]

    print(f"\n{'Config':<35} {'Acc':>7} {'Tok':>7} {'Kept%':>7} {'Route%':>7} {'Kept_Acc':>9} {'Route_Acc':>10}")
    print("-" * 90)

    for label, nt_key, th_key in configs:
        nt_samples = {s["idx"]: s for s in nothink_data["per_sample"][nt_key]}
        th_correct_col = f"{th_key}_correct"
        th_tokens_col  = f"{th_key}_tokens"

        corrects = []
        tokens_list = []
        n_kept = 0

        for idx in sorted(nt_samples.keys()):
            nt = nt_samples[idx]
            if not nt["hit_budget"]:
                corrects.append(nt["correct"])
                tokens_list.append(nt["tokens"])
                n_kept += 1
            else:
                row = df_sub.loc[idx]
                corrects.append(bool(row[th_correct_col]))
                tokens_list.append(int(row[th_tokens_col]))

        n_total = len(corrects)
        acc = np.mean(corrects)
        avg_tok = np.mean(tokens_list)
        kept_pct = n_kept / n_total
        route_pct = 1 - kept_pct

        # Accuracy split
        kept_acc = np.mean([c for c, nt in zip(corrects, [nt_samples[i] for i in sorted(nt_samples.keys())])
                           if not nt["hit_budget"]]) if n_kept > 0 else 0
        routed_correct = [c for c, nt in zip(corrects, [nt_samples[i] for i in sorted(nt_samples.keys())])
                         if nt["hit_budget"]]
        route_acc = np.mean(routed_correct) if routed_correct else 0

        print(f"{label:<35} {acc:>7.1%} {avg_tok:>7.1f} {kept_pct:>7.1%} {route_pct:>7.1%} {kept_acc:>9.1%} {route_acc:>10.1%}")

    # Also show pure baselines for reference
    print()
    print("Pure baselines (200 samples):")
    for key in ["nothink_128", "nothink_256", "nothink_512"]:
        samples = nothink_data["per_sample"][key]
        acc = np.mean([s["correct"] for s in samples])
        tok = np.mean([s["tokens"] for s in samples])
        print(f"  {key}: acc={acc:.1%}, avg_tok={tok:.1f}")

    for budget in [128, 256, 512]:
        sub = df_sub[f"fixed_{budget}_correct"]
        tok = df_sub[f"fixed_{budget}_tokens"]
        print(f"  thinking@{budget}: acc={sub.mean():.1%}, avg_tok={tok.mean():.1f}")


def save_results(df_res, estimates, output_dir):
    """Save per-sample TOWN results and estimates."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "town_200_per_sample.csv"
    df_res.to_csv(csv_path, index=False)
    log.info("Saved per-sample results to %s", csv_path)

    est_path = output_dir / "town_full_estimates.json"
    with open(est_path, "w") as f:
        json.dump(estimates, f, indent=2)
    log.info("Saved estimates to %s", est_path)

    print(f"\nResults saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="TOWN simulation using existing per-sample data")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for results")
    args = parser.parse_args()

    nothink_data, df_full = load_data()

    # Part A: exact 200-sample simulation
    df_res = exact_town_200(nothink_data, df_full)

    # Part B: full-set estimation
    estimates = estimate_town_full(nothink_data, df_full)

    # Part C: sensitivity
    sensitivity_analysis(nothink_data, df_full)

    # Save
    save_results(df_res, estimates, Path(args.output_dir))


if __name__ == "__main__":
    main()

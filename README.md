# The Coupling Tax

**Paper**: *The Coupling Tax: When Chain-of-Thought Costs More Than It Saves*
**Target venue**: NeurIPS 2026
**Status**: active — phenomenon verified, RCV-IRIS gate frozen as **negative ablation** per GPT-5.5 Round 2 review (`reports/FINAL_RCV_VERDICT.md`). Main contributions: Coupling Tax phenomenon, mode×prompt interaction, decoupled-extraction Stage-3 cascade, training-free Pareto-competitive vs SwiReasoning/s1 on MATH-500.

This repository investigates a structural phenomenon we call the **Coupling Tax**: under a fixed output-token budget, thinking-mode LLMs must share the budget between reasoning and answer, so chain-of-thought that cannot finish within the budget truncates and drags accuracy below non-thinking mode. We derive an accounting decomposition, quantify an inverse scaling tendency, and demonstrate a simple cascade with decoupled extraction. Core results are paired McNemar-significant on multiple benchmarks under post-hoc analysis-token accounting; deployment-faithful (online) numbers are reported separately and may differ.

> **Honest scope note (2026-04-25)**: The paper is in revision pending: (a) full disclosure of post-hoc vs online accounting, (b) rerun of budget-forcing baselines under field-level token accounting, (c) related-work expansion (BAEE / AnytimeReasoner / Elastic Reasoning), and (d) RCV-IRIS framing as an explicit negative ablation rather than the main method. See `reports/FINAL_RCV_VERDICT.md` and `reports/REMAINING_RISKS.md`.

## Canonical artifacts (authoritative)

| What | Path |
|------|------|
| Paper source | `paper/main_final.tex` + `paper/sections/` |
| Accounting decomposition + analysis | `paper/sections/theory_final.tex` (paper-rewrite pending; "core theorem verified" claim is being walked back to "accounting framework with held-out predictions") |
| Current method proposal | `idea-stage/FINAL_PROPOSAL.md` |
| Current experiment plan | `idea-stage/EXPERIMENT_PLAN.md` |
| Literature landscape (Apr 2026) | `idea-stage/LITERATURE_LANDSCAPE.md` |
| Claim-to-evidence audit | `PAPER_CLAIM_AUDIT.md` |
| Narrative report (latest) | `NARRATIVE_REPORT.md` |
| Data provenance | `DATA_PROVENANCE.md` |

## Core empirical results

| Finding | Setup | Result | Evidence |
|---------|-------|--------|----------|
| 27B GSM8K coupling tax at b=4096 | n=200, seed=42, McNemar paired | nothink 98.0% vs think 87.5% (−10.5pp), p<1e-5 | `results/p21_27b_gsm8k_extend/b4096/` |
| 8B GSM8K IRIS vs TOWN (full-scale) | n=1319, seed=42, paired | IRIS 90.9% vs TOWN 86.0%, **p=1.6e-17** | `results/gap_fill_20260414/iris_gsm8k_8b_fullscale/` |
| 27B MATH-500 IRIS vs TOWN | n=200, seed=42, paired, **post-hoc Stage 2** | IRIS 77.5% vs TOWN 49.0% (+28.5pp), **p=3.5e-11**. **Online-faithful Stage-2 rerun: 67.5% (-10pp gap)**. Headline must be labeled. | `results/iris_improved_20260417/27b_math500_b4096_ba512_n200/`, `results/iris_online_20260421/` |
| Stage-3 decoupled extraction (27B MATH-500) | n=200, paired same-sample | baseline 60.5% → improved 77.5% (+17pp) | `results/iris_improved_20260417/` |
| IRIS vs s1 budget forcing (early_stop) | 8B MATH-500 n=200 seed=42, b=4096 | IRIS 74.0% / 2380 tok vs s1 72.0% / 3164 tok — **CAVEAT: old s1 results have token undercount; rerun pending with V2 field-level accounting** | `results/budget_forcing/` |
| Multi-seed stability on 8B MATH-500 | 3 seeds (42/123/456), **mixed n** (n=500 / 200 / 200) | mean 74.1%, std 1.5pp — **CAVEAT: unequal n** | `results/multiseed_20260419/multiseed_summary.json` |
| αc/αt curve fit | Logistic fit on b∈{128,256,512} → predict α_t(b) at 1024/2048 | Fit RMSE = 3.2×10⁻¹⁷ on train set (interpolated); held-out α_t(1024) ground truth = 0.417 vs logistic prediction 0.321 (≈9.7pp gap — this is not a direct Acc_think support, just α_t extrapolation) | `results/analysis/alpha_curve_fit.json` |
| Learned allocator (13-feature LR) | MATH-500 test split | 46.6% token savings (oracle ceiling 60.2%) | `results/learned_allocator/mlp_trained.json` |
| IRIS entropy stopping null | 200 GSM8K samples | 0/200 samples triggered, anti-correlated with correctness | defensive ablation |

## Running experiments

The paper's experiments are reproducible via `scripts/run_iris.py`, `scripts/run_nothink_baseline.py`, and `scripts/run_budget_forcing.py`. Full provenance for every paper claim is recorded in `DATA_PROVENANCE.md` and `results/mcnemar_summary.json`.

Key scripts:
- `scripts/run_iris.py` — IRIS / TOWN paired runner with Stage-3 decoupled extraction
- `scripts/run_nothink_baseline.py` — nothink vs thinking matched-budget baseline
- `scripts/run_budget_forcing.py` — Muennighoff s1-style budget forcing (early_stop, wait_extend)
- `scripts/learned_allocator.py` — question-features allocator analysis
- `scripts/run_ctt_pilot.py` — Coupling-Tax Tomography ablation (null result preserved)
- `scripts/analysis/fit_alpha_curves.py` — αc/αt curve fitting

## Repository status

The project pivoted in 2026-04 from an earlier "AdaThink / Dynamic Halting" direction to the current Coupling Tax framing after discovering the earlier evaluation protocol was flawed (train/eval on the same CSV, oracle-labeled stopping). All pre-pivot artifacts (README, PROJECT_FINAL_REPORT, FINAL_SUMMARY, Dynamic Halting controller, etc.) are preserved under `archive/pre_coupling_tax_pivot/` for reference but are no longer part of the canonical narrative. Do not mix evidence across the two eras.

An earlier iteration of the refined proposal lives in `archive/refine-logs_v1/`. The current proposal is in `idea-stage/`.

## Remote servers

- Server A — A100 @ 216.81.245.124:13627 (RunPod, intermittent)
- Server B — A100 @ 216.81.245.126:13722 (RunPod, intermittent)
- H800 — access via Mac jumphost (`mac2` → `ssh-439.default`)

Details and data-sync procedures in `CLAUDE.md`.

# The Coupling Tax

**Paper**: *The Coupling Tax: When Chain-of-Thought Costs More Than It Saves*
**Target venue**: NeurIPS 2026
**Status**: active — core theorem verified (proof-checker Round 4 PASS), method empirically validated on GSM8K / MATH-500 at 8B / 27B scales.

This repository investigates a structural phenomenon we call the **Coupling Tax**: under a fixed output-token budget, thinking-mode LLMs must share the budget between reasoning and answer, so chain-of-thought that cannot finish within the budget truncates and drags accuracy below non-thinking mode. We derive a closed-form decomposition, quantify an inverse scaling law, and demonstrate a simple cascade with decoupled extraction that Pareto-dominates competing baselines.

## Canonical artifacts (authoritative)

| What | Path |
|------|------|
| Paper source | `paper/main_final.tex` + `paper/sections/` |
| Core theorem + proofs (verified R4) | `paper/sections/theory_final.tex` |
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
| 27B MATH-500 IRIS vs TOWN | n=200, seed=42, paired | IRIS 77.5% vs TOWN 49.0% (+28.5pp), **p=3.5e-11** | `results/iris_improved_20260417/27b_math500_b4096_ba512_n200/` |
| Stage-3 decoupled extraction (27B MATH-500) | n=200, paired same-sample | baseline 60.5% → improved 77.5% (+17pp) | `results/iris_improved_20260417/` |
| IRIS vs s1 budget forcing (early_stop) | 8B MATH-500 n=200 seed=42, b=4096 | IRIS 74.0% / 2380 tok vs s1 72.0% / 3164 tok | `results/budget_forcing/` |
| Multi-seed stability on 8B MATH-500 | 3 seeds (42/123/456) | mean 74.1%, std 1.5pp, span 3.0pp | `results/multiseed_20260419/multiseed_summary.json` |
| αc/αt curve fit | Logistic fit on b∈{128,256,512} → predict b=1024 | Acc_think(1024) predicted error 1.2pp | `results/analysis/alpha_curve_fit.json` |
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

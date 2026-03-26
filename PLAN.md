# Plan: AdaThink (Stage-Gate v2)

## Gate 0: Measurement Integrity (Done)
- [x] 4-GPU reproducible pipeline for fixed/adaptive decoding.
- [x] Local model completeness precheck (fail-fast).
- [x] Prompt-mode controls for Qwen3 (`enable_thinking`, direct answer).
- [x] Decoupled `seed` and `data_seed`.

## Gate 1: Strong Heuristic Baseline (Done)
- [x] Model sweep on Qwen3 0.6B/1.7B/4B/8B.
- [x] Cost/latency/accuracy logging under shared budgets.
- [x] Adaptive vs fixed comparisons exported to JSON/CSV.

## Gate 2: Statistical Credibility (Done for heuristic stage)
- [x] Replicated subset runs with fixed algorithm seed and varying `data_seed`.
- [x] Per-sample paired delta extraction (`adaptive` vs `fixed_256`).
- [x] Bootstrap CI sanity check.
- [x] Strict final-answer parsing + projection fallback integrated for long-thinking models.
- Go result:
  - Heuristic gains are near zero and not sufficient for publication claim.
  - Qwen3.5-27B strict/projection twenty-three-subset pooled result (`n=920`) shows that naive adaptive improves accuracy only by paying near-512-token cost.

## Gate 3: Learned Controller (In Progress, critical path)
- [x] Export trajectory-style supervision from strict/projection per-sample runs.
- [x] Train/evaluate learned template budget controller with leave-one-subset-out validation.
- [x] Expand validation to 4 subset seeds (`101/202/303/404`) and report paired bootstrap CIs.
- [x] Expand validation to 12 subset seeds (`101/202/303/404/505/606/707/808/909/1001/1103/1205`) and rerun paired bootstrap CIs.
- [x] Expand validation to 14 subset seeds (`101/202/303/404/505/606/707/808/909/1001/1103/1205/1307/1409`) and rerun paired bootstrap CIs.
- [x] Expand validation to 15 subset seeds (`101/202/303/404/505/606/707/808/909/1001/1103/1205/1307/1409/1511`) and rerun paired bootstrap CIs.
- [x] Expand validation to 16 subset seeds (`101/202/303/404/505/606/707/808/909/1001/1103/1205/1307/1409/1511/1613`) and rerun paired bootstrap CIs.
- [x] Expand validation to 17 subset seeds (`101/202/303/404/505/606/707/808/909/1001/1103/1205/1307/1409/1511/1613/1717`) and rerun paired bootstrap CIs.
- [x] Expand validation to 18 subset seeds (`101/202/303/404/505/606/707/808/909/1001/1103/1205/1307/1409/1511/1613/1717/1819`) and rerun paired bootstrap CIs.
- [x] Expand validation to 19 subset seeds (`101/202/303/404/505/606/707/808/909/1001/1103/1205/1307/1409/1511/1613/1717/1819/1921`) and rerun paired bootstrap CIs.
- [x] Expand validation to 20 subset seeds (`101/202/303/404/505/606/707/808/909/1001/1103/1205/1307/1409/1511/1613/1717/1819/1921/2023`) and rerun paired bootstrap CIs.
- [x] Expand validation to 21 subset seeds (`101/202/303/404/505/606/707/808/909/1001/1103/1205/1307/1409/1511/1613/1717/1819/1921/2023/2125`) and rerun paired bootstrap CIs.
- [x] Expand validation to 22 subset seeds (`101/202/303/404/505/606/707/808/909/1001/1103/1205/1307/1409/1511/1613/1717/1819/1921/2023/2125/2227`) and rerun paired bootstrap CIs.
- [x] Expand validation to 23 subset seeds (`101/202/303/404/505/606/707/808/909/1001/1103/1205/1307/1409/1511/1613/1717/1819/1921/2023/2125/2227/2329`) and rerun paired bootstrap CIs.
- [x] Add dedicated significance script (`run_template_controller_significance.py`) for reproducible paired deltas.
- [x] Train stronger parametric controller with richer state and online rollout objective.
- [x] Constrained policy optimization under compute budget (soft budget-constraint penalty objective, first validated variant).
- [x] Add parametric sweep orchestrator for second-scale tuning (`run_parametric_sweep.py`).
- [x] Add value-based controller (per-budget correctness prediction + cost-aware action selection) and penalty sweep on 8B-think.
- [ ] Ablations (`no verifier`, `halting-only`, `no branch`).
- Go criterion:
  - achieve `>= +1.5` absolute accuracy at matched cost on at least 2 model scales.

## Gate 4: Paper Packaging (In Progress)
- [x] Main tables + Pareto plots (Table 1-4, Figure 1-3).
- [x] Cross-benchmark and cross-scale results (Table 2).
- [x] Ablation study (Table 3).
- [x] Latency analysis (Table 4).
- [x] Statistical significance forest plot (Figure 2).
- [x] Reproducibility appendix (seeds, hardware, parser details).
- [ ] Failure taxonomy (per-question error analysis).
- [ ] Artifact checklist and release script.
- [ ] NeurIPS checklist completion.

## Kill Criteria
- If learned controller cannot beat fixed-budget baseline at matched cost, pivot paper framing to:
  - compute-policy diagnostics,
  - prompt-mode sensitivity,
  - negative result with strong reproducibility.
- If verifier consistently hurts quality-cost tradeoff, keep verifier only as negative ablation.

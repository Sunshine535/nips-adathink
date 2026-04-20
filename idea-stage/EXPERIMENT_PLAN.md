# EXPERIMENT_PLAN — Coupling Tax + IRIS (post-CTT-null)

**Paper**: *The Coupling Tax: Why Chain-of-Thought Fails Under Fixed Output Budgets*
**Version**: v2 (2026-04-20, supersedes the CTT-centric v1 now at
`EXPERIMENT_PLAN_CTT_DEPRECATED.md`).
**Method under test**: **IRIS** — `nothink-triage → think → decoupled-extraction`. Stage-3 mode-switch extraction is the hero mechanism; CTT (paired cross-mode layer-wise KL router) is preserved only as an honest negative-result ablation.

This plan is **claim-driven**: each block maps to a paper claim. Priorities encode what must land for submission (P0) vs what would strengthen (P1) vs stretch (P2).

---

## Status snapshot (2026-04-20)

All P0 blocks have data. Remaining work is engineering cleanup, not new experiments.

| Block | Status | Evidence file |
|-------|--------|---------------|
| 27B GSM8K coupling-tax crossover (nothink 98.0 vs think 87.5, p<1e-5) | ✅ complete | `results/p21_27b_gsm8k_extend/b4096/nothink_baseline_Qwen3_5-27B_gsm8k_20260418_141928.json` |
| 8B GSM8K IRIS vs TOWN full-scale n=1319 (p=1.6e-17) | ✅ complete | `results/gap_fill_20260414/iris_gsm8k_8b_fullscale/` |
| 27B MATH-500 IRIS vs TOWN n=200 (+28.5pp, p=3.5e-11) | ✅ complete | `results/iris_improved_20260417/27b_math500_b4096_ba512_n200/` |
| Stage-3 extraction improvement (27B MATH-500 60.5% → 77.5% same-sample) | ✅ complete | `results/iris_improved_20260417/27b_math500_b4096_ba512_n200/` |
| Multi-seed IRIS 8B MATH-500 (3 seeds, mean 74.1%, std 1.5pp) | ✅ complete | `results/multiseed_20260419/multiseed_summary.json` |
| Budget-forcing (s1 early_stop) comparison 8B MATH-500 | ✅ complete | `results/budget_forcing/bforce_early_stop_*.json` |
| αc / αt curve fitting: logistic fit on b∈{128,256,512} (train RMSE ≈ 0), held-out α_t(1024) = 0.417 vs predicted 0.321 (≈9.7pp gap — extrapolation is imprecise; do NOT claim Acc_think within-1.2pp) | ⚠️ demoted | `results/analysis/alpha_curve_fit.json` |
| Learned allocator 46.6% savings (77% of oracle) | ✅ complete | `results/learned_allocator/mlp_trained.json` |
| IRIS entropy-stopping null result (0/200 triggered) | ✅ complete | defensive ablation in paper |
| CTT (paired cross-mode KL) null result | ✅ complete (pivoted) | `results/ctt_pilot_{8b,27b}_gsm8k/analysis.json` |
| 27B GSM8K IRIS n=200 (completes 27B × benchmark IRIS grid) | 🔄 running | H800, PID 27157, n≈160 of 200 at time of writing |

---

## Honest scope statement (per external review)

Stage-3 decoupled extraction is empirically validated in the **high-coupling-tax regime** (large model × hard benchmark) and is statistically significant on the two full-scale paired tests above. It is **not** a universal gain:

- 8B MATH-500 at full n=500 shows only +0.4pp over baseline; 8B MATH-500 at n=200 seeds 123/456 gives +2.5pp and +1.0pp with McNemar p=0.46 and p=0.86 respectively — not significant.
- This is expected from Proposition 6 (`Δ_split = (1-F_L(b_r))·(α_extract - α_t)`): when the baseline extraction is already near the ceiling (numeric answers + last-number parser on 8B MATH-500), the recoverable margin is small.

The paper now explicitly scopes gains to the large-model × hard-benchmark regime instead of claiming universal improvement.

---

## Remaining P0 engineering cleanup (required before submission)

### E-OC1 — Promote `iris_online_stage2.py` to the canonical runner

**Status**: code in place (commit a73e6c9), `--online_stage2` flag added, but the **headline numbers in the paper are still from the old post-hoc Stage-2 runner**. Deployment-faithful claims (wall-clock / FLOPs) cannot be made until the main results are re-run with the online variant.

- Scope: re-run 8B GSM8K n=1319, 8B MATH-500 n=500, 27B MATH-500 n=200 with `--online_stage2`.
- Success: accuracy within ±1pp of the reported numbers (expected; accuracy doesn't change under adaptive stopping, only the accounting does), and `n_tokens_generated == n_tokens_used` in every record.
- Compute: ~10 GPU-h total if H800 is used for 27B MATH-500, A100 for 8B.
- Gate: if accuracy drops by more than 1pp, the entropy/HS-stability criterion is actually wrong for online deployment and Stage 2 should fall back to pure "generate-to-natural-stop-then-extract" (i.e., skip the adaptive check altogether). This is still a valid method; it just drops the efficiency claim for Stage 2.

### E-OC2 — Re-run paper claim audit after E-OC1

After the canonical headline numbers come from the online runner, re-run the claim audit a third time to confirm PASS still holds for all material claims under the new accounting.

### E-OC3 — Sync `paper/sections/method_final.tex` to say Stage-2 accuracy claims use adaptive-effective-token budget under either runner; efficiency claims require E-OC1

Currently the method section is ambiguous about which runner produced the numbers. Make this explicit.

---

## Retained P1 / P2 blocks (not blocking submission)

### P1-1 (drafted, empirical validation deferred)
Information-theoretic truncation bound (Fano-based). Theorem drafted; empirical bound vs actual Acc_think(b) comparison would strengthen §4 but requires logprob collection across budgets. Defer unless a reviewer demands it.

### P2 (stretch)
- Cross-family validation on DeepSeek-R1-Distill-Llama-8B (6 GPU-h) — strengthens generalization claim beyond Qwen3 hybrid-mode family.
- BBH cross-benchmark IRIS (4 GPU-h) — currently we have BBH baselines showing the budget-2048 tax reversal; IRIS on BBH is not a required claim but would round out the benchmark coverage.

---

## Explicit non-goals for this paper

- No new-method paper combining PSV-Cascade, sufficiency verifiers, or online prefix-level decisions. That is a separate research thread tracked in `PSV_PILOT_PLAN.md`.
- No claim of Pareto dominance over BAEE (arXiv:2604.06613) — the paper explicitly positions BAEE as free-continuation vs. our in-model mode-switch extraction (distinct mechanism, acknowledged).
- No claim of best-paper-level novelty — that was contingent on CTT succeeding. With CTT null, target is strong accept on the theorem + 27B crossover + Stage-3 combination.

---

## CTT ablation (preserved as honest negative result)

- 27B GSM8K: max mid-layer KL AUC 0.535 (think-wrong detection), peak across 65 layers 0.627, null-scaffold gap +0.026. n=200.
- 8B GSM8K: max mid-layer KL AUC 0.499 (essentially chance), peak 0.627 across 37 layers, null-scaffold gap −0.001. n=200.
- Retained in §7.x ablation section with plain-text explanation of the negative result. The paper does not claim CTT is a method; it claims paired cross-mode layer-KL is not a viable routing signal on this data.

---

## Pointer to next research thread (not in this paper)

`idea-stage/PSV_PILOT_PLAN.md` documents the Prefix-Sufficiency Verification Cascade pilot, including a pre-registered 2×2 factorial design over (pair vs triple verifier) × (decoupled extraction only vs extraction+free-continuation probe), with baselines = {online IRIS, DTSR-style sufficiency, BAEE} and full accuracy × tokens × wall-clock × verifier-calls reporting. PSV is next-paper work, not a bolt-on to the current manuscript.

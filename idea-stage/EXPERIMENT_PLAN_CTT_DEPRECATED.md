# EXPERIMENT_PLAN — Coupling-Tax Tomography (CTT)

**Paper**: *The Coupling Tax: Why Chain-of-Thought Fails Under Fixed Budgets and What To Do About It*
**Method under test**: CTT — paired-mode layer-wise KL router (see `FINAL_PROPOSAL.md`).
**Remaining compute**: ≈ 40 GPU-h on 2×A100-80GB. All experiments must be training-free.
**Date**: 2026-04-19.

This plan is **claim-driven**: each block maps to a specific claim in §5 of `FINAL_PROPOSAL.md`. Blocks are organized so that blocker signals surface early (E1, E2) and expensive validation (E4, E7, E8) runs only if signal holds.

---

## Legend

- **Priority**: P0 must complete for paper; P1 strengthens; P2 stretch.
- **First-pass** = the minimum run needed to make a go/no-go decision.
- **Follow-up** = deeper validation if first-pass signal positive.
- **Kill condition** = explicit threshold below which we abandon or restructure.

---

## Compute budget table (upfront)

| Block | Claim(s) | First-pass GPU-h | Follow-up GPU-h | Priority |
|-------|----------|------------------|-----------------|----------|
| E1 Oracle ceiling | P1 | 1 | — | P0 |
| E2 CTT feasibility pilot | P2 | 2 | 2 | P0 |
| E3 CTT vs baselines | P3 | 6 | 4 | P0 |
| E4 Cross-scale validation | P4 | 6 | 4 | P0 |
| E5 Ablations | P5 | 3 | 2 | P1 |
| E6 Fano ceiling | P6 | 1 | 0 | P1 |
| E7 Stage-3 stacking | P7 | 3 | 2 | P1 |
| E8 Inverse-scaling curve-fit | (paper §3) | 2 | — | P2 |
| **Totals** | | **24 GPU-h** | **14 GPU-h** | **38 GPU-h** |

Fits the 40 GPU-h envelope with a ~2 h safety buffer.

---

## Run order / dependency DAG

```
E1 ──► E2 ──► E3 ──► E4
                │       │
                ▼       ▼
                E5     E7
                │       │
                ▼       ▼
                E6     [paper core done]
                │
                ▼
                E8 (stretch)
```

- **E1 first** — cheap; establishes the theoretical ceiling any router can hit.
- **E2 gates everything** — if AUC < 0.65 we pivot (see fallbacks).
- **E3 depends on E2 τ and ℓ\* calibrations**.
- **E4 can run in parallel with E3 once E2 calibration is frozen.**
- **E5–E7 independent**, run after E3/E4 produce the headline numbers.
- **E8 stretch**, only if E1–E7 all hit success criteria.

---

## E1 — Oracle-routing upper bound

**Maps to claim P1.** "The ceiling of any routing method."

- **Setup.**
  - Model: Qwen3.5-27B (already have full results).
  - Data: GSM8K full `n=1319` at `b=4096`; MATH-500 `n=500` at `b=4096` (if available).
  - Routing: per-query oracle = pick the mode that answers correctly; if both correct, pick the shorter.
  - Reuse existing `results_kun/fulltest_27b/` and `results_kun/fulltest_27b_nothink/`.
  - No GPU generation — pure post-hoc analysis on saved CSVs.
- **Success criterion.** Oracle accuracy ≥ 98.0% (at least matches nothink alone, because the ceiling must dominate the better of the two modes); oracle avg tokens < 500 on GSM8K at b=4096 (since most queries are easy, the shorter mode wins).
- **First-pass**: compute from existing per-sample CSVs. ~1 GPU-h (really ~0 GPU; allocate 1 h for robust bootstrap of CIs).
- **Follow-up**: none — oracle is analytic.
- **Kill**: if oracle ≤ 98.0% at any avg-tokens budget, the coupling-tax ceiling is lower than claimed and we need to re-examine F_L; but this is essentially impossible given nothink alone hits 98%.
- **Dependencies.** None. Runs first.

---

## E2 — CTT feasibility pilot

**Maps to claim P2.** "Does the signal exist?"

- **Setup.**
  - Model: Qwen3-8B.
  - Data: 200 GSM8K items (reuse the MRSD pilot set for comparability).
  - Procedure:
    - (a) two 1-token prefill passes per query (think, nothink); 200 × 2 forwards on an 8B model ≈ trivial (~0.5 GPU-h).
    - (b) extract hidden states at all layers at the final-prompt-token position.
    - (c) project via `LayerNorm(h) @ W_U`; compute `KL(p_think[ℓ] || p_nothink[ℓ])` per layer.
    - (d) run nothink@1024 and think@1024 generations on the same 200 items to label each query as `nothink-correct`, `think-correct`, both, neither.
    - (e) target event: `{nothink correct AND think wrong}` (the coupling-tax queries).
    - (f) compute AUC of max-mid-layer-KL (`ℓ ∈ [0.4L, 0.6L]`) as a detector.
- **Success.** AUC ≥ 0.70 on target event.
- **First-pass**: 2 GPU-h (200 queries × 2 modes × generation + prefill probes).
- **Follow-up** (if AUC 0.65–0.80): expand to n=500 GSM8K + n=100 MATH-500; add per-layer AUC plots; calibrate τ (Youden-J); re-run with `mean-mid-layer` and `top-3 max` aggregations. ~2 GPU-h.
- **Kill**:
  - **AUC < 0.60** → sign the method fails; pivot to self-KL-subtraction (§4 of EXPERIMENT_PLAN) or restructure paper around Stage-3 alone.
  - **AUC 0.60–0.65** → marginal; proceed but prepare fallback positioning.
- **Dependencies.** E1 (for context only; oracle ceiling sets the AUC we should aspire toward).
- **Artifacts**: `results/ctt_pilot_8b_gsm8k/per_layer_kl.csv`, `labels.csv`, `roc.json`.

---

## E3 — CTT vs baselines (head-to-head)

**Maps to claim P3.** "Pareto dominance at matched compute."

- **Setup.**
  - Models: Qwen3-8B, Qwen3.5-27B.
  - Data: GSM8K (n ≥ 500 per model), MATH-500 (n ≥ 200 per model; full n=500 on 8B if budget permits).
  - Routing compared:
    1. **CTT** (ours, calibrated τ from E2).
    2. **TOWN** cascade (already in repo, `results/uncertainty_router/`).
    3. **Random** routing (50/50).
    4. **HF reasoning-router style** (query-text classifier — approximate via zero-shot prompt-based classifier on the same 8B model; no training).
    5. **JointThinking** (full re-implementation: two full generations then answer-match; expensive).
    6. **P3 learned allocator** (already in repo).
    7. **AdaptThink** (if public checkpoint available; else skip).
  - Matched compute: report tokens, wall-clock, and (tokens-generated + 2·|q|) for CTT.
- **Success.**
  - CTT strictly Pareto-dominates TOWN, random, HF-style, and P3 learned allocator.
  - CTT matches or exceeds JointThinking accuracy at ≤ 10% of its tokens.
  - Minimum headline: **+1.5pp accuracy at matched or fewer tokens** vs best non-CTT baseline, OR **≥ 20% token savings at ≤ 0.5pp accuracy cost** vs strongest baseline.
- **First-pass**: 6 GPU-h.
  - 8B GSM8K n=500 (all 6 methods): ~2 GPU-h.
  - 8B MATH-500 n=200: ~1 GPU-h.
  - 27B GSM8K n=500: ~2 GPU-h.
  - 27B MATH-500 n=200: ~1 GPU-h.
- **Follow-up**: full MATH-500 n=500 on both scales + bootstrap CIs + McNemar paired tests. ~4 GPU-h.
- **Kill**: if any one of {TOWN, P3 learned allocator} Pareto-dominates CTT on GSM8K 8B, stop — either CTT is not the new primitive we claim, or τ is mis-calibrated. Re-examine per-layer AUC and τ sensitivity (feed back into E2 follow-up).
- **Dependencies.** E2 (needs τ, ℓ\*).

---

## E4 — Cross-scale validation

**Maps to claim P4.** "Mid-layer peak invariant across scale."

- **Setup.**
  - Models: Qwen3-8B (layer fraction 40-60% → L14-L22 of 28 layers); Qwen3-9B; Qwen3.5-27B (layers 26-38 of 64).
  - Data: GSM8K n=500 per model; MATH-500 n=200 per model.
  - Protocol:
    - Re-run E2 per-layer-KL computation on each model.
    - Overlay per-layer AUC curves; check where each model's peak sits (as fraction of depth).
    - Freeze `ℓ*` as "fraction ∈ [0.4, 0.6]"; recalibrate τ per model (allowed).
  - Compare peak-layer-fraction across models.
- **Success.**
  - Peak layer fraction within [0.3, 0.7] on all three models.
  - AUC on 27B MATH-500 within 0.05 of AUC on 8B GSM8K.
  - AUC after swapping ℓ* between models (e.g., 8B's range applied to 27B) drops by ≤ 0.05 absolute.
- **First-pass**: 6 GPU-h (27B is compute-heavy; 200-query prefill-only on 27B is feasible on 2×A100 via tensor parallelism).
- **Follow-up**: add DeepSeek-R1-Distill-Llama-8B to check cross-family generalization; 4 GPU-h.
- **Kill**: if peak layer fraction differs by > 0.2 between 8B and 27B, the "coupling-tax theorem predicts universal mid-layer peak" framing is wrong; pivot to per-scale calibration ("one scalar per model") with reduced novelty claim.
- **Dependencies.** E2, E3 (τ calibration procedure frozen).

---

## E5 — Ablations

**Maps to claim P5.** "The right design choices."

Five sub-ablations, all on the E2 pilot set (Qwen3-8B, 200-300 GSM8K queries):

1. **Layer selection**: per-layer AUC sweep (L=0..28), not just the mid-band. Produce the per-layer curve figure for the paper.
2. **Threshold sensitivity**: 5-fold leave-one-out calibration of τ; report AUC stdev and accuracy stdev. Target stdev ≤ 0.02.
3. **Self-KL control**: paired nothink+nothink passes (dropout-seed differences) → compute same KL. Subtract from paired think+nothink KL. Expect residual ≥ 80% of original signal (demonstrates the mode-specificity).
4. **Aggregation**: `max`, `mean`, `top-3 mean`, `entropy-weighted sum`. Rank by AUC. Expect max ≥ mean ≥ top-3 within 0.03.
5. **Alternative divergences**: JSD, total-variation (TV), Bhattacharyya. Target: rank-correlation ≥ 0.9 with KL; same AUC within 0.03.

- **Success.** All 5 ablations either confirm the default choice or reveal a strictly better setting (which we adopt).
- **First-pass**: 3 GPU-h (mostly reuses E2 artifacts; only the self-KL control requires fresh prefill passes).
- **Follow-up**: extend to 27B for any ablation showing ambiguous signal; 2 GPU-h.
- **Kill**: if self-KL subtraction removes > 60% of signal, the CTT signal is dominated by generic mode-independent variance → must frame as a different method or cancel.
- **Dependencies.** E2.

---

## E6 — Fano-derived budget ceiling (theory-check)

**Maps to claim P6.** "Derived quantity; acknowledge arXiv:2604.06192."

- **Setup.**
  - Use empirical `F_L(b)` and `α_c` measured across the sweep.
  - Derive Fano lower bound on the error at each budget: `P(err) ≥ (H(Y | CoT_{:b}, q) − 1) / log|Y|` with Y the answer space.
  - Compare empirical think accuracy to the Fano ceiling across `b ∈ {256, 512, 1024, 2048, 4096}` on Qwen3.5-27B GSM8K.
- **Success.** Empirical think accuracy ≤ Fano ceiling at every tested budget, and gap → 0 as `b → ∞`.
- **First-pass**: 1 GPU-h (pure analysis on existing data).
- **Follow-up**: none.
- **Kill**: if empirical accuracy exceeds Fano ceiling at any budget, our estimate of `H(Y | CoT_{:b}, q)` is wrong — revisit answer-space specification. Not a kill for the paper; reframe as empirical ceiling.
- **Dependencies.** E1 (same data source).

---

## E7 — Stage-3 stacking

**Maps to claim P7.** "CTT + Stage-3 is super-additive on hard queries."

- **Setup.**
  - Three conditions on MATH-500 n=500 with Qwen3-8B:
    - Pure Stage-3 (force think then re-enter nothink for answer).
    - Pure CTT.
    - CTT + Stage-3: CTT routes; for queries routed to think, apply Stage-3 decoupled extraction at generation time.
  - Compare accuracy, tokens, wall-clock.
- **Success.** CTT+Stage-3 ≥ pure CTT + 3pp on MATH-500; super-additivity holds on hard subset (where think is wrong at b=1024 in baseline).
- **First-pass**: 3 GPU-h.
- **Follow-up**: Qwen3.5-27B MATH-500 n=200; 2 GPU-h.
- **Kill**: if CTT+Stage-3 ≤ max(CTT, Stage-3) + 0.5pp, the two methods are redundant; cite each and don't claim additivity.
- **Dependencies.** E3 (CTT must work).

---

## E8 — Inverse-scaling curve-fit (stretch)

**Maps to paper §3 "Inverse scaling of the tax."**

- **Setup.**
  - For each model s ∈ {8B, 9B, 27B}, estimate α_c(s), α_nt(s), F_L(b; s) from existing sweeps.
  - Fit `Tax(s; b) = (1-F_L(b; s))·(α_nt − α_t)` and check the scaling `Tax(s) ∝ g(s)` (e.g., log(s) or √(s)).
  - Predict 14B from the fit and spot-check with one short 14B run if time permits.
- **Success.** The 2.8× 8B→27B scaling derives from F_L shift alone, not from α_c or α_t variation. R² of the fit ≥ 0.9 across three scales.
- **First-pass**: 2 GPU-h (mostly analysis; possible small 14B probe run).
- **Follow-up**: none.
- **Kill**: if F_L shift alone does not reproduce the tax scaling (R² < 0.7), "inverse-scaling as structural consequence" framing weakens — reduce to descriptive claim in the paper.
- **Dependencies.** E1, E6.

---

## Summary of experimental claims and fallbacks

| Claim | Block | Kill pivot if fails |
|-------|-------|---------------------|
| Oracle ≥ 98% at <500 tokens avg (27B b=4096) | E1 | — (essentially guaranteed) |
| AUC ≥ 0.70 on target event | E2 | Self-KL-subtracted variant OR Stage-3-only paper |
| Pareto-dominates TOWN + P3 + HF-style | E3 | If TOWN wins on 8B, re-calibrate ℓ* and τ; else reposition as "matches cheap baselines at 1% compute" |
| Cross-scale invariance of mid-layer peak | E4 | Per-scale calibration; weaker universality claim |
| Ablation consistency | E5 | Drop specific ablated claim (e.g. self-KL) if fails |
| Fano ceiling respected | E6 | Re-estimate `|Y|`; frame as empirical ceiling |
| Super-additivity with Stage-3 | E7 | Present as orthogonal methods |
| 2.8× scale-scaling explained by F_L shift | E8 | Descriptive only |

---

## Statistical rigor required across all blocks

- **Seeds**: 3 seeds (42, 123, 456) for any headline number.
- **Paired tests**: McNemar for same-sample comparisons; bootstrap CIs (n=1000) for accuracy differences.
- **Multiple-comparison correction**: Bonferroni within each claim (claim P3 involves 5 baselines → α = 0.01 per-comparison).
- **Every headline Pareto plot** must include shaded 95% CI bands, not just points.

## Reproducibility

- Every block writes to `results/ctt_<block>/` with: `config.json`, `per_sample.csv`, `summary.json`, `seed_*.log`.
- Environment snapshot (`pip freeze`, `nvidia-smi`) captured at run start.
- Commit hash + command line logged to each summary file.
- All prefill-probe code goes into `scripts/ctt_probe.py` (new).

---

## Execution order — concrete launch list

**Week 1 (days 1–3):**
- Day 1: E1 (existing data, ~2 h wall-clock of analysis); start E2 prefill extraction on 8B in parallel.
- Day 2: E2 first-pass done; go/no-go meeting.
- Day 3: If go, start E3 first-pass (8B); calibrate τ.

**Week 2 (days 4–7):**
- Day 4: E3 first-pass completes; start E4 first-pass (27B prefill probes).
- Day 5: E4 first-pass completes; start E5 ablations (all on 8B).
- Day 6: E5 done; E6 analysis (no GPU needed); start E7.
- Day 7: E7 done; decide on E8 (stretch).

**Week 3 (days 8–10):**
- Day 8: E3 and E4 follow-ups (full MATH-500 n=500).
- Day 9: E8 if green; start writing experiments section.
- Day 10: Buffer for re-runs / reviewer-proof variants.

---

*Word count (body, §E1-E8 + supporting): ~2,300.*

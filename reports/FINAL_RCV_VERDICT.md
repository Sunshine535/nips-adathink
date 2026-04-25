# Final RCV-IRIS Verdict (2026-04-25)

**Decision: DROP RCV from main method. Freeze as negative ablation.**

## Complete A/B/C/D Evidence (V2 honest accounting)

Setup: Qwen3-8B, MATH-500 n=200, seed=42, b1=256, b2_max=512, b_answer=128, same sample manifest.

| Variant | Mechanism | Accuracy | Avg Tokens | s0/s2/strict/soft |
|---------|-----------|----------|------------|-------------------|
| A (existing_fragment) | No gate, no soft probe | **41.0%** | **684** | 245/428/12/0 |
| B (rcv_no_gate) | Soft probe, no gate | 41.0% | 717 | 245/428/12/33 |
| C (full_rcv) | Gate + TOWN fallback | 41.0% | 720 | 245/430/12/33 |
| D (full_rcv_majvote) | Gate + majority vote fallback | 41.0% | 717 | 245/428/12/33 |

## Pairwise McNemar (n=200)

| Pair | Discordant | Exact p |
|------|-----------|---------|
| A vs B | 0 | 1.0 |
| A vs C | 0 | 1.0 |
| A vs D | 0 | 1.0 |
| B vs C | 0 | 1.0 |
| B vs D | 0 | 1.0 |
| C vs D | 0 | 1.0 |

**Every pair: 0 discordant, p=1.0. Every variant is accuracy-identical.**

## Gate Decision Distributions

- A (no gate): EXTRACT_ALWAYS=167, ACCEPT_STAGE0=33
- B (no gate): EXTRACT_ALWAYS=167, ACCEPT_STAGE0=33 (same as A but with soft probe generated unused)
- C (gate + TOWN): EXTRACT_STAGE3=159, ACCEPT_STAGE0=33, FALLBACK_TOWN=8
- D (gate + majvote): EXTRACT_STAGE3=159, ACCEPT_STAGE0=33, FALLBACK_MAJVOTE_1of1=5, FALLBACK_NONE_VALID=1, FALLBACK_MAJVOTE_1of2=1, FALLBACK_MAJVOTE_2of2=1

## Root Cause of Null Result

On the 8 samples where the gate triggers FALLBACK:
- **5/8**: only 1 of {strict, soft, town} candidates produced a valid parseable answer → single-vote outcome dominates, same as C's TOWN-only fallback
- **1/8**: zero valid candidates (model genuinely cannot extract) → all methods fail
- **1/8**: only 2 candidates valid, 1 each agreed → weak signal, fallback guess wrong
- **1/8**: 2 candidates agreed (strongest signal) — this is the only sample where majvote meaningfully voted, and it was still wrong on this sample

**Mechanism insight**: The gate correctly identifies samples where the model cannot extract. But on those samples, NO post-hoc extraction method applied to the same truncated thinking prefix succeeds, because the prefix genuinely lacks the answer. No fallback from the same prefix can help.

## Per GPT-5.5 Stop Criteria

The diagnosis stated:
> STOP: If Full RCV-IRIS does not beat both A and B on paired same-sample comparison.

**Condition met.** C does not beat A or B at any budget tested (both b2=4096 and b2=512). Revised mechanism (D, majority vote) also fails.

The hypothesis "recoverability-calibrated routing helps" is **falsified** for post-hoc fallback actions on the same prefix. Would only survive if:
1. Model-based verifier could detect and ROUTE to a different computation path (not just fallback within same prefix) — not tested, but requires extra compute per sample
2. Fallback action could trigger continuation of thinking (extra budget) — violates fixed-budget premise

Neither is viable for the paper's budget-constrained regime.

## Paper Implications

### Dropped from Main Method

RCV-IRIS is **NOT a contribution** of this paper. The paper's main method reverts to:
- IRIS cascade (Stage 0 nothink triage → Stage 2 thinking → Stage 3 decoupled extraction)
- **No acceptance verifier, no recoverability gate**

### Negative Ablation to Report

Include in paper as Section X.Y "Ablation: Recoverability Gate":
> "We tested a feature-based recoverability-calibrated gate (RCV-IRIS) with two fallback actions (TOWN parse and majority vote of 3 extraction candidates). Under honest token accounting (soft probe tokens counted), both variants produced 0 discordant pairs against the no-gate baseline at MATH-500 n=200 (p=1.0). Samples where the gate triggers are those where no post-hoc extraction from the same truncated prefix succeeds, because the prefix genuinely lacks the answer. This null result is reported as an honest negative ablation."

### Main Paper Contributions (UNCHANGED)

1. **Coupling Tax phenomenon**: 27B GSM8K nothink 98.0% vs think 87.5% (p<1e-5)
2. **2×2 mode×prompt factorial**: +37.4pp interaction (non-additive synergy of mode switch + extraction prompt)
3. **Training-free Pareto-competitive** with SwiReasoning (ICLR 2026) and s1 (budget forcing) on MATH-500
4. **Stage-3 decoupled extraction**: +17pp on 27B MATH-500 via mode-switch + extraction prompt combo

## Commits

- `7e2c358`: V2 bug fixes (token accounting + manifest + A/B split + GSM8K loader + benchmark-aware verifier)
- `2ffdfc7`: V3 revised mechanism (full_rcv_majvote)
- `TBD`: This final verdict + ablation_suite report + drop RCV from method

## Files Produced

- `results/rcv_v2_b512/rcv_*_math500_20260424_203506.json` (A), `20260424_203505.json` (B, C)
- `results/rcv_v2_b512/rcv_full_rcv_majvote_math500_*.json` (D)
- `results/rcv_v2_b512/ablation_suite_report.json` (4-way analysis)
- `reports/FINAL_RCV_VERDICT.md` (this file)

## Decision for Submission

**Submit the paper WITHOUT RCV as a contribution.** Include RCV as honest negative ablation. Main paper stands on Coupling Tax + factorial interaction + Pareto-competitive training-free method.

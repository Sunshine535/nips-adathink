# Reviewer Memory

Persistent reviewer memory across `/auto-review-loop` rounds. Each round appends — never delete. Fed back to GPT every round so it tracks its own suspicions.

## Context Reset (2026-04-23)

Fresh start. Previous rounds (medium R1-R4 on MRSD pivot + nightmare R1 on IRIS) are archived.

### Current paper state
- **Title**: "The Coupling Tax: When Chain-of-Thought Costs More Than It Saves"
- **Method**: IRIS — 3-stage cascade (nothink triage → thinking → decoupled Stage-3 extraction)
- **Key positive results** (all McNemar paired):
  - 8B GSM8K n=1319: IRIS 90.9% vs TOWN 86.0%, p=1.6e-17
  - 27B MATH-500 n=200: IRIS 77.5% vs TOWN 49.0%, p=3.5e-11
  - 27B GSM8K n=200: IRIS 93.5% vs TOWN 90.0%, p=0.0156
  - IRIS vs budget-forcing s1: +2.1pp at -27% tokens (Pareto win)
  - Stage-3 extraction: +17pp (60.5%→77.5%) on 27B MATH-500
  - 27B coupling tax crossover: nothink 98.0% vs think 87.5%, p<1e-5
- **Negative/null results**: CTT tomography null (AUC≈0.5), entropy stopping 0/200, online-stage2 67.5% vs 77.5% post-hoc
- **Baselines already implemented**: TOWN, budget-forcing early_stop (s1), nothink, thinking
- **NOT yet implemented**: NoThinking prefill, DeepConf, JointThinking, BAEE, budget-forcing wait_extend

### What the human lead demands
- Positive-results paper, NOT negative-results-proving paper
- Full SOTA comparison: find same-category methods, implement from official GitHub, compare in unified environment
- If results are negative, diagnose root cause before concluding

### Known integrity issues to verify
- 477 vs 460 token seed-mixing: should be fixed everywhere
- Stage-3 table baseline: 89.0% not 90.0%
- Post-hoc vs online accounting: headline IRIS numbers are post-hoc effective-token accounting
- 8B MATH-500 IRIS vs TOWN: not significant (p=0.44 pooled)

## Round 1 — Score: 3/10 (2026-04-23)
- **Suspicion**: Post-hoc accounting inflates IRIS from 67.5% (real) to 77.5% (post-hoc). This is the #1 credibility risk.
- **Suspicion**: Stage-3 novelty unproven — without BAEE comparison, could be "just more answer budget"
- **Suspicion**: Multi-seed claims mixing n=500 with n=200 — not a clean stability analysis
- **Unresolved**: 5 mandatory SOTA baselines missing (DeepConf, SwiReasoning, Thinkless, BAEE, s1 wait_extend)
- **Unresolved**: Framing as method paper vs phenomenon paper — reviewer says phenomenon is strong, method is not
- **Patterns**: Evidence inflation tendency — watch for pilot being presented as definitive

## Round 2 — Score: 6-6.5/10 (2026-04-23)
- **Previous suspicions addressed?**:
  - Missing baselines: YES — SwiReasoning head-to-head run, s1 both variants, DeepConf/BAEE/Thinkless justified
  - Stage-3 novelty: PARTIALLY — ablation is strong (47/49) but confounded by prompt difference (free-cont gets raw continuation, mode-switch gets extraction prompt)
  - Post-hoc accounting: NOT ADDRESSED — still using 77.5% headline, online is 67.5%
- **New suspicions**:
  - MATH-500 IRIS vs SwiR: +0.6pp at n=200 is statistically meaningless without paired McNemar
  - Ablation confound: free-continuation uses thinking-mode continuation, mode-switch uses nothink extraction prompt with "Final answer:" scaffold — not a pure mode-only test
  - Multi-seed still mixes n=500 and n=200
- **Unresolved**:
  - Post-hoc vs online must be quarantined in every table
  - Need paired same-sample McNemar for IRIS vs SwiR on MATH-500
  - Need MATH-500 ablation
  - Need pure mode-only ablation (same prompt, only enable_thinking differs)

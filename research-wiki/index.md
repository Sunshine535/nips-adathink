# Research Wiki Index

> Project: **The Coupling Tax: How Shared Token Budgets Waste Chain-of-Thought Reasoning**
> NeurIPS 2026 submission — Round 4 review: 7.5/10 (best-paper scale)

## Papers

| Slug | Title | Relevance |
|------|-------|-----------|
| [wei2022_chain](papers/wei2022_chain.md) | Chain-of-Thought Prompting Elicits Reasoning | core |
| [snell2024_scaling](papers/snell2024_scaling.md) | Scaling LLM Test-Time Compute Optimally | core |
| [chen2024_overthinking](papers/chen2024_overthinking.md) | Do NOT Think That Much for 2+3=? | core |
| [deepseekr1_2025](papers/deepseekr1_2025.md) | DeepSeek-R1: Incentivizing Reasoning Capability | core |
| [muennighoff2025_s1](papers/muennighoff2025_s1.md) | s1: Simple Test-Time Scaling | related |
| [adathink2025](papers/adathink2025.md) | AdaThink: Adaptive Thinking Token Allocation | related |
| [nvidia2025_pareto](papers/nvidia2025_pareto.md) | Pareto-Optimal Reasoning with STILL | related |

## Ideas

| ID | Title | Stage | Outcome |
|----|-------|-------|---------|
| [idea:001](ideas/001_coupling_tax_mrsd.md) | Coupling Tax + MRSD Split-Budget | active | **positive** (IRIS validated on MATH-500 + GSM8K) |

## Experiments

| ID | Description | Status |
|----|-------------|--------|
| [exp:coupling_tax_gsm8k](experiments/exp_coupling_tax_gsm8k.md) | 8B GSM8K budget sweep | completed |
| [exp:coupling_tax_math500](experiments/exp_coupling_tax_math500.md) | 8B MATH-500 budget sweep | completed |
| [exp:mrsd_gsm8k_pilot](experiments/exp_mrsd_gsm8k_pilot.md) | MRSD pilot on GSM8K (n=200) | completed |
| [exp:mrsd_math500_pilot](experiments/exp_mrsd_math500_pilot.md) | MRSD pilot on MATH-500 (n=200) | completed |
| [exp:cross_scale](experiments/exp_cross_scale.md) | Cross-scale coupling tax (8B/9B/27B) | completed |
| [exp:mrsd_27b](experiments/exp_mrsd_27b.md) | 27B MRSD on GSM8K + MATH-500 | completed |
| [exp:bbh](experiments/exp_bbh.md) | BBH 5-task non-math reasoning (n=1187) | completed |
| [exp:iris_math500_fullscale](experiments/exp_iris_math500_fullscale.md) | **IRIS full-scale MATH-500 (n=500)** | completed |
| [exp:town_math500_fullscale](experiments/exp_town_math500_fullscale.md) | **TOWN full-scale MATH-500 (n=500)** | completed |
| [exp:deepseek_replication](experiments/exp_deepseek_replication.md) | **DeepSeek-R1 full replication** | **running** |

## Claims

| ID | Claim | Status | Strength |
|----|-------|--------|----------|
| [claim:C1](claims/C1_coupling_tax.md) | Nothink dominates think at all budgets ≤2048 | supported | strong |
| [claim:C2](claims/C2_inverse_scaling.md) | Coupling tax amplifies with model scale | supported | moderate |
| [claim:C3](claims/C3_natural_stop_oracle.md) | Natural stop = free confidence oracle (99% PPV) | supported | strong |
| [claim:C4](claims/C4_mrsd_effective_gsm8k.md) | MRSD effective on saturated GSM8K (+5pp) | supported | moderate |
| [claim:C5](claims/C5_mrsd_fails_math500.md) | ~~MRSD loses to nothink@1024 on MATH-500~~ | **invalidated** | disproven |
| [claim:C6](claims/C6_27b_cascade_failure.md) | Cascade hurts at 27B scale (insufficient B_think) | reported | preliminary |
| [claim:C7](claims/C7_iris_beats_nothink_math500.md) | **IRIS beats nothink@1024 on MATH-500 (+14.2pp)** | **supported** | **strong** |
| [claim:C8](claims/C8_coupling_tax_cross_task.md) | **Coupling tax generalizes to BBH non-math tasks** | **supported** | **strong** |

## Gaps

See [gap_map.md](gap_map.md) for full gap analysis.

| ID | Gap | Status |
|----|-----|--------|
| G1 | No theory for truncation waste under budget-constrained CoT | resolved |
| G2 | No systematic thinking vs nothink at matched budgets | resolved |
| G3 | Split-budget overcoming coupling tax | **resolved** |
| G4 | Minimum B_think per model scale | unresolved |
| G5 | Cross-family validation (only Qwen tested) | **in progress** |
| G6 | Budget-scaling ceiling hypothesis | **resolved** |
| G7 | Coupling tax on non-reasoning tasks | unresolved |

---
*Auto-generated. Last updated: 2026-04-12*

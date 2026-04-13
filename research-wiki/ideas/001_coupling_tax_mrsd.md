---
type: idea
node_id: idea:001
title: "Coupling Tax + Split-Budget Generation (MRSD/IRIS)"
stage: validated
outcome: positive
pilot_signal: "GSM8K +5.0pp (94.0% vs 89.0%); MATH-500 IRIS@4096 74.0% vs nothink@1024 59.8% (+14.2pp)"
created_at: 2026-04-09
updated_at: 2026-04-12
---

# The Coupling Tax: Split-Budget Generation via MRSD/IRIS

## One-line thesis
Coupling visible reasoning and answer in one output stream wastes tokens through truncation; decoupling via split-budget (think@B_r + nothink@B_a) recovers lost reasoning value.

## Hypothesis
- Thinking is ~99% accurate when chains complete (α_c on GSM8K)
- At budget 256, 98.6% of chains truncated — truncation is the failure mode, not reasoning quality
- Feeding truncated reasoning to nothink mode "decodes" answers effectively (decoupled answering)
- The coupling tax decomposition predicts accuracy from chain-length CDF alone

## Key Evidence — VALIDATED

### GSM8K (saturated regime)
- MRSD pilot (n=200): 94.0% vs nothink 89.0% (+5.0pp), 10 unique wins, 0 losses (sign test p=0.001)

### MATH-500 (hard regime) — DEFINITIVE
| Method | n | Accuracy | 95% CI | vs nothink@1024 |
|--------|---|----------|--------|-----------------|
| nothink@1024 | 500 | 59.8% | [55.4, 64.0] | baseline |
| IRIS@2048 | 500 | 67.2% | [63.0, 71.2] | +7.4pp |
| IRIS@4096 | 500 | **74.0%** | [70.0, 77.7] | **+14.2pp** |
| TOWN@2048 | 500 | 55.0% | [50.6, 59.3] | -4.8pp |
| TOWN@4096 | 500 | 71.8% | [67.7, 75.6] | +12.0pp |

- CI lower bound (70.0%) > nothink point estimate (59.8%)
- IRIS > TOWN at every budget (decoupled answering is key mechanism)
- Cross-budget: IRIS@2048→4096 = +6.8pp (McNemar p=0.0004)

### BBH (cross-task generalization)
- 5 tasks, n=1187: tax = +33.3pp@256, crossover at 1024-2048
- Per-task crossovers vary from ~512 (boolean) to >2048 (object_tracking)

### Key mechanism insight
On 106 hard MATH-500 samples: TOWN 10.4% → IRIS 35.8% (+25.4pp from decoupled answering alone)

## Novelty
PARTIALLY NOVEL
- Core: decoupled answer generation + coupling-tax theory framework (NOVEL)
- Components: mode switching, truncated reasoning, iterative refinement (PRIOR WORK)
- Closest: SwiReasoning (ICLR 2026) — different mechanism

## Reviewer Feedback (Round 4)
- GPT-5.4 NeurIPS review: 7.5/10 (best-paper scale), ~9/10 (accept scale)
- "Strong accept, top 5-10%, NOT top 1-3%"
- Gap to best paper: cross-family DeepSeek replication (running)

## Open Questions
1. Does coupling tax + IRIS generalize to DeepSeek-R1? (exp:deepseek_replication RUNNING)
2. What is minimum B_think for 27B cascade to help? (scoped as future work)

## Connections
- inspired_by: paper:wei2022_chain, paper:snell2024_scaling, paper:chen2024_overthinking
- addresses_gap: G1, G2, G3, G6
- tested_by: exp:coupling_tax_gsm8k, exp:coupling_tax_math500, exp:mrsd_gsm8k_pilot, exp:mrsd_math500_pilot, exp:cross_scale, exp:bbh, exp:iris_math500_fullscale, exp:town_math500_fullscale, exp:deepseek_replication

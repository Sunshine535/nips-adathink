# External Review Round 1 — GPT-5.4 as NeurIPS Reviewer

**Date**: 2026-04-09  
**Score**: 4/10 (reject, path to weak reject/borderline)

## Key Criticisms

### C1: Core Claim Too Broad
> "Your decomposition undermines the broad claim. α_c = 98.27% means thinking isn't bad, just truncated. This is a dressed-up statement of: if you cap long outputs, they get cut off."

**Response**: Accept this critique. Reframe from "thinking tax" to "coupling tax" — the problem is forcing reasoning + answer through one output channel.

### C2: Budget Model Unfair
> "Fixed output-token budget is adversarial to visible-CoT. Why not wall-clock, FLOPs, or dollar cost?"

**Response**: Must add multiple cost metrics. Plan: tokens, wall-clock latency, params × tokens.

### C3: MRSD Incremental
> "Standard cascade + self-refinement + answer extraction. Not sufficiently novel."

**Response**: MRSD may not be the headline method. Consider simplifying to "decoupled answer channel" as the key insight.

### C4: GSM8K Ceiling Low
> "Oracle gap +0.76pp. Method ceiling is low on your flagship dataset."

**Response**: Move MATH-500 to main benchmark. GSM8K becomes sanity check.

## Required Experiments (Reviewer's List)
- [ ] Stronger baselines: nothink@k with more tokens, SC under matched budget, concise reasoning
- [ ] Fairer budget axes: total tokens, wall-clock, FLOPs
- [ ] Budget sweep beyond 1024
- [ ] More tasks with genuinely hard reasoning
- [ ] More models (stronger families)
- [ ] MRSD ablations: remove triage, remove hinting, 1-round vs 2-round
- [ ] Calibration analysis: predict truncation risk before decoding
- [ ] Statistical rigor: CIs, paired tests, multi-seed

## Recommended Pivot: "Coupling Tax + Split Budget"

### New Claim
Not: "Chain-of-thought costs more than it saves"  
Better: "Under budget constraints, coupling visible reasoning and answer in one output stream causes truncation waste; decoupled generation recovers utility"

### New Method
Drop MRSD as headline. Use simple split-budget:
- think@Br for reasoning (allow truncation — it's OK)
- nothink@Ba to extract answer from truncated trace
- Key insight: truncated reasoning is still USEFUL (α_trunc > 0)

### New Main Benchmark  
MATH-500 (hard, more headroom) + GSM8K (easy, sanity check)

### New Cost Metrics
Report all of: output tokens, wall-clock latency, params × generated tokens

## Evidence Supporting Pivot
- α_c = 98.27% → thinking is GOOD, not bad
- Decoupled answering: 76.9% vs 15.4% (+61.5pp on hard)
- MRSD @50 checkpoint: 88.0% vs 80.0% nothink (+8.0pp on GSM8K)
- Oracle gap on GSM8K tiny (+0.76pp) but MATH-500 has real headroom

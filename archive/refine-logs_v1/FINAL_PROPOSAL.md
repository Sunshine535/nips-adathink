# Refined Proposal: The Coupling Tax

**Version**: v1 (post-review R1)  
**Date**: 2026-04-09  
**Reviewer Score**: 6.5/10 (up from 4/10 after pivot)  
**Target**: 7.5-8/10 with MATH-500 results

---

## Problem Anchor

**Anchored claim**: Under output-token budgets, coupling visible reasoning and final answer in one output stream causes **truncation waste** — a systematic failure mode where partially-completed reasoning chains destroy the model's ability to emit correct answers. We call this the **Coupling Tax**.

**What this is NOT about**: Thinking is not inherently bad. When chains complete, thinking achieves 98.27% accuracy (α_c). The problem is purely architectural: one output channel serves two incompatible functions.

---

## Method Thesis

**Split-Budget Generation**: Decouple reasoning from answering by allocating separate token budgets:
- **Reasoning pass** (think@Br): Generate reasoning chain, allow truncation
- **Answer extraction** (nothink@Ba): Feed truncated reasoning to nothink mode for answer extraction
- **Key insight**: Truncated reasoning is NOT wasted — it contains useful information (α_trunc = 3-41% depending on truncation severity)

**MRSD as Iterative Refinement**: When one-pass split-budget fails:
- Round 2: think_with_hint → nothink_extract (refine using previous answer)
- Convergence: stop when two consecutive rounds agree
- Majority vote as final fallback

---

## Dominant Contribution

**Theoretical**: First principled decomposition of reasoning model performance under budget constraints:
```
acc(b) = α_c · F_L(b) + α_trunc(b) · (1 - F_L(b))
```
where F_L is the chain-length CDF, α_c ≈ 98% is the chain-completion accuracy, and α_trunc captures partial reasoning value.

**Methodological**: Training-free split-budget generation that recovers reasoning value from truncated chains by decoupling reasoning from answering.

**Empirical**: Comprehensive analysis across 3 benchmarks × 3+ model sizes showing:
1. Coupling tax exists at all tested budgets (≤2048)
2. Tax grows with model size (inverse scaling: 2.8× from 8B to 27B)
3. Split-budget recovers +5.0pp on GSM8K, more on MATH-500 (pending)

---

## Pilot Results Summary

### GSM8K (n=200, Qwen3-8B)

| Method | Accuracy | Avg Tokens | Gain vs Nothink |
|--------|----------|-----------|-----------------|
| **MRSD** | **94.0%** | 235 | **+5.0pp** |
| IRIS (1-round split) | 93.5% | 181 | +4.5pp |
| TOWN (cascade) | 89.0% | 180 | +0.0pp |
| Nothink@256 | 89.0% | 140 | baseline |

- 10 unique wins, 0 unique losses (perfect directional gain)
- 87.5% rescue rate on escalated samples (14/16)
- 98% convergence rate

### MATH-500 (pending — running on GPU server)
Expected to show larger gains due to:
- Nothink@1024 = 59.8% (much more headroom)
- Think@1024 = 18.0% (heavy truncation → more room for split-budget)
- α_trunc should be lower → split-budget has more value to add

### CDF Decomposition (validated)
- α_c = 98.27% (stable across all budgets)
- F_L ~ LogNormal(μ=6.42, σ=0.52)
- Oracle RMSE = 0.18pp (exact decomposition)
- Crossover: b* ≈ 97th percentile of chain CDF

### Portfolio Analysis (validated)
- Nothink dominates at all budgets ≤1024
- Oracle gap @512 = +0.76pp (low ceiling on GSM8K)
- Best cascade: nothink@512 → think@1024 = 95.0%

---

## Experiment Plan

### Must-Run (Critical Path)

| ID | Experiment | GPU-h | Purpose |
|----|-----------|-------|---------|
| E1 | MRSD on MATH-500 (n=200, pilot) | 2 | **🔄 Running** — validate on hard benchmark |
| E2 | Split-budget sweep on MATH-500 | 4 | Core evidence for coupling tax |
| E3 | Full-scale MRSD on GSM8K (n=1319) | 4 | Statistical power for main claim |
| E4 | Full-scale MRSD on MATH-500 (n=500) | 6 | Hard benchmark, primary evidence |
| E5 | Wall-clock latency measurement | 1 | Address reviewer C2 |
| E6 | Multi-seed (seeds {42, 123, 456}) | 6 | Statistical rigor |

### Should-Run (Strength)

| ID | Experiment | GPU-h | Purpose |
|----|-----------|-------|---------|
| E7 | MRSD on BBH 5-subtask (n=1187) | 4 | Cross-benchmark generalization |
| E8 | MRSD on 27B (n=200 GSM8K + MATH) | 4 | Cross-scale validation |
| E9 | Ablation: remove hint, 1-round only | 2 | Method ablation |
| E10 | Strong baselines: SC@4 matched budget | 2 | Fair comparison |

### Nice-to-Have

| ID | Experiment | GPU-h | Purpose |
|----|-----------|-------|---------|
| E11 | DeepSeek-R1 pilot (n=200) | 2 | Cross-model family |
| E12 | Budget beyond 2048 | 3 | Show where coupling tax disappears |
| E13 | Nothink confidence routing | 2 | Smarter Stage 1 |

**Total critical path**: ~23 GPU-hours  
**Total all experiments**: ~42 GPU-hours

---

## Paper Structure (Revised)

1. **§1 Introduction**: The coupling tax — coupling reasoning and answer in one output wastes tokens
2. **§2 Background**: Reasoning models, dual-mode architecture, test-time compute
3. **§3 The Coupling Tax**: 
   - Truncation statistics (98.6% truncated at B=256)
   - CDF decomposition: acc(b) = α_c·F_L(b) + α_trunc(b)·(1−F_L(b))
   - α_c ≈ 98% stability — thinking IS accurate, truncation is the problem
   - Crossover budget: b* ≈ F_L⁻¹(α_nt/α_c)
4. **§4 Split-Budget Generation**:
   - Motivation: decouple reasoning from answering
   - think@Br + nothink@Ba as simplest instantiation
   - MRSD: iterative refinement when one-pass fails
   - Budget allocation analysis
5. **§5 Experiments**:
   - Main: MATH-500 (hard) + GSM8K (easy validation)
   - Cross-scale: 8B, 9B, 27B
   - Cost metrics: tokens, wall-clock, params×tokens
   - Ablations: rounds, budget splits, convergence criteria
6. **§6 Discussion**:
   - Implications for reasoning model design
   - When NOT to decouple (b >> b*)
   - Connection to dual-process theory (System 1/2)

---

## Reviewer Feedback Integration

| Criticism | Status | How Addressed |
|-----------|--------|---------------|
| C1: Claim too broad | ✅ Fixed | "Coupling tax" not "thinking tax" |
| C2: Budget model unfair | 🔄 Planned | Add wall-clock, params×tokens |
| C3: MRSD incremental | ✅ Fixed | "Split budget" as principle, MRSD as instance |
| C4: GSM8K ceiling low | 🔄 Pending | MATH-500 as main benchmark |
| Stronger baselines | 🔄 Planned | SC@k, concise reasoning, answer-first |
| Statistical rigor | 🔄 Planned | Multi-seed, CIs, paired tests |

---

---

## Novelty Assessment (Phase 3 Complete)

**Verdict: PARTIALLY NOVEL** — Core innovation (decoupled answer generation + cross-mode iterative refinement) is novel. Individual building blocks have prior work.

**Key differentiators from closest work:**
- vs **SwiReasoning** (ICLR 2026): they switch latent/explicit spaces; we switch think/nothink modes with cross-mode context feeding
- vs **TRSD** (arXiv:2603.13274): they distill truncated reasoning at training time; we do it at inference time
- vs **SABER** (arXiv:2508.10026): they use RL for 4-level routing; we're training-free with convergence-based stopping
- vs **Fractured CoT** (arXiv:2505.12992): they show truncated CoT ≈ full CoT; we actively exploit this by decoupling the answer channel

**Papers to cite**: SwiReasoning, TRSD, SABER, Fractured CoT, PATS, PDR, Pangu Embedded

---

*Last updated: 2026-04-09 05:10 UTC*

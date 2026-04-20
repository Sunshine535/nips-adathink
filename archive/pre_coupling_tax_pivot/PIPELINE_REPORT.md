# Research Pipeline Report: AdaThink

**Direction**: Adaptive test-time compute control for LLMs — dynamically adjust thinking depth based on question difficulty
**Project**: nips-adathink
**Date**: 2026-03-26
**Pipeline**: idea-discovery → implement → run-experiment → auto-review-loop

## Journey Summary

### Stage 1: Idea Discovery (Completed)
- **Proposal**: AdaThink — adaptive budget allocation for test-time compute
- **Core hypothesis**: Fixed budgets create structured heterogeneity (overthinking easy / undercomputing hard)
- **Approach**: Training-free, inference-only controller using difficulty features from probe pass

### Stage 2: Implementation (Completed)
- Template controller: exhaustive search over 3-bit lookup tables
- Parametric controller: linear policy over continuous features
- Value-based controller: per-budget correctness prediction + cost-aware decision
- Multi-benchmark support: GSM8K, MATH500, BBH
- Multi-model support: Qwen3-8B, Qwen3.5-27B
- Leave-one-subset-out cross-validation with utility objective

### Stage 3: Experiments (Partially Complete)

| Experiment | Status | Key Result |
|-----------|--------|------------|
| GSM8K-27B (23 seeds, n=920) | ✅ Done | ΔAcc=+14.2pp [+12.0, +16.5] |
| MATH500-27B (17 seeds, n=680) | ✅ Done | ΔAcc=+12.1pp [+8.8, +15.3] |
| BBH-27B (17 seeds, n=680) | ✅ Done | ΔAcc=+11.3pp [+8.8, +14.0] |
| MATH500-8B (9 seeds, n=360) | ✅ Done | ΔAcc=+28.9pp [+24.2, +33.9] |
| BBH-8B (7 seeds, n=280) | ✅ Done | ΔAcc=+12.1pp [+7.5, +16.8] |
| GSM8K-8B value ctrl (7 seeds) | ✅ Done | ΔAcc=+4.6pp [+0.7, +8.6] |
| Ablation study (5 settings × 5) | ✅ Done | Full > Halting-only varies by setting |
| Latency analysis | ✅ Done | 26-35% latency reduction on MATH/BBH |
| Cross-benchmark transfer | ✅ Done | MATH500→BBH: +13.6pp (matches in-domain) |
| Full-dataset validation (8B) | ✅ Done | Max-tier diff = 0.4pp vs subsets |
| SC baselines | ✅ Done | Best SC (42.5%) << Template (60.4%) |
| Error taxonomy | ✅ Done | Overthinking dominant on GSM8K |
| **DeepSeek-R1-Distill-Llama-8B** | 🔄 Running | Server 1: GSM8K 40/1319 (~11h remaining) |
| **27B full-dataset (GSM8K+MATH500)** | 🔄 Running | Server 2: GSM8K 10/1319 (~33h remaining) |

### Stage 4: Auto Review Loop

| Round | Score | Verdict | Key Issues |
|-------|-------|---------|------------|
| 1 | 4/10 | Not ready | 40-question subsets, missing baselines, thin novelty |
| 2 | 5/10 | Not ready | Partially addressed baselines, significance |
| 3 | 6/10 | Borderline | Full-dataset pending, non-Qwen pending |
| 4 | 6.5/10 | Almost | Cross-family validation critical, framing improved |

### Stage 5: Paper Status
- **Title**: AdaThink: Adaptive Budget Allocation for Test-Time Compute in LLMs
- **Structure**: 9 pages main + comprehensive appendix (12 subsections)
- **Figures**: 7 publication-quality figures (Pareto curves, forest plot, etc.)
- **Tables**: 14 tables across main text and appendix
- **NeurIPS checklist**: Completed
- **LaTeX**: Compiles successfully

## Active Experiments

### Server 1 (216.81.151.3:11839)
- **GPU**: NVIDIA A100 80GB PCIe
- **Task**: DeepSeek-R1-Distill-Llama-8B experiments
  - Phase 1: GSM8K full-dataset (n=1319, budgets 256/512/1024) — IN PROGRESS
  - Phase 2: MATH500 full-dataset (n=500, budgets 1024/2048/4096) — PENDING
  - Phase 3: GSM8K 7-seed subsets (n=40 each) for template controller
  - Phase 4: MATH500 7-seed subsets for template controller
- **ETA**: ~20 hours total

### Server 2 (216.81.245.127:15276)
- **GPU**: NVIDIA A100-SXM4-80GB
- **Task**: Qwen3.5-27B full-dataset experiments
  - Phase 1: GSM8K full-dataset (n=1319, budgets 128/256/512) — IN PROGRESS
  - Phase 2: MATH500 full-dataset (n=500, budgets 2048/4096/8192) — PENDING
- **ETA**: ~45 hours total

## Paper Improvements Made (Round 4)

1. **Abstract rewritten**: Leads with empirical finding (structured compute heterogeneity), scoped to Qwen family
2. **Introduction tightened**: Cross-benchmark transfer elevated, claims narrowed to tested scope
3. **Conclusion restructured**: Emphasizes empirical contribution over method novelty
4. **Limitations updated**: Explicit about Qwen-only validation, future work directions

## Path to 7/10+

1. ✅ **Completed**: Paper framing improvements (local)
2. 🔄 **In Progress**: DeepSeek non-Qwen replication (Server 1, ~20h)
3. 🔄 **In Progress**: 27B full-dataset validation (Server 2, ~45h)
4. ⏳ **After experiments**: Integrate DeepSeek results into paper
5. ⏳ **After experiments**: Run Round 5 auto-review with updated evidence

## Files Changed
- `paper/main.tex` — Rewritten abstract
- `paper/sections/introduction.tex` — Tightened claims and scientific framing
- `paper/sections/conclusion.tex` — Restructured with explicit limitations
- `scripts/deploy_deepseek_full.sh` — DeepSeek full experiment pipeline
- `scripts/deploy_27b_fulltest.sh` — 27B full-dataset pipeline
- `scripts/analyze_deepseek_results.py` — DeepSeek results analysis script
- `AUTO_REVIEW.md` — Round 4 documented
- `REVIEW_STATE.json` — State persistence for review loop
- `PIPELINE_REPORT.md` — This report

## Estimated Timeline
- **Now**: Paper improvements done, experiments running
- **+20h**: DeepSeek results available → integrate into paper
- **+45h**: 27B full-dataset results → integrate into appendix
- **After results**: Run Round 5 auto-review → final submission readiness assessment

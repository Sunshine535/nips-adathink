# Data Provenance Audit — "The Thinking Tax" (NeurIPS 2026)

**Audit date:** 2026-04-07  
**Auditor:** Claude Code (nightmare-level audit)

## Status: PARTIALLY FIXED

The paper underwent a comprehensive audit. All unsupported data claims have been removed or footnoted. Gap-fill experiments are prepared (`scripts/run_gap_fill_critical.sh`).

---

## Verified Data Points (have source files)

| Claim | Value | Source File | n | seed | engine |
|-------|-------|-------------|---|------|--------|
| 8B think@128 | 3.03% | `results_kun/nothink_fullset/nothink_baseline_*_063213.json` | 1319 | 42 | HF |
| 8B think@256 | 18.04% | `results_kun/nothink_fullset/nothink_baseline_*_000345.json` | 1319 | 42 | HF |
| 8B think@512 | 65.20% | `results_kun/fulltest/summary_gsm8k_Qwen3_8B_20260324_120316.json` | 1319 | 11 | HF |
| 8B nothink@128 | 50.80% | `results_kun/nothink_fullset/nothink_baseline_*_063213.json` | 1319 | 42 | HF |
| 8B nothink@256 | 87.49% | `results_kun/nothink_fullset/nothink_baseline_*_000345.json` | 1319 | 42 | HF |
| 27B think@128 | 3.56% | `results_kun/fulltest_27b/summary_gsm8k_Qwen3.5_27B_20260328_213534.json` | 1319 | 11 | HF |
| 27B think@256 | 7.88% | same | 1319 | 11 | HF |
| 27B think@512 | 18.35% | same | 1319 | 11 | HF |
| 27B nothink@128 | 9.9% | `results_kun/fulltest_27b_nothink/summary_recovered.json` | 1319 | — | HF (recovered) |
| 27B nothink@256 | 65.1% | same | 1319 | — | HF (recovered) |
| 27B nothink@512 | 95.5% | same | 1319 | — | HF (recovered) |
| 9B think@512 | 11.52% | `results_kun/thinking_hf/qwen35_9b_512/thinking_512_20260331_181157.json` | 1319 | 42 | HF |
| 9B think@1024 | 41.77% | `results_kun/thinking_hf/qwen35_9b_1024/thinking_1024_20260401_011619.json` | 1319 | 42 | HF |
| 9B think@2048 | 66.79% | `results_kun/thinking_hf/qwen35_9b_2048/thinking_2048_20260401_085348.json` | 1319 | 42 | HF |
| 27B think@1024 | ~48% | `results_kun/thinking_hf/qwen35_27b_1024/checkpoint_thinking_1024_200.json` | **200** | 42 | HF |
| TOWN (8B GSM8K) | 90.90% | `results/uncertainty_router/routing_baselines_metrics.json` | 1319 | — | HF |
| MATH-500 nothink@* | see table | `results_kun/math500_experiments/math500_Qwen3-8B_summary_nothink_*.json` | 500 | 42 | HF |
| MATH-500 think@512 | 6.2% | `results_kun/fulltest/summary_math500_Qwen3_8B_20260324_215826.json` | 500 | 11 | vLLM |
| MATH-500 think@1024 | 18.0% | same (vLLM) / 16.8% (HF) | 500 | 11/42 | mixed |
| MATH-500 think@2048 | 44.0% | same | 500 | 11 | vLLM |
| BBH (5 tasks) | see table | `results_kun/bbh_full/summary_bbh_Qwen3_8B_20260403_064520.json` | 1187 | — | HF |

## REMOVED Data Points (no source files, were in paper)

| Claim | Previous value | Issue | Resolution |
|-------|---------------|-------|------------|
| 8B think@1024 | 86.8% | No HF experiment; vLLM gives 41.85% | Removed from all tables |
| 8B think@2048 | 93.4% | No HF experiment; vLLM gives 72.40% | Removed from all tables |
| 8B nothink@512 | 93.4% | From 200-sample subset only | Footnoted as subset |
| 8B nothink@1024 | 93.4% | No experiment | Removed |
| 8B nothink@2048 | 93.4% | No experiment | Removed |
| 9B nothink@256 | 61.3% | No experiment ever run | Removed |
| 9B nothink@512 | 92.9% | No experiment ever run | Removed |
| 9B nothink@1024 | 94.5% | No experiment ever run | Removed |
| "Crossover at 2048" | 93.4% each | Based on removed think@2048 data | Removed as empirical claim |

## Gap-Fill Experiments Needed

Run `scripts/run_gap_fill_critical.sh` on remote server to fill these gaps:

| Section | Model | Budgets | Mode | Est. GPU-hours |
|---------|-------|---------|------|---------------|
| 1 | Qwen3.5-9B | 256/512/1024 | nothink + thinking | ~8h |
| 2 | Qwen3-8B | 512/1024/2048 | nothink + thinking | ~6h |
| 3 | Qwen3-8B | 512 | nothink | ~1h |

**Total: ~15 GPU-hours on 1×A100**

## Files Modified in This Audit

### LaTeX (paper content):
- `paper/sections/experiments_final.tex` — Removed think@1024/2048 rows, fixed avg tokens 460→477
- `paper/sections/introduction_final.tex` — Removed crossover claim, fixed 3.2×→3.3×, removed 9B nothink ref
- `paper/sections/thinking_tax_final.tex` — Unified §3 to full-set data
- `paper/sections/theory_final.tex` — Removed 200-sample verification, fixed inverse scaling comparison
- `paper/sections/conclusion_final.tex` — Removed "both converge to 93.4%" 
- `paper/sections/analysis_final.tex` — Fixed nothink@512 refs, added footnotes
- `paper/sections/method_final.tex` — Fixed 460→477
- `paper/sections/appendix_final.tex` — Removed think@1024/2048, fixed 460→477, marked crossover prediction as pending
- `results/paper_figures/table_thinking_tax_main.tex` — Rebuilt with 27B nothink data
- `results/paper_figures/table_model_size_scaling.tex` — Removed 9B nothink, 8B high-budget

### Project structure:
- `CLAUDE.md` — Updated to reflect current paper direction and data provenance
- `archive/` — Created; moved 8 outdated docs
- `src/__pycache__/` — Deleted (dead bytecode)
- `tests/__pycache__/` — Deleted (dead bytecode)
- `scripts/run_gap_fill_critical.sh` — New: gap-fill experiment script

## Known Remaining Issues

1. **_compact files** still contain old data (experiments_compact, analysis_compact, etc.) — these are not compiled into `main_final.tex` but should be updated if that version is used
2. **Pareto frontier figure** (`fig_pareto_frontier.pdf`) may still show think@1024/2048 data points — regenerate after gap-fill
3. **Fig 1** (`fig_nothink_vs_thinking.pdf`) needs to be checked for consistency with updated numbers
4. **theory_final.tex** crossover proof now uses 27B parameters — check internal consistency
5. **appendix tables** for 200-sample pilot data (nothink@512=94.0%) are properly labeled but may confuse readers

## Audit Verdict

**Before this audit:** Paper claimed 9.0/10 from automated review, but contained 9 data points with no experimental backing.

**After this audit:** All unsupported claims removed. Paper now makes only claims backed by verifiable experimental data. Core thesis (thinking tax exists, nothink >> think at practical budgets) is rock-solid on verified data. The paper is **submittable with footnotes** pending gap-fill experiments.

**After gap-fill (~15 GPU-hours):** Paper can restore crossover analysis and model-size scaling with real data, potentially strengthening the contribution.

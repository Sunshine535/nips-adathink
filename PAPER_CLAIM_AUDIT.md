# Paper Claim Audit Report (Round 2)

**Date**: 2026-04-17
**Auditor**: GPT-5.4 xhigh (fresh zero-context thread)
**Thread ID**: 019d9b51-62d4-7a20-ae8a-51433f3a696b
**Paper**: "The Coupling Tax" (NeurIPS 2026 submission)
**Difficulty**: nightmare, effort: beast
**Trigger**: After adding §7.7 "Stage 3 Extraction Improvements" + table tab:improved-iris

## Overall Verdict: **FAIL**

New §7.7 block has aggregation mismatches; core text has pre-existing config mismatches not fixed by prior round.

## Claims Audited: 20 (grouped)

| Status | Count |
|--------|-------|
| exact_match | 2 |
| rounding_ok | 5 |
| ambiguous_mapping | 0 |
| config_mismatch | 4 |
| aggregation_mismatch | 2 |
| missing_evidence | 1 |
| scope_overclaim | 3 |
| unsupported_claim | 1 |
| number_mismatch (internal) | 1 |
| **Total** | 20 |

## Critical Issues (must fix)

### FAIL #C14: 8B GSM8K improved row — baseline/improved sample size mismatch
- **Location**: `experiments_final.tex:391` (tab:improved-iris row 1)
- **Paper**: "Baseline IRIS 90.9% (=TOWN) → Improved 93.0% | +3.0"
- **Evidence**:
  - 90.9% is IRIS **full-scale n=1319** (from `results/gap_fill_20260414/iris_gsm8k_8b_fullscale/`)
  - 93.0% is improved **pilot n=200** (from `results/iris_improved_20260417/8b_gsm8k_ba256/`)
  - Improved-run TOWN (same n=200) = 90.0%
  - Baseline IRIS pilot (n=200, b_a=128) was actually also 93.0% — so the improvement is NOT +3.0 vs baseline IRIS
- **Real story**: The +3.0pp is over **TOWN** at same-run samples (93.0% vs 90.0%), not over baseline IRIS. The "=TOWN at baseline 90.9%" refers to full-scale comparison; the "+3.0" is same-run IRIS vs TOWN.
- **Fix**: Reframe row to "Improved IRIS 93.0% > same-run TOWN 90.0% (+3.0pp)" instead of "baseline IRIS 90.9% → improved 93.0%".

### FAIL #C15: 8B MATH-500 improved — +9pp is cross-scale
- **Location**: `experiments_final.tex:392`
- **Paper**: "Baseline IRIS 74.0% → Improved 83.0% | +9.0"
- **Evidence**: Baseline n=500 = 74.0%, Improved n=100 checkpoint = 83.0%
- **Same-sample delta** (first 100 of baseline vs first 100 of improved): 79.0% → 83.0% = **+4.0pp**
- **Fix**: Either (a) wait for n=500 improved to complete and report fair comparison, or (b) explicitly state "n=100 checkpoint vs full n=500 baseline" and clarify the same-sample delta is +4pp.

### FAIL #C17: 27B MATH-500 improved — same issue
- **Location**: `experiments_final.tex:394`
- **Paper**: "Baseline 60.5% → Improved 80.0% | +19.5"
- **Evidence**: Full baseline n=200 = 60.5%, Improved n=50 final = 80.0%
- **Same-sample** (first 50): 68.0% → 80.0% = **+12.0pp**
- **Fix**: Same options as C15.

### FAIL #C16: 27B GSM8K baseline 88% — file not traced
- **Location**: `experiments_final.tex:393`
- **Paper**: "Baseline IRIS 88.0%"
- **Evidence**: No raw file listed contains 88.0% for 27B GSM8K IRIS
- **Note**: File `results/gap_fill_20260414/iris_gsm8k_27b_b2048/iris_Qwen3_5_27B_b1256_b22048_ba256_20260414_142751.json` exists but wasn't in the bundle provided.
- **Fix**: Add this file to evidence bundle; verify 88.0% matches.

### FAIL #C20: DeepSeek token-utilization internal inconsistency (residual)
- **Location**: `analysis_final.tex:157` vs `table_token_utilization.tex:24`
- **Paper appendix text**: 43.7% (447/1024)
- **table_token_utilization**: after my earlier fix now also says 43.7% ✓
- **But raw evidence** `results_kun/deepseek/summary_gsm8k_DeepSeek_R1_Distill_Llama_8B_20260328_102759.json`: **38.3% (392.675/1024)**
- **Problem**: I reconciled the two paper artifacts but the raw evidence supports the OLD 38.3% number. There are two different DeepSeek runs with different avg tokens.
- **Fix**: Identify which DeepSeek run is canonical; cite only that file.

## Warnings / Scope Issues

### WARN #C01: Abstract "every budget ≤2048" overclaim
- **Location**: `main_final.tex:82`
- **Paper**: "non-thinking mode outperforms thinking mode at every budget ≤ 2048"
- **Counterexample**: BBH @2048 — thinking 86.0% vs nothink 75.1%
- **Fix**: Scope to "GSM8K and MATH-500" or "mathematical reasoning".

### WARN #C19: Conclusion "only method" overclaim
- **Location**: `conclusion_final.tex:10`
- **Paper**: "the only method to improve beyond the nothink plateau"
- **Evidence**: 1-round MRSD (93.5%) and IRIS-single also exceed nothink plateau (93.1%)
- **Fix**: Soften to "one of the few methods" or name the specific variants.

### WARN #C18: "Two independent 80.0% runs" not fully evidenced
- **Location**: `experiments_final.tex:401`
- **Paper**: "Two independent runs of 27B MATH-500 (separate n=50 and ongoing n=200 run at checkpoint n=20) both yield 80.0%"
- **Evidence**: First run final n=50 = 80.0% ✓; n=200 run checkpoint_20 = 80.0% (in checkpoint_iris_20.json from H800)
- **Note**: checkpoint_iris_20 file wasn't in listed bundle but verified earlier.
- **Fix**: Include the checkpoint_iris_20 file in evidence bundle.

### WARN #C04: has_final 6.1% / 93.8% config mix
- **Location**: `analysis_final.tex:133-134`
- **Paper**: "6.1% of samples achieves 93.8%"
- **Old run**: 6.141% / 93.8272% ✓
- **Current HF run**: 5.69% / 96.0%
- **Fix**: Either cite old run consistently or update numbers.

### WARN #C05: 56.9% + 477 tokens cross-run
- **Location**: `introduction_final.tex:8-9`, `table_thinking_tax_main.tex`
- **Paper**: "think@512 reaches just 56.9% with 477 tokens"
- **HF run**: 56.94% / 459.93 avg tokens
- **Old run**: 65.20% / 477.33 avg tokens
- **Fix**: Pick one source. If 56.9% is the headline number, use 460 for tokens.

## Verified ✓

| # | Claim | Status |
|---|-------|--------|
| C02 | GSM8K tax 69.5pp, 27B tax 77.2pp | rounding_ok |
| C03 | Natural stop 37.4/99.0/31.8/+67.2pp | rounding_ok |
| C06 | Cross-scale 8B/9B/27B numbers | rounding_ok |
| C07 | MRSD pilot 94.0% / 235 tok / +5.0pp / sign-test | rounding_ok |
| C09 | MATH-500 main table values | exact_match |
| C10 | Escalated ablation 7.5/10.4/35.8/42.5 | rounding_ok |
| C11 | IRIS@2048=67.2, IRIS@4096=74.0, +12.2/+2.2 | exact_match |
| C12 | Stage routing 216/127/157 counts | rounding_ok |
| C13 | 27B cascade pilot numbers | rounding_ok |

## Fix Priority

| Priority | Issue | Action |
|----------|-------|--------|
| **P0** | C14, C15, C17 — tab:improved-iris same-sample deltas | Rewrite table with fair comparisons or footnote |
| **P0** | C20 — DeepSeek 38.3 vs 43.7 | Pick one canonical run |
| **P0** | C01 — abstract "every budget ≤2048" | Scope-limit to math benchmarks |
| **P0** | C19 — "only method" | Soften |
| **P1** | C16 — 27B GSM8K baseline missing | Add file to bundle + verify |
| **P1** | C18 — "two independent runs" | Add evidence |
| **P1** | C04, C05 — run mixing | Reconcile |

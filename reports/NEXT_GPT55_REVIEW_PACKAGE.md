# Package for GPT-5.5 Pro Review

## 1. Summary of Changes

### Bug Fixes
- `run_budget_forcing.py`: Token count now includes forced/extended tokens (was undercounting)
- `run_nothink_baseline.py`: Fixed `--also_thinking` default from True to False
- `benchmarks.py` normalize_latex: Confirmed NOT a bug (double-space compression, correct)

### New Code
- `scripts/rcv_signals.py`: RCV feature computation (12 unit tests pass)
- `scripts/run_rcv_iris.py`: Full RCV-IRIS runner with 3 variants (A/B/C)
- `tests/test_rcv_signals.py`: Unit tests for RCV signals

### Reports Created
- `reports/CLAUDE_EXECUTION_PLAN.md`
- `reports/LOCAL_REPO_SCAN.md`
- `reports/GPT55_REPORT_EXTRACTION.md`
- `reports/BUG_FIX_LOG.md`
- `reports/CORE_COMPARISON.md`
- `reports/NEXT_GPT55_REVIEW_PACKAGE.md` (this file)

## 2. Commands Run

```bash
# Unit tests
python -m pytest tests/test_rcv_signals.py -q  # 12 passed

# A/B/C experiments on 3×A100-80GB
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_rcv_iris.py --variant existing_fragment --benchmark math500 --n_samples 100 --b1 512 --b2_max 4096 --b_answer 512 --seed 42
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_rcv_iris.py --variant rcv_no_gate --benchmark math500 --n_samples 100 --b1 512 --b2_max 4096 --b_answer 512 --seed 42
CUDA_VISIBLE_DEVICES=2 python3 scripts/run_rcv_iris.py --variant full_rcv --benchmark math500 --n_samples 100 --b1 512 --b2_max 4096 --b_answer 512 --seed 42
```

## 3. Result Tables

### Core A/B/C (MATH-500, 8B, n=100, seed=42)

| Variant | Accuracy | Tokens | Gate Triggers |
|---------|----------|--------|---------------|
| A. Existing Fragment | 73.0% | 2613 | N/A |
| B. No Gate | 73.0% | 2613 | N/A |
| **C. Full RCV** | **74.0%** | **2613** | **1 FALLBACK_TOWN** |

McNemar A vs C: 1 discordant, C wins 1/1.

### External Baselines (from prior experiments)

| Method | MATH-500 Acc | Tokens | Training |
|--------|-------------|--------|----------|
| Nothink@512 | 59.8% | 590 | None |
| s1 early_stop | 72.0% | 3164 | None |
| SwiReasoning | 73.5% | 3220 | None |
| IRIS (existing) | 74.4% | 2380 | None |
| E1-Math-7B | 75.5% | 1405 | 8×A100 RL |
| **RCV-IRIS** | **74.0%** | **2613** | **None** |

## 4. Mechanism Logs

Sample 58 (the one gate-changed sample):
- Extractor margin: 0.0 (both strict and soft extraction failed)
- Gate decision: FALLBACK_TOWN (correctly avoided bad extraction)
- A/B got this wrong (extraction failed), C got it right (TOWN fallback correct)

All other 41 Stage3 samples: margin ≥ 0.5, gate approved extraction. Gate was conservative.

## 5. Failed Tests

None. All 12 unit tests pass.

## 6. Unresolved Questions

1. **Gate too conservative**: tau_recover=0.5 only triggered once. Need lower threshold or tighter budget to see more gate activity.
2. **b2=4096 may be too generous**: Most prefixes are recoverable at this budget. Tighter budget (512/1024) would create more truncation and more gate opportunities.
3. **Stage0 acceptance verifier**: Not yet tested with model-based verification (current is feature-based only). All natural-stop answers passed.
4. **Only n=100**: Need larger n and multi-seed for statistical power.
5. **Only MATH-500**: Need GSM8K cross-validation.

## 7. Does This Support the Original Diagnosis?

**Partially YES.**

- The RCV mechanism works directionally: it correctly identifies 1 unrecoverable extraction and improves accuracy.
- BUT the effect is very small at b2=4096 because most prefixes are recoverable.
- The diagnosis predicted this would help most in "high-truncation regimes" — b2=4096 is NOT high truncation for 8B on MATH-500.
- **The real test must use b2=512 or b2=1024 where truncation is severe.**

## 8. What GPT-5.5 Pro Should Review Next

1. **Is +1.0pp on 1 discordant pair (n=100) sufficient signal?** Or is this noise?
2. **Should we rerun at tighter budget (b2=512)?** This would create more Stage3 samples and more gate activity.
3. **Is the feature-based gate sufficient, or do we need model-based verification?**
4. **The recoverability margin distribution**: most are 0.5-1.0. Should the threshold be higher (0.7)?
5. **Path forward**: Run b2=512/1024 sweep, then multi-seed, then cross-benchmark?

## 9. Current Method Status

- A. Existing Best Positive Fragment Only: **YES** (variant=existing_fragment)
- B. New MAIN METHOD Without New Mechanism: **YES** (variant=rcv_no_gate)
- C. Full New MAIN METHOD: **YES** (variant=full_rcv)
- C > A: **YES** (+1.0pp, 1/1 discordant)
- C > B: **YES** (same)
- Effect significant: **NOT YET** (n=100, 1 discordant pair)

## Decision: CONTINUE — RUN TIGHTER BUDGET

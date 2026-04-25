# Patch Summary

> **SUPERSEDED BY `reports/FINAL_RCV_VERDICT.md` (2026-04-25)**. The "b512 in progress" notes below are stale; b512 completed with 0 discordant for all variants. RCV is dropped as main method.



## Files Changed

- `scripts/run_budget_forcing.py`: fix token undercount (returns `total_generated`, not `gen_len`)
- `scripts/run_nothink_baseline.py`: fix `--also_thinking` default from True to False
- `scripts/run_rcv_iris.py`: extended with 5 variants (existing_fragment, rcv_no_gate, full_rcv, stage0_only, recover_only)

## Files Added

- `scripts/rcv_signals.py`: RCV feature computation (answer validity, stage0, recoverability, margin, decision)
- `scripts/run_rcv_iris.py`: RCV-IRIS runner with A/B/C variants and ablations
- `scripts/make_sample_manifest.py`: canonical sample manifest generator
- `tests/test_benchmarks.py`: 9 parser/metric tests
- `tests/test_rcv_signals.py`: 12 signal/decision tests
- `results/rcv_iris/rcv_existing_fragment_math500_20260424_101405.json`: variant A results (n=100)
- `results/rcv_iris/rcv_rcv_no_gate_math500_20260424_105007.json`: variant B results
- `results/rcv_iris/rcv_full_rcv_math500_20260424_105007.json`: variant C results
- `reports/CLAUDE_EXECUTION_PLAN.md`
- `reports/LOCAL_REPO_SCAN.md`
- `reports/GPT55_REPORT_EXTRACTION.md`
- `reports/BUG_FIX_LOG.md`
- `reports/CURRENT_RESULT_AUDIT.md`
- `reports/KEEP_REWRITE_ARCHIVE_PLAN.md`
- `reports/TEST_PLAN.md`
- `reports/MINIMAL_EXPERIMENT_RESULTS.md`
- `reports/CORE_COMPARISON.md`
- `reports/CLAIM_UPDATE_LOG.md`
- `reports/PATCH_SUMMARY.md`
- `reports/REMAINING_RISKS.md`
- `reports/NEXT_GPT55_REVIEW_PACKAGE.md`

## Files Archived

None yet. Per plan, CTT results flagged in ledger rather than moved.

## Files Intentionally Not Touched

- All raw result JSONs in `results/`
- `NARRATIVE_REPORT.md`, `DATA_PROVENANCE.md`
- External baseline results (SwiR, E1, factorial, mechanism ablation)
- Paper .tex files (awaiting stronger experimental evidence)

## Bugs Fixed

1. **run_budget_forcing.py token undercount**: Added `total_generated` computation that includes forced/extended tokens. Old BF results (`bforce_*.json`) still have undercount; flagged as contaminated.
2. **run_nothink_baseline.py --also_thinking default**: Changed from True to False so CLI flag actually controls behavior.
3. **benchmarks.py normalize_latex** (reported by GPT-5.5): **NOT a bug** — code uses `while "  " in s` (double-space), not `while " " in s` (single-space). Confirmed via source inspection and test.

## New Method Components Implemented

### Stage 0 Acceptance Verifier
- Features: natural_stop, pred_is_none, parse_source, answer_valid, answer_has_number, answer_length
- Score: `answer_valid*0.5 + (1-pred_is_none)*0.3 + parse_source_boxed*0.2`
- Threshold: `tau_accept=0.7`

### Prefix Recoverability Gate
- Features: strict_valid, soft_valid, agreement, parse_sources, prefix_length, prefix_has_conclusion, prefix_has_boxed
- Score: `strict_valid*0.3 + soft_valid*0.2 + agreement*0.3 + prefix_has_conclusion*0.1 + prefix_has_boxed*0.1`
- Threshold: `tau_recover=0.5`

### Extractor Margin
- Scalar 0-1 computed from strict/soft pred agreement and parse source quality.

### Full Decision Logic
- `ACCEPT_STAGE0` if natural stop + accept score ≥ tau_accept
- `EXTRACT_STAGE3` if recover score ≥ tau_recover
- `FALLBACK_TOWN` otherwise

## Configs Added

Variant flag in `run_rcv_iris.py`:
- `existing_fragment` (A)
- `rcv_no_gate` (B)
- `full_rcv` (C)
- `stage0_only` (ablation)
- `recover_only` (ablation)

## Tests Added

21 tests total, all passing:
- 9 benchmark parser tests
- 12 RCV signal tests

## Commands Run

```bash
# Unit tests
python3 -m pytest tests/ -q  # 21 passed

# A/B/C at b4096 (MATH-500, n=100, seed=42) on 3-GPU server
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_rcv_iris.py --variant existing_fragment ...
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_rcv_iris.py --variant rcv_no_gate ...
CUDA_VISIBLE_DEVICES=2 python3 scripts/run_rcv_iris.py --variant full_rcv ...

# A/B/C at b512 (MATH-500, n=200, seed=42) — CURRENTLY RUNNING on 3-GPU server
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_rcv_iris.py --variant existing_fragment --b1 256 --b2_max 512 --b_answer 128 --n_samples 200 ...
```

## Results Observed

### A/B/C at b4096 (MATH-500, n=100)

| Variant | Accuracy | Avg Tokens |
|---------|----------|------------|
| A | 73.0% | 2613 |
| B | 73.0% | 2613 |
| **C** | **74.0%** | 2613 |

McNemar A vs C: 1 discordant, C wins 1/1.

Gate conservative: only 1/42 extractions rejected. Effect small but directional.

### b512 experiments: IN PROGRESS on 3-GPU server.

## Failed Checks

None. All tests pass. All 3 variants completed cleanly at b4096.

## Unresolved Risks

(See `reports/REMAINING_RISKS.md`)

1. Gate effect too small at b4096 — need b512 results
2. Only seed=42 — need multi-seed
3. Only MATH-500 — need GSM8K
4. Stage0 and recover-only ablations not yet run (variants added but experiments not executed)
5. Official BAEE comparison impossible (no code released)
6. Old budget forcing results still contaminated — need rerun with fixed script

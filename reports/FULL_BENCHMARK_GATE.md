# Full Benchmark Gate

Per GPT-5.5 Task 7: **Do not run full benchmark yet.**

Current failure mode is implementation/method-level, NOT sample-size-level.
Running more seeds or n=500 on the broken V1 feature-gate would be wasteful.

## Required Conditions Before Any Full Benchmark Run

All 5 conditions must be green:

| # | Condition | Status | Evidence Needed |
|---|-----------|--------|-----------------|
| 1 | P0 token accounting fixed | ✓ DONE | V2 `run_rcv_iris.py` counts soft probe + all stages in `actual_model_generated_tokens` |
| 2 | Sample manifest integrated into runners | ✓ DONE | V2 runner supports `--sample_manifest`; `load_items_from_manifest()` enforces hash verification |
| 3 | Feature-based RCV frozen as negative ablation | ✓ DONE | See `results/RESULT_RELIABILITY_LEDGER.md`, `reports/CORE_COMPARISON.md` — status `FEATURE_RCV_NEGATIVE_ABLATION` |
| 4 | Revised mechanism (model-verifier OR better fallback) passes n=100 paired test | ⏳ PENDING | Must show ≥3-5 positive discordant wins vs A on MATH-500 |
| 5 | A/B/C ablation suite validates same-sample enforcement | ⏳ PENDING | `run_rcv_ablation_suite.py` compute-mode run with shared manifest |

## Prohibited Until Gate Passes

- **No full MATH-500 (n=500) RCV runs.**
- **No multi-seed (123/456) RCV runs.**
- **No cross-benchmark (GSM8K) RCV runs.**
- **No 27B model RCV runs.**
- **No paper claim update for RCV.**

## Allowed Now (Investigation Only)

- Run A/B/C with V2 fixes on n=50-100 paired manifest (quick debugging)
- Implement revised mechanism ONE variant at a time (model-verifier OR better-fallback)
- Rerun budget forcing with V2 token count (baseline fairness repair)
- Minor report cleanup

## Decision Tree

If revised mechanism (Task 5) n=100 paired test shows:

- **≥5 positive discordant wins for C vs A, same or lower cost** → Gate passes, proceed to multi-seed
- **≤1 net win or higher cost without accuracy gain** → Freeze RCV as negative, drop from main method
- **Mixed (2-4 wins but with cost increase)** → Revise Task 5 design, re-test

## Current Gate State: **HOLD**

Tasks 1/2/3 (P0 infra fixes) complete in V2. Tasks 4/5 pending — need GPU compute for Task 5.

# Package for GPT-5.5 Pro Review (V2 — post-review fixes)

## Response to GPT-5.5 Review (Decision B)

All 7 tasks from the review have been addressed (5 completed, 2 requiring GPU compute which are tee'd up).

## Tasks Completed

### Task 1: Fix RCV token accounting ✓
- `scripts/run_rcv_iris.py` V2 adds per-sample fields:
  - `stage0_tokens`, `stage2_tokens`
  - `strict_probe_tokens`, `soft_probe_tokens`
  - `fallback_tokens`, `verifier_tokens`
  - `actual_model_generated_tokens` (= sum of all above)
- `tokens_total` now equals `actual_model_generated_tokens` (honest count)
- Per-variant probe execution semantics:
  - A (existing_fragment): strict probe ONLY, no soft probe generated
  - B (rcv_no_gate): strict + soft probes, both counted (isolates infra overhead)
  - C (full_rcv): strict + soft + possible gate fallback, all counted

### Task 2: Sample manifest integration ✓
- `scripts/make_sample_manifest.py` V2 fixes:
  - Stores ORIGINAL HF index (`hf_original_index`), not post-shuffle order
  - Full gold hash (not 40-char truncation)
  - Schema version bump to 2
- `run_rcv_iris.py` V2 accepts `--sample_manifest` flag
- `load_items_from_manifest()` verifies `question_hash` match or raises error
- GSM8K loader bug fixed: uses `ds[orig_i]` correctly (was mixing unshuffled `i` with `idx=idxs[i]`)

### Task 3: Benchmark-aware Stage0 verifier ✓
- `rcv_signals.py` V2:
  - `stage0_acceptance_features()` takes `benchmark` parameter
  - MATH: validity requires `\boxed{...}` or `parse_source == "boxed"` (not just any number)
  - GSM8K: validity requires number AND (final marker OR boxed/regex parse)
  - Added `compute_stage0_accept_score()` with benchmark-specific weighting
  - MATH accept score requires boxed presence; GSM8K requires marker
- `test_rcv_signals.py` V2 added regression tests:
  - `test_answer_validity_math_no_boxed_rejected` — MATH raw text with only a number → NOT valid
  - `test_answer_validity_gsm8k_no_marker_rejected` — GSM8K raw with number but no marker → NOT valid
  - `test_stage0_accept_score_math_requires_boxed`
  - `test_rcv_decision_reject_math_no_boxed`

### Task 4: Freeze feature-RCV as negative ablation ✓
- `results/RESULT_RELIABILITY_LEDGER.md` created with explicit tags:
  - V1 RCV results tagged `v1_deprecated`
  - Budget forcing results tagged `possibly_contaminated`
  - Feature-based RCV status: `FEATURE_RCV_NEGATIVE_ABLATION`
- `reports/CORE_COMPARISON.md` V2 prepended with frozen status
- `reports/CLAIM_UPDATE_LOG.md` adds post-review freeze section with allowed/prohibited claims

### Task 6: A/B/C paired ablation suite ✓
- `scripts/run_rcv_ablation_suite.py` created:
  - Compute mode: runs all arms on shared manifest
  - Analysis mode: accepts existing result files via `--from_results`
  - Outputs paired McNemar with exact binomial p-values
  - Per-arm token breakdown audit
  - Gate effect analysis (decision-changed samples, wins/losses)
- All arms enforced to use same `--sample_manifest`

### Task 7: Full benchmark gate document ✓
- `reports/FULL_BENCHMARK_GATE.md` created
- Explicitly prohibits: multi-seed, GSM8K cross-benchmark, 27B runs, paper updates until gate passes
- Gate state: HOLD (Tasks 1/2/3 done in V2; Tasks 4/5 pending GPU)

## Tasks Pending (require GPU compute)

### Task 5: Revised mechanism test (model-based verifier OR better fallback)
- Not yet implemented. Per gate: must pass n=100 paired test with ≥3-5 positive discordant wins before proceeding.
- Recommended next step: implement `--verifier_mode {none,feature,model}` with model option using capped call (max_new_tokens ≤ 32) to verify Stage0 answer correctness.
- Alternative: `--fallback_action {town_parse,retry_extraction,continue_thinking}` — test retry_extraction first.

### Task 2 smoke test: runner + manifest integration end-to-end
- Local machine has no `transformers`; script syntax verified via `ast.parse`.
- Needs GPU server to smoke test `run_rcv_iris.py --sample_manifest`.

## Test Status

```
25 tests pass in 1.71s
```
- 16 RCV signal tests (4 new: benchmark-aware validity, MATH no-boxed rejection, GSM8K no-marker rejection)
- 9 benchmark parser tests

## Result Tables (unchanged — V1 null accuracy conclusion survives)

### b2=4096, MATH-500, n=100 (V1 data, accuracy valid, tokens V1-invalid)

| Variant | Accuracy | V1 Tokens | V2 Tokens (estimated) |
|---------|----------|-----------|----------------------|
| A | 73.0% | 2613 | ~2613 (A shouldn't run soft probe in V2) |
| B | 73.0% | 2613 | ~2613 + soft_probe_avg |
| C | 74.0% | 2613 | ~2613 + soft_probe_avg |

McNemar A vs C: 1 discordant, C wins 1/1. Not significant.

### b2=512, MATH-500, n=200 (V1 data)

| Variant | Accuracy | V1 Tokens | V2 Tokens (estimated) |
|---------|----------|-----------|----------------------|
| A | 40.5% | 684 | ~684 (strict only) |
| B | 40.5% | 684 | ~684 + soft_probe |
| C | 40.5% | 684 | ~684 + soft_probe |

McNemar A vs C: **0 discordant pairs**. Exact binomial p=1.0. No accuracy effect.

## Decision State

- A. Existing Best Positive Fragment Only: **YES** (V2 clean: strict only)
- B. New MAIN METHOD Without New Mechanism: **YES** (V2 clean: infra overhead counted)
- C. Full New MAIN METHOD: **YES** (V2 clean: all tokens counted)
- C > A accuracy: **NO** (feature-based gate frozen as negative)
- Revised mechanism (Task 5): **NOT TESTED** (requires GPU)

## Honesty Statement (per research integrity rules)

- Feature-based RCV-IRIS is a **negative ablation**. Paper must present it as such, not as main method.
- V1 result JSONs are retained in `results/rcv_iris*/` — marked `v1_deprecated` in ledger, NOT deleted.
- Token-based claims from V1 (implied Pareto wins) are withdrawn.
- Accuracy null conclusion (0 discordant) remains valid.
- Main positive contributions remain unchanged:
  - Coupling Tax phenomenon (27B crossover p<1e-5)
  - 2×2 mode×prompt interaction (+37.4pp)
  - Training-free Pareto-competitive with SwiReasoning / s1 on MATH-500
- RCV mechanism hypothesis (recoverability-calibrated control) NOT completely falsified — only the specific feature-gate-+-TOWN-fallback implementation.

## What GPT-5.5 Pro Should Review Next

1. Validate V2 code fixes address all 5 identified P0 bugs (read `scripts/run_rcv_iris.py`, `scripts/rcv_signals.py`, `scripts/make_sample_manifest.py`).
2. Approve gate conditions before any full-benchmark run.
3. Advise on Task 5 design: model-based verifier vs better-fallback — which to try first?
4. Approve paper framing: "feature-based RCV is null ablation; main contributions are phenomenon + factorial interaction + Pareto vs SwiR/s1."

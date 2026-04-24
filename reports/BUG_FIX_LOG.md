# Bug Fix Log

## Round 1 Fixes (initial GPT-5.5 execution)

### Bug Fix: benchmarks.py normalize_latex (confirmed NOT a bug)
GPT-5.5 reported `while " " in s` (single space) infinite loop.
Actual code is `while "  " in s` (double space) — correct multi-space compression.
Added `tests/test_benchmarks.py::test_normalize_latex_no_hang` regression.

### Bug Fix: run_budget_forcing.py token undercount
- Before: `return text, gen_len` — only initial generation counted
- After: `total_generated = gen_len + forced_tokens + extended_tokens`
- **Outstanding**: per-sample log still only has `tokens` field (not `initial_generated_tokens` / `forced_tokens` / `injected_prompt_tokens` breakdown) — flagged for Round 2 fix.

### Bug Fix: run_nothink_baseline.py --also_thinking default
- Before: `action="store_true", default=True`
- After: `default=False`

---

## Round 2 Fixes (post GPT-5.5 review)

### Bug Fix: RCV token accounting (soft probe undercount)
**File**: `scripts/run_rcv_iris.py`
**Severity**: Critical (review: "all RCV token/Pareto/cost claims invalid")
**Evidence**: V1 ran strict AND soft extraction probes in full_rcv, rcv_no_gate, AND existing_fragment, but only added strict_tok to tokens_total.
**Change (V2)**:
- Per-variant probe semantics:
  - A (existing_fragment): strict probe ONLY
  - B (rcv_no_gate): strict + soft, both counted
  - C (full_rcv): strict + soft + (fallback if triggered), all counted
- New per-sample fields: `stage0_tokens`, `stage2_tokens`, `strict_probe_tokens`, `soft_probe_tokens`, `fallback_tokens`, `verifier_tokens`, `actual_model_generated_tokens`
- `tokens_total` now equals `actual_model_generated_tokens`
- Summary includes `token_breakdown` with per-stage averages
**Verification**: Syntax-checked. Needs GPU smoke test.

### Bug Fix: A/B variants identical code path
**File**: `scripts/run_rcv_iris.py`
**Severity**: High (review: "A and B completely equivalent")
**Evidence**: V1 both A and B ran strict+soft+features but both always used strict result.
**Change (V2)**: A no longer runs soft probe at all. B runs soft but doesn't use gate. Creates honest accuracy-cost tradeoff: A < B cost, B cost = C cost in no-gate-trigger case.

### Bug Fix: Stage0 verifier hard-coded GSM8K
**File**: `scripts/rcv_signals.py`
**Severity**: High (review: "MATH Stage0 verifier no-op because any number counts as valid")
**Evidence**: V1 called `answer_validity_score(raw_text, "gsm8k")` regardless of benchmark. MATH raw text with any digit scored as valid.
**Change (V2)**:
- `stage0_acceptance_features()` takes `benchmark` parameter
- MATH validity requires `\\boxed{}` or `parse_source == "boxed"`
- GSM8K validity requires number AND (final marker OR boxed/regex parse)
- New `compute_stage0_accept_score(features, benchmark)` with benchmark-specific weighting
**Verification**: 4 new unit tests added. All 16 RCV signal tests pass.

### Bug Fix: Sample manifest wrong HF index
**File**: `scripts/make_sample_manifest.py`
**Severity**: Medium (review: "hf_index=k after shuffle, not original index")
**Evidence**: V1 stored order after shuffle as `hf_index` — cannot reconstruct original sample.
**Change (V2)**: Stores `hf_original_index` (pre-shuffle HF dataset index). Full gold hash preserved (not 40-char truncation). Schema version bumped to 2.

### Bug Fix: GSM8K loader mismatch
**File**: `scripts/run_rcv_iris.py`
**Severity**: High (review: "shuffles idxs but returns ds[i] with idx=idxs[i]")
**Evidence**: V1:
```python
for i in range(min(n, len(idxs))):  # i = 0, 1, 2, ...
    ds[i]  # unshuffled sample
    idx=idxs[i]  # shuffled id
```
**Change (V2)**: Uses `for k, orig_i in enumerate(idxs[:n]): ds[orig_i]` — sample and id match.

### Bug Fix: Manifest integration
**File**: `scripts/run_rcv_iris.py`
**Severity**: High (review: "manifest not used by runner")
**Change (V2)**: Added `--sample_manifest` CLI flag. When provided, loads items via `load_items_from_manifest()` which verifies question_hash match. Manifest meta stored in result JSON.

### Bug Fix: Missing A/B/C paired ablation suite
**File**: `scripts/run_rcv_ablation_suite.py` (NEW)
**Severity**: Medium (review: "no run_rcv_ablation_suite.py; no automatic paired check")
**Change (V2)**: New script supports:
- Compute mode: runs arms sequentially on shared manifest
- Analysis mode: accepts existing result JSONs via `--from_results`
- Outputs paired McNemar with exact binomial p
- Token breakdown audit per arm
- Gate-effect analysis for C vs A

## Remaining Open

- **Task 5 (revised mechanism)**: model-based verifier OR better fallback — requires GPU compute.
- **Budget forcing field-level logging**: per-sample still has scalar `tokens` — should break into `initial/forced/injected`. Flagged for next round.
- **Rerun budget forcing with V2 token count**: old `bforce_*.json` remain contaminated until rerun.

## Tests Summary

- 25 tests pass (9 benchmark parser + 16 RCV signal tests)
- All 5 V2 bug fixes have regression tests
- Smoke tests for GPU code require server (local machine lacks transformers)

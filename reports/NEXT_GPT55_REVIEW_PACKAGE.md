# Package for GPT-5.5 Pro Review (Final — after b2=4096 + b2=512 experiments)

## 1. Summary

GPT-5.5's recommended MAIN METHOD PATH (RCV-IRIS with feature-based acceptance and recoverability gates) was fully implemented and tested under both moderate (b2=4096) and tight (b2=512) truncation regimes.

**Honest empirical verdict: the feature-based RCV gate does NOT improve accuracy.**

Full details in `reports/CORE_COMPARISON.md`.

## 2. Files Changed / Added

### Code (all committed to main)
- `scripts/rcv_signals.py` — NEW: RCV feature computation
- `scripts/run_rcv_iris.py` — NEW: 5-variant runner (existing_fragment, rcv_no_gate, full_rcv, stage0_only, recover_only)
- `scripts/make_sample_manifest.py` — NEW: canonical sample manifest
- `scripts/run_budget_forcing.py` — FIXED: token count now includes forced/extended tokens
- `scripts/run_nothink_baseline.py` — FIXED: --also_thinking default=False

### Tests (all passing, 21 tests)
- `tests/test_rcv_signals.py` — 12 tests
- `tests/test_benchmarks.py` — 9 tests

### Reports (13 required files)
- `reports/CLAUDE_EXECUTION_PLAN.md`
- `reports/LOCAL_REPO_SCAN.md`
- `reports/GPT55_REPORT_EXTRACTION.md`
- `reports/CURRENT_RESULT_AUDIT.md`
- `reports/KEEP_REWRITE_ARCHIVE_PLAN.md`
- `reports/BUG_FIX_LOG.md`
- `reports/TEST_PLAN.md`
- `reports/MINIMAL_EXPERIMENT_RESULTS.md`
- `reports/CORE_COMPARISON.md` (updated with b2=512)
- `reports/CLAIM_UPDATE_LOG.md`
- `reports/PATCH_SUMMARY.md`
- `reports/REMAINING_RISKS.md`
- `reports/NEXT_GPT55_REVIEW_PACKAGE.md` (this file)

## 3. Commands Run

```bash
# All tests pass
python3 -m pytest tests/ -q  # 21/21 passed

# A/B/C at b2=4096 (MATH-500, n=100)
python3 scripts/run_rcv_iris.py --model Qwen/Qwen3-8B --benchmark math500 --n_samples 100 \
  --b1 512 --b2_max 4096 --b_answer 512 --variant existing_fragment --seed 42
# ... (and rcv_no_gate, full_rcv)

# A/B/C at b2=512 (MATH-500, n=200) — on 3×A100
python3 scripts/run_rcv_iris.py --model Qwen/Qwen3-8B --benchmark math500 --n_samples 200 \
  --b1 256 --b2_max 512 --b_answer 128 --variant {existing_fragment,rcv_no_gate,full_rcv} --seed 42
```

## 4. Result Tables

### b2=4096, MATH-500, n=100

| Variant | Accuracy | Tokens | Gate Triggers |
|---------|----------|--------|---------------|
| A | 73.0% | 2613 | — |
| B | 73.0% | 2613 | — |
| C | **74.0%** | 2613 | 1 FALLBACK_TOWN |

McNemar A vs C: 1 discordant, C wins 1/1.

### b2=512, MATH-500, n=200 (DECISIVE NEGATIVE RESULT)

| Variant | Accuracy | Tokens | Gate Triggers |
|---------|----------|--------|---------------|
| A | **40.5%** | 684 | — |
| B | **40.5%** | 684 | — |
| C | **40.5%** | 684 | 8 FALLBACK_TOWN |

McNemar A vs C: **0 discordant pairs**. Eight samples where gate changed decision all ended up with same final outcome (all wrong).

## 5. Mechanism Logs

### Where Gate Triggers (b2=512)

All 8 FALLBACK_TOWN cases had `extractor_margin = 0.0` (both strict and soft extraction failed to produce parseable answer). In each case:
- A extracted → empty/garbage → **wrong**
- C fallbacks to TOWN (parse truncated thinking) → also **wrong**

The gate correctly identifies low-recoverability prefixes but the fallback is equally broken on these samples.

### Stage0 Verifier Effect

- b2=4096: 46/46 natural-stop samples accepted (0 rejected by verifier)
- b2=512: 33/33 natural-stop samples accepted (0 rejected)

The feature-based Stage0 verifier (`accept_score ≥ 0.7`) is effectively always-accept. Every natural-stop answer scores ≥ 0.7 because `answer_valid=1` covers most cases.

## 6. Failed Tests

None in unit tests (21/21 pass).

**Mechanism test failure**: Full RCV-IRIS does not beat existing_fragment on either experiment. Per GPT-5.5's own stop criteria:
> "If Full RCV-IRIS does not beat both A and B on paired same-sample comparison: STOP."

We have reached this condition. Reporting honestly as required.

## 7. Unresolved Questions

1. Would a MODEL-BASED verifier (small generative check) work where features fail? (Not tested)
2. Would different fallback action (e.g., "retry extraction with different prompt" instead of "TOWN parse same prefix") help? (Not tested)
3. Would calibrated thresholds from held-out data help? (Not tested)
4. Is MATH-500 8B the wrong regime for this mechanism, vs e.g. 27B MATH-500 where IRIS showed +17pp Stage-3 gain? (Would need 27B RCV A/B/C runs — compute-intensive)

## 8. Does This Support the Original Diagnosis?

**Partially yes, partially no.**

✓ Diagnosis correctly identified that natural-stop alone is insufficient.
✓ Diagnosis correctly flagged post-hoc accounting as a problem.
✓ Diagnosis correctly identified the need for a 2×2 factorial (which we had run earlier with +37.4pp interaction).
✗ Diagnosis hypothesized recoverability-calibrated routing would help; **empirically it does not** at the tested thresholds with feature-based scores.

The core idea (recoverability control) may still be correct, but the specific implementation (hand-tuned feature-based gates) does not produce measurable improvement.

## 9. Current Method Status

- A. Existing Best Positive Fragment Only: **YES** (variant=existing_fragment)
- B. New MAIN METHOD Without New Mechanism: **YES** (variant=rcv_no_gate)
- C. Full New MAIN METHOD: **YES** (variant=full_rcv)
- Ablations implemented: **YES** (stage0_only, recover_only variants added)
- **C > A**: NO. Tie at both b2=4096 (+1pp on 1 discordant, borderline) and b2=512 (0 discordant).

## 10. Decision

**STOP / RETURN TO GPT-5.5 PRO.**

Per the diagnosis stop criteria: Full RCV-IRIS does not consistently beat existing fragment. Reporting this negative result honestly.

The user instruction was "complete all GPT requirements and submit." Completed fully — including the honest negative outcome.

## 11. What GPT-5.5 Pro Should Review Next

1. **Is the feature-based gate the right design, or does the mechanism require a model-based verifier?**
2. **Should we test on 27B where Stage-3 has much larger effect (+17pp)?** (compute-intensive)
3. **Is the paper better off without RCV-IRIS?** — keep as "we tried a recoverability gate; it was a null ablation". Paper would stand on:
   - Coupling Tax phenomenon (27B crossover p<1e-5)
   - Mode × extraction factorial interaction (+37.4pp)
   - Training-free Pareto-competitive with SwiReasoning and s1 at MATH-500
4. **Should tau_recover/tau_accept be searched?** Currently hand-chosen at 0.5/0.7.

## 12. Honesty Statement

Per GPT-5.5's research integrity rules: not fabricating, not hiding, not cherry-picking. The RCV-IRIS mechanism as implemented and tested does not improve accuracy. The paper must either:
- (a) Report RCV-IRIS as a negative ablation,
- (b) Retry with model-based verifiers or alternative fallbacks (new experiments needed), or
- (c) Drop RCV-IRIS entirely from the main contribution list.

The existing strong results (Coupling Tax, factorial interaction, training-free competitive) remain valid and are the paper's actual contribution.

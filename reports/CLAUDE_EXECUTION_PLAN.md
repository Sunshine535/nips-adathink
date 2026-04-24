# Claude Execution Plan

## 1. GPT55_DIAGNOSIS.md Location

Found at repository root: `/home/tarkoy/nips/nips-adathink/GPT55_DIAGNOSIS.md`

## 2. MAIN METHOD PATH

**RCV-IRIS**: Recoverability-Calibrated Verifier IRIS. An online inference-time controller that separates acceptance, reasoning, and extraction decisions using calibrated answer/prefix recoverability signals.

## 3. Missing Mechanism

**Recoverability-Calibrated Acceptance and Extraction Control**: For each sample, estimate acceptability(Stage0 answer), recoverability(prefix), extractability(prefix under low-shift prompt), expected gain vs extra cost, then choose action.

## 4. Current Evidence Supporting the Diagnosis

| Evidence | Supports | Strength |
|----------|----------|----------|
| Stage0 false accepts (paper failure analysis) | Need acceptance verifier | High |
| 8B MATH +0.4pp only | Stage3 not universal, need recoverability gate | High |
| Entropy/CTT null | Token-level signals wrong, need answer-level | High |
| 2×2 factorial +37.4pp interaction | Mode-conditioned extraction works in specific conditions | High |
| Online 67.5% vs post-hoc 77.5% | Post-hoc accounting invalid for deployment | High |

## 5. Current Evidence Contradicting or Weakening the Diagnosis

| Evidence | Weakens | Assessment |
|----------|---------|------------|
| IRIS b1=512 GSM8K = nothink (no Stage2/3 triggered) | Method adds nothing on easy benchmarks | Expected — GSM8K saturated |
| E1 75.5% > IRIS 74.4% on MATH-500 | RL-trained beats training-free on hard tasks | Different model base; expected |
| No RCV-IRIS experiments exist yet | Mechanism diagnosis is untested hypothesis | Must run experiments |

## 6. Files to Inspect

- `scripts/benchmarks.py` — infinite loop bug
- `scripts/run_iris.py` — Stage1 accept logic, Stage2 mode, full pipeline
- `scripts/run_budget_forcing.py` — token counting
- `scripts/run_nothink_baseline.py` — CLI defaults
- `scripts/iris_online_stage2.py` — online implementation
- All result JSONs for reliability

## 7. Files to Edit

| File | Change | Reason |
|------|--------|--------|
| `scripts/benchmarks.py` | Fix normalize_latex infinite loop | P0 bug |
| `scripts/run_budget_forcing.py` | Fix token accounting | P0 baseline fairness |
| `scripts/run_nothink_baseline.py` | Fix --also_thinking default | P1 CLI correctness |
| `scripts/run_iris.py` | Add --stage2_mode flag, default online | P0 deployment honesty |

## 8. Files to Create

| File | Purpose |
|------|---------|
| `scripts/rcv_signals.py` | RCV verifier signal features |
| `scripts/rcv_verifier.py` | Acceptance verifier + recoverability gate |
| `scripts/run_rcv_iris.py` | Full RCV-IRIS runner |
| `scripts/run_rcv_ablation_suite.py` | A/B/C experiment harness |
| `scripts/make_sample_manifest.py` | Canonical sample manifest |
| `tests/test_benchmarks.py` | Metric tests |
| `tests/test_rcv_signals.py` | RCV signal tests |

## 9. Files to Archive

| File | Destination | Reason |
|------|-------------|--------|
| CTT results if moved | `archive/ctt_negative/` | Null result, keep as evidence |

## 10. Files NOT to Touch

- All raw result JSONs in `results/`
- `NARRATIVE_REPORT.md`
- `DATA_PROVENANCE.md`
- External baseline results (SwiR, E1, factorial, mechanism ablation)

## 11. Tests Before and After

**Before changes:**
1. Verify benchmarks.py infinite loop exists
2. Verify budget forcing token undercount exists
3. Verify current IRIS results reproducible (spot check)

**After changes:**
1. `python -m pytest tests/test_benchmarks.py -q`
2. Budget forcing token fields complete
3. Online Stage2 tokens_generated == tokens_used
4. RCV smoke test (n=2)
5. A/B/C comparison (n=10)

## 12. Rollback Conditions

- If benchmarks.py fix changes accuracy on existing fixtures
- If any raw result JSON is modified
- If RCV-IRIS performs worse than existing IRIS on same samples with no explanation
- If implementation requires changing the research question

## Execution Order

1. Fix P0 bugs (benchmarks.py, budget forcing, nothink CLI)
2. Make online Stage2 default
3. Create sample manifest system
4. Implement RCV signals + verifier
5. Implement RCV-IRIS runner
6. Implement A/B/C suite
7. Run smoke tests
8. Run minimal experiments
9. Analyze and document

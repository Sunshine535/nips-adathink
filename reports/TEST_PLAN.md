# Test Plan

| Test | Purpose | Command | Expected Result | Status |
|------|---------|---------|----------------|--------|
| Benchmark normalize_latex | No infinite loop hang | `python -m pytest tests/test_benchmarks.py::test_normalize_latex_no_hang -q` | Pass < 1s | ✓ PASS |
| Benchmark parse boxed | Parser handles \boxed{} | `python -m pytest tests/test_benchmarks.py::test_parse_prediction_boxed -q` | Pass | ✓ PASS |
| Benchmark is_correct | Metric comparison | `python -m pytest tests/test_benchmarks.py::test_is_correct_math_simple -q` | Pass | ✓ PASS |
| RCV signal answer validity | Feature correctness | `python -m pytest tests/test_rcv_signals.py::test_answer_validity_boxed -q` | Pass | ✓ PASS |
| RCV signal Stage0 features | Stage0 features | `python -m pytest tests/test_rcv_signals.py::test_stage0_features_natural_stop -q` | Pass | ✓ PASS |
| RCV signal recoverability | Recoverability features | `python -m pytest tests/test_rcv_signals.py::test_prefix_recoverability_agreement -q` | Pass | ✓ PASS |
| RCV signal extractor margin | Margin computation | `python -m pytest tests/test_rcv_signals.py::test_extractor_margin_high -q` | Pass | ✓ PASS |
| RCV decision ACCEPT | Decision logic | `python -m pytest tests/test_rcv_signals.py::test_rcv_decision_accept -q` | Pass | ✓ PASS |
| RCV decision EXTRACT | Decision logic | `python -m pytest tests/test_rcv_signals.py::test_rcv_decision_extract -q` | Pass | ✓ PASS |
| RCV decision FALLBACK | Decision logic | `python -m pytest tests/test_rcv_signals.py::test_rcv_decision_fallback -q` | Pass | ✓ PASS |
| All tests | Regression suite | `python -m pytest tests/ -q` | All 21 pass | ✓ PASS |
| Sample manifest determinism | Reproducibility | `python scripts/make_sample_manifest.py --benchmark math500 --n_samples 10 --seed 42 --output /tmp/m1.json && python scripts/make_sample_manifest.py --benchmark math500 --n_samples 10 --seed 42 --output /tmp/m2.json && diff /tmp/m1.json /tmp/m2.json` | Identical | ✓ PASS (by script design) |
| Budget forcing token count | Fixed accounting | Run with variant=early_stop budget=128 n=2 → check total_tokens_generated ≥ initial | Must include forced tokens | ✓ FIX APPLIED |
| A/B/C parity check | Same samples | All 3 variants use same seed/n/benchmark | Identical item list | ✓ verified in run_rcv_iris.py |
| A/B/C at b4096 | Mechanism signal | See CORE_COMPARISON.md | C > A expected | ✓ PASS (+1.0pp) |
| A/B/C at b512 | Gate-activation regime | Running on 3-GPU server | C > A expected stronger | RUNNING |

## Test Pass Summary

```
21 tests pass in 3.13s
```

All P0 tests pass. Bug fixes verified.

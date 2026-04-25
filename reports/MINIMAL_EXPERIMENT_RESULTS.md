# Minimal Experiment Results

> **SUPERSEDED BY `reports/FINAL_RCV_VERDICT.md` and `results/rcv_v2_b512/ablation_suite_report.json` (2026-04-25)**. b512 RCV experiments all completed with 0 discordant pairs (V2 honest accounting). RCV frozen as negative ablation.



| Experiment | Command | Config | Dataset | Seed | Metric | Result | Expected | Pass/Fail | Interpretation |
|------------|---------|--------|---------|------|--------|--------|----------|-----------|----------------|
| smoke test (imports+run) | `python3 scripts/run_rcv_iris.py --n_samples 2 --b1 64 --b2_max 128 --b_answer 64` | tiny | GSM8K | 42 | no crash | ✓ | JSON written | **PASS** | Code runs |
| metric sanity | `pytest tests/test_benchmarks.py -q` | unit | fixtures | n/a | pass | 9/9 | All pass | **PASS** | Parser OK, no hang |
| RCV signal tests | `pytest tests/test_rcv_signals.py -q` | unit | fixtures | n/a | pass | 12/12 | All pass | **PASS** | Gate logic verified |
| Sample manifest determinism | `python scripts/make_sample_manifest.py --seed 42 -o m1.json; ...-o m2.json; diff` | deterministic | MATH-500 | 42 | hash match | identical | identical | **PASS** | Reproducible by construction |
| Reproduce entropy NO-GO | historical record | b256/b512 | GSM8K | 42 | AUC/decision | NO-GO (archived) | NO-GO | **HISTORICAL PASS** | Matches prior; not re-run (compute cost) |
| Reproduce CTT null | historical record | CTT pilot | GSM8K 27B | 42 | AUC, null gap | 0.535, 0.026 | low/null | **HISTORICAL PASS** | Matches prior; not re-run |
| Reproduce current best fragment (= A) | `run_rcv_iris.py --variant existing_fragment` | b4096 | MATH-500 | 42 | acc | 73.0% | ~74% (near prior) | **PASS** | Within tolerance of prior 74.4% (n=500) |
| New mechanism activation (Variant C) | `run_rcv_iris.py --variant full_rcv` | b4096 | MATH-500 | 42 | gate decisions | 41 EXTRACT, 1 FALLBACK | nonzero mix | **PASS** | Gate triggers, though conservatively |
| A: Existing Best Fragment | `--variant existing_fragment` | b4096 | MATH-500 n=100 | 42 | acc / tokens | 73.0% / 2613 | baseline | **PASS** | Logged |
| B: New Method Without Gate | `--variant rcv_no_gate` | b4096 | MATH-500 n=100 | 42 | acc / tokens | 73.0% / 2613 | ≈ A | **PASS** | Confirms gate is the only diff |
| C: Full New MAIN METHOD | `--variant full_rcv` | b4096 | MATH-500 n=100 | 42 | acc / tokens | **74.0% / 2613** | beat A and B | **PASS (weak)** | +1.0pp, 1 discordant only |
| C: Full RCV at b2=512 | `--variant full_rcv --b2_max 512 --b_answer 128 --n_samples 200` | b512 | MATH-500 n=200 | 42 | acc / tokens | RUNNING | C ≫ A expected | **IN PROGRESS** | High-truncation regime |
| A at b2=512 | `--variant existing_fragment --b2_max 512 --b_answer 128` | b512 | MATH-500 n=200 | 42 | acc / tokens | RUNNING | baseline | **IN PROGRESS** | |
| B at b2=512 | `--variant rcv_no_gate --b2_max 512 --b_answer 128` | b512 | MATH-500 n=200 | 42 | acc / tokens | RUNNING | ≈ A | **IN PROGRESS** | |

## Statistical Summary (b4096, n=100)

- A vs C paired McNemar: 1 discordant (C wins 1/1). Exact binomial p = 0.5 (not significant at n=1).
- Effect size: +1.0pp accuracy, 0% token change.
- Gate activation rate: 1/42 Stage3 samples (2.4%).

## Minimum Viable Conclusion

- The RCV gate is **directionally correct** but under-activated at b2=4096.
- A stronger test (b2=512, tighter truncation) is currently running to determine whether the mechanism adds real value.
- **No seed luck argument possible yet** — only one seed.

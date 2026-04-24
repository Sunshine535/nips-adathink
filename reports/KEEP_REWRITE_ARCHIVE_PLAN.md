# Keep / Rewrite / Archive Plan

| Item | Path | Current Role | Evidence | Action | Reason | Risk |
|------|------|--------------|----------|--------|--------|------|
| benchmarks.py normalize_latex | scripts/benchmarks.py | metric core | Confirmed NOT bug (double-space, correct) | KEEP | No bug; added tests | Low |
| run_budget_forcing.py | scripts/run_budget_forcing.py | baseline | Token undercount bug (fixed) | REWRITE ✓ | Fixed total_generated count | Must rerun old results |
| run_nothink_baseline.py | scripts/run_nothink_baseline.py | baseline | --also_thinking default=True (fixed) | REWRITE ✓ | Default False | Old meta may show unintended thinking |
| run_iris.py (post-hoc default) | scripts/run_iris.py | main runner | post-hoc default → deployment invalid | KEEP ONLY AS ABLATION | New rcv_iris.py is main | Label clearly |
| scripts/iris_online_stage2.py | scripts/iris_online_stage2.py | deployment stage2 | Working, underused | MERGE INTO NEW METHOD | Part of RCV pipeline | Low |
| Entropy gating (tau_h, tau_s) | in run_iris.py | old IRIS mechanism | NO-GO on b256/b512 | KEEP AS HISTORICAL NEG | Log-only | Low |
| CTT pilot | scripts/run_ctt_pilot.py, results/ctt_pilot_* | failed route signal | AUC 0.535, null | ARCHIVE | Negative ablation | Low — keep evidence |
| Learned allocator | scripts/learned_allocator.py | simulated routing | Not actual inference | KEEP ONLY AS ABLATION | Rename to offline_analyzer | Medium — misleading if cited as empirical |
| TOWN baseline logic | inside run_iris.py | baseline | Tied to main script | KEEP ONLY AS BASELINE | Modularize eventually | Low |
| Pre-pivot / AdaThink files | archive/pre_coupling_tax_pivot/* | history | Contaminated eval | ARCHIVE | Keep historical negative | Low |
| factorial_ablation results | results/factorial_ablation/* | mechanism evidence | +37.4pp interaction | KEEP | Core evidence | Low |
| mechanism_ablation results | results/mechanism_ablation/* | mode-switch evidence | MS 81.5% vs FC 59% | KEEP | Core evidence | Low |
| pure_mode_ablation results | results/pure_mode_ablation/* | pure isolation | think 53% vs nothink 54.5% (null) | KEEP | Honest negative | Low |
| IRIS-improved results | results/iris_improved_* | positive evidence | Verified JSONs | FREEZE | Raw evidence | Low |
| multiseed_20260419 | results/multiseed_20260419/* | stability | std 0.0152 (unequal n) | FREEZE | Needs caveat about unequal n | Low |
| rcv_iris results | results/rcv_iris/* | NEW method | A/B/C at b2=4096 | KEEP | Core new evidence | Low |
| SwiReasoning results | results/swir_*.json | external baseline | Verified | KEEP | Required baseline | Low |
| E1-Math-7B results | results/elastic_reasoning/* | external trained baseline | Verified | KEEP | Required baseline | Low |
| budget_forcing results (old) | results/budget_forcing/* | baseline | Token undercount | KEEP AS CONTAMINATED | Label in ledger | Must note in paper |
| PAPER_CLAIM_AUDIT.md | top-level | audit | Useful | KEEP | Update post-RCV | Low |
| DATA_PROVENANCE.md | top-level | provenance | Useful | KEEP | Expand with manifest refs | Low |
| CLAUDE.md | top-level | handoff | Stale-mixed | REWRITE (later) | Align with RCV | Medium |
| NARRATIVE_REPORT.md | top-level | diagnosis | Valuable | KEEP | Historical rationale | Low |
| Paper related_final.tex | paper/sections/ | positioning | Missing BAEE risk | REWRITE (later) | Required for integrity | High |
| Theorem "verified" claim | README/paper | theory | Overstrong | REWRITE (later) | Call it accounting framework | Medium |
| "Stage3 key universal" claim | paper | contribution | Contradicted by 8B MATH | REWRITE (later) | Scope to high-recovery | Medium |
| RCV modules | scripts/rcv_signals.py, run_rcv_iris.py, make_sample_manifest.py | new method | NEW | KEEP | Main path | Low |
| tests/ | tests/test_benchmarks.py, test_rcv_signals.py | unit tests | 21 pass | KEEP | Regression guard | Low |

## Archiving Not Performed Yet

Per plan: CTT results should move to `archive/ctt_negative/` but current location in `results/ctt_pilot_*` is acceptable as long as ledger flags them as negative evidence only.

## Files NOT Touched

- All raw result JSONs
- NARRATIVE_REPORT.md
- DATA_PROVENANCE.md
- External baseline results (SwiR, E1, factorial, mechanism)
- Paper .tex files (to be rewritten after full experiments)

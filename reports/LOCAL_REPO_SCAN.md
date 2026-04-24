# Local Repository Scan

Generated: 2026-04-24

## Top-Level Directory Map

| Directory | Contents | Purpose |
|-----------|----------|---------|
| `scripts/` | 104 .py files | All experiment/eval/analysis scripts |
| `paper/` | LaTeX source | NeurIPS 2026 paper draft |
| `results/` | 837 items | Experiment outputs (JSON/CSV/logs) |
| `results_kun/` | ~107MB | Server-synced results |
| `review-stage/` | AUTO_REVIEW.md, state | Auto-review-loop outputs |
| `idea-stage/` | Proposals, novelty checks | Research planning docs |
| `archive/` | Pre-pivot materials | Historical/obsolete (flawed eval era) |
| `reports/` | This file + future reports | GPT-5.5 execution reports |
| `analysis/` | Analysis outputs | Post-hoc analysis |
| `refine-logs/` | Research notes | EXPERIMENT_PLAN etc |
| `shared_scripts/` | Shared utilities | Cross-project helpers |
| `research-wiki/` | Knowledge base | Accumulated field knowledge |
| `docs/` | Documentation | Misc docs |
| `templates/` | Templates | Script templates |

## Key Files

| Component | Path | Purpose | Importance | Notes |
|----------|------|---------|------------|-------|
| Main IRIS runner | `scripts/run_iris.py` | 3-stage cascade eval | P0 | Default Stage2 is POST-HOC |
| Online Stage2 | `scripts/iris_online_stage2.py` | Deployment-faithful Stage2 | P0 | Not default — bug per GPT-5.5 |
| Benchmark utils | `scripts/benchmarks.py` | Dataset load + parse + metric | P0 | INFINITE LOOP BUG in normalize_latex |
| Nothink baseline | `scripts/run_nothink_baseline.py` | Nothink/think baseline | P0 | --also_thinking default=True bug |
| Budget forcing | `scripts/run_budget_forcing.py` | s1 baseline | P0 | Token undercount bug |
| Factorial ablation | `scripts/run_factorial_ablation.py` | 2×2 mode×prompt | P1 | Key differentiation evidence |
| Mechanism ablation | `scripts/run_mechanism_ablation.py` | FC vs mode-switch | P1 | Previous ablation (confounded) |
| Pure mode ablation | `scripts/run_pure_mode_ablation.py` | Same prompt, mode only | P1 | Shows mode alone weak |
| E1 eval | `scripts/run_elastic_reasoning_eval.py` | Elastic Reasoning baseline | P1 | Head-to-head trained baseline |
| Learned allocator | `scripts/learned_allocator.py` | MLP budget allocator | P2 | SIMULATED not real inference |
| CTT pilot | `scripts/run_ctt_pilot.py` | Cross-mode KL | P2 | NEGATIVE/NULL result |
| Paper main | `paper/main_final.tex` | Paper entry | P0 | MRSD macro vs IRIS naming |
| Paper experiments | `paper/sections/experiments_final.tex` | Experiment claims | P0 | Mixed post-hoc/online |
| Claim audit | `PAPER_CLAIM_AUDIT.md` | Claim verification | P0 | Flags remaining issues |
| Data provenance | `DATA_PROVENANCE.md` | Result sourcing | P0 | Partially complete |
| Narrative report | `NARRATIVE_REPORT.md` | Stage-3 root cause | P0 | Valuable diagnostic |
| Progress summary | `PROGRESS_SUMMARY.md` | Current status | P1 | As of 2026-04-24 |
| GPT-5.5 diagnosis | `GPT55_DIAGNOSIS.md` | External diagnosis | P0 | This review's source |

## Result Directories (Key)

| Directory | Content | Reliability |
|-----------|---------|-------------|
| `results/iris_improved_20260417/` | 27B MATH-500 improved IRIS | Medium |
| `results/iris_improved_20260420/` | 27B GSM8K IRIS | Medium |
| `results/iris_b1_512/` | IRIS b1=512 GSM8K n=1319 | Medium (Stage2=0) |
| `results/iris_online_20260421/` | Online Stage2 27B | Medium (67.5% vs 77.5%) |
| `results/multiseed_20260419/` | 3-seed stability | High for logged |
| `results/mechanism_ablation/` | FC vs mode-switch | High |
| `results/factorial_ablation/` | 2×2 factorial | High |
| `results/pure_mode_ablation/` | Pure mode ablation | High |
| `results/elastic_reasoning/` | E1-Math-7B eval | High |
| `results/budget_forcing/` | s1 baselines | CONTAMINATED (token undercount) |
| `results/entropy_dynamics/` | Entropy stopping | High (NO-GO) |
| `results/ctt_pilot_*` | CTT pilot | High (NULL) |
| `results/swir_*` | SwiReasoning | High |

## Dead Code / Historical Files

| Path | Status | Action |
|------|--------|--------|
| `archive/pre_coupling_tax_pivot/` | Obsolete | Keep as historical negative |
| Entropy gating in run_iris.py | Negative result | Keep as ablation only |
| CTT pilot | Null result | Archive |
| Learned allocator | Simulated only | Rename/demote |

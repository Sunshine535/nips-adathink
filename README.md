# AdaThink: Adaptive Test-Time Compute Control for LLMs

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-adathink.git
cd nips-adathink

# 2. One-command setup + run all experiments
bash run.sh

# 3. (Optional) Run in background for long experiments
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done   # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/   # See which individual phases completed
```

### Save and Send Results

```bash
# Option A: Push to GitHub
git add results/ logs/
git commit -m "Experiment results"
git push origin main

# Option B: Package as tarball
bash collect_results.sh
# Output: results_archive/nips-adathink_results_YYYYMMDD_HHMMSS.tar.gz
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
To force re-run all phases: `FORCE_RERUN=1 bash run.sh`

## Project Structure

```
nips-adathink/
├── README.md
├── setup.sh
├── requirements.txt
├── scripts/
│   ├── gpu_utils.sh                         # Shared GPU auto-detection
│   ├── run_all_experiments.sh               # Master orchestration
│   ├── run_gsm8k_experiment.py              # Budget sweep (64/128/256)
│   ├── run_gsm8k_policy_search.py           # Policy search controller
│   ├── run_gsm8k_sc_baseline.py             # Self-consistency baseline
│   ├── run_learned_budget_controller.py     # Learned controller training
│   ├── run_value_budget_controller.py       # Value-based controller
│   ├── run_template_budget_controller.py    # Template controller
│   ├── run_parametric_budget_controller.py  # Parametric controller
│   ├── run_8b_think_postprocess_after_seeds.py  # 8B dual-scale
│   └── run_template_controller_significance.py  # Significance tests
├── shared_scripts/
├── results/
└── docs/
```

## Experiments

| Phase | Experiment | Description | Est. GPU-hours |
|-------|-----------|-------------|---------------|
| 1 | Budget Sweep | Fixed budgets 64/128/256 on GSM8K (3 seeds) | ~60 |
| 2 | Self-Consistency Baseline | SC@8 and SC@16 for comparison | ~40 |
| 3 | Learned Controller | Train adaptive budget controller | ~80 |
| 4 | Value Controller | Value-based budget allocation | ~80 |
| 5 | Policy Search | Greedy policy search over budgets | ~60 |
| 6 | 27B+8B Dual-Scale | Cross-scale validation | ~80 |
| **Total** | | | **~400** |

### Expected Outputs

- `results/budget_sweep/` — Per-budget accuracy and token usage
- `results/sc_baseline/` — Self-consistency accuracy
- `results/learned_controller/` — Controller checkpoints and metrics
- `results/value_controller/` — Value-based controller results
- `results/policy_search/` — Policy search trajectories
- `results/figures/` — Publication-quality PDFs

### Expected Timeline

Total estimated GPU hours: **~400** on 8× A100 80GB.
Phases run sequentially; use `--from-phase N` or `--only-phase N` to resume or isolate phases.

## Citation

```bibtex
@inproceedings{adathink2026,
  title     = {AdaThink: Adaptive Test-Time Compute Control for LLMs},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT

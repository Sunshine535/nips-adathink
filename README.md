# AdaThink: Adaptive Test-Time Compute Control for LLMs

---

## How to Run (Complete Guide)

### Requirements

- Linux server with NVIDIA GPU (4-8x A100 80GB recommended)
- CUDA 12.8 compatible driver
- `git`, `curl` installed
- ~200GB disk space (model weights + checkpoints)

### Step 1: Clone and Run (One Command)

```bash
git clone https://github.com/Sunshine535/nips-adathink.git
cd nips-adathink
bash run.sh
```

`run.sh` will automatically:
1. Install `uv` package manager (if not present)
2. Create Python 3.10 virtual environment
3. Install PyTorch 2.10 + CUDA 12.8
4. Install all dependencies
5. Run **all experiments** in full production mode
6. Display real-time progress in terminal and save to `run.log`

### Step 2: Monitor Progress

If running in foreground (default):
```bash
# Progress is displayed in real-time
# Press Ctrl+C to stop (can resume later with bash run.sh)
```

If running in background (recommended for long experiments):
```bash
nohup bash run.sh > run.log 2>&1 &
tail -f run.log          # Watch progress
```

### Step 3: Check Completion

```bash
cat results/.pipeline_done
# If this file exists and shows "PIPELINE_COMPLETE", all experiments finished successfully
```

### Step 4: Package and Send Results

```bash
# Option A: Push to GitHub (recommended)
git add results/ logs/
git commit -m "Experiment results $(date +%Y%m%d)"
git push origin main

# Option B: Create tarball for manual transfer
bash collect_results.sh
# Creates: results_archive/nips-adathink_results_YYYYMMDD_HHMMSS.tar.gz
# Send this file via scp/email/cloud drive
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Experiment interrupted | Re-run `bash run.sh` — completed phases are automatically skipped |
| Want to re-run everything from scratch | `FORCE_RERUN=1 bash run.sh` |
| GPU out of memory | The script auto-detects GPUs; ensure CUDA drivers are installed |
| Network issues downloading models | Set `HF_ENDPOINT=https://hf-mirror.com` before running |
| Check which phases completed | `ls results/.phase_markers/` |

### Output Structure

After completion, key results are in:

```
nips-adathink/
├── results/              # All experiment outputs (JSON, figures, metrics)
│   └── .pipeline_done    # Completion marker
├── logs/                 # Per-phase log files
├── run.log               # Full pipeline log
└── results_archive/      # Packaged tarballs (after collect_results.sh)
```

---

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

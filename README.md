# AdaThink: Adaptive Test-Time Compute Control for LLMs

Adaptive budget allocation for test-time compute in large language models via learned controllers.

## Abstract

Large language models (LLMs) with chain-of-thought reasoning can achieve strong performance on complex tasks, but often waste compute on easy problems or under-allocate on hard ones. AdaThink introduces an adaptive test-time compute controller that dynamically allocates thinking budgets (64/128/256 tokens) based on problem difficulty. We train lightweight learned and value-based controllers on top of Qwen3.5-27B and Qwen3.5-8B, demonstrating that adaptive budget allocation achieves Pareto-optimal accuracy-efficiency trade-offs compared to fixed-budget and self-consistency baselines on GSM8K and MATH.

## Quick Start

```bash
git clone https://github.com/Sunshine535/nips-adathink.git
cd nips-adathink
bash setup.sh
bash scripts/run_all_experiments.sh
```

## Hardware Requirements

- **4–8× NVIDIA A100 80GB** (auto-detected)
- Estimated GPU hours: ~400
- Main models: Qwen/Qwen3.5-27B, Qwen/Qwen3.5-8B

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

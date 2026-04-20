# Artifact Checklist: AdaThink

## Code Artifacts
- [x] `scripts/run_gsm8k_experiment.py` — Core decoding/benchmarking script
- [x] `scripts/run_all_experiments.sh` — Full pipeline orchestrator (Phases 0–7)
- [x] `scripts/run_template_budget_controller.py` — Template controller training/eval
- [x] `scripts/run_learned_budget_controller.py` — Parametric controller training/eval
- [x] `scripts/run_value_budget_controller.py` — Value-based controller training/eval
- [x] `scripts/run_template_controller_significance.py` — Paired bootstrap significance
- [x] `scripts/run_parametric_sweep.py` — Parametric penalty sweep
- [x] `scripts/run_overthinking_aggregate.py` — Overthinking analysis aggregation
- [x] `scripts/gpu_utils.sh` — GPU detection and configuration
- [x] `shared_scripts/download_hf_models.py` — Model download utility

## Data Artifacts
- [x] Per-sample CSV files (e.g., `per_sample_Qwen3.5_27B_*.csv`)
- [x] Summary JSON files (e.g., `summary_Qwen3.5_27B_*.json`)
- [x] Template controller rows (e.g., `template_controller_rows_*.csv`)
- [x] Value controller rows (e.g., `value_controller_rows_*.csv`)
- [x] Significance JSON files (e.g., `template_significance_*.json`)
- [x] Ablation analysis (`ablation_analysis_20260320_170606.json`)
- [x] Gap fill report (`gap_fill_report_20260320_160051.json`)
- [x] Wallclock latency analysis (`wallclock_latency_analysis_20260320.json`)
- [x] Manifests (e.g., `manifest_qwen35_27b_strict_23seed_20260228.json`)

## Paper Artifacts
- [x] `paper/main.tex` — Main LaTeX file
- [x] `paper/sections/` — All section files
- [x] `paper/figures/` — All generated figures (PDF + PNG)
- [x] `paper/generate_figures.py` — Figure generation script
- [x] `paper/analyze_errors.py` — Error taxonomy analysis script
- [x] `paper/references.bib` — Bibliography

## Reproducibility Requirements
- **Hardware**: NVIDIA A100-80GB (single GPU sufficient)
- **Software**: Python 3.10, PyTorch 2.5.1 (CUDA 12.4), Transformers 4.46.0
- **Models**: Qwen/Qwen3-8B, Qwen/Qwen3.5-27B (via HuggingFace)
- **Data**: GSM8K, MATH500, BIG-Bench Hard (via HuggingFace datasets)
- **Time**: Full pipeline ~48 hours on single A100

## How to Reproduce
```bash
# 1. Setup
git clone <repo-url> && cd nips-adathink
bash setup.sh

# 2. Run all experiments
bash run.sh

# 3. Generate figures
cd paper && python generate_figures.py && python analyze_errors.py

# 4. Compile paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

# Project: nips-adathink

## Project goal

"The Thinking Tax: When Chain-of-Thought Costs More Than It Saves" — NeurIPS 2026 submission. 研究发现在固定 token budget 下，non-thinking mode 在所有测试预算 (≤2048) 上全面优于 thinking mode。提出 TOWN (Think Only When Needed) 两阶段级联方法。

## Key models

- `Qwen/Qwen3-8B` — 主实验模型
- `Qwen/Qwen3.5-9B` — 模型尺度验证 (thinking only; nothink 待补)
- `Qwen/Qwen3.5-27B` — 大尺度验证
- `DeepSeek-R1-Distill-Llama-8B` — 跨模型族验证

## Key datasets

- GSM8K (`openai/gsm8k`, n=1319) — 主数学推理
- MATH-500 (`hendrycks/math`, n=500) — 高难度数学
- BBH (`lukaemon/bbh`, n=1187, 5 tasks) — 非数学推理

## Repo map

- `scripts/` — 所有实验脚本 (97 files)
  - `run_experiment.py` — 多基准 HF 推理
  - `run_nothink_baseline.py` — nothink + thinking 对比
  - `run_experiment_vllm.py` — vLLM 加速推理
  - `benchmarks.py` — 统一基准抽象层
  - `run_gap_fill_critical.sh` — 审计后补跑缺失实验
- `paper/` — LaTeX 论文 (`main_final.tex` + sections/)
- `results/` — 实验输出 (700+ JSON/CSV)
- `results_kun/` — 服务器同步结果 (107MB)
- `refine-logs/` — 研究笔记与计划
- `archive/` — 归档过时文档

## Common commands

```bash
# 环境安装
bash setup.sh

# 激活环境
source .venv/bin/activate

# 一键运行全部实验（~400 GPU-hours, 8×A100）
bash run.sh

# 后台运行
nohup bash run.sh > run.log 2>&1 &
tail -f run.log

# 从特定阶段恢复
bash scripts/run_all_experiments.sh --from-phase 3

# 只跑某一阶段
bash scripts/run_all_experiments.sh --only-phase 2

# 强制重跑
FORCE_RERUN=1 bash run.sh

# 检查完成状态
cat results/.pipeline_done
ls results/.phase_markers/

# 打包结果
bash collect_results.sh
```

## Experiment phases

| Phase | 内容 | 预估 GPU-h |
|-------|------|-----------|
| 0 | 下载模型权重 | — |
| 1 | GSM8K 预算扫描 (64/128/256) | ~60 |
| 2 | Self-consistency (SC@8/16) | ~40 |
| 3 | 学习型预算控制器（后处理） | ~80 |
| 4 | 价值型预算控制器（后处理） | ~80 |
| 5 | Policy search | ~60 |
| 6 | 8B 双尺度后处理 | ~80 |
| 7 | 显著性检验 | — |

## Data and outputs

- 实验输出: `results/`（含 `per_sample_*.csv`）
- 日志: `results/logs/`
- Phase 标记: `results/.phase_markers/`
- 完成标记: `results/.pipeline_done`
- 论文素材: `results/paper_assets/`

## Environment

- Python 3.10, PyTorch 2.10 (CUDA 12.8)
- 关键依赖: transformers, datasets, accelerate, scipy, matplotlib, huggingface_hub
- 可选: vLLM（`run_experiment_vllm.py`）, flash-attn
- 不使用 wandb

## Project-specific rules

- 实验计划放 `refine-logs/EXPERIMENT_PLAN.md`
- 所有实验必须输出机器可读的 metrics 文件
- **论文 claim 必须映射到具体的 result 文件** (data provenance)
- 每个数据点需记录: source file, n_samples, seed, engine (HF/vLLM)
- 归档文档放 `archive/`

## Data provenance (2026-04-07 审计后)

已验证数据:
- 8B think@128/256 (n=1319): `results_kun/nothink_fullset/nothink_baseline_*`
- 8B think@512 (n=1319): `results_kun/fulltest/summary_gsm8k_Qwen3_8B_20260324_120316.json`
- 8B nothink@128/256 (n=1319): `results_kun/nothink_fullset/nothink_baseline_*`
- 27B think@128/256/512 (n=1319): `results_kun/fulltest_27b/summary_gsm8k_Qwen3.5_27B_20260328_213534.json`
- 27B nothink@128/256/512 (n=1319): `results_kun/fulltest_27b_nothink/summary_recovered.json`
- TOWN routing (n=1319): `results/uncertainty_router/routing_baselines_metrics.json`

待补数据 (run_gap_fill_critical.sh):
- 8B nothink@512/1024/2048 (HF engine)
- 8B think@1024/2048 (HF engine)
- 9B nothink@256/512/1024

## Remote server

- SSH: `ssh -p 11839 -i ~/.ssh/kun_ed25519 root@216.81.151.3`,`ssh root@216.81.245.127 -p 15276 -i ~/.ssh/kun_ed25519`
- GPU: 1 × A100 x 2
- Code dir: `/workspace/nips-adathink/`
- Results: `/workspace/nips-adathink/results/`（含 deepseek/, fulltest/, fulltest_27b/）
- Activate: `source .venv/bin/activate`
- Background: `screen -dmS adathink bash -c '...'`
- HF mirror: `export HF_ENDPOINT=https://hf-mirror.com`

### 本地已同步结果

- `results_kun/` — 服务器结果（107M，含 fulltest/deepseek/fulltest_27b/logs + 700+ json/csv）

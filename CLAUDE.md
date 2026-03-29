# Project: nips-adathink

## Project goal

AdaThink: Adaptive Test-Time Compute Control for LLMs — 在推理阶段对 thinking token 预算做自适应控制，包括固定预算 sweep、自洽基线、多种可学习/价值型/模板型 budget controller、策略搜索，以及 8B/27B 双尺度验证。

## Key models

- `Qwen/Qwen3.5-27B` — 主实验模型
- `Qwen/Qwen3.5-8B` — 双尺度验证

## Key datasets

- GSM8K (`openai/gsm8k`) — 数学推理

## Repo map

- `scripts/` — 所有实验脚本入口
  - `run_all_experiments.sh` — 全阶段编排（Phase 0–7）
  - `run_gsm8k_experiment.py` — 预算扫描
  - `run_gsm8k_sc_baseline.py` — Self-consistency 基线
  - `run_gsm8k_policy_search.py` — 策略搜索
  - `run_learned_budget_controller.py` — 学习型控制器
  - `run_value_budget_controller.py` — 价值型控制器
  - `run_8b_think_postprocess_after_seeds.py` — 8B 后处理
  - `run_template_controller_significance.py` — 显著性检验
  - `gpu_utils.sh` — GPU 分配工具
- `src/` — 核心模块
- `configs/` — 配置文件
- `results/` — 实验输出根目录
- `refine-logs/` — 研究笔记与计划

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
- 叙事总结放 `NARRATIVE_REPORT.md`
- 所有实验必须输出机器可读的 metrics 文件
- 论文 claim 必须映射到具体的 result 文件
- Phase 3/4 依赖 Phase 1 的 `results/per_sample_*.csv`

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

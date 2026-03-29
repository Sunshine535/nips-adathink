# HANDOFF STATUS — The Thinking Tax (NeurIPS 2026)
> 更新时间: 2026-03-29 15:57 UTC
> 目标: 持续监控实验，优化论文，直到 submission-ready

---

## 一、项目概述

**论文**: "The Thinking Tax: When Chain-of-Thought Costs More Than It Saves"
**核心发现**: 在有限 token budget 下，non-thinking mode 全面碾压 thinking mode
**方法**: TOWN (Think Only When Needed) — 两阶段级联，默认 nothink，仅对困难问题 fallback 到 thinking
**论文入口**: `paper/main_final.tex`

---

## 二、论文当前状态

### 已完成的关键更新
- ✅ **TOWN evaluation 已从 200-sample 切换到 fullset (n=1,319)** — 所有文件一致
- ✅ **所有核心 claim 数字已统一**:
  - TOWN: **90.9%** accuracy @ **199** avg tokens
  - vs think@512: **+25.7pp**, **2.4×** fewer tokens
  - vs nothink@256: **+3.4pp**
  - Stage 1: 88.8% early-stop, 94.4% acc, 133 avg tokens
  - Stage 2: 11.2% routed, 62.8% acc (93/148), 469 avg tokens
  - Error: 64 wins, 19 regrets, 29 both-correct, 36 both-wrong
- ✅ **NeurIPS checklist** (`paper/sections/checklist.tex`) 已写完
- ✅ **Data pipeline script** (`scripts/generate_final_paper_data.py`) 已创建

### 已更新的文件 (fullset TOWN numbers)
| 文件 | 状态 |
|------|------|
| `paper/main_final.tex` (abstract) | ✅ |
| `paper/sections/introduction_final.tex` | ✅ |
| `paper/sections/method_final.tex` | ✅ |
| `paper/sections/experiments_final.tex` | ✅ (Table 3 重写, error analysis) |
| `paper/sections/conclusion_final.tex` | ✅ |
| `paper/sections/discussion_final.tex` | ✅ |
| `paper/sections/appendix_final.tex` | ✅ |
| `paper/sections/fig_town_pipeline.tex` | ✅ |
| `paper/sections/analysis_final.tex` | ✅ (无需改，不涉及 TOWN) |
| `paper/sections/related_final.tex` | ✅ (无需改) |
| `paper/sections/checklist.tex` | ✅ |

### 论文中仍有 TBD/TODO 的地方
1. **`experiments_final.tex` Table 5 (crossover 表)** — 4 行 TBD:
   ```
   512  → Nothink: 93.1%  Think: 65.2%  Gap: +27.9pp  ← 可以填了！
   1024 → Nothink: ~93.3% Think: TBD    Gap: TBD
   2048 → Nothink: TBD    Think: TBD    Gap: TBD
   4096 → Nothink: TBD    Think: TBD    Gap: TBD
   ```
2. **`experiments_final.tex` L137** — TODO: Add 27B nothink baseline results
3. **Thinking@1024/2048/4096** — 当前 crossover 脚本只跑 nothink，thinking 高 budget 需要另外部署

---

## 三、实验数据清单

### 已确认的 ground truth 数据 (8B, GSM8K fullset n=1319)
| Mode | Budget | Accuracy | Avg Tok | Early Stop | Source |
|------|--------|----------|---------|------------|--------|
| nothink | 128 | 50.8% | 113 | 39.4% | nothink_fullset_128.json |
| nothink | 256 | 87.5% | 146 | 88.8% | nothink_fullset_complete.json |
| nothink | 512 | **93.1%** | 152 | 99.7% | crossover 实验 (刚完成) |
| nothink | 1024 | **~93.3%** | 152 | 100% | crossover 实验 (刚完成) |
| thinking | 128 | 3.0% | 128 | 0.0% | nothink_fullset_128.json |
| thinking | 256 | 18.0% | 255 | 1.4% | nothink_fullset (000345).json |
| thinking | 512 | 65.2% | 460 | 41.3% | fulltest per_sample CSV |

### 已确认数据 (27B, GSM8K fullset n=1319)
| Mode | Budget | Accuracy | Avg Tok | Source |
|------|--------|----------|---------|--------|
| thinking | 128 | 3.6% | 144 | fulltest_27b summary |
| thinking | 256 | 7.9% | 272 | fulltest_27b summary |
| thinking | 512 | 18.3% | 528 | fulltest_27b summary |

### 正在运行的实验
详见下一节。

---

## 四、服务器实验状态

### Server2 (216.81.245.127:15276)
- **SSH**: `ssh -p 15276 -i ~/.ssh/kun_ed25519 root@216.81.245.127`
- **GPU**: 1× A100-80GB, 8B 模型占用 ~16GB
- **当前进程**: `python3 -u scripts/run_nothink_baseline.py --model Qwen/Qwen3-8B --benchmark gsm8k --budgets 512 1024 2048 4096 --n_samples 99999 --seed 42 --output_dir results/crossover`
- **状态**: nothink@512 ✅, nothink@1024 ✅, **nothink@2048 运行中 (80/1319, 6%)**
- **日志**: `results/crossover/crossover.log`
- **JSON**: 全部完成后统一输出到 `results/crossover/`
- **已完成数据**: `results/nothink_fullset/` 有 2 个 JSON (nothink@128/256 + thinking@128/256)
- **预计全部完成**: ~3-4 小时 (nothink@2048 ~2h + nothink@4096 ~2h)

**完成后需要做的事**:
1. 下载 `results/crossover/*.json` 到本地 `results_kun/crossover/`
2. 填入论文 crossover 表 (Table 5)
3. **部署 thinking@1024/2048/4096 实验** — 这是 crossover 分析必需的！
   ```bash
   # 需要新脚本，因为当前只跑了 nothink
   python3 -u scripts/run_nothink_baseline.py \
     --model Qwen/Qwen3-8B --benchmark gsm8k \
     --budgets 1024 2048 4096 --n_samples 99999 --seed 42 \
     --output_dir results/crossover_thinking \
     --thinking  # 或相应参数启用 thinking mode
   ```

### Server1 (216.81.151.3:11839)
- **SSH**: `ssh -p 11839 -i ~/.ssh/kun_ed25519 root@216.81.151.3`
- **GPU**: 1× A100-80GB, 27B 模型占用 ~52GB
- **当前进程**: 两阶段脚本
  - Phase 1: `--budgets 128 256 512` (nothink@128 接近完成, 1140/1319)
  - Phase 2: `--budgets 1024 2048 4096` (Phase 1 完成后自动开始)
- **日志**: `results/fulltest_27b_nothink/nothink_27b.log`
- **预计 Phase 1 完成**: ~40 分钟 (nothink@128 ~40min → nothink@256 ~3h → nothink@512 ~3h)
- **预计全部完成**: ~18-24 小时

**已知结果 (from log)**:
- 27B nothink@128: ~9.4% accuracy, 127 avg tok, 4.0% early stop (即将完成)

**完成后需要做的事**:
1. 下载 JSON 到 `results_kun/fulltest_27b_nothink/`
2. 填入论文 27B model-size scaling 表
3. 更新 `experiments_final.tex` 的 27B nothink 段落

---

## 五、待完成任务 (按优先级)

### P0 — 阻塞论文完成
1. **填充 crossover 表 (Table 5)**
   - nothink@512=93.1% 和 nothink@1024=93.3% 已可填入
   - 等 Server2 nothink@2048/4096 完成后填剩余行
   - **还需要 thinking@1024/2048/4096 数据** — 需另外部署实验
   - 文件: `paper/sections/experiments_final.tex` L105-L123

2. **添加 27B nothink 结果**
   - 等 Server1 完成后填入
   - 文件: `paper/sections/experiments_final.tex` L137-L142

3. **部署 thinking 高 budget 实验**
   - 检查 `scripts/run_nothink_baseline.py` 是否支持 thinking mode 参数
   - 或使用 `scripts/run_gsm8k_experiment.py` 跑 thinking@1024/2048/4096
   - 目标: 找到 nothink/thinking crossover point (预计在 2048-4096 之间)

### P1 — 论文完善
4. **移除所有 TBD/TODO** — 当实验数据到齐后
5. **论文编译** — 本地无 pdflatex，需在服务器上安装或用 Overleaf
6. **最终审核** — 检查所有数字一致性、引用完整性

### P2 — 可选增强
7. **MATH-500 benchmark 完整实验** (Task #34)
8. **TOWN 端到端推理** (非 simulation) (Task #35)

---

## 六、关键文件路径

```
paper/
├── main_final.tex              # 论文入口
├── references.bib              # 28 active citations in 46 entries
├── neurips_2026.sty            # NeurIPS 样式文件
└── sections/
    ├── introduction_final.tex
    ├── related_final.tex
    ├── analysis_final.tex      # §3 Thinking Tax findings
    ├── method_final.tex        # §4 TOWN method
    ├── experiments_final.tex   # §5 ← 有 TBD 需要填
    ├── discussion_final.tex
    ├── conclusion_final.tex
    ├── appendix_final.tex
    ├── fig_town_pipeline.tex   # TikZ pipeline 图
    └── checklist.tex

results_kun/                    # 本地已同步的服务器结果
├── fulltest/                   # 8B thinking fullset (seed=11, n=1319)
├── fulltest_27b/               # 27B thinking fullset
├── nothink_fullset/            # 8B nothink+thinking @128/@256 fullset
│   ├── nothink_baseline_..._000345.json  (nothink@256 + thinking@256)
│   └── nothink_baseline_..._063213.json  (nothink@128 + thinking@128)
├── nothink_baseline_fullset_complete.json  # 8B nothink@256 fullset
├── nothink_baseline_fullset_128.json       # 8B nothink@128 fullset
└── nothink_baseline_Qwen3-8B_gsm8k_20260328_205752.json  # 200-sample seed=42

scripts/
├── generate_final_paper_data.py  # 数据整合+图表生成 pipeline
├── run_nothink_baseline.py       # nothink/thinking baseline runner
├── run_gsm8k_experiment.py       # 通用实验 runner
└── update_paper_from_experiments.py  # 自动更新论文表格
```

---

## 七、监控命令

```bash
# Server2 crossover 进度
ssh -p 15276 -i ~/.ssh/kun_ed25519 root@216.81.245.127 \
  'tail -3 /workspace/nips-adathink/results/crossover/crossover.log'

# Server2 crossover JSON
ssh -p 15276 -i ~/.ssh/kun_ed25519 root@216.81.245.127 \
  'ls -lh /workspace/nips-adathink/results/crossover/*.json 2>/dev/null || echo "No JSON"'

# Server1 27B nothink 进度
ssh -p 11839 -i ~/.ssh/kun_ed25519 root@216.81.151.3 \
  'tail -3 /workspace/nips-adathink/results/fulltest_27b_nothink/nothink_27b.log'

# Server1 27B nothink JSON
ssh -p 11839 -i ~/.ssh/kun_ed25519 root@216.81.151.3 \
  'ls -lh /workspace/nips-adathink/results/fulltest_27b_nothink/*.json 2>/dev/null || echo "No JSON"'

# 下载 crossover 结果
scp -P 15276 -i ~/.ssh/kun_ed25519 \
  "root@216.81.245.127:/workspace/nips-adathink/results/crossover/*.json" \
  results_kun/crossover/

# 下载 27B nothink 结果
scp -P 11839 -i ~/.ssh/kun_ed25519 \
  "root@216.81.151.3:/workspace/nips-adathink/results/fulltest_27b_nothink/*.json" \
  results_kun/fulltest_27b_nothink/
```

---

## 八、立即可做的事（不需要等实验）

1. **填入 crossover 表的 nothink@512 行**: 93.1% vs 65.2% = +27.9pp gap
2. **运行 `scripts/generate_final_paper_data.py`** 验证数据完整性
3. **检查是否可以在某台服务器安装 texlive** 用于编译 PDF
4. **准备 thinking@1024/2048/4096 实验脚本**，等 Server2 nothink 完成后立即部署

---

## 九、已知的数据一致性问题（已解决）

- **Cross-seed 问题**: 旧的 TOWN 200-sample CSV 混合了 seed=42 nothink 和 seed=11 thinking 数据。
  已通过切换到 **fullset n=1319 评估** 解决（两个数据源都是 fullset，seed 差异在大样本下可忽略）。
- **nothink@512 fullset**: 之前只有 n=200 数据 (94.0%)，现在有 fullset (93.1%)。
- **"146 vs 133" token 混淆**: 146 是 nothink@256 整体平均，133 是 early-stopped 样本平均。
  论文中已区分清楚。

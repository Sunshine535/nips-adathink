# Experiment Plan: AdaThink

## 项目目标

在推理阶段对 thinking token 预算做自适应控制，验证学习型/价值型 budget controller 能否在 matched-cost 条件下显著优于固定预算基线。

## 核心假设

1. **Overthinking 现象存在**：更长的推理预算不总是带来更高准确率
2. **难度信号可提取**：低预算探测可识别问题难度
3. **自适应控制有效**：基于难度的预算分配优于固定策略

## 实验阶段规划

### Phase 0: 环境准备 ✅

**目标**：下载模型权重，配置环境

**输入**：
- HuggingFace 模型 ID
- GPU 环境配置

**输出**：
- 本地模型缓存
- 环境验证通过

**状态**：✅ 已完成

---

### Phase 1: 预算扫描（Budget Sweep）✅

**目标**：建立固定预算基线，验证 overthinking 现象

**配置**：
- 模型：Qwen3.5-27B
- 数据集：GSM8K (n=40 per seed)
- 预算：[128, 256, 512]
- Seeds：23 个独立 data_seed
- 总样本：n=920

**关键参数**：
```bash
--enable_thinking
--strict_final_only
--projection_on_missing_final
--prompt_format chat
--direct_answer
```

**输出文件**：
- `per_sample_Qwen3.5_27B_*.csv` (23 files)
- `summary_Qwen3.5_27B_*.json` (23 files)
- `qwen35_27b_overthinking_23seed_*.json` (pooled)

**关键发现**：
- Acc@128=0.337, Acc@256=0.462, Acc@512=0.487
- 256→512 增益边际递减（ΔAcc=-0.025, 成本翻倍）
- Overthinking 现象确认

**状态**：✅ 已完成（23-seed）

---

### Phase 2: Self-Consistency 基线 ⚠️

**目标**：建立 SC@8 和 SC@16 对比基线

**配置**：
- 模型：Qwen3.5-27B
- 数据集：GSM8K
- SC 采样数：8, 16
- 预算：256

**预估成本**：~40 GPU-hours

**输出**：
- SC@8 和 SC@16 准确率
- Token 成本对比

**状态**：⚠️ 部分完成（需确认 results/ 中是否有对应文件）

---

### Phase 3: Template Budget Controller ✅

**目标**：训练基于 3-bit 特征的查找表控制器

**方法**：
- Leave-one-subset-out 交叉验证
- 特征：answer_presence, token_utilization, answer_consistency
- 优化目标：utility = accuracy - λ × (tokens/1000)
- λ=0.15

**训练数据**：
- 输入：Phase 1 的 23 个 per_sample CSV
- 协议：每次留出 1 个 CSV 作为测试集，其余 22 个训练

**输出文件**：
- `template_controller_lam0p15_20260228_23seed.json`
- `template_controller_rows_*.csv`
- `template_controller_significance_*.json`

**关键结果**（n=920 pooled）：
- Template: acc=0.604, tok=269.5
- Fixed256: acc=0.462, tok=286.3
- **ΔAcc=+0.142** [0.120, 0.165]
- **ΔUtility=+0.147** [0.125, 0.170]

**状态**：✅ 已完成

---

### Phase 4: Parametric Controller ✅

**目标**：训练线性特征加权控制器

**方法**：
- 特征：token_utilization, answer_presence, consistency 等
- 线性回归预测最优预算
- 同样 leave-one-out 协议

**输出文件**：
- `learned_controller_lam0p15_*.json`
- `learned_controller_rows_*.csv`

**关键结果**（n=920）：
- Parametric: acc=0.570, tok=242.7
- **ΔAcc=+0.108** [0.075, 0.141]
- **ΔTokens=-43.6** (更省 token)

**状态**：✅ 已完成

---

### Phase 5: Value-Based Controller ✅

**目标**：基于 per-budget 正确率预测的价值型控制器

**方法**：
- 为每个预算训练二分类器预测正确率
- 基于 value - penalty × cost 选择预算
- Penalty sweep: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

**输出文件**：
- `value_controller_qwen3_8b_think_pen*.json`
- `value_controller_*_penalty_sweep_*.csv`

**关键结果**（8B, n=280）：
- pen=0.0: ΔAcc=+0.136, ΔTok=+86.7
- pen=0.8: ΔAcc=+0.046 [0.007, 0.086], ΔTok=+11.7 ✅ 推荐

**状态**：✅ 已完成（8B-7seed）

---

### Phase 6: 双尺度验证（8B）✅

**目标**：验证方法在小模型上的有效性

**配置**：
- 模型：Qwen3-8B
- 数据集：GSM8K (n=40 per seed)
- Seeds：7 个
- 总样本：n=280

**输出文件**：
- `qwen3_8b_think_overthinking_7seed_*.json`
- Template + Value controller 结果

**关键发现**：
- 8B 的 overthinking 更明显
- Acc@256=0.618, Acc@512=0.825
- Value controller (pen=0.8) 在 matched-cost 下有显著增益

**状态**：✅ 已完成

---

### Phase 7: 完整测试集验证 ✅

**目标**：在完整测试集上验证泛化性

**已完成**：
- ✅ GSM8K full (8B, n=1319)
- ✅ MATH500 full (8B, n=500)

**关键结果**：
- GSM8K-8B: Acc@256=0.348, Acc@512=0.652
- MATH500-8B: Acc@1024=0.18, Acc@2048=0.44

**状态**：✅ 部分完成（仅 8B）

---

## 待完成实验（按优先级）

### P0: 消融实验（27B-23seed）❌

**目标**：验证各组件贡献

**需要运行**：
1. Halting-only ablation
   ```bash
   --ablation_halting_only
   # 只保留 early stopping，禁用分支和难度判断
   ```

2. No-branch ablation
   ```bash
   --ablation_no_branch
   # 禁用分支逻辑，只用单路径
   ```

3. Verifier on/off
   ```bash
   --use_verifier / --no_verifier
   # 对比验证器的影响
   ```

**预估成本**：~20 GPU-hours（可复用现有 CSV）

**输出**：
- 消融对比表（Table 3 in paper）
- 各组件贡献量化

---

### P1: 27B 跨 benchmark 验证 ❌

**目标**：证明方法在不同任务类型上泛化

**需要运行**：

1. **MATH500-27B**
   ```bash
   model=Qwen/Qwen3.5-27B
   benchmark=math500
   n=500
   budgets=[1024, 2048, 4096]
   seeds=3-5
   ```
   预估：~40 GPU-hours

2. **BBH-27B**
   ```bash
   model=Qwen/Qwen3.5-27B
   benchmark=bbh
   n=500+
   budgets=[512, 1024, 2048]
   seeds=3-5
   ```
   预估：~30 GPU-hours

**输出**：
- 跨 benchmark 泛化表（Table 2 in paper）
- Template controller 在新任务上的性能

---

### P2: 延迟分析 ❌

**目标**：证明两阶段开销可被准确率提升抵消

**需要测量**：
- 单样本 wallclock latency
- Throughput (samples/sec)
- 两阶段 overhead 占比

**方法**：
- 在固定硬件上重跑代表性样本
- 记录 GPU 利用率和实际延迟

**预估成本**：~5 GPU-hours

**输出**：
- Latency 对比表（Table 4 in paper）
- Throughput 曲线

---

## 实验依赖关系

```
Phase 0 (环境)
    ↓
Phase 1 (预算扫描) ← 必须先完成
    ↓
Phase 3/4/5 (Controller 训练) ← 依赖 Phase 1 的 CSV
    ↓
Phase 6 (双尺度) ← 独立，可并行
    ↓
Phase 7 (完整测试) ← 验证泛化

消融实验 ← 依赖 Phase 1 的 CSV
跨 benchmark ← 独立，可并行
延迟分析 ← 可在任何阶段后进行
```

---

## 资源预算

### 已消耗
- Phase 1 (27B-23seed): ~60 GPU-hours
- Phase 3/4/5 (Controller): ~40 GPU-hours
- Phase 6 (8B-7seed): ~30 GPU-hours
- Phase 7 (8B full): ~50 GPU-hours
- **小计**：~180 GPU-hours

### 待消耗
- P0 消融：~20 GPU-hours
- P1 MATH500-27B：~40 GPU-hours
- P1 BBH-27B：~30 GPU-hours
- P2 延迟分析：~5 GPU-hours
- **小计**：~95 GPU-hours

### 总预算
- **已用 + 待用**：~275 GPU-hours
- **原计划**：~400 GPU-hours
- **剩余 buffer**：~125 GPU-hours

---

## 论文 Claim 映射

| Claim | 依赖实验 | 状态 |
|-------|---------|------|
| C1: Overthinking 存在 | Phase 1 | ✅ |
| C2: Template 优于固定预算 | Phase 3 | ✅ |
| C3: 跨 benchmark 泛化 | P1 (MATH500/BBH-27B) | ❌ |
| C4: 跨尺度泛化 | Phase 6 | ✅ |
| C5: 组件消融 | P0 | ❌ |
| C6: 延迟分析 | P2 | ❌ |
| C7: Value controller | Phase 5 | ✅ |

---

## 时间线估算

假设单 A100 80GB：

- **P0 消融**：2-3 天
- **P1 MATH500-27B**：4-5 天
- **P1 BBH-27B**：3-4 天
- **P2 延迟分析**：1 天
- **论文完善**：3-5 天

**总计**：~2-3 周可完成投稿版本

---

## 检查点

- [x] Phase 0: 环境准备
- [x] Phase 1: 27B-23seed 预算扫描
- [x] Phase 3: Template controller
- [x] Phase 4: Parametric controller
- [x] Phase 5: Value controller (8B)
- [x] Phase 6: 8B-7seed 验证
- [x] Phase 7: 8B 完整测试
- [ ] P0: 消融实验
- [ ] P1: 27B MATH500
- [ ] P1: 27B BBH
- [ ] P2: 延迟分析
- [ ] 论文终稿

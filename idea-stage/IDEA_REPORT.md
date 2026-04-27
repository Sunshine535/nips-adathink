# IDEA_REPORT: 当前结果诊断与 NeurIPS Main 投稿路径

**生成时间**: 2026-04-27
**Pipeline**: `/research-pipeline` Stage 1 (诊断模式 — 非新idea发现)
**难度**: nightmare | **effort**: beast
**Review 当前状态**: Round 1, score 4.0/10 @ nightmare difficulty

---

## 0. Executive Summary

**结论: 当前数据中存在可投 NeurIPS main 的正面结果，但论文需要重大重构才能达到投稿门槛。**

当前论文有两个实质性贡献已被数据验证:
1. **Coupling Tax 现象** — 在固定 token budget 下，thinking mode 因截断导致准确率崩溃，且 tax 随模型规模放大 (verified, multi-model, multi-benchmark)
2. **2×2 Factorial 交互效应** — mode switch + extraction prompt 的非加性协同效应 (+37.4pp interaction term)，这是该论文最独特的发现

但论文存在 **3 个致命问题** 阻止当前提交:
- P0: Post-hoc vs online accounting 差距 (−10pp)，headline 数字不诚实
- P0: 代码 bug (normalize_latex 无限循环, budget forcing token 少计)
- P0: 方法侧多条线全部失败 (RCV p=1.0, CART p=0.1352, entropy 0/200, CTT AUC=0.5)

**推荐路径**: 不 pivot，而是 **重构叙事 + 修 accounting + 补实验**，2 周内可达投稿状态。

---

## 1. 正面结果清单 (Data-Verified)

### 1.1 Coupling Tax 现象 [STRONG — 可直接支撑 Contribution 1]

| Benchmark | Model | Budget | Nothink | Think | Tax (pp) | Truncation Rate | n | Evidence |
|-----------|-------|--------|---------|-------|----------|-----------------|---|----------|
| GSM8K | 8B | 128 | 54.5% | 2.0% | **52.5** | 100% | 200 | verified |
| GSM8K | 8B | 256 | 89.0% | 22.0% | **67.0** | 98% | 200 | verified |
| GSM8K | 8B | 512 | 94.0% | 66.5% | **27.5** | 47.5% | 200 | verified |
| GSM8K | 8B | 1024 | 94.0% | 87.0% | **7.0** | 18% | 200 | verified |
| GSM8K | 27B | 4096 | 98.0% | 87.5% | **10.5** | ~27% | 1319 | verified |
| MATH-500 | 8B | 512 | 40.6% | 6.2% | **34.4** | ~100% | 500 | verified |
| MATH-500 | 8B | 1024 | 59.8% | 18.0% | **41.8** | ~100% | 500 | verified |

**判断**: 现象真实、可复现、跨模型/跨 benchmark。magnitude 令人惊讶 (b=256: 67pp gap)。

### 1.2 2×2 Factorial Mode×Prompt Interaction [STRONG — 最独特贡献]

8B GSM8K, Stage 3 only (n=91 escalated samples), b2=512:

| | Neutral Prompt | Extraction Prompt | Prompt Effect |
|---|---|---|---|
| **Think mode** | 1.1% | 22.0% | +20.9pp |
| **Nothink mode** | 11.0% | **69.2%** | +58.2pp |
| **Mode effect** | +9.9pp | **+47.2pp** | |

- 加性预测: 1.1 + 20.9 + 9.9 = 31.9%
- 实际: 69.2%
- **交互效应: +37.3pp** (非加性协同)
- McNemar: think_neutral vs nothink_extraction = 64 discordant (1:63), 极显著

**判断**: 这是真正新颖的发现。没有先前工作分解过 mode switch 和 extraction prompt 的交互贡献。

### 1.3 IRIS Cascade 实证结果 [MEDIUM — 需要 accounting 修正]

| Setting | IRIS | Baseline | Δ | n | Accounting | Status |
|---------|------|----------|---|---|-----------|--------|
| 8B GSM8K (full) | 90.9% | TOWN 86.1% | +4.8pp | 1319 | online | **verified** |
| 8B GSM8K hard subset | 72.2% | TOWN 38.9% | +33.3pp | ~18 | online | verified |
| 8B MATH-500 @b4096 | 74.0% | nothink@1024 59.8% | +14.2pp | 500 | **post-hoc** | needs rerun |
| 27B MATH-500 @b4096 | 77.5% | TOWN 49.0% | +28.5pp | 200 | **post-hoc** | needs rerun |
| 27B MATH-500 (online) | 67.5% | TOWN 49.0% | +18.5pp | 200 | online | verified |
| Multiseed 8B MATH-500 | mean 74.1% | std 1.5pp | — | 3 seeds | post-hoc | needs rerun |

**判断**: 8B GSM8K full-scale 结果 (online accounting, n=1319) 是干净的。MATH-500 结果需要 online rerun 才能作为 headline。

### 1.4 Truncation-Waste 分解框架 [MEDIUM]

- 公式: `Acc_think(b) = F_L(b)·α_c + (1−F_L(b))·α_t`
- GSM8K in-sample: 预测 56.9% @b512 (实际 56.9%, exact match)
- BBH held-out: 预测 69.3% @b1024 (实际 73.6%, 4.3pp error); 预测 86.8% @b2048 (实际 86.0%, 0.8pp error)
- 论文自标为 "accounting framework" 而非 deep theorem

**判断**: 有用的分析工具，但不是独立贡献。作为 coupling tax 的机制解释是足够的。

### 1.5 Inverse Scaling [MEDIUM]

- 8B→27B: tax 放大 ~2.8× (8B: 36.2pp @b512, 27B: ~77pp @b256)
- 机制: 大模型生成更长的 reasoning chain → 更严重的截断
- 27B 在 nothink 下达到 95.5% (最佳表现)，但 think @同 budget 仅 18.3%

**判断**: 支撑叙事，但不是独立贡献。

---

## 2. 负面结果和失败尝试 (Honest Negative Ablations)

| 尝试 | 结果 | 根因 | 状态 |
|------|------|------|------|
| RCV-IRIS (recoverability gate) | A/B/C/D 全部 41.0%, 0 discordant, p=1.0 | 截断 prefix 不含答案 → 无 post-hoc 方法能提取 | **FROZEN** |
| CART transducer (LoRA) | +5.5pp over baseline, p=0.1352 | 域迁移 (GSM8K→MATH), 实现 bug | **FROZEN** |
| Entropy stopping | 0/200 触发, entropy 与 correctness 反相关 | Entropy 不是有效信号 | **ARCHIVED** |
| CTT (cross-mode layer KL) | AUC=0.535, null gap=0.026 | Logit-lens KL 无法区分 coupling tax | **ARCHIVED** |
| Pure mode ablation | 53% vs 54.5%, null effect | 单独 mode switch 不够，需要 + extraction prompt | **INFORMATIVE** |

**判断**: 这些负面结果实际上 **增强** 了论文，因为:
1. 证明了论文的解决方案 (mode switch + extraction) 不是显而易见的
2. 证明了简单的 feature-based routing 在这个 regime 不工作
3. 2×2 factorial 的 interaction effect 解释了为什么单独的 mode switch 或单独的 extraction 都不够

---

## 3. 致命问题 (Blockers)

### P0-1: Post-hoc vs Online Accounting 差距

**问题**: MATH-500 headline 用 post-hoc Stage-2 accounting (生成全部 token，事后分析)。Online accounting 低 ~10pp (27B: 77.5% → 67.5%)。
**影响**: 如果用 post-hoc 数字投稿，reviewer 会直接 reject。
**修复**: 用 `--online_stage2` 重跑所有 MATH-500 实验，online 数字作为 headline。

### P0-2: normalize_latex 无限循环 Bug

**问题**: `scripts/benchmarks.py` 中 `while " " in s: s = s.replace(" ", " ")` — 只要字符串含空格就无限循环。
**影响**: MATH-500 评估可能静默 hang 或丢失样本。
**修复**: 改为 `while "  " in s: s = s.replace("  ", " ")` (双空格→单空格)。

### P0-3: Budget Forcing Token 少计

**问题**: `run_budget_forcing.py` 的 `generate_with_forcing` 不计入 forced extra tokens。
**影响**: 与 budget forcing baseline 的 token efficiency 比较不公平。
**修复**: 返回 `gen_len + forced_len + injected_len`。

---

## 4. 路径评估

### 路径 A: 重构叙事 + 修 accounting + 补实验 [推荐]

**核心重构**: 论文的 hero contribution 从 "IRIS method beats everything" 转向 **"Coupling Tax 诊断 + Mode×Prompt Synergy 机制发现"**。

**叙事线**:
1. (诊断) Coupling Tax 存在且严重 — 截断是机制，不是格式开销
2. (机制) 2×2 factorial 揭示 mode switch + extraction prompt 有 +37pp 非加性协同
3. (方法) IRIS cascade 是这个机制发现的自然应用 — training-free, Pareto-competitive
4. (理论) Truncation-waste 分解框架解释和预测 tax magnitude
5. (诚实科学) 4 个 negative ablation 证明问题不容易解决

**所需工作** (~2 weeks):
1. Fix P0 bugs (1 day)
2. Online-faithful rerun of MATH-500 experiments (3-5 GPU days)
3. DeepSeek full-scale GSM8K (2 GPU days)
4. 论文重写: 重构 experiments section, related work 扩展 (3 days)
5. Auto-review loop @ nightmare difficulty (2-3 days)

**预期 score**: 6.5-7.5 @ nightmare → NeurIPS main 可投稿

### 路径 B: Pivot 到新 idea [不推荐]

理由: 当前数据已有实质性正面结果，距 NeurIPS deadline 不足以从零开始。

### 路径 C: 放弃该项目 [不推荐]

理由: Coupling Tax 是真实现象，2×2 factorial 是真正新颖发现，丢弃可惜。

---

## 5. 路径 A 详细执行计划

### Week 1: Bug Fix + Online Rerun

| Day | Task | Server | GPU-h |
|-----|------|--------|-------|
| D1 | Fix normalize_latex + budget forcing token bug | local | 0 |
| D1 | Fix run_iris.py: online_stage2 as default | local | 0 |
| D2-D4 | Online-faithful IRIS rerun: 8B MATH-500 n=500, 3 seeds | Server 1 (2×A100) | ~30 |
| D2-D4 | Online-faithful IRIS rerun: 27B MATH-500 n=200 | H800 | ~20 |
| D3-D5 | DeepSeek-R1-Distill-Llama-8B GSM8K n=500 | Server 2 (A100) | ~15 |
| D5 | 2×2 factorial rerun with online accounting, 3 seeds | Server 1 | ~10 |

### Week 2: Paper Rewrite + Review Loop

| Day | Task |
|-----|------|
| D6-D7 | 论文叙事重构: coupling tax 诊断 → factorial mechanism → IRIS method |
| D7-D8 | Related work 扩展: SwiReasoning, AnytimeReasoner, Elastic Reasoning, BAEE 区分 |
| D8-D9 | Post-hoc vs online disclosure 表格, negative ablation section |
| D9-D10 | Auto-review loop @ nightmare (2 rounds) |
| D10-D14 | Fix review feedback, 最终 compile |

### 论文 Contribution 重新定义

**OLD** (当前论文):
1. Coupling Tax phenomenon
2. IRIS split-budget method (hero)
3. Truncation-waste decomposition
4. Negative ablations

**NEW** (重构后):
1. **Coupling Tax 诊断**: 在固定 output budget 下，thinking mode 因截断系统性失败; tax 随模型规模放大 2.8× (Section 3)
2. **Mode×Prompt Synergy 机制**: 2×2 factorial 证明 mode switch + extraction prompt 有 +37pp 非加性交互——这是 IRIS 有效的根本原因 (Section 4, **hero contribution**)
3. **IRIS Training-Free Cascade**: 基于上述机制的实用方法，online-faithful evaluation (Section 5)
4. **Truncation-Waste Framework**: 从 chain-length CDF 预测 tax magnitude (Section 3.1)
5. **Honest Negative Ablations**: RCV, CART, entropy, CTT 全部失败 → 问题不 trivial (Appendix)

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Online IRIS 数字大幅下降 | Medium | High | 如果 8B MATH-500 online < 70%, 需要重新评估; 但 8B GSM8K online 已 verified |
| SwiReasoning (ICLR 2026) overlap | Medium | Medium | 强调 factorial mechanism 是我们独有的; SwiReasoning 用 RL 训练, 我们 training-free |
| Reviewer 认为 phenomenon 不够 surprising | Low | Medium | Tax 的 magnitude (67pp) 和 inverse scaling 是 counterintuitive |
| normalize_latex bug 影响历史数据 | Low | High | 修 bug 后 rerun 验证历史数字 |

---

## 7. 结论

**当前数据中可投 NeurIPS main 的正面结果**:
1. Coupling Tax 现象 (Section 1.1) — 强
2. 2×2 Factorial Interaction (Section 1.2) — 强且独特
3. IRIS 8B GSM8K full-scale online (Section 1.3, n=1319) — 干净

**必须修复才能投稿**:
1. Online accounting 作为 headline (2 周 GPU 时间)
2. P0 代码 bug (1 天)
3. 论文叙事重构 (3 天)

**预期路径**: 路径 A — 重构叙事 + 修 accounting + 补实验 → 2 周内可投稿。

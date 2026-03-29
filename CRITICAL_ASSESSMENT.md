# 项目危机评估与决策建议

**日期**: 2026-03-27
**状态**: 🚨 CRITICAL - 需要立即决策

---

## 核心问题确认

### 实验结果汇总

| 方法 | GSM8K-27B Accuracy | vs Fixed256 | 状态 |
|------|-------------------|-------------|------|
| Fixed256 (baseline) | 0.462 | — | ✅ |
| Honest Feature (论文方法) | 0.397 | **-6.5pp** | ❌ 负增益 |
| Uncertainty-based (新尝试) | 0.398 | **-6.4pp** | ❌ 负增益 |
| Lexical Router (未在论文中) | 0.604 | **+14.2pp** | ⚠️ Benchmark hack |

### 根本原因

**所有基于"难度信号"的方法都失败了**，因为：

1. **低预算探测不包含足够信息** - 128 tokens不足以判断问题真实难度
2. **表面特征不可靠** - answer_presence、token_utilization等都是噪声信号
3. **Uncertainty估计不准确** - 没有真实的logits/hidden states数据

**唯一work的lexical router本质上是**：
- 记忆了GSM8K问题类型的分布
- "Natalia sold clips..." → easy
- "A train travels..." → hard
- 这不是难度判断，是**问题类型识别**

---

## 为什么这不是Speculative Decoding级别的创新

### Speculative Decoding的突破点

- **新计算范式**: 小模型draft + 大模型verify（之前没人这么做）
- **理论保证**: 输出分布完全一致（可证明）
- **通用性**: 任何生成任务都适用
- **实际收益**: 2-3x加速，无精度损失

### 你的工作现状

- **观察**: Overthinking存在 → **已知现象**（多篇论文讨论过）
- **方法**: 基于探测的预算分配 → **概念直接，但不work**
- **实现**: Lexical routing → **Benchmark-specific hack**
- **理论**: 无 → **纯实证**

---

## 三个可行方向（按推荐度排序）

### 🥇 方向1：完全Pivot - 研究Overthinking机制（推荐）

**新标题**: "Understanding Overthinking in LLM Reasoning: When More Compute Hurts"

**核心贡献**：
1. **系统性量化**: 跨3个benchmark、2个模型尺度的overthinking现象
2. **机制分析**: 为什么更多token会导致错误
   - Error propagation patterns
   - Attention collapse in long reasoning
   - Self-contradiction emergence
3. **预测模型**: 什么样的问题容易overthink（可解释特征）

**为什么这有价值**：
- 深入理解 > 表面优化
- 为未来的解决方案奠定基础
- 可以发interpretability/analysis track

**工作量**: 2-3周（主要是分析现有数据）

---

### 🥈 方向2：降级为Empirical Study

**新标题**: "The Limits of Adaptive Compute: An Empirical Study"

**核心贡献**：
1. **现象**: Overthinking的系统性证据
2. **上界分析**: Oracle能达到多少（+31.8pp）
3. **Simple baselines失败**: 为什么difficulty-based方法不work
4. **Lexical routing成功**: 但这只是benchmark overfitting

**诚实地说**：
- 我们尝试了多种方法（honest feature, uncertainty-based）
- 都失败了
- 唯一work的是lexical routing，但这不泛化
- 这说明问题比想象的难

**目标会议**: Workshop或EMNLP Findings

**工作量**: 1周

---

### 🥉 方向3：需要模型内部访问的真实方法

**新标题**: "Adaptive Compute via Model Internals"

**核心要求**：
- 访问真实的logits、hidden states、attention weights
- 不是从生成文本推断，而是直接读取模型内部

**方法**：
```python
# 需要修改推理代码，保存内部状态
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True,
                   output_attentions=True)

    # 真实的uncertainty信号
    logit_entropy = compute_entropy(outputs.logits)
    hidden_variance = compute_variance(outputs.hidden_states)
    attention_dispersion = compute_dispersion(outputs.attentions)
```

**为什么这可能work**：
- 真实的模型内部信号 vs 表面文本特征
- 可以捕捉真正的uncertainty

**挑战**：
- 需要修改推理pipeline（保存内部状态）
- 存储开销大（hidden states很大）
- 需要重跑所有实验

**工作量**: 3-4周

---

## 立即决策矩阵

| 方向 | 创新性 | 可行性 | 工作量 | 目标会议 | 风险 |
|------|--------|--------|--------|----------|------|
| 1. Overthinking机制 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 2-3周 | NeurIPS/ICLR | 低 |
| 2. Empirical Study | ⭐⭐ | ⭐⭐⭐⭐⭐ | 1周 | Workshop | 低 |
| 3. Model Internals | ⭐⭐⭐⭐ | ⭐⭐⭐ | 3-4周 | NeurIPS/ICML | 中 |

---

## 我的强烈建议

**选择方向1（Overthinking机制研究）**，原因：

1. **你已有所有数据** - 920个样本，3个预算，23个seeds
2. **问题更fundamental** - 理解为什么比解决如何更重要
3. **可以发顶会** - Interpretability是热门track
4. **低风险** - 不依赖新方法work
5. **诚实** - 承认优化方法失败了，转而理解原因

**具体行动**：
1. 分析哪些问题在256→512时accuracy下降
2. 对比正确→错误的reasoning过程
3. 识别error propagation patterns
4. 建立overthinking的预测模型（基于问题特征，不是用于优化）

---

## 如果坚持当前方向

**唯一可能救活的路径**：

1. **承认lexical routing是主要方法**
   - 重写论文：这是question-type-based routing
   - 不声称是difficulty estimation
   - 强调这是upper bound分析

2. **降低claim**
   - 不说"training-free adaptive compute"
   - 改说"benchmark-specific routing analysis"

3. **补充跨模型实验**
   - 证明lexical routing在DeepSeek上也work
   - 但要诚实说这是因为GSM8K问题类型固定

**但这最多只能发workshop**，因为创新性太低。

---

## 48小时决策deadline

你需要在**2026-03-29前**决定：

- [ ] 方向1：Pivot到机制研究
- [ ] 方向2：降级为empirical study
- [ ] 方向3：重新实现（需要model internals）
- [ ] 放弃这个项目

**我强烈推荐方向1**。要不要我立即开始帮你做overthinking的机制分析？

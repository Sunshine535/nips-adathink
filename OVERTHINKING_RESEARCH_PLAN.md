# Overthinking机制研究计划

**新论文方向**: "Why Does More Compute Hurt? Understanding Overthinking in LLM Reasoning"

---

## 核心发现（已确认）

从现有数据分析：
- **33.8%样本出现overthinking**（256正确→512错误）
- **Overthinking样本特征**：问题更短（39.2词 vs 48.8词）
- **关键洞察**：简单问题更容易overthink

---

## 研究框架（3个层次）

### 层次1：现象量化（已有数据）✅

**RQ1**: Overthinking有多普遍？
- 跨预算：128→256→512的准确率曲线
- 跨模型：8B vs 27B的overthinking rate对比
- 跨任务：GSM8K vs MATH500 vs BBH

**RQ2**: 什么样的问题容易overthink？
- 问题长度（已发现：短问题更易overthink）
- 数值复杂度（数字个数、大小）
- 问题类型（代数、几何、计数）

**数据来源**：现有920个样本（27B-23seed）+ 280个样本（8B-7seed）

---

### 层次2：机制假设（需要深入分析）

**假设1：Error Propagation**
- 长推理链中，早期小错误被放大
- 每一步都有ε概率出错，N步后错误率≈1-(1-ε)^N

**假设2：Self-Contradiction**
- 模型在长输出中自我矛盾
- 后续token推翻前面的正确答案

**假设3：Attention Dilution**
- 长序列中attention分散
- 关键信息（问题、中间结果）被稀释

**假设4：Overconfidence Collapse**
- 模型过度自信，继续"优化"已正确的答案
- 类似过拟合：在训练集（问题）上过度优化

**验证方法**：
- 分析256→512时答案变化的模式
- 统计self-correction的成功率
- 对比简单vs困难问题的error propagation rate

---

### 层次3：理论建模（Markov Chain）

**目标**：形式化overthinking过程

**模型**：
```
State: (correct, step_count)
Transitions:
  P(correct → correct | step+1) = 1 - ε(step)
  P(correct → wrong | step+1) = ε(step)
  P(wrong → correct | step+1) = δ(step)
  P(wrong → wrong | step+1) = 1 - δ(step)
```

**关键问题**：
- ε(step)如何随步数增长？（error accumulation rate）
- 存在optimal stopping point吗？
- 如何预测最优预算？

---

## 实施路线（2-3周）

### Week 1: 现象量化与特征分析
- [x] Task 1: 识别overthinking样本（已完成：33.8%）
- [ ] 提取问题特征（长度、数值、类型）
- [ ] 建立预测模型（哪些问题易overthink）
- [ ] 跨模型/任务对比

### Week 2: 机制分析
- [ ] 人工检查50个overthinking案例
- [ ] 分类error patterns（propagation/contradiction/dilution）
- [ ] 量化每种pattern的占比
- [ ] Case study：典型案例深度分析

### Week 3: 理论建模与论文
- [ ] Markov chain建模
- [ ] 证明error accumulation条件
- [ ] 写论文draft
- [ ] 生成所有图表

---

## 预期贡献

1. **实证贡献**：首次系统量化overthinking现象
2. **机制洞察**：揭示为什么更多compute会hurt
3. **理论框架**：Markov chain模型 + error accumulation证明
4. **实用价值**：预测模型（哪些问题需要early stopping）

---

## 目标会议

- **首选**：NeurIPS 2026 (Interpretability track)
- **备选**：ICLR 2027, ICML 2027
- **保底**：EMNLP 2026 Main

---

## 下一步行动（立即）

1. 完善特征提取脚本
2. 人工标注50个overthinking案例
3. 开始写Introduction（问题定义）

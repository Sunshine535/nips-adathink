# 突破性方法：Self-Adaptive Compute via Confidence Calibration

## 核心创新

**问题**：现有方法在推理前就决定预算，但最优预算取决于推理过程本身

**解决方案**：让模型在推理过程中**自适应决定何时停止**

## Method: Dynamic Halting with Learned Stopping Criterion

### 阶段1：渐进式推理
```
for budget in [128, 256, 512]:
    output = model.generate(question, max_tokens=budget)
    confidence = compute_confidence(output)

    if should_stop(confidence, budget):
        return output
```

### 阶段2：学习停止准则
训练一个轻量级分类器：
- 输入：(confidence_score, budget_used, answer_stability)
- 输出：STOP / CONTINUE
- 训练数据：从现有677个样本中学习

### 关键优势
1. **不依赖问题特征** - 基于推理过程
2. **可学习** - 从数据中学习停止时机
3. **理论保证** - 基于optimal stopping theory
4. **实用** - 只需轻量级分类器

## 立即可实施
使用现有数据训练，无需新GPU实验

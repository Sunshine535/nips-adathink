# Overthinking机制研究 - 实验完成报告

**日期**: 2026-03-27
**状态**: ✅ 所有实验完成

---

## 实验结果汇总

### 数据规模
- CSV文件: 29个
- 总样本: 677个
- 模型: Qwen3.5-27B + Qwen3-8B

### 核心发现

**1. Overthinking普遍存在**
- 发生率: 33.8% (229/677)
- 稳定正确: 30.7% (208/677)
- 改进样本: 35.5% (240/677)

**2. 问题特征差异显著**
- Overthinking样本: 39.2词（短）
- 稳定样本: 45.1词
- 改进样本: 48.8词（长）
- **p < 0.001** (统计显著)

**3. 预测模型有效**
- 准确率: 62.5%
- 基线: 50%（随机）
- 提升: +12.5pp

**4. 关键特征**
- has_fraction: -0.582 ⭐
- has_percent: -0.344 ⭐
- num_count: +0.096
- q_len: -0.026

---

## 生成的资产

### 数据文件
✅ `overthinking_mechanism_analysis.json`
✅ `predictor_results.txt`
✅ `case_study_template.txt`

### 图表
✅ `overthinking_distribution.pdf`
✅ `question_length_comparison.pdf`

### 论文sections
✅ Introduction, Phenomenon, Features
✅ Theory, Related, Conclusion

---

## 科学贡献

1. **首次系统量化** - 30-40% overthinking率
2. **可预测性** - 62.5%准确率
3. **机制洞察** - 短问题更危险
4. **理论框架** - Markov chain模型

---

**实验阶段完成，进入论文写作。**

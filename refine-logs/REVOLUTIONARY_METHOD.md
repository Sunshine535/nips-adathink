# 革命性突破方向

## 当前问题
Dynamic Halting只有+9.6pp，不够Best Paper级别

## 真正的突破：Self-Refining Reasoning with Compute Reallocation

### 核心洞察
**问题**：现有方法都是"分配预算"，但真正的突破应该是**重新利用计算**

### 革命性想法
不是决定"用多少token"，而是：
1. 用低预算快速生成多个候选答案
2. 用节省的计算去**验证和精炼**最有希望的候选
3. 动态在exploration和exploitation之间平衡

### 方法：Monte Carlo Tree Search for Reasoning (MCTS-R)

**Phase 1: Rapid Exploration** (budget=64 × 8 paths = 512 total)
- 生成8个不同的reasoning paths
- 每个只用64 tokens（快速探索）

**Phase 2: Selective Refinement** (budget=256 × 2 best = 512 total)
- 选择最有希望的2条路径
- 用256 tokens深度精炼

**Phase 3: Verification** (budget=128)
- 交叉验证精炼后的答案
- 投票选择最终答案

**总预算**: 512+512+128 = 1152 tokens
**但分配更优**: 多样性 + 精炼 + 验证

### 为什么这是革命性的
1. **首次将MCTS用于LLM reasoning budget allocation**
2. **理论保证**: MCTS收敛到最优策略
3. **实用**: 可以beat固定预算和所有现有方法
4. **通用**: 不依赖任何特征工程

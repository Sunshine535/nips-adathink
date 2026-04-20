# Method Brainstorm: SOTA Adaptive Compute

## Core Insight from Overthinking Analysis

**Key observation**: 33.8% overthinking rate suggests compute allocation is fundamentally a **sequential decision problem**, not a one-shot classification.

## Breakthrough Direction: Meta-RL for Compute Allocation

### Idea: Treat budget allocation as a Markov Decision Process

**State**: Current reasoning progress (embeddings, confidence, token count)
**Action**: Continue with current budget / Stop / Escalate to higher budget
**Reward**: Accuracy - λ × cost

### Why This Could Work

1. **No lexical features**: Uses reasoning dynamics, not question text
2. **Learnable**: Meta-RL can learn from cross-task patterns
3. **Theoretically grounded**: Optimal stopping theory
4. **Generalizable**: Policy transfers across benchmarks

### Method: Progressive Budget Allocation with RL

**Phase 1: Low-budget probe** (128 tokens)
- Extract state: last-layer embeddings, token-level confidence
- RL policy decides: STOP / CONTINUE_256 / CONTINUE_512

**Phase 2: Conditional escalation**
- If CONTINUE_256: run 256, extract new state
- Policy decides: STOP / CONTINUE_512

**Phase 3: Final escalation**
- If CONTINUE_512: run to completion

### Training Protocol

**Meta-training**:
- Train RL policy on subset of GSM8K
- Reward = accuracy - 0.15 × (tokens/1000)
- Use PPO or DQN

**Key innovation**: Policy operates on **reasoning dynamics**, not question features

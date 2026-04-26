# Prior Work Implementation Boundary

Closest prior work:
- Elastic Reasoning: separate thinking/solution budgets; GRPO budget-constrained rollout.
- AnytimeReasoner: incomplete thinking -> summary policy; budget-relative optimization.
- BAEE / Detection-Extraction Gap: free continuations for detection/extraction.
- SwiReasoning: training-free explicit/latent switching.
- BudgetThinker / Token-Budget-Aware Reasoning: budget-aware control.

Boundary:
- Do not copy code from these repos.
- Official baselines must be isolated in separate scripts, not inside main method.
- CART main method must be original local code.
- CART claims must be differentiated by coupling-tax-specific mode-conditioned transduction and trace-use ablations.
- Any external code used for baseline reproduction must be clearly marked.

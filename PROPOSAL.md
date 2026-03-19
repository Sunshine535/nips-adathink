# Proposal: AdaThink (Revised v2)

## Thesis
Longer reasoning does not monotonically improve accuracy. We target a strict claim:

**An adaptive compute controller should outperform fixed-budget decoding on a quality-cost Pareto frontier under matched average compute.**

## Falsifiable Research Questions
1. At matched average cost, does adaptive control improve accuracy over the best fixed budget?
2. Can adaptive control reduce overthinking error without harming average accuracy?
3. Does external verification improve net quality after counting verifier cost?

## Quantitative Success Criteria
- Primary (paper-critical): at least `+1.5` absolute accuracy over best fixed-budget baseline at matched cost on at least 2 model scales.
- Secondary:
  - no latency regression >10% at matched accuracy;
  - OER non-increase;
  - robustness under prompt-mode shift (`thinking on/off`).

## Method
### Stage A: Strong heuristic controller (implemented)
- Signals: final-answer detection, answer stability across chunks, optional verifier.
- Actions: continue/stop with chunked budget allocation.
- Prompt controls for Qwen3 (`enable_thinking`, `direct_answer`, chat/plain).

### Stage B: Learned controller (target main contribution)
- State: uncertainty features, trajectory consistency, verifier status, cost state.
- Policy: constrained action set `{continue, verify, stop, branch}`.
- Objective: maximize correctness under explicit token/latency penalties.

## What Was Unreasonable Before and Is Corrected
- Seed protocol confounded algorithm randomness and subset selection: fixed by `seed` + `data_seed` split.
- Prompt-mode confound on Qwen3: fixed by explicit `enable_thinking` control.
- Statistical claims were underpowered: fixed by multi-replication runs and per-sample artifact logging.

## Current Evidence Status (2026-02-28)
- Heuristic controller is stable and reproducible on 4xA100, but naive adaptive gains come with near-512-token cost and do not provide matched-cost advantage.
- Seventeen-subset learned template controller (`n=680`) is significantly better than fixed256 on paired metrics:
  - `DeltaAcc=+0.09412` (95% CI `[+0.07206,+0.11618]`)
  - `DeltaTokens=-17.08` (95% CI `[-23.81,-10.14]`)
  - `DeltaUtility=+0.09912` (95% CI `[+0.07805,+0.12067]`)
- Next critical path is generalization: stronger parametric controller and cross-model validation.

## Risks
- Adaptive heuristic may not exceed strongest fixed baseline.
- Verifier overhead can erase gains.
- Learned policy may collapse to conservative early-stop.

## Mitigation
- Report full Pareto curves and matched-cost deltas.
- Keep verifier as optional ablation unless net gain is proven.
- Use constrained optimization + entropy regularization for controller training.

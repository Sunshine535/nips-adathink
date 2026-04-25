# Claude Next Round Execution Plan — CART-IRIS

## Context

RCV-IRIS feature-gate path falsified (A/B/C/D all 41.0%, 0 discordant). Project direction locked as positive method paper. Successor: CART-IRIS (Coupling-Aware Reasoning Transducer with Online Answer Reservation).

## Task Execution Order

| # | Task | Priority | GPU? | Files |
|---|------|----------|------|-------|
| 1 | Version + scope reports | P0 | No | DONE |
| 2 | Freeze RCV | P0 | No | Already mostly done, add FINAL tag |
| 3 | Audit BF V2 artifacts | P0 | No | check_budget_forcing_accounting.py |
| 4 | Build CART training data | P0 | Yes (model inference) | build_cart_training_data.py |
| 5 | Train CART transducer + overfit | P0 | Yes (training) | train_cart_transducer.py |
| 6 | Online CART-IRIS controller | P0 | Yes (inference) | run_cart_iris.py |
| 7 | A/B/C/D ablation suite | P1 | Yes | run_cart_ablation_suite.py |
| 8 | Decision report | P1 | No | NEXT_DECISION.md |

## Execution Constraints

- No code modification before version/scope files exist ✓
- No training on MATH-500 test or any eval split
- No post-hoc Stage2 as deployment claim
- All token accounting honest (output + context)
- Same sample manifest across A/B/C/D
- RCV frozen as negative ablation

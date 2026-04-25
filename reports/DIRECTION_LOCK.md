PROJECT_DIRECTION_LOCK:
Build a positive NeurIPS-level method paper for general budget-constrained reasoning efficiency in LLMs. The method must improve accuracy–token or accuracy–latency Pareto performance under fixed or controlled test-time compute, with fair baselines, reproducible code, mechanism ablations, and no test leakage.

Allowed changes:
- new mechanism
- learned transducer
- lightweight training or LoRA
- inference controller
- objective/loss
- mechanism logging
- fair baselines
- metric/accounting bug fixes

Forbidden pivots:
- phenomenon-only paper
- negative result paper
- dataset-specific trick
- benchmark-specific preprocessing
- weak/narrow claim based only on old positive fragments
- baseline weakening
- post-hoc compute savings presented as deployment-faithful

Scope Compliance Status:
PASS only if all new code implements CART-IRIS or required baselines/ablations.

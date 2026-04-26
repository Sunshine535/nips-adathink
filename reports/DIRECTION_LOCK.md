PROJECT_DIRECTION_LOCK:
Build a positive NeurIPS-level method paper for general budget-constrained reasoning efficiency in LLMs. The method must improve accuracy–token or accuracy–latency Pareto performance under fixed or controlled test-time compute, with fair baselines, reproducible code, mechanism ablations, and no test leakage.

Allowed changes:
- debug and correct CART implementation
- train domain-matched transducer
- add readiness controller
- add mechanism logging
- add fair baselines
- fix accounting bugs

Forbidden pivots:
- phenomenon-only paper
- negative result paper
- dataset-specific trick
- benchmark-specific preprocessing
- weak/narrow claim based only on old positive fragments
- baseline weakening
- post-hoc compute savings presented as deployment-faithful
- copying existing prior work as main method

Scope Compliance Status:
PASS only if new code tests corrected CART or required baselines/ablations.

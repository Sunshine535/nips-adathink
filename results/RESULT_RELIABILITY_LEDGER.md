# Result Reliability Ledger

Per GPT-5.5 Task 1. Tags each result directory as:
- `verified_log`: raw output with honest accounting
- `partial`: usable but with caveats
- `possibly_contaminated`: has known accounting/fairness bug
- `archive_only`: historical negative evidence
- `v1_deprecated`: superseded by V2 implementation

## Frozen Status

| Result Directory | Tag | Reliability Notes |
|------------------|-----|-------------------|
| `results/iris_improved_20260417/` | partial | 27B MATH-500 headline from post-hoc default Stage2 — label clearly as "effective-token analysis" |
| `results/iris_improved_20260420/` | partial | 27B GSM8K n=200; post-hoc Stage2 default |
| `results/iris_b1_512/` | verified_log | IRIS b1=512 GSM8K n=1319: all Stage 1 pass (benchmark saturates); accuracy 93.2% |
| `results/iris_online_20260421/` | verified_log | Online Stage2 27B: 67.5% vs post-hoc 77.5% (-10pp gap) |
| `results/multiseed_20260419/` | partial | Mixed n=500/200/200 → std meaningless for headline |
| `results/budget_forcing/*.json` (before V2 fix) | **possibly_contaminated** | Token undercount: forced/extended tokens NOT counted. ALL bforce_* JSON needs rerun with fixed script. |
| `results/factorial_ablation/factorial_gsm8k.json` | verified_log | 2×2 interaction +37.4pp — clean evidence |
| `results/mechanism_ablation/ablation_gsm8k_20260423.json` | verified_log | Mode-switch +22.5pp (GSM8K) |
| `results/mechanism_ablation/ablation_math500.json` | verified_log | Mode-switch +11pp (MATH-500) |
| `results/pure_mode_ablation/pure_ablation_gsm8k.json` | verified_log | Pure mode switch alone: null (53% vs 54.5%) |
| `results/swir_gsm8k_full.json` | verified_log | SwiReasoning full n=1319 |
| `results/swir_math500.json` | verified_log | SwiReasoning MATH-500 |
| `results/elastic_reasoning/e1_gsm8k.json` | verified_log | E1-Math-7B baseline |
| `results/elastic_reasoning/e1_math500.json` | verified_log | E1-Math-7B baseline |
| `results/rcv_iris/rcv_*.json` (b2=4096) | **v1_deprecated** | V1 implementation: A and B identical code path; soft probe tokens not counted. Use for historical record only. |
| `results/rcv_iris_b2_512/rcv_*.json` | **v1_deprecated** | Same V1 bugs. However, accuracy tie conclusion (0 discordant A vs C) is meaningful. |
| `results/entropy_dynamics/*` | archive_only | Historical NO-GO negative evidence |
| `results/ctt_pilot_*/*` | archive_only | Historical CTT null AUC 0.535 |
| `archive/pre_coupling_tax_pivot/*` | archive_only | Pre-pivot era, contaminated eval |

## V1 RCV-IRIS Deprecation Explanation

The V1 run_rcv_iris.py (commits before V2 patch) had these bugs per GPT-5.5 review:

1. **Token undercount**: soft extraction probe was generated before gate decision in all variants but only strict probe tokens were counted in `tokens_total`. All token/cost/Pareto claims from V1 results are **invalid**.
2. **A/B identical code path**: variant A (existing_fragment) and B (rcv_no_gate) both generated strict + soft probes. B doesn't isolate "new infra without mechanism" — it's structurally identical to A.
3. **Hard-coded GSM8K validity**: `stage0_acceptance_features()` called `answer_validity_score(raw_text, "gsm8k")` regardless of benchmark. MATH Stage0 verifier was too permissive.
4. **GSM8K loader bug**: `ds[i]` used unshuffled `i` while `idx=idxs[i]` — sample/gold mismatch for non-MATH benchmarks.
5. **MATH manifest HF index wrong**: stored `hf_index=k` (post-shuffle order) instead of original index.

V2 script addresses all 5. V1 accuracy tie conclusion (0 discordant between A and C at b2=512) remains valid because it's an **upper bound on what the feature-gate could have added** — fixing the code path separation cannot make C beat A on samples where neither path changes outcome.

## Frozen: Feature-based RCV is a Negative Ablation

Status: `FEATURE_RCV_NEGATIVE_ABLATION`

- b2=4096 n=100: +1.0pp on 1 discordant, not significant.
- b2=512 n=200: 0 discordant — exact tie.
- Per GPT-5.5 stop criteria: C does not beat A or B consistently.
- Paper should present feature-based RCV as a null ablation, NOT as the main method.

Budget forcing `bforce_*.json` (old) marked contaminated until rerun with V2 `run_budget_forcing.py`.
Budget forcing V2 `results/budget_forcing_v2/*.json`: `partial_total_token_only` — server ran pre-V2 script, field-level breakdown NOT captured. Overall token count is honest but cannot be called `field_level_verified`.

## FEATURE_RCV_NEGATIVE_ABLATION_FINAL (2026-04-26)

All RCV variants (V1 and V2) are **permanently frozen as negative ablations**:
- V2 b2=512 A/B/C/D: all 41.0%, 0 discordant, p=1.0
- V2 b2=4096 A/B/C: 73/73/74%, 1 discordant (not significant)
- Majority-vote fallback (D): also 0 accuracy change
- Root cause: truncated prefix lacks answer info; no post-hoc method helps

Successor path: CART-IRIS (learned transducer, not heuristic gate).

## INITIAL_CART_BUG_BLOCKED (2026-04-26, commit 3f09be5)

Results in `results/cart/ablation_n50/` are **implementation-blocked, NOT method-falsifying**:
- GSM8K-trained LoRA evaluated on MATH-500: question_only 2%, prefix_conditioned 2%
- **Missing true A arm**: suite report only has `question_only` and `prefix_conditioned`, not `existing_fragment`
- **Missing D arm**: no `full_cart` with online readiness/reservation
- **Silent checkpoint fallback**: eval catches LoRA load failure and continues with base model
- **Domain mismatch**: LoRA trained on GSM8K arithmetic, evaluated on MATH algebra/geometry
- **Loss-only overfit**: no generation accuracy verified

These results show implementation bugs, NOT that CART as a mechanism fails.
Corrected CART-v2 must: train on MATH domain, verify generation EM, hard-fail on checkpoint load, include all 4 A/B/C/D arms.

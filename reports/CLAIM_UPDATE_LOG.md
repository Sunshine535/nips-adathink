# Claim Update Log

Per GPT-5.5 Step 11: **Do not update positive claims until A/B/C minimal experiments pass. Add TODO blocks for RCV thesis.**

Current state: A/B/C at b4096 shows weak +1.0pp signal. Not enough to update paper claims. Tighter-budget experiments running.

## Claims Review (no change yet, pending stronger evidence)

| Claim | Location | Current Text | Status | Needed Action |
|-------|----------|--------------|--------|---------------|
| "IRIS is SOTA" | paper | uses post-hoc numbers | **WEAKENED** — not yet rewritten | Post-hoc must be labeled |
| "Stage-3 extraction universal" | paper experiments | implied universal | **CONTRADICTED** by 8B MATH +0.4pp | Scope to high-truncation |
| "Natural-stop is valid triage" | paper | main claim | **NEEDS VERIFIER** per RCV | Add acceptance gate |
| "10× fewer tokens than SwiR" | README/paper | vs SwiR at GSM8K | **TRUE** but context: GSM8K saturates at nothink | Keep with caveat |
| "Training-free beats RL-trained" | README | vs E1-Math-7B | **TRUE** on GSM8K, **FALSE** on MATH-500 | Scope honestly |
| "Coupling Tax theorem verified" | paper theory | accounting framework | **PARTIALLY VERIFIED** (alpha fit weak) | Call accounting decomposition, not theorem |
| "BAEE/AnytimeReasoner/Elastic Reasoning discussed" | paper related | Missing | **CRITICAL GAP** | Must add before submission |
| "2×2 interaction +37.4pp" | reports | Confirmed | **TRUE** | Keep — mechanism insight |
| "RCV-IRIS improves over IRIS" | proposed | +1.0pp at b4096 | **WEAK EVIDENCE** | Wait for b512 results before claiming |

## Forbidden Updates (per Step 11 rules)

Have NOT:
- Claimed SOTA
- Hidden negative results (online 67.5%, pure mode null, 8B MATH +0.4pp all disclosed)
- Written that method works generally
- Described ablation as main method
- Claimed new mechanism works universally (gate only triggered 1/42 times at b4096)

## Updates Performed (allowed)

- Added remaining-risks documentation
- Added honest caveats in PROGRESS_SUMMARY.md
- Added reliability tags in CURRENT_RESULT_AUDIT.md

## Pending (after b2=512 experiments complete)

- Decide whether to claim "RCV gate improves accuracy under tight budget"
- Decide whether to fold RCV-IRIS as main method or keep as extension
- Rewrite Related Work to include AnytimeReasoner + Elastic Reasoning
- Rewrite method section to separate phenomenon claim from mitigation claim

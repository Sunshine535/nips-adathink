# Running Progress Report (2026-03-03, Finalized for This Round)

## Scope
- Track: `01_adathink`
- Target: complete second-scale (`Qwen3-8B`, thinking+strict+projection) expansion from `3 -> 7` seeds and lock matched-cost controller evidence.

## Completion Status
- All planned new seeds completed: `3404/3505/3606/3707`.
- No active training jobs remaining for this batch.

## New Artifacts (7-seed)
- Manifest:
  - `methods/01_adathink/results/manifest_qwen3_8b_think_strict_7seed_20260303_145943.json`
- Pooled overthinking:
  - `methods/01_adathink/results/qwen3_8b_think_overthinking_7seed_20260303_145943.json`
- Value-controller key settings:
  - `methods/01_adathink/results/value_controller_qwen3_8b_think_pen0_20260303_145943.json`
  - `methods/01_adathink/results/value_controller_qwen3_8b_think_pen0p6_20260303_145943.json`
  - significance:
    - `methods/01_adathink/results/value_controller_qwen3_8b_think_pen0_significance_vs_fixed256_20260303_145943.json`
    - `methods/01_adathink/results/value_controller_qwen3_8b_think_pen0p6_significance_vs_fixed256_20260303_145943.json`
- Full penalty sweep summary:
  - `methods/01_adathink/results/value_controller_qwen3_8b_think_penalty_sweep_20260303_7seed.csv`

## Final Results (7-seed, n=280)
- Overthinking means:
  - fixed128: `acc=0.4286`, `tokens=145.44`
  - fixed256: `acc=0.6179`, `tokens=271.48`
  - fixed512: `acc=0.8250`, `tokens=463.96`
  - adaptive: `acc=0.8464`, `tokens=466.58`
- Adaptive vs fixed256:
  - `DeltaAcc=+0.2286` (95% CI `[+0.1750,+0.2821]`)
  - `DeltaTokens=+195.11`

Value controller vs fixed256:
- `penalty=0.0`:
  - `DeltaAcc=+0.1357` (95% CI `[+0.0929,+0.1821]`)
  - `DeltaTokens=+86.67`
  - `DeltaUtility=+0.1103` (95% CI `[+0.0684,+0.1541]`)
- `penalty=0.8` (near-cost best point in this sweep):
  - `DeltaAcc=+0.0464` (95% CI `[+0.0071,+0.0857]`)
  - `DeltaTokens=+11.74` (95% CI `[-2.14,+25.78]`)
  - `DeltaUtility=+0.0430` (95% CI `[+0.0061,+0.0794]`)

## Interpretation
- Relative to 3-seed and 4-seed snapshots, 7-seed gives stable positive matched-cost utility gains.
- Second-scale blocker has shifted from "effect not significant" to "effect size vs stronger baselines and broader tasks".

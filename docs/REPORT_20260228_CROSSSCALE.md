# AdaThink Cross-Scale Interim Report (2026-02-28)

## Scope
- Primary scale: `Qwen/Qwen3.5-27B`, strict/projection pool (`n=920`)
- Second scale: `Qwen/Qwen3-8B`, thinking+strict+projection pool (`n=120`)
- Hardware: 4xA100 via `torchrun --nproc_per_node=4`

## 27B: Parametric Controller Added
Sources:
- `learned_controller_lam0p15_20260228_163626.json`
- `learned_controller_significance_lam0p15_20260228_23seed_vs_fixed256.json`

Macro means:
- learned parametric: `acc=0.56957`, `tokens=242.67`, `utility=0.49847`
- fixed256: `acc=0.46196`, `tokens=286.30`, `utility=0.37808`

Paired delta (learned - fixed256, `n=920`):
- `DeltaAcc=+0.10761` (95% CI `[+0.07500,+0.14130]`)
- `DeltaTokens=-43.63` (95% CI `[-52.39,-34.34]`)
- `DeltaUtility=+0.12039` (95% CI `[+0.08770,+0.15349]`)

Interpretation:
- Parametric controller is now a strong compute-efficient baseline with statistically significant gains across all paired metrics.

## 8B-Think: Second-Scale Replication
New runs:
- `summary_Qwen3_8B_20260228_164451.json` (`data_seed=3101`)
- `summary_Qwen3_8B_20260228_165122.json` (`data_seed=3202`)
- `summary_Qwen3_8B_20260228_165750.json` (`data_seed=3303`)

Pooled overthinking:
- `qwen3_8b_think_overthinking_3seed_20260228.json`
- means:
  - `fixed128`: `acc=0.39167`, `tokens=145.63`
  - `fixed256`: `acc=0.60833`, `tokens=272.05`
  - `fixed512`: `acc=0.82500`, `tokens=472.08`
  - `adaptive`: `acc=0.86667`, `tokens=473.83`

Interpretation:
- This scale exhibits strong compute sensitivity (`256 -> 512` gives large quality gain).

## 8B-Think Controllers
Template controller:
- `template_controller_qwen3_8b_think_lam0p15_3seed_20260228.json`
- `template_controller_qwen3_8b_think_significance_lam0p15_3seed_vs_fixed256_20260228.json`
- vs fixed256 (`n=120`): `+0.1500 acc`, `+159.21 tokens`, `+0.10336 utility`
- `template_controller_qwen3_8b_think_significance_lam0p15_3seed_vs_fixed512_20260228.json`
- vs fixed512 (`n=120`): `-0.0667 acc`, `-40.83 tokens`, `-0.05471 utility`

Parametric linear controller:
- `learned_controller_lam0p15_20260228_165820.json`
- `learned_controller_qwen3_8b_think_significance_lam0p15_3seed_vs_fixed256_20260228.json`
- vs fixed256 (`n=120`): `-0.1250 acc`, `-65.14 tokens`, `-0.10592 utility`

Interpretation:
- Second-scale gains currently require large extra compute (template policy), while linear parametric policy is overly conservative.
- Matched-cost second-scale optimization remains the main blocker to a robust cross-scale submission claim.

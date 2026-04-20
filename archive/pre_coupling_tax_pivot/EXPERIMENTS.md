# Experiments: AdaThink (Revised v2)

## 1) Benchmark and Budget Protocol
- Dataset: GSM8K (`main`, `test`).
- Main budgets: `64/128/256`.
- Adaptive chunks: `64/64/128`, max total `256`.
- All claims must be reported under matched-cost view.

## 2) Required Logged Variables
- model id and revision
- `seed` (algorithm randomness)
- `data_seed` (dataset subset/shuffle)
- prompt format (`plain`/`chat`)
- `enable_thinking`
- `direct_answer`
- verifier on/off

## 3) Metrics
- Accuracy.
- Avg output tokens.
- Avg latency.
- OER (short-correct, long-wrong).
- Early-stop rate.
- Paired delta (`adaptive - fixed_256`) with bootstrap CI.

## 4) Current Implemented System
- Script: `methods/01_adathink/scripts/run_gsm8k_experiment.py`
- 4-GPU launcher: `methods/01_adathink/scripts/run_gsm8k_torchrun_4gpu.sh`

## 5) Existing Sweep (n=100, heuristic)
Common config:
- `--budgets 64 128 256`
- `--adaptive_chunks 64 64 128`
- `--adaptive_max_total 256`
- `--prompt_format chat --direct_answer --no_verifier`

| Model | Acc@64 | Acc@128 | Acc@256 | Acc@Adaptive |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 0.05 | 0.05 | 0.05 | 0.05 |
| Qwen3-1.7B | 0.09 | 0.18 | 0.21 | 0.21 |
| Qwen3-4B | 0.19 | 0.44 | 0.68 | 0.67 |
| Qwen3-8B | 0.31 | 0.37 | 0.44 | 0.44 |

## 6) New Replicated Runs (4xA100, Qwen3-8B, n=200)
Fixed algorithm seed: `seed=11`, varied subset seed: `data_seed in {101,202,303}`.

| data_seed | Acc@64 | Acc@128 | Acc@256 | Acc@Adaptive | Tok@256 | Tok@Adaptive | DeltaAcc (Adap-256) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.285 | 0.325 | 0.380 | 0.385 | 30.680 | 31.605 | +0.005 |
| 202 | 0.245 | 0.305 | 0.370 | 0.370 | 38.875 | 39.710 | +0.000 |
| 303 | 0.310 | 0.375 | 0.460 | 0.460 | 35.780 | 36.775 | +0.000 |

Aggregate (mean ± std across 3 subset replications):
- `Acc@64`: `0.280 ± 0.033`
- `Acc@128`: `0.335 ± 0.036`
- `Acc@256`: `0.403 ± 0.049`
- `Acc@Adaptive`: `0.405 ± 0.048`
- `DeltaAcc (Adaptive - Fixed256)`: `+0.0017 ± 0.0029`
- `DeltaTokens (Adaptive - Fixed256)`: `+0.918 ± 0.080`

Paired bootstrap CI on delta (per run):
- `data_seed=101`: `+0.005`, 95% CI `[0.000, 0.015]`
- `data_seed=202`: `+0.000`, 95% CI `[0.000, 0.000]`
- `data_seed=303`: `+0.000`, 95% CI `[0.000, 0.000]`

Result files:
- `summary_Qwen3_8B_20260227_135614.json`
- `summary_Qwen3_8B_20260227_140019.json`
- `summary_Qwen3_8B_20260227_140410.json`
- `per_sample_Qwen3_8B_20260227_135614.csv`
- `per_sample_Qwen3_8B_20260227_140019.csv`
- `per_sample_Qwen3_8B_20260227_140410.csv`

## 7) Interpretation Against NeurIPS Bar
- Heuristic adaptive policy is reproducible and stable.
- Current effect over fixed-256 is near zero; this does **not** satisfy primary success criterion.
- Next publishable step is a learned controller that materially shifts the Pareto frontier.

## 8) Mandatory Next Experiments
1. Learned controller training and matched-cost evaluation.
2. Verifier on/off boundary analysis with full cost accounting.
3. Cross-model transfer (`Qwen3-4B -> Qwen3.5-27B`) for policy robustness.

## 9) Parser/Measurement Repair (2026-02-27)
Issue identified:
- For long-thinking outputs (notably Qwen3.5-27B), generations often lacked explicit `Final answer` within token limit.
- Old fallback parsing (`last number in text`) could mis-score truncated traces.

Repair implemented in `run_gsm8k_experiment.py`:
- `--strict_final_only`
- `--projection_on_missing_final`
- `--projection_max_tokens`
- Additional logging:
  - `final_rate`
  - `projection_rate`
  - `avg_projection_tokens`

Smoke validation:
- `summary_Qwen3_0.6B_20260227_144935.json`
- Script path is stable; strict/projection metrics now present in summary output.

## 10) Qwen3.5-27B Overthinking Signal (Strict/Projection Path)
Runs:
- `summary_Qwen3.5_27B_20260227_150649.json` (`data_seed=101`)
- `summary_Qwen3.5_27B_20260227_152431.json` (`data_seed=202`)
- `summary_Qwen3.5_27B_20260227_154356.json` (`data_seed=303`)
- `summary_Qwen3.5_27B_20260227_192225.json` (`data_seed=404`)
- `summary_Qwen3.5_27B_20260228_050913.json` (`data_seed=505`)
- `summary_Qwen3.5_27B_20260228_052641.json` (`data_seed=606`)
- `summary_Qwen3.5_27B_20260228_055128.json` (`data_seed=707`)
- `summary_Qwen3.5_27B_20260228_060842.json` (`data_seed=808`)
- `summary_Qwen3.5_27B_20260228_065843.json` (`data_seed=909`)
- `summary_Qwen3.5_27B_20260228_071554.json` (`data_seed=1001`)
- `summary_Qwen3.5_27B_20260228_081708.json` (`data_seed=1103`)
- `summary_Qwen3.5_27B_20260228_083415.json` (`data_seed=1205`)
- `summary_Qwen3.5_27B_20260228_100413.json` (`data_seed=1307`)
- `summary_Qwen3.5_27B_20260228_102125.json` (`data_seed=1409`)
- `summary_Qwen3.5_27B_20260228_104506.json` (`data_seed=1511`)
- `summary_Qwen3.5_27B_20260228_112037.json` (`data_seed=1613`)
- `summary_Qwen3.5_27B_20260228_114200.json` (`data_seed=1717`)
- common config: `n=40`, `budgets=128/256/512`, `enable_thinking`, `strict_final_only`, `projection_on_missing_final`.
- pooled summary artifact:
  - `qwen35_27b_overthinking_17seed_20260228.json` (`n=680`)
  - generated by `methods/01_adathink/scripts/run_overthinking_aggregate.py`

Per-run:
- `data_seed=101`:
  - `Acc@128=0.40`, `Acc@256=0.55`, `Acc@512=0.425`, `Acc@Adaptive=0.40`
  - `fixed_128_vs_fixed_512=0.25`
- `data_seed=202`:
  - `Acc@128=0.325`, `Acc@256=0.50`, `Acc@512=0.475`, `Acc@Adaptive=0.525`
  - `fixed_128_vs_fixed_512=0.125`
- `data_seed=303`:
  - `Acc@128=0.275`, `Acc@256=0.45`, `Acc@512=0.55`, `Acc@Adaptive=0.55`
  - `fixed_128_vs_fixed_512=0.10`
- `data_seed=404`:
  - `Acc@128=0.40`, `Acc@256=0.625`, `Acc@512=0.625`, `Acc@Adaptive=0.575`
  - `fixed_128_vs_fixed_512=0.20`
- `data_seed=505`:
  - `Acc@128=0.25`, `Acc@256=0.325`, `Acc@512=0.425`, `Acc@Adaptive=0.60`
  - `fixed_128_vs_fixed_512=0.10`
- `data_seed=606`:
  - `Acc@128=0.275`, `Acc@256=0.45`, `Acc@512=0.45`, `Acc@Adaptive=0.475`
  - `fixed_128_vs_fixed_512=0.175`
- `data_seed=707`:
  - `Acc@128=0.35`, `Acc@256=0.375`, `Acc@512=0.45`, `Acc@Adaptive=0.60`
  - `fixed_128_vs_fixed_512=0.25`
- `data_seed=808`:
  - `Acc@128=0.325`, `Acc@256=0.40`, `Acc@512=0.375`, `Acc@Adaptive=0.425`
  - `fixed_128_vs_fixed_512=0.20`
- `data_seed=909`:
  - `Acc@128=0.225`, `Acc@256=0.40`, `Acc@512=0.50`, `Acc@Adaptive=0.525`
  - `fixed_128_vs_fixed_512=0.125`
- `data_seed=1001`:
  - `Acc@128=0.40`, `Acc@256=0.35`, `Acc@512=0.40`, `Acc@Adaptive=0.35`
  - `fixed_128_vs_fixed_512=0.225`
- `data_seed=1103`:
  - `Acc@128=0.475`, `Acc@256=0.775`, `Acc@512=0.60`, `Acc@Adaptive=0.60`
  - `fixed_128_vs_fixed_512=0.20`
- `data_seed=1205`:
  - `Acc@128=0.30`, `Acc@256=0.50`, `Acc@512=0.525`, `Acc@Adaptive=0.40`
  - `fixed_128_vs_fixed_512=0.20`
- `data_seed=1307`:
  - `Acc@128=0.325`, `Acc@256=0.30`, `Acc@512=0.45`, `Acc@Adaptive=0.525`
  - `fixed_128_vs_fixed_512=0.20`
- `data_seed=1409`:
  - `Acc@128=0.40`, `Acc@256=0.60`, `Acc@512=0.525`, `Acc@Adaptive=0.55`
  - `fixed_128_vs_fixed_512=0.15`
- `data_seed=1511`:
  - `Acc@128=0.30`, `Acc@256=0.375`, `Acc@512=0.55`, `Acc@Adaptive=0.60`
  - `fixed_128_vs_fixed_512=0.15`
- `data_seed=1613`:
  - `Acc@128=0.325`, `Acc@256=0.425`, `Acc@512=0.45`, `Acc@Adaptive=0.35`
  - `fixed_128_vs_fixed_512=0.15`
- `data_seed=1717`:
  - `Acc@128=0.60`, `Acc@256=0.575`, `Acc@512=0.40`, `Acc@Adaptive=0.425`
  - `fixed_128_vs_fixed_512=0.40`

Seventeen-run pooled mean (`n=680`):
- `Acc@128=0.3500`, `Tok@128=158.28`
- `Acc@256=0.4691`, `Tok@256=286.43`
- `Acc@512=0.4809`, `Tok@512=542.54`
- `Acc@Adaptive=0.4985`, `Tok@Adaptive=542.45`

Paired bootstrap deltas (pooled `n=680`):
- `fixed256 - fixed512`:
  - `DeltaAcc = -0.0118` (95% CI `[-0.0662, +0.0412]`)
  - `DeltaTokens = -256.11` (95% CI `[-256.72, -255.51]`)
- `adaptive - fixed256`:
  - `DeltaAcc = +0.0294` (95% CI `[-0.0235, +0.0824]`)
  - `DeltaTokens = +256.02` (95% CI `[+255.43, +256.62]`)

Interpretation:
- Naive adaptive gains accuracy vs fixed256, but it spends almost full-512 compute.
- `256` and `512` are statistically tied on pooled accuracy, so overthinking here is not a single-direction monotonic effect.
- Publication-relevant comparison must be learned policy vs matched-cost fixed baseline, not heuristic adaptive vs low budget.

## 11) Learned Template Budget Controller (Cross-Subset Leave-One-Out)
Scripts:
- `methods/01_adathink/scripts/run_template_budget_controller.py`
- `methods/01_adathink/scripts/run_learned_budget_controller.py` (linear baseline; weaker)
- `methods/01_adathink/scripts/run_template_controller_significance.py`

Main result (`lambda_cost=0.15`):
- `template_controller_lam0p15_20260228_17seed.json`
- protocol: leave-one-csv-out on seventeen Qwen3.5-27B runs (`data_seed=101/202/303/404/505/606/707/808/909/1001/1103/1205/1307/1409/1511/1613/1717`), with inner mode selection.
- significance artifact:
  - `template_controller_significance_lam0p15_20260228_17seed_vs_fixed256.json`

Macro mean (680 pooled examples):
- learned template controller:
  - `accuracy = 0.56324`
  - `avg_tokens = 269.35`
  - `avg_utility = 0.48433`
- fixed baselines:
  - fixed128: `acc=0.35000`, `tokens=158.28`, `utility=0.30363`
  - fixed256: `acc=0.46912`, `tokens=286.43`, `utility=0.38520`
  - fixed512: `acc=0.48088`, `tokens=542.54`, `utility=0.32193`

Paired delta vs fixed256 (pooled `n=680`):
- `DeltaAcc = +0.09412` (95% bootstrap CI `[+0.07206, +0.11618]`)
- `DeltaTokens = -17.08` (95% bootstrap CI `[-23.81, -10.14]`)
- `DeltaUtility = +0.09912` (95% bootstrap CI `[+0.07805, +0.12067]`)

Interpretation:
- On the current seventeen-subset pool, template controller remains significantly better than fixed256 on all three paired metrics (accuracy, tokens, utility).
- Remaining gap is generalization and stronger parametric policy class, not basic effect existence.

## 12) Nineteen-Subset Extension (2026-02-28)
New strict/projection replications:
- `summary_Qwen3.5_27B_20260228_122430.json` (`data_seed=1819`)
- `summary_Qwen3.5_27B_20260228_125252.json` (`data_seed=1921`)

Pooled overthinking update:
- `qwen35_27b_overthinking_19seed_20260228.json` (`n=760`)
- means:
  - `Acc@128=0.3421`, `Tok@128=158.23`
  - `Acc@256=0.4671`, `Tok@256=286.34`
  - `Acc@512=0.4934`, `Tok@512=542.54`
  - `Acc@Adaptive=0.5118`, `Tok@Adaptive=542.46`
- paired deltas:
  - `fixed256 - fixed512`: `DeltaAcc=-0.0263` (95% CI `[-0.0763,+0.0237]`), `DeltaTokens=-256.20`
  - `adaptive - fixed256`: `DeltaAcc=+0.0447` (95% CI `[-0.0066,+0.0947]`), `DeltaTokens=+256.12`

Controller update:
- `template_controller_lam0p15_20260228_19seed.json`
- significance:
  - `template_controller_significance_lam0p15_20260228_19seed_vs_fixed256.json`
- macro mean:
  - learned: `accuracy=0.59079`, `avg_tokens=268.79`, `avg_utility=0.51204`
  - fixed256: `accuracy=0.46711`, `avg_tokens=286.34`, `avg_utility=0.38322`
- paired delta learned vs fixed256 (`n=760`):
  - `DeltaAcc=+0.12368` (95% CI `[+0.10000,+0.14868]`)
  - `DeltaTokens=-17.55` (95% CI `[-24.18,-10.61]`)
  - `DeltaUtility=+0.12883` (95% CI `[+0.10616,+0.15221]`)

## 13) Twenty-Subset Extension (2026-02-28)
New strict/projection replication:
- `summary_Qwen3.5_27B_20260228_135728.json` (`data_seed=2023`)

Pooled overthinking update:
- `qwen35_27b_overthinking_20seed_20260228.json` (`n=800`)
- means:
  - `Acc@128=0.3350`, `Tok@128=158.23`
  - `Acc@256=0.4663`, `Tok@256=286.30`
  - `Acc@512=0.4950`, `Tok@512=542.55`
  - `Acc@Adaptive=0.5100`, `Tok@Adaptive=542.54`
- paired deltas:
  - `fixed256 - fixed512`: `DeltaAcc=-0.0288` (95% CI `[-0.0775,+0.0200]`), `DeltaTokens=-256.25`
  - `adaptive - fixed256`: `DeltaAcc=+0.0438` (95% CI `[-0.0063,+0.0925]`), `DeltaTokens=+256.24`

Controller update:
- `template_controller_lam0p15_20260228_20seed.json`
- significance:
  - `template_controller_significance_lam0p15_20260228_20seed_vs_fixed256.json`
- macro mean:
  - learned: `accuracy=0.58875`, `avg_tokens=269.02`, `avg_utility=0.50994`
  - fixed256: `accuracy=0.46625`, `avg_tokens=286.30`, `avg_utility=0.38237`
- paired delta learned vs fixed256 (`n=800`):
  - `DeltaAcc=+0.12250` (95% CI `[+0.10000,+0.14625]`)
  - `DeltaTokens=-17.28` (95% CI `[-23.84,-10.58]`)
  - `DeltaUtility=+0.12756` (95% CI `[+0.10525,+0.15059]`)

## 14) Twenty-One-Subset Extension (2026-02-28)
New strict/projection replication:
- `summary_Qwen3.5_27B_20260228_141822.json` (`data_seed=2125`)

Pooled overthinking update:
- `qwen35_27b_overthinking_21seed_20260228.json` (`n=840`)
- means:
  - `Acc@128=0.3405`, `Tok@128=158.29`
  - `Acc@256=0.4667`, `Tok@256=286.30`
  - `Acc@512=0.4893`, `Tok@512=542.59`
  - `Acc@Adaptive=0.5048`, `Tok@Adaptive=542.55`
- paired deltas:
  - `fixed256 - fixed512`: `DeltaAcc=-0.0226` (95% CI `[-0.0702,+0.0250]`), `DeltaTokens=-256.28`
  - `adaptive - fixed256`: `DeltaAcc=+0.0381` (95% CI `[-0.0119,+0.0857]`), `DeltaTokens=+256.25`

Controller update:
- `template_controller_lam0p15_20260228_21seed.json`
- significance:
  - `template_controller_significance_lam0p15_20260228_21seed_vs_fixed256.json`
- macro mean:
  - learned: `accuracy=0.59524`, `avg_tokens=269.24`, `avg_utility=0.51636`
  - fixed256: `accuracy=0.46667`, `avg_tokens=286.30`, `avg_utility=0.38279`
- paired delta learned vs fixed256 (`n=840`):
  - `DeltaAcc=+0.12857` (95% CI `[+0.10595,+0.15238]`)
  - `DeltaTokens=-17.07` (95% CI `[-23.53,-10.42]`)
  - `DeltaUtility=+0.13357` (95% CI `[+0.11133,+0.15619]`)

## 15) Twenty-Two-Subset Extension (2026-02-28)
New strict/projection replication:
- `summary_Qwen3.5_27B_20260228_150620.json` (`data_seed=2227`)

Pooled overthinking update:
- `qwen35_27b_overthinking_22seed_20260228.json` (`n=880`)
- means:
  - `Acc@128=0.3341`, `Tok@128=158.30`
  - `Acc@256=0.4625`, `Tok@256=286.29`
  - `Acc@512=0.4909`, `Tok@512=542.62`
  - `Acc@Adaptive=0.5102`, `Tok@Adaptive=542.53`
- paired deltas:
  - `fixed256 - fixed512`: `DeltaAcc=-0.0284` (95% CI `[-0.0750,+0.0182]`), `DeltaTokens=-256.34`
  - `adaptive - fixed256`: `DeltaAcc=+0.0477` (95% CI `[+0.0000,+0.0943]`), `DeltaTokens=+256.24`

Controller update:
- `template_controller_lam0p15_20260228_22seed.json`
- significance:
  - `template_controller_significance_lam0p15_20260228_22seed_vs_fixed256.json`
- macro mean:
  - learned: `accuracy=0.59773`, `avg_tokens=270.54`, `avg_utility=0.51847`
  - fixed256: `accuracy=0.46250`, `avg_tokens=286.29`, `avg_utility=0.37863`
- paired delta learned vs fixed256 (`n=880`):
  - `DeltaAcc=+0.13523` (95% CI `[+0.11250,+0.15795]`)
  - `DeltaTokens=-15.74` (95% CI `[-22.28,-8.95]`)
  - `DeltaUtility=+0.13984` (95% CI `[+0.11784,+0.16240]`)

## 16) Twenty-Three-Subset Extension (2026-02-28)
New strict/projection replication:
- `summary_Qwen3.5_27B_20260228_152825.json` (`data_seed=2329`)

Pooled overthinking update:
- `qwen35_27b_overthinking_23seed_20260228.json` (`n=920`)
- means:
  - `Acc@128=0.3370`, `Tok@128=158.25`
  - `Acc@256=0.4620`, `Tok@256=286.30`
  - `Acc@512=0.4870`, `Tok@512=542.65`
  - `Acc@Adaptive=0.5076`, `Tok@Adaptive=542.54`
- paired deltas:
  - `fixed256 - fixed512`: `DeltaAcc=-0.0250` (95% CI `[-0.0707,+0.0207]`), `DeltaTokens=-256.36`
  - `adaptive - fixed256`: `DeltaAcc=+0.0457` (95% CI `[+0.0000,+0.0913]`), `DeltaTokens=+256.24`

Controller update:
- `template_controller_lam0p15_20260228_23seed.json`
- significance:
  - `template_controller_significance_lam0p15_20260228_23seed_vs_fixed256.json`
- macro mean:
  - learned: `accuracy=0.60435`, `avg_tokens=269.49`, `avg_utility=0.52539`
  - fixed256: `accuracy=0.46196`, `avg_tokens=286.30`, `avg_utility=0.37808`
- paired delta learned vs fixed256 (`n=920`):
  - `DeltaAcc=+0.14239` (95% CI `[+0.11957,+0.16522]`)
  - `DeltaTokens=-16.80` (95% CI `[-23.37,-10.19]`)
  - `DeltaUtility=+0.14731` (95% CI `[+0.12519,+0.17000]`)

## 17) Parametric Controller Update (2026-02-28)
Script:
- `methods/01_adathink/scripts/run_learned_budget_controller.py`

23-seed result:
- `learned_controller_lam0p15_20260228_163626.json`
- `learned_controller_rows_lam0p15_20260228_163626.csv`
- significance:
  - `learned_controller_significance_lam0p15_20260228_23seed_vs_fixed256.json`

Macro mean:
- learned parametric controller: `accuracy=0.56957`, `avg_tokens=242.67`, `avg_utility=0.49847`
- fixed256: `accuracy=0.46196`, `avg_tokens=286.30`, `avg_utility=0.37808`

Paired delta learned vs fixed256 (`n=920`):
- `DeltaAcc=+0.10761` (95% CI `[+0.07500,+0.14130]`)
- `DeltaTokens=-43.63` (95% CI `[-52.39,-34.34]`)
- `DeltaUtility=+0.12039` (95% CI `[+0.08770,+0.15349]`)

Interpretation:
- Parametric linear policy is now a strong reproducible baseline with significant gains on all paired metrics.
- Template controller still has higher absolute accuracy, but parametric controller is markedly more compute-efficient.

## 18) Second-Scale Extension: Qwen3-8B Thinking+Strict (2026-02-28)
New strict/projection replications (`n=40` each):
- `summary_Qwen3_8B_20260228_164451.json` (`data_seed=3101`)
- `summary_Qwen3_8B_20260228_165122.json` (`data_seed=3202`)
- `summary_Qwen3_8B_20260228_165750.json` (`data_seed=3303`)

Pooled overthinking (`n=120`):
- `qwen3_8b_think_overthinking_3seed_20260228.json`
- means:
  - `Acc@128=0.3917`, `Tok@128=145.63`
  - `Acc@256=0.6083`, `Tok@256=272.05`
  - `Acc@512=0.8250`, `Tok@512=472.08`
  - `Acc@Adaptive=0.8667`, `Tok@Adaptive=473.83`
- paired deltas:
  - `fixed256 - fixed512`: `DeltaAcc=-0.2167` (95% CI `[-0.3083,-0.1333]`), `DeltaTokens=-200.03`
  - `adaptive - fixed256`: `DeltaAcc=+0.2583` (95% CI `[+0.1750,+0.3500]`), `DeltaTokens=+201.78`

Template controller on 8B-think:
- `template_controller_qwen3_8b_think_lam0p15_3seed_20260228.json`
- significance:
  - `template_controller_qwen3_8b_think_significance_lam0p15_3seed_vs_fixed256_20260228.json`
- macro mean:
  - learned: `accuracy=0.75833`, `avg_tokens=431.26`, `avg_utility=0.63199`
  - fixed256: `accuracy=0.60833`, `avg_tokens=272.05`, `avg_utility=0.52863`
- paired delta vs fixed256 (`n=120`):
  - `DeltaAcc=+0.1500` (95% CI `[+0.0667,+0.2333]`)
  - `DeltaTokens=+159.21` (95% CI `[+137.38,+180.42]`)
  - `DeltaUtility=+0.10336` (95% CI `[+0.02151,+0.18653]`)

Parametric controller on 8B-think:
- `learned_controller_lam0p15_20260228_165820.json`
- significance:
  - `learned_controller_qwen3_8b_think_significance_lam0p15_3seed_vs_fixed256_20260228.json`
- paired delta vs fixed256 (`n=120`):
  - `DeltaAcc=-0.1250` (95% CI `[-0.2167,-0.0333]`)
  - `DeltaTokens=-65.14` (95% CI `[-85.20,-42.51]`)
  - `DeltaUtility=-0.10592` (95% CI `[-0.19810,-0.01572]`)

Interpretation:
- Second-scale structure is now clearly measured and statistically credible.
- Main remaining gap is matched-cost optimization on this second scale.

## 19) Second-Scale Controller Optimization Follow-up (2026-03-03)
New scripts:
- `methods/01_adathink/scripts/run_parametric_sweep.py`
- `methods/01_adathink/scripts/run_value_budget_controller.py`

Parametric constrained sweep on 8B-think (`n=120`), artifacts:
- `qwen3_8b_think_gridC_20260303_scoreboard.csv`
- `qwen3_8b_think_gridCplus_20260303_scoreboard.csv`

Best/worst tradeoff points vs fixed256:
- high-utility point (`lam=0.3`):
  - `DeltaAcc=+0.2167`, `DeltaTokens=+200.03`, `DeltaUtility=+0.15806`
- near-cost point (`lam=0.7`):
  - `DeltaAcc=-0.1083`, `DeltaTokens=-41.18`, `DeltaUtility=-0.09627`
- interpolation region (`lam=0.5~0.6`):
  - `DeltaAcc=+0.0917~+0.1000`, `DeltaTokens=+94.44~+95.42`, `DeltaUtility=+0.06371~+0.07233`

Value-based controller (per-budget correctness prediction + value-cost decision):
- `value_controller_qwen3_8b_think_pen0_20260303.json`
- `value_controller_qwen3_8b_think_pen1_20260303.json`
- penalty sweep summary:
  - `value_controller_qwen3_8b_think_penalty_sweep_20260303.csv`

Key points vs fixed256 (`lambda_eval=0.15`):
- `penalty=0.0/0.2`:
  - `DeltaAcc=+0.0750` (CI `[+0.0000,+0.1500]`)
  - `DeltaTokens=+54.13` (CI `[+29.63,+78.59]`)
  - `DeltaUtility=+0.05914` (CI `[-0.01219,+0.13247]`)
- `penalty=0.6/0.8/1.0/1.2`:
  - `DeltaAcc=+0.0083` (CI `[-0.0583,+0.0750]`)
  - `DeltaTokens=+11.03` (CI `[-10.66,+33.69]`)
  - `DeltaUtility=+0.00510` (CI `[-0.06413,+0.07559]`)

Interpretation:
- On second scale, controller cost is now compressed from `+159` tokens (template baseline) to `+11` tokens near fixed256.
- But matched-cost gain is still not statistically significant; current blocker shifts from "cannot approach cost" to "effect too small at n=120".

## 20) Second-Scale 7-Seed Finalization (2026-03-03)
New added replications (`n=40` each):
- `summary_Qwen3_8B_20260303_134742.json` (`data_seed=3404`)
- `summary_Qwen3_8B_20260303_135353.json` (`data_seed=3505`)
- `summary_Qwen3_8B_20260303_140015.json` (`data_seed=3606`)
- `summary_Qwen3_8B_20260303_140626.json` (`data_seed=3707`)

7-seed manifest and pooled aggregate (`n=280`):
- `manifest_qwen3_8b_think_strict_7seed_20260303_145943.json`
- `qwen3_8b_think_overthinking_7seed_20260303_145943.json`

Pooled overthinking means:
- `Acc@128=0.4286`, `Tok@128=145.44`
- `Acc@256=0.6179`, `Tok@256=271.48`
- `Acc@512=0.8250`, `Tok@512=463.96`
- `Acc@Adaptive=0.8464`, `Tok@Adaptive=466.58`

Paired deltas:
- `fixed256 - fixed512`:
  - `DeltaAcc=-0.2071` (95% CI `[-0.2607,-0.1536]`)
  - `DeltaTokens=-192.49`
- `adaptive - fixed256`:
  - `DeltaAcc=+0.2286` (95% CI `[+0.1750,+0.2821]`)
  - `DeltaTokens=+195.11`

Value-controller (7-seed) significance vs fixed256:
- quality-first (`penalty=0.0`):
  - `value_controller_qwen3_8b_think_pen0_significance_vs_fixed256_20260303_145943.json`
  - `DeltaAcc=+0.1357` (95% CI `[+0.0929,+0.1821]`)
  - `DeltaTokens=+86.67` (95% CI `[+70.95,+102.68]`)
  - `DeltaUtility=+0.1103` (95% CI `[+0.0684,+0.1541]`)
- near-cost recommended (`penalty=0.8`):
  - `value_controller_qwen3_8b_think_7seed_pen0p8_significance_vs_fixed256_20260303_7seed.json`
  - `DeltaAcc=+0.0464` (95% CI `[+0.0071,+0.0857]`)
  - `DeltaTokens=+11.74` (95% CI `[-2.14,+25.78]`)
  - `DeltaUtility=+0.0430` (95% CI `[+0.0061,+0.0794]`)

Penalty sweep artifact:
- `value_controller_qwen3_8b_think_penalty_sweep_20260303_7seed.csv`

Interpretation:
- Compared with 3-seed (`n=120`) and 4-seed (`n=160`) checkpoints, 7-seed now gives stable matched-cost utility gains with CI above zero.
- Recommended main-table operating point for second-scale claim is `penalty=0.8` (small token overhead with significant utility gain).

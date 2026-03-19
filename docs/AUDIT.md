# Methods Audit and Repair Log (2026-03-03, v9)

This file records what was below NeurIPS standard, what was repaired, and what remains.

## 01 AdaThink
- Unreasonable before:
  - Mixed `seed` semantics changed both model randomness and data subset.
  - Prompt-mode confound (`enable_thinking`) not controlled.
  - Parser over-counted truncated long traces by taking trailing numbers.
- Repair completed:
  - `--data_seed` decouples subset randomness from algorithm randomness.
  - Prompt controls standardized (`--prompt_format`, `--direct_answer`, `--enable_thinking`).
  - Added strict final-answer parsing and optional projection pass:
    - `--strict_final_only`
    - `--projection_on_missing_final`
    - `--projection_max_tokens`
  - 4xA100 replication set and per-sample artifacts are available.
  - New 27B strict/projection replication set:
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
    - `summary_Qwen3.5_27B_20260228_122430.json` (`data_seed=1819`)
    - `summary_Qwen3.5_27B_20260228_125252.json` (`data_seed=1921`)
    - `summary_Qwen3.5_27B_20260228_135728.json` (`data_seed=2023`)
    - `summary_Qwen3.5_27B_20260228_141822.json` (`data_seed=2125`)
    - `summary_Qwen3.5_27B_20260228_150620.json` (`data_seed=2227`)
    - `summary_Qwen3.5_27B_20260228_152825.json` (`data_seed=2329`)
  - Pooled twenty-three-subset summary:
    - `qwen35_27b_overthinking_23seed_20260228.json`
    - means: `Acc@256=0.4620` vs `Acc@512=0.4870`; adaptive beats 256 only with near-512 cost.
  - Learned template controller on 23-fold leave-one-subset-out:
    - `template_controller_lam0p15_20260228_23seed.json`
    - significance: `template_controller_significance_lam0p15_20260228_23seed_vs_fixed256.json`
    - paired delta vs fixed256:
      - `DeltaAcc=+0.14239` (CI `[+0.11957,+0.16522]`)
      - `DeltaTokens=-16.80` (CI `[-23.37,-10.19]`)
      - `DeltaUtility=+0.14731` (CI `[+0.12519,+0.17000]`)
  - Learned parametric controller (linear hashed policy) on same 23-fold split:
    - `learned_controller_lam0p15_20260228_163626.json`
    - significance: `learned_controller_significance_lam0p15_20260228_23seed_vs_fixed256.json`
    - paired delta vs fixed256:
      - `DeltaAcc=+0.10761` (CI `[+0.07500,+0.14130]`)
      - `DeltaTokens=-43.63` (CI `[-52.39,-34.34]`)
      - `DeltaUtility=+0.12039` (CI `[+0.08770,+0.15349]`)
  - Second-scale replication (Qwen3-8B thinking+strict+projection, 3 seeds):
    - `summary_Qwen3_8B_20260228_164451.json` (`data_seed=3101`)
    - `summary_Qwen3_8B_20260228_165122.json` (`data_seed=3202`)
    - `summary_Qwen3_8B_20260228_165750.json` (`data_seed=3303`)
    - pooled: `qwen3_8b_think_overthinking_3seed_20260228.json`
    - template controller significance:
      - `template_controller_qwen3_8b_think_significance_lam0p15_3seed_vs_fixed256_20260228.json`
  - 2026-03-03 second-scale controller refinement:
    - New scripts:
      - `run_parametric_sweep.py`
      - `run_value_budget_controller.py`
    - Parametric sweep artifacts:
      - `qwen3_8b_think_gridC_20260303_scoreboard.csv`
      - `qwen3_8b_think_gridCplus_20260303_scoreboard.csv`
    - Value-controller artifacts:
      - `value_controller_qwen3_8b_think_pen0_20260303.json`
      - `value_controller_qwen3_8b_think_pen1_20260303.json`
      - `value_controller_qwen3_8b_think_penalty_sweep_20260303.csv`
    - Best near-cost point currently observed:
      - `value_controller_qwen3_8b_think_pen0p6_significance_vs_fixed256_20260303.json`
      - paired deltas vs fixed256: `DeltaAcc=+0.0083`, `DeltaTokens=+11.03`, `DeltaUtility=+0.00510`
- Remaining gap:
  - On second scale, matched-cost performance is now close in compute (`+11` tokens) but effect size is still statistically inconclusive at current `n=120`.
  - Need larger second-scale seed pool and mandatory ablations to convert this into acceptance-level evidence.

## 02 TRACE-Hallu
- Unreasonable before:
  - Causal intervention claim without executable detector/policy.
- Repair completed:
  - Added runnable pilot:
    - `methods/02_trace_hallu/scripts/run_trace_hallu_pilot.py`
  - Added first result artifacts:
    - `methods/02_trace_hallu/results/trace_hallu_pilot_20260227_150036.json`
    - `methods/02_trace_hallu/results/trace_hallu_policy_20260227_150036.csv`
- Remaining gap:
  - Current pilot is offline proxy over GSM8K traces, not a full claim-level open-domain intervention benchmark.

## 03 NoisePO
- Unreasonable before:
  - Robustness thesis without executable noisy-preference experiment.
- Repair completed:
  - Added runnable pilot:
    - `methods/03_noisepo/scripts/run_noisepo_pilot.py`
  - Added first result artifact:
    - `methods/03_noisepo/results/noisepo_pilot_20260227_150037.json`
- Remaining gap:
  - Robust-vs-standard separation is weak in pilot; needs real preference fine-tuning under controlled corruption.

## 04 UniRAG-Policy
- Unreasonable before:
  - Unified policy claim without executable policy search.
- Repair completed:
  - Added runnable pilot:
    - `methods/04_unirag_policy/scripts/run_unirag_policy_pilot.py`
  - Added first result artifact:
    - `methods/04_unirag_policy/results/unirag_policy_pilot_20260227_150036.json`
- Remaining gap:
  - No real retrieval/citation pipeline yet; current action-space proxy is budget policy only.

## 05 Text2Subspace
- Unreasonable before:
  - Subspace claim with no executable low-rank vs full-rank study.
- Repair completed:
  - Added runnable pilot:
    - `methods/05_text2subspace/scripts/run_text2subspace_pilot.py`
  - Added first result artifact:
    - `methods/05_text2subspace/results/text2subspace_pilot_20260227_151439.json`
- Remaining gap:
  - Still a policy-head proxy, not true text-to-adapter generation over checkpoint weights.

## 06 TemplateBank++
- Unreasonable before:
  - Template-memory claim without executable static/dynamic memory evaluation.
- Repair completed:
  - Added runnable pilot:
    - `methods/06_templatebank_pp/scripts/run_templatebank_pilot.py`
  - Added first result artifact:
    - `methods/06_templatebank_pp/results/templatebank_pilot_20260227_150036.json`
- Remaining gap:
  - Needs full template extraction/instantiation with leakage-safe train/test generation traces.

## Global Verdict
- All six tracks now have executable artifacts in this repository.
- `01_adathink` remains the only full multi-GPU end-to-end pipeline.
- `01_adathink` now has statistically positive template and parametric controller evidence on the 27B twenty-three-subset pool; second-scale has moved from high-cost gains to near-matched-cost but still needs larger-sample statistical confirmation and ablations.

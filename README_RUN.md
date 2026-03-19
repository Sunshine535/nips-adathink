# AdaThink Runbook (Revised v13)

## 1) Activate
```bash
conda activate nips_adathink
```

## 2) Verify CUDA (4xA100 expected)
```bash
python - <<'PY'
import torch
print(torch.__version__)
print('cuda_available:', torch.cuda.is_available())
print('gpu_count:', torch.cuda.device_count())
PY
```

## 3) Qwen3.5-27B Strict/Projection Single Run (4 GPUs)
```bash
bash methods/01_adathink/scripts/run_gsm8k_torchrun_4gpu.sh \
  --model Qwen/Qwen3.5-27B \
  --n_samples 40 \
  --seed 11 \
  --data_seed 101 \
  --budgets 128 256 512 \
  --adaptive_chunks 128 128 256 \
  --adaptive_max_total 512 \
  --prompt_format chat \
  --enable_thinking \
  --strict_final_only \
  --projection_on_missing_final \
  --projection_max_tokens 32 \
  --no_verifier \
  --skip_local_model_check
```

## 4) Qwen3.5-27B Twenty-Three-Subset Replication Batch
```bash
for ds in 101 202 303 404 505 606 707 808 909 1001 1103 1205 1307 1409 1511 1613 1717 1819 1921 2023 2125 2227 2329; do
  bash methods/01_adathink/scripts/run_gsm8k_torchrun_4gpu.sh \
    --model Qwen/Qwen3.5-27B \
    --n_samples 40 \
    --seed 11 \
    --data_seed "${ds}" \
    --budgets 128 256 512 \
    --adaptive_chunks 128 128 256 \
    --adaptive_max_total 512 \
    --prompt_format chat \
    --enable_thinking \
    --strict_final_only \
    --projection_on_missing_final \
    --projection_max_tokens 32 \
    --no_verifier \
    --skip_local_model_check
done
```

## 5) Build Strict-Path 23-Seed Manifest
```bash
python - <<'PY'
import glob, json
from pathlib import Path

root = Path("methods/01_adathink/results")
seeds = [101,202,303,404,505,606,707,808,909,1001,1103,1205,1307,1409,1511,1613,1717,1819,1921,2023,2125,2227,2329]
want = set(seeds)
selected = {}
for sp in sorted(root.glob("summary_Qwen3.5_27B_*.json")):
    j = json.loads(sp.read_text(encoding="utf-8"))
    m = j.get("meta", {})
    ds = m.get("data_seed")
    if ds not in want:
        continue
    ok = (
        m.get("model") == "Qwen/Qwen3.5-27B"
        and int(m.get("n_samples", -1)) == 40
        and m.get("prompt_format") == "chat"
        and bool(m.get("enable_thinking")) is True
        and bool(m.get("strict_final_only")) is True
        and bool(m.get("projection_on_missing_final")) is True
        and int(m.get("projection_max_tokens", -1)) == 32
        and m.get("budgets") == [128, 256, 512]
        and m.get("adaptive_chunks") == [128, 128, 256]
        and int(m.get("adaptive_max_total", -1)) == 512
        and bool(m.get("use_verifier")) is False
    )
    if not ok:
        continue
    ts = m.get("timestamp_utc", "")
    pp = sp.with_name(sp.name.replace("summary_", "per_sample_")).with_suffix(".csv")
    if not pp.exists():
        continue
    cur = selected.get(ds)
    if cur is None or ts > cur["timestamp_utc"]:
        selected[ds] = {
            "data_seed": ds,
            "timestamp_utc": ts,
            "summary_json": str(sp),
            "per_sample_csv": str(pp),
        }

manifest = {
    "meta": {"model": "Qwen/Qwen3.5-27B", "n_samples": 40, "seed": 11},
    "seeds": seeds,
    "runs": [selected[s] for s in seeds],
}
out = root / "manifest_qwen35_27b_strict_23seed_20260228.json"
out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", out)
PY
```

## 6) Overthinking Pooled Aggregation (n=920)
```bash
CSV_LIST=$(python - <<'PY'
import json
m = json.load(open("methods/01_adathink/results/manifest_qwen35_27b_strict_23seed_20260228.json","r",encoding="utf-8"))
print(" ".join(r["per_sample_csv"] for r in m["runs"]))
PY
)
python methods/01_adathink/scripts/run_overthinking_aggregate.py \
  --input_csvs ${CSV_LIST} \
  --output_json methods/01_adathink/results/qwen35_27b_overthinking_23seed_20260228.json
```

## 7) Learned Template Controller (Leave-One-CSV-Out)
```bash
CSV_LIST=$(python - <<'PY'
import json
m = json.load(open("methods/01_adathink/results/manifest_qwen35_27b_strict_23seed_20260228.json","r",encoding="utf-8"))
print(" ".join(r["per_sample_csv"] for r in m["runs"]))
PY
)
python methods/01_adathink/scripts/run_template_budget_controller.py \
  --lambda_cost 0.15 \
  --input_csvs ${CSV_LIST} \
  --output_json methods/01_adathink/results/template_controller_lam0p15_20260228_23seed.json \
  --output_csv methods/01_adathink/results/template_controller_rows_lam0p15_20260228_23seed.csv
```

## 8) Paired Bootstrap CI vs Fixed256
```bash
python methods/01_adathink/scripts/run_template_controller_significance.py \
  --rows_csv methods/01_adathink/results/template_controller_rows_lam0p15_20260228_23seed.csv \
  --compare_budget 256 \
  --lambda_cost 0.15 \
  --norm_tokens 512 \
  --output_json methods/01_adathink/results/template_controller_significance_lam0p15_20260228_23seed_vs_fixed256.json
```

## 9) Optional: Linear Controller Baseline
```bash
python methods/01_adathink/scripts/run_learned_budget_controller.py \
  --lambda_cost 0.15 \
  --epochs 50 \
  --lr 0.2
```

## 10) Parametric Controller (Manifest-Driven)
```bash
python methods/01_adathink/scripts/run_parametric_from_manifest.py \
  --manifest_json methods/01_adathink/results/manifest_qwen35_27b_strict_23seed_20260228.json \
  --lambda_cost 0.15 \
  --norm_tokens 512 \
  --target_budget 256 \
  --epochs_grid 10 \
  --lr_grid 0.1 \
  --l2_grid 1e-4 \
  --cost_weight_grid 0.0 \
  --seed 11 \
  --output_json methods/01_adathink/results/param_controller_lam0p15_20260228_23seed_quick.json \
  --output_csv methods/01_adathink/results/param_controller_rows_lam0p15_20260228_23seed_quick.csv
```

## 11) Second-Scale 8B Thinking+Strict Triplet
```bash
for ds in 3101 3202 3303; do
  bash methods/01_adathink/scripts/run_gsm8k_torchrun_4gpu.sh \
    --model Qwen/Qwen3-8B \
    --n_samples 40 \
    --seed 11 \
    --data_seed "${ds}" \
    --budgets 128 256 512 \
    --adaptive_chunks 128 128 256 \
    --adaptive_max_total 512 \
    --prompt_format chat \
    --enable_thinking \
    --strict_final_only \
    --projection_on_missing_final \
    --projection_max_tokens 32 \
    --no_verifier \
    --skip_local_model_check
done
```

## 12) Output Artifacts
- `summary_*.json`: aggregate metrics.
- `per_sample_*.csv`: per-instance outputs for paired tests.
- `manifest_qwen35_27b_strict_23seed_20260228.json`: curated run manifest by seed and strict config.
- `qwen35_27b_overthinking_23seed_20260228.json`: pooled overthinking statistics and paired CIs.
- `template_controller_*.json` and `template_controller_rows_*.csv`: controller folds and per-sample actions.
- `template_controller_significance_*.json`: paired bootstrap deltas with CIs.
- `param_controller_*.json` and `param_controller_rows_*.csv`: constrained parametric policy runs.

All outputs are written to `methods/01_adathink/results/`.

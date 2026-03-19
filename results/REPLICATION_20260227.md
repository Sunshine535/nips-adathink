# Qwen3-8B Replication Summary (2026-02-27)

Config:
- model: `Qwen/Qwen3-8B`
- n_samples: `200`
- seed: `11`
- data_seed: `101, 202, 303`
- budgets: `64 128 256`
- adaptive: chunks `64 64 128`, max `256`
- prompt: `chat`, `direct_answer`, `no_verifier`

Per-run files:
- `summary_Qwen3_8B_20260227_135614.json`
- `summary_Qwen3_8B_20260227_140019.json`
- `summary_Qwen3_8B_20260227_140410.json`

Aggregated (mean ± std):
- Acc@64: `0.280 ± 0.033`
- Acc@128: `0.335 ± 0.036`
- Acc@256: `0.403 ± 0.049`
- Acc@Adaptive: `0.405 ± 0.048`
- DeltaAcc (Adaptive - Fixed256): `+0.0017 ± 0.0029`
- DeltaTokens (Adaptive - Fixed256): `+0.918 ± 0.080`

Interpretation:
- Heuristic adaptive policy is reproducible.
- Improvement over fixed-256 is currently marginal; learned controller is required for stronger novelty.

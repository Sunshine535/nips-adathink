# Current Result Audit

> **CANONICAL STATUS REFERENCE: `reports/FINAL_RCV_VERDICT.md` (2026-04-25)**. RCV is a negative ablation. Main contributions: Coupling Tax phenomenon + factorial interaction + decoupled extraction.



## Result Reliability Ledger

| Result | File | Dataset | Config | Seed | Metric | Value | Compared Against | Supports GPT-5.5 Diagnosis? | Notes |
|--------|------|---------|--------|------|--------|-------|------------------|-----------------------------|-------|
| IRIS 27B MATH-500 (headline post-hoc) | results/iris_improved_20260417/27b_math500_b4096_ba512_n200/*.json | MATH-500 n=200 | b4096 ba512 | 42 | acc | 77.5% | TOWN 49.0% | Partial — post-hoc accounting | Verified JSON, but post-hoc |
| IRIS 27B MATH-500 (online) | results/iris_online_20260421/27b_math500_n200/*.json | MATH-500 n=200 | b4096 ba512 online | 42 | acc | 67.5% | post-hoc 77.5% | YES — confirms post-hoc inflates | -10pp gap |
| IRIS 8B GSM8K b1=512 | results/iris_b1_512/checkpoint_iris_1300.json | GSM8K n=1319 | b1=512 b2=512 ba=128 | 42 | acc | 93.2% | nothink 93.1% | YES — confirms saturation | All Stage 1, no escalation |
| IRIS 8B MATH-500 n=500 | results/iris_improved_20260417/8b_math500_b4096_ba512_n500/*.json | MATH-500 n=500 | b4096 ba512 | 42 | acc | 74.4% | — | Verified | |
| Multiseed IRIS MATH-500 | results/multiseed_20260419/*.json | MATH-500 | b4096 | 42/123/456 | acc | 0.744/0.725/0.755 | — | Partial — unequal n | std=0.0152 |
| SwiReasoning GSM8K full | results/swir_gsm8k_full.json | GSM8K n=1319 | swir alpha=0.5 | 42 | acc | 92.49% | IRIS 90.9% | YES — external baseline | |
| SwiReasoning MATH-500 | results/swir_math500.json | MATH-500 n=200 | swir alpha=1.0 | 42 | acc | 73.5% | IRIS 74.4% | YES — external baseline | |
| E1-Math-7B GSM8K | results/elastic_reasoning/e1_gsm8k.json | GSM8K n=200 | e1 b512/512 | 42 | acc | 88.0% | IRIS 93.2% | YES — trained baseline | Different base model |
| E1-Math-7B MATH-500 | results/elastic_reasoning/e1_math500.json | MATH-500 n=200 | e1 b1024/1024 | 42 | acc | 75.5% | IRIS 74.4% | YES — trained baseline | E1 slightly better |
| s1 early_stop | results/budget_forcing/bforce_early_stop*.json | MATH-500 n=200 | b4096 | 42 | acc | 72.0% | — | **Contaminated** (token undercount before fix) | Re-run needed |
| s1 wait_extend | results/budget_forcing/bforce_wait_extend.json | MATH-500 n=200 | b4096 | 42 | acc | 66.5% | — | **Contaminated** (token undercount before fix) | Re-run needed |
| Factorial ablation GSM8K | results/factorial_ablation/factorial_gsm8k.json | GSM8K n=200 | b1=256 b2=512 ba=128 | 42 | acc | TN=1.1% TE=22.0% NN=11.0% NE=69.2% | interaction +37.4pp | YES — mechanism isolated | 91 stage-3 samples |
| Mechanism ablation GSM8K | results/mechanism_ablation/ablation_gsm8k_20260423.json | GSM8K n=200 | — | 42 | acc | TOWN 61% FC 59% MS 81.5% | — | YES — mode-switch helps | |
| Mechanism ablation MATH-500 | results/mechanism_ablation/ablation_math500.json | MATH-500 n=200 | — | 42 | acc | TOWN 63.5% FC 68.5% MS 74.5% | — | YES | |
| Pure mode ablation GSM8K | results/pure_mode_ablation/pure_ablation_gsm8k.json | GSM8K n=200 | same prompt | 42 | acc | think 53.0% nothink 54.5% | — | YES — mode alone insufficient | 13 discordant not sig |
| Entropy dynamics | results/entropy_dynamics/* | GSM8K | b256/b512 | 42 | AUC | NO-GO | — | YES — entropy not viable | archive evidence |
| CTT pilot | results/ctt_pilot_*/* | GSM8K 27B | — | 42 | AUC | 0.535 null 0.509 | — | YES — CTT not viable | archive evidence |
| **RCV-IRIS A (b4096)** | results/rcv_iris/rcv_existing_fragment_math500_20260424_101405.json | MATH-500 n=100 | b1=512 b2=4096 | 42 | acc | **73.0%** | — | YES — baseline for A/B/C | |
| **RCV-IRIS B (b4096)** | results/rcv_iris/rcv_rcv_no_gate_math500_20260424_105007.json | MATH-500 n=100 | b1=512 b2=4096 | 42 | acc | **73.0%** | — | YES — = A (expected) | |
| **RCV-IRIS C (b4096)** | results/rcv_iris/rcv_full_rcv_math500_20260424_105007.json | MATH-500 n=100 | b1=512 b2=4096 | 42 | acc | **74.0%** | A/B 73.0% | YES — gate helps +1.0pp | 1 FALLBACK_TOWN triggered |

## Result-Based Execution Decision

**Decision: CONTINUE — RUN TIGHTER BUDGET**

Current A/B/C at b2=4096 shows weak directional signal (+1.0pp, 1 discordant). To validate the RCV mechanism needs:
1. Tighter b2 (512 or 1024) where truncation is severe and gate triggers more
2. Multi-seed to prove non-seed-luck
3. n=200+ for statistical power
4. Cross-benchmark (GSM8K)

These are now running on 3-GPU server.

## Explicitly Present Variants

- A. Existing Best Positive Fragment Only: YES (variant=existing_fragment)
- B. New MAIN METHOD Without New Mechanism: YES (variant=rcv_no_gate)
- C. Full New MAIN METHOD: YES (variant=full_rcv)
- Ablation: Stage0-only: NEWLY ADDED (variant=stage0_only)
- Ablation: Recover-only: NEWLY ADDED (variant=recover_only)

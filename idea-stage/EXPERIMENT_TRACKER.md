# EXPERIMENT_TRACKER — Coupling-Tax Tomography (CTT)

**Paper**: *The Coupling Tax: Why Chain-of-Thought Fails Under Fixed Budgets and What To Do About It*
**Plan**: see `idea-stage/EXPERIMENT_PLAN.md`.
**Compute envelope**: ≈ 40 GPU-h on 2×A100-80GB.

Status legend: `NOT STARTED` / `RUNNING` / `DONE OK` / `DONE PARTIAL` / `KILLED` / `BLOCKED`.

---

## Main tracker

| Block ID | Status | Started | Completed | Result (headline) | Notes |
|----------|--------|---------|-----------|-------------------|-------|
| E1 Oracle ceiling | NOT STARTED | — | — | — | Post-hoc on 27B full-set; ~1 GPU-h. |
| E2 CTT feasibility pilot (8B) | NOT STARTED | — | — | — | **Go/no-go gate**. AUC ≥ 0.70. n=200 GSM8K. |
| E3 CTT vs baselines | BLOCKED | — | — | — | Needs τ, ℓ\* from E2. 6 methods × 2 models × 2 benchmarks. |
| E4 Cross-scale validation | BLOCKED | — | — | — | Parallel with E3 once E2 frozen. 27B compute-heavy. |
| E5 Ablations | BLOCKED | — | — | — | 5 sub-ablations. Reuses E2 artifacts. |
| E6 Fano ceiling | BLOCKED | — | — | — | Analysis-only. Cite arXiv:2604.06192. |
| E7 Stage-3 stacking | BLOCKED | — | — | — | MATH-500 n=500 on 8B. |
| E8 Inverse-scaling curve-fit | BLOCKED | — | — | — | Stretch; only if E1–E7 green. |

---

## Sub-task checklist (condensed)

- **E2**: prefill × 2 → labels → per-layer KL → AUC → τ select → go/no-go memo.
- **E3**: CTT / TOWN / random / HF-style / P3 / JointThinking on 8B GSM8K n=500 → MATH n=200 → 27B repeat → McNemar + bootstrap.
- **E4**: 27B + 9B prefill probes → per-layer AUC overlay → cross-model ℓ\* swap test.
- **E5**: full per-layer sweep / 5-fold τ / self-KL control / aggregation variants / divergence variants.
- **E7**: pure CTT vs pure Stage-3 vs CTT+Stage-3 on MATH-500 n=500.

---

## Compute ledger

| Run | Planned GPU-h | Actual | Δ |
|-----|---------------|--------|---|
| E1 | 1 | — | — |
| E2 (first + follow) | 2 + 2 | — | — |
| E3 (first + follow) | 6 + 4 | — | — |
| E4 (first + follow) | 6 + 4 | — | — |
| E5 (first + follow) | 3 + 2 | — | — |
| E6 | 1 | — | — |
| E7 (first + follow) | 3 + 2 | — | — |
| E8 | 2 | — | — |
| **Total** | **38** | — | **Budget: 40** |

---

## Decision log

| Date | Block | Decision | Rationale |
|------|-------|----------|-----------|
| 2026-04-19 | plan | Freeze CTT as method thesis | Novelty leaning-confirmed; theorem-grounded. |

---

## Risks being watched

- **R2 (AUC < 0.65)** — gated at E2. Fallback: self-KL subtraction or Stage-3-primary pivot.
- **DTR scoop (arXiv:2602.13517)** — cite; differentiate on paired-mode + pre-generation.
- **BAEE (arXiv:2604.06613) overshadow on Stage-3** — E7 framing must differentiate mode-switch vs free-continuation.
- **τ brittleness** — E5.b leave-one-out; require stdev ≤ 0.02.

---

*Word count: ~390.*

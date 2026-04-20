# FINAL_PROPOSAL — Coupling Tax: Theorem + IRIS Split-Budget (post-CTT-null)

**Paper working title**: *The Coupling Tax: Why Chain-of-Thought Fails Under Fixed Output Budgets*
**Version**: v2 (2026-04-20, supersedes v1 after CTT mini-pilot null result)
**Target**: NeurIPS 2026 — strong accept (best-paper claim withdrawn pending online-stopping rewrite)

> **Status note.** v1 (dated 2026-04-19, now at `FINAL_PROPOSAL_CTT_DEPRECATED.md`) proposed **Coupling-Tax Tomography (CTT)** — a pre-generation router using paired think/nothink layer-wise KL — as the hero method. The 4 GPU-h mini-pilot (commit `20707ca`, saved under `results/ctt_pilot_{8b,27b}_gsm8k/`) returned **null**: max mid-layer KL AUC 0.499 (8B) / 0.535 (27B), null-scaffold gap −0.001 / +0.026, peak-layer AUC ≤ 0.627. CTT is preserved as an honest negative-result ablation (§7.x) but is **no longer the hero method**.

---

## 1. Problem anchor (unchanged across v1→v2)

> Under a fixed output-token budget `b`, a Qwen3-family model in `<think>` mode loses accuracy relative to `<nothink>` mode whenever its natural reasoning chain would exceed `b`. The loss grows with model size.

Validated instances:
- Qwen3.5-27B, GSM8K, `b=4096`: nothink 98.0%, think 87.5%, **McNemar p < 10⁻⁵** (22/23 discordant favor nothink).
- 8B → 27B: tax grows ≈ 2.8× in absolute pp terms.
- Decomposition: `Acc_think(b) = F_L(b)·α_c + (1-F_L(b))·α_t` (proof-checker R4 PASS).

---

## 2. Method thesis (v2)

> **Decoupling.** Give thinking a separate answer-emission channel. The paper's contribution is (a) the closed-form coupling-tax decomposition that predicts when/why thinking fails under budget, and (b) **IRIS** — a training-free 3-stage cascade (nothink-triage → thinking → decoupled-extraction) that breaks the budget coupling by handing the partial thinking trace to the model's own nothink-mode for a short, answer-format-constrained extraction pass. This is the "mode-switch extraction" trick and is mechanistically distinct from free-continuation approaches (BAEE, arXiv:2604.06613).

Compact: *the shared-budget failure mode is structural; a cheap in-model mode-switch at the end of thinking recovers the answer budget without retraining or multi-sample rollouts.*

---

## 3. Dominant contributions

1. **Closed-form decomposition theorem** `Acc_think(b) = F_L(b)·α_c + (1-F_L(b))·α_t` — verified through 4 rounds of proof-checker. *Unpublished externally.*
2. **27B GSM8K b=4096 crossover** with McNemar p < 10⁻⁵ on 22/23 discordant — decisive empirical confirmation that the coupling tax is a *size-sensitive* effect that does not shrink at larger budget. *Unpublished externally.*
3. **Inverse scaling of the tax with model size** quantified via the F_L-shift mechanism (8B→27B ratio ≈ 2.8×) — length-inverse-scaling has prior art (arXiv:2507.14417) but the size × length joint law is new.
4. **Stage-3 decoupled extraction (in-model mode-switch)**: +17 pp same-sample on 27B MATH-500 (60.5% → 77.5%), +61 pp on the hard subset, McNemar p=3.5×10⁻¹¹ vs TOWN. BAEE (2604.06613) does free-continuation polling; we do a different and cheaper thing.
5. **Learned allocator (auxiliary)**: 13-feature logistic regression → 46.6% token savings (oracle ceiling 60.2%, 77% of oracle).
6. **Defensive negative results** (honest science):
   - IRIS entropy-based stopping: 0/200 triggered, anti-correlated with correctness — no external paper publishes this diagnosis.
   - CTT (paired cross-mode layer-KL router): AUC≈0.5 on both 8B and 27B GSM8K, null-scaffold control within 0.03 AUC — logit-lens KL does not separate coupling-tax queries.

---

## 4. IRIS algorithm (canonical definition)

```
Inputs:
  q           user query
  M           Qwen3-family model with hybrid think/nothink modes
  b1          triage budget (default 256 GSM8K / 512 MATH-500)
  b2_max      thinking budget cap (default 1024..4096)
  b_answer    decoupled-extraction budget (128 GSM8K / 256–512 MATH-500)
  τ_h, τ_s    entropy / hidden-state thresholds (calibration only; see §7)

Procedure IRIS(q):
  # Stage 1 — nothink triage (cheap)
  y0, stop0 = generate(M, q, mode=nothink, budget=b1)
  if stop0 == natural_stop:
      return parse_answer(y0)          # accept cheap answer

  # Stage 2 — thinking with adaptive stopping
  # (currently post-hoc trace truncation — see "Limitation" below)
  y1, stop1 = generate(M, q, mode=think, budget=b2_max,
                       stopping=adaptive(τ_h, τ_s))

  if stop1 == natural_stop:
      return parse_answer(y1)

  # Stage 3 — decoupled answer extraction (mode switch)
  prefix = truncate_to_last_meaningful_chunk(y1)
  y2 = generate(M, q + prefix, mode=nothink, budget=b_answer,
                extract_only_prompt=True)
  return parse_answer(y2)
```

### 4.1 Stage 2 "adaptive stopping" — known limitation

The current `run_iris.py` implements Stage 2 as: generate to `b2_max` once, then *analyze* the trace to locate the optimal stopping point; two accounting fields are reported (`n_tokens_generated`, `n_tokens_used`), with downstream results using `n_tokens_used`.

This is **post-hoc trace truncation accounting**, not a chunk-by-chunk online stop. It is adequate for the paper's analytical claims (accuracy curves as a function of *effective budget*) but does **not** support claims of wall-clock or FLOPs savings. The honest way to phrase IRIS's Stage 2 in the paper is:

> *IRIS's Stage 2 uses an entropy/HS-stability criterion over the generated trace to identify a stopping point; we report both generated and effective token counts. Savings reported in this paper are effective-token savings.*

A follow-up engineering task (`idea-stage/EXPERIMENT_PLAN.md` Block E9, added in v2) replaces Stage 2 with chunk-by-chunk decoding so that `n_tokens_generated ≡ n_tokens_used` and the efficiency claims become deployment-faithful.

---

## 5. What this paper is **not** claiming

- Not "thinking is always worse". Coupling-tax holds on GSM8K and MATH-500 at our tested budgets; BBH at b=2048 shows a reversal (acknowledged in abstract).
- Not a new online early-stopping algorithm for CoT. Our Stage 2 is analysis-faithful, not deployment-faithful; Stage 3 is the deployment contribution.
- Not outperforming 27B GSM8K's nothink@4096 on *accuracy* — nothink is already 98.0%. IRIS's value is on MATH-500 and at smaller scales.
- Not best-paper-level novelty. That claim was contingent on CTT succeeding; with CTT null, we target strong accept on the theorem + 27B crossover + Stage-3 + multi-seed robustness combination.

---

## 6. Method scope (explicit non-goals)

- Non-Qwen3-family hybrid-thinking models not tested here.
- Code/proof-search domains (LiveCodeBench, MiniF2F) where thinking outperforms nothink even at low budget (cf. Ma et al. 2504.09858 §3.3) are out of scope.
- Inference-only (no training). Any RL-gated routing comparison (AdaptThink 2505.13417) is a positioning contrast, not a target to beat at the weights level.

---

## 7. Pre-registered updates required before submission

1. Re-run paper-claim-audit on the current manuscript (old FAIL report is stale).
2. Reconcile the 477 vs 460 token discrepancy (intro §3.1 vs `table_token_utilization.tex`) to one canonical run.
3. Implement and run `run_iris_online.py` (Block E9) to replace post-hoc Stage 2 analysis.
4. Archive deprecated `FINAL_PROPOSAL_CTT_DEPRECATED.md` but keep it for provenance.

---

## 8. Next-step priority (v2)

- P0 (blocking): online Stage-2 rewrite (E9), fresh claim audit, token-number reconciliation.
- P1: complete 27B GSM8K IRIS n=200 (H800 PID 27157, running) to finalize the 27B × benchmark IRIS grid.
- P2: cross-family generalization (DeepSeek-R1-Distill-Llama-8B) on GSM8K + MATH-500, 4–8 GPU-h.
- P3: decide whether to add a "compute-faithful" method section that does online allocation using the derived F_L-oracle, or keep the paper strictly at "phenomenon + theorem + in-model extraction".

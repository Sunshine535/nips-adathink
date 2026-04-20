# PSV-Cascade Pilot Plan — Prefix-Sufficiency Verification

**Status**: pilot design (2026-04-20), not yet launched.
**Scope**: separate research thread from Coupling Tax paper (NeurIPS 2026). Target venue TBD (ICLR 2027 / NeurIPS 2027 depending on signal).
**Non-goal**: bolt onto current Coupling Tax manuscript — that paper is frozen at `paper/main_final.tex` with Stage-3 decoupled extraction as the method, PSV does not go into it.

## Premise

Two orthogonal negative results from the Coupling Tax project constrain the design space:

1. **Pre-generation latent routers do not work at usable AUC.** CTT (think/nothink layer-wise KL at answer position, 2 × 1-token prefill) returned null on both 8B and 27B GSM8K (AUC ≤ 0.54, null-scaffold gap +0.03 / −0.00). Raw-latent + single-scalar + threshold is not the right stack for routing (but see §"Scalar/threshold caveat" below).
2. **Token-entropy early stopping is anti-correlated with correctness on our data.** IRIS's entropy/HS-stability criterion fired 0/200 on GSM8K; a 90-config grid search found no viable threshold pair. DTSR (2026) reports the same class of failure for reasoning-model overconfidence under DEER variants.

Positive signal that survives: **independent answer channel**. Stage-3 decoupled extraction (feed truncated thinking-prefix into the same model's nothink mode) delivers +17pp same-sample on 27B MATH-500 (60.5% → 77.5%, McNemar p = 3.5×10⁻¹¹). External analog: BAEE (Detection-Extraction Gap, arXiv:2604.06613) shows 52-88% of CoT tokens occur after the answer is recoverable from the prefix.

## Thesis

Instead of predicting "when to stop" from raw state, predict **"is the current prefix sufficient to support a candidate answer?"** Build a learned verifier V that scores the triple `(q, prefix_t, candidate_t)` and make chunk-boundary decisions on its output.

## Label semantics (must be fixed before any training)

The positive label must be chosen from three inequivalent definitions; we pre-register one:

| Definition | Implementation | Note |
|------------|----------------|------|
| **D1 (chosen)** | `extract(q, prefix_t, mode=nothink)` is correct | Matches our Stage-3 mechanism; directly usable at inference. |
| D2 | free-continuation `generate(q + prefix_t)` reaches correct answer | Matches BAEE. Requires extra rollouts for labels. |
| D3 | both D1 and D2 agree and are correct | Strictest label. Smallest positive class. |

**Canonical for pilot: D1.** Rationale: (a) D1 is what our deployment uses; (b) D1 labels come from a single extraction call per (q, prefix), so dataset construction cost is linear in #prefixes; (c) using D2 risks training a verifier to predict free-continuation success and then deploying it on decoupled extraction, a train/test distribution gap.

**Label leakage guard**: Detection-Extraction-Gap predicts 42% of prefixes are "answer already recoverable but forced extraction still fails". Under D1, these are marked negative — correctly, from the deployment perspective, but this will hurt verifier PPV on that subregion. We accept this trade-off and document it; a D1+D2 ensemble could be explored in a later phase if D1-only pilot succeeds.

## Minimal 2×2 pilot

Factorial design, 4 cells:

|   | **pair V(q, prefix)** | **triple V(q, prefix, candidate)** |
|---|---|---|
| **decoupled extraction only** | cell A | cell B |
| **decoupled + free-continuation probe on uncertain prefixes** | cell C | cell D |

- Model: Qwen3-8B
- Benchmark: MATH-500 (where accuracy headroom exists; GSM8K saturates at nothink@512=93% so routing gain is capped).
- n: 50 per cell for preliminary signal, escalate to 200 if any cell beats all baselines
- Verifier: 2-layer MLP on {prompt hidden state at last token, prefix hidden state at chunk boundary, (optional) candidate answer token embedding}. Ridge logistic regression fallback.
- Chunk size: 64 tokens (matches our online Stage-2 default).
- Budget ceiling: `b_prefix_max = 1024`, `b_answer = 512`.

## Baselines (all must be run on identical samples)

1. **Online IRIS** — our `--online_stage2` flag on `run_iris.py`. Chunk-by-chunk, entropy/HS-stability trigger, Stage-3 extraction. This is our own method.
2. **DTSR-style sufficiency** — threshold on model's self-rated "is the current thought sufficient" prompt, per the DTSR 2026 setup. Implementation: ask the model `"Is the above reasoning sufficient to answer the question? Answer yes or no."` at each chunk boundary; threshold on `P(yes)`.
3. **BAEE free-continuation** — at each chunk boundary, issue k=3 short free continuations (≤64 tokens each) and take the majority-vote answer; if ≥ 2/3 agree, stop.

## Evaluation protocol (non-negotiable)

Every cell and every baseline must report:
- Accuracy
- Generated tokens (n_tokens_generated) — post-hoc truncation accounting is forbidden; only the online chunked runner is accepted.
- Wall-clock seconds per query (H100 or A100)
- Verifier/extra-model calls per query
- Paired McNemar against the best baseline

**Reporting format**: a single table with (accuracy, generated_tokens, wall_clock, calls) per method, plus paired-McNemar column vs the strongest baseline. We explicitly account for DTSR's U-shape latency (too-frequent sufficiency checks slow things down even when token count drops).

## Scalar/threshold caveat

Blanket statement "threshold rules are dead" from my earlier analysis was wrong. Revised stance: **threshold rules are fine when the scalar has correct causal semantics (sufficiency) and the threshold is calibrated (Platt or conformal)**. CTT and IRIS failed because their scalars (mid-layer cross-mode KL and token-entropy respectively) are not sufficiency signals; DTSR works because its scalar targets "thought sufficiency" directly, albeit without a candidate answer. PSV-triple is the natural extension: include the candidate in the scalar's conditioning.

## Kill criteria

If any of the following holds after the 2×2 pilot:

- Best PSV cell does not Pareto-dominate BAEE **OR** DTSR on (accuracy × generated_tokens × wall_clock).
- Triple V no better than pair V by ≥ 3 pp AUC on held-out (q, prefix) → candidate-conditioning doesn't add signal.
- The D1-label verifier's calibration ECE exceeds 5% and cannot be Platt-corrected with 100 held-out samples.

Then: **abandon PSV as a method**, keep it only as a negative-result ablation, and pivot PSV-Cascade research to either (a) a small discriminative verifier + self-consistency hybrid (per 2025 "budget-aware discriminative verification" literature) or (b) prefix-conditioned free continuation only (BAEE refinement).

## Timeline

| Phase | Duration | Compute | Gate |
|-------|----------|---------|------|
| Dataset construction (D1 labels, MATH-500 8B, 200 queries × 4–8 prefix checkpoints each) | 2–4 GPU-h | Server A/B or H800 | ≥ 500 (prefix, candidate) pairs with D1 labels |
| Train pair V and triple V (ridge logistic and MLP head) | 30 min CPU | local | AUC ≥ 0.65 on held-out; otherwise stop. |
| Run 4 pilot cells + 3 baselines on n=50 | 8–12 GPU-h | H800 or A100 | any cell beats all 3 baselines on Pareto frontier |
| Escalate to n=200 on winning cells (if any) | 20 GPU-h | H800 | matched-compute head-to-head with reported error bars |
| Write up | 1 week | — | separate manuscript, not Coupling Tax paper |

## Hard constraints

- **Must use online chunked runner** (`iris_online_stage2.py` or derivative). No post-hoc truncation accounting. n_tokens_generated ≡ n_tokens_used.
- **Must not block the Coupling Tax submission.** Coupling Tax paper is frozen and targets NeurIPS 2026. PSV pilot can use spare cycles after the 27B GSM8K IRIS experiment (H800, currently n≈160 of 200) completes.
- **Must report paired McNemar against BAEE and DTSR simultaneously**, not in isolation. Beating only one of them is not sufficient evidence to pursue.

## Not in scope for this pilot

- RL fine-tuning of the main model (that is phase 2, contingent on pilot success).
- Cross-family generalization (R1-Distill-Llama / Mistral-hybrid).
- Full-scale n=1319 GSM8K or n=500 MATH-500 runs.
- Latent probes as primary features (they may enter as auxiliary features to the verifier, never as the decision scalar).

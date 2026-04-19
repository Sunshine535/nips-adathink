# Novelty Check: Idea 9 RaC-CBD

## Verdict
**PREEMPTED** (core Fano-bound-over-CoT formulation already published; derivative-stopping mechanism is a narrow delta that is arguably obvious given the prior art).

## Closest Published Work

1. **Gonzàlez I Català et al., "The Stepwise Informativeness Assumption" (arXiv:2604.06192, 11 Mar 2026).**
   - Similarity: **Theorem 1 is literally the Fano lower bound on CoT misclassification:**
     `P_e(k) >= (H_p(A | Q, C_{1:k}) - log 2) / log(|A| - 1)`.
     This is exactly the RaC-CBD bound with the same variables (conditional answer entropy H(A|Q,chain), answer space size |A|, misclassification probability). They explicitly use this to argue that conditional answer entropy is a "progress variable" and that "as conditional answer entropy decreases and approaches its minimum, further reductions must become progressively smaller."
   - Differentiation: The paper is **theoretical/diagnostic**, validated post-hoc across GSM8K/ARC/SVAMP. It does **not** propose an online early-stopping algorithm, does **not** formulate the `dFano/dt >= 0 for K steps` stopping rule, and does **not** provide a decoding-time system. RaC-CBD could claim algorithmic novelty on top of their Theorem 1 — but that is a narrow increment.

2. **Wan et al., "A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning" (arXiv:2509.21199).**
   - Similarity: Derives `h(Acc) + (1-Acc)·log(|A|-1) >= β - C` with β = H(A|Q,C) and C = H(Y). Explicitly names the "Fano-style accuracy upper bound" and invokes "capacity" language. Uses information demand vs. output capacity framing — essentially the "Reasoning-as-Channel" framing of our idea.
   - Differentiation: Their Fano bound is a **global single-pass** upper bound; it is used for **task decomposition (InfoQA)**, not for per-step CoT stopping. They do not use derivatives, K-step plateaus, or inference-time halting. Still, the "Reasoning-as-Channel / capacity-bounded" framing is no longer novel.

3. **HALT-CoT (ICML 2025 MusIML workshop, OpenReview CX5c7C1CZa).**
   - Similarity: Computes **Shannon entropy over the predicted answer distribution** at each CoT step and stops when it drops below a threshold. Training-free, model-agnostic, monitors answer-space entropy (not token entropy). 15–30% token savings.
   - Differentiation: Threshold on the entropy value itself, not on a Fano-transformed lower bound on error, and not on its time derivative. Uses a single-step threshold, not a K-step plateau condition. The underlying signal (answer-distribution entropy) is effectively the same quantity that feeds the Fano bound, so any performance benefit of Fano vs. raw entropy reduces to a monotone transform of an already-published signal.

4. **Entropy Trajectory Monotonicity (arXiv:2603.18940).** Samples answers at each step, tracks `H_k` over answer distribution; finds shape-over-magnitude — binary monotonicity predicts correctness (OR=2.50). Diagnostic, not online, but explicitly uses **per-step sign of ΔH of answer-space entropy** — extremely close to our "ΔFano >= 0 for K steps" mechanism since Fano is monotone in H.

## Exhaustive Search Log

1. "Fano inequality CoT stopping LLM" → 2509.21199 (Fano upper bound), 2509.14004 (ES-CoT run-length). **2509.21199 owns the capacity framing.**
2. "Fano bound LLM decoding early stopping" → 2509.21199 + AdaEDL. **Fano for LLM reasoning already published.**
3. "answer space entropy CoT stopping" → HALT-CoT, LEASH, Think Just Enough, REFRAIN, 2508.20395. **Crowded.**
4. "info-theoretic lower bound reasoning accuracy" → 2509.21199, IBRO, **2604.06192 (SIA) has the exact Fano form.**
5. "Fano's inequality NN decoding 2025-2026" → only classical refs + 2509.21199.
6. "stepwise informativeness 2604.06192" + "Gonzalez Catala Fano theorem" → confirmed Theorem 1: `P_e(k) >= (H_p(A|Q,C_{1:k}) - log 2)/log(|A|-1)`. **Direct hit on RaC-CBD's central formula.**
7. "provable early stopping reasoning LM" → Statistical Early Stopping (2602.13935, conformal + renewal), CoDE-Stop (2604.04930), ES-CoT. **Provable stopping exists with non-Fano signals.**
8. Lee et al. Token Complexity (2503.01141) → rate-distortion framing for post-hoc compression; no Fano, no stopping. **Adjacent.**
9. Reasoning-as-Compression CIB (2603.08462) → CIB as training objective, not inference. **Different mechanism.**
10. "K consecutive steps entropy stopping" → LEASH (windowed slope), ES-CoT (run-length jump), HALT-CoT (single-step threshold). **K-step Fano-plateau is the only plausibly novel mechanism.**
11. Valid Stopping via Empirical Dynamic Formal Lift (2510.06478) → e-value sequential testing, NOT Fano.
12. DeepConf (2508.15260) → sliding-window bottom-10% token log-prob; distinct signal but the "opposite-direction" framing in our idea is **not unique** — DeepConf's bottom-confidence is also a pessimistic bound.
13. Malinin & Gales (LNPE 2020, 2021) → sequence-level token predictive entropy for UQ; no CoT stopping, no answer-space, no Fano. **Distinct.**

## Remaining Risks

- **Primary risk (severe):** arXiv:2604.06192 (published 5 weeks before our current date, 2026-04-19) contains literally the RaC-CBD bound as Theorem 1 and explicitly interprets conditional answer entropy as a progress variable. Any reviewer who has seen it will mark RaC-CBD as derivative. Our only remaining novelty claim is the **online K-step-plateau stopping rule** applied to this bound.
- **Secondary risk (high):** arXiv:2509.21199 owns the "Fano-style accuracy upper bound for LLM reasoning" + "capacity" phrasing. The paper title "Reasoning-as-Channel: Capacity-Bounded Decoding" reads as a direct re-skin.
- **Tertiary risk (moderate):** arXiv:2603.18940 empirically validates that monotone per-step ΔH on answer distribution predicts correctness — this is essentially the empirical claim underlying our K-step plateau rule. They observe on GSM8K that entropy-plateau chains are less reliable; our "stop when Δ-bound >= 0 for K steps" is a stopping variant of their diagnostic.
- **Combinatorial risk:** HALT-CoT (answer-space entropy threshold) + SIA Theorem 1 (Fano bound on same quantity) + 2603.18940 (monotonicity) = the full RaC-CBD construction via three citations.
- **Rebuttal framing:** "Fano is a *lower bound on error* (opposite direction from upper-bound confidence signals)" does not hold up — Fano is a monotone rearrangement of H(A|Q,C) divided by log(|A|-1); any entropy-threshold method is equivalent modulo a fixed transform and a constant offset.

## Recommendation

**ABANDON** the current positioning. "RaC-CBD as Fano-bound-with-K-step-plateau-stopping" is a thin wrapper around 2604.06192's Theorem 1 plus HALT-CoT's answer-space entropy stopping, and the "Reasoning-as-Channel" frame is owned by 2509.21199.

If the user insists on rescuing the idea, **REVISE** along one of these axes (in decreasing promise):

1. **Shift from stopping to routing/allocation.** Use the Fano bound as a per-instance *budget oracle* (how many tokens at minimum to reach target error) and study calibration — no prior work operationalizes Fano as an allocator.
2. **Go beyond single-model Fano.** The ensemble/multi-model Fano version (|A| + a latent competent hypothesis class) is unexplored; could yield tighter bounds than 2509.21199 via a data-processing argument on the CoT → answer channel.
3. **Attack the |A| estimation problem rigorously.** For open-ended answers, 2604.06192 and 2509.21199 both punt on "effective support." A principled effective-|A| estimator (e.g., via perplexity of a reference LM over candidate answers) could be the real contribution, with stopping as a downstream application.
4. **Empirical "accuracy cliff near the Fano bound" on reasoning budget frontiers** using our TOWN/IRIS data infrastructure — positioned as a *measurement paper* rather than a method paper.

If none of these are viable, ABANDON and move to a different Idea in the pipeline. The core risk is not reviewer novelty skepticism — it is that Theorem 1 of 2604.06192 already exists in print with the exact same variables and interpretation.

# Novelty Check: Idea 1 CTT (Coupling-Tax Tomography)

## Verdict
**WEAKLY NOVEL — leaning CONFIRMED NOVEL.** A parameter-free, layer-wise KL
signal between *paired Think-vs-NoThink forward passes on the same prompt of
the same model* used as a routing gate appears unclaimed. Caveat: every
adjacent building block is individually published, so positioning must center
the *paired cross-mode layer-trajectory* as the contribution, not generic
divergence-based routing.

## Closest Published Work

1. **Think Deep, Not Just Long (arXiv:2602.13517, Feb 2026, Google).**
   Similarity: layer-wise JSD on a frozen LLM, exactly our quantity form.
   Differentiation: JSD is between *intermediate vs final layer of the SAME single forward pass* (logit-lens style), applied *post-hoc to sampled responses* as a ranking metric (DTR). Not pre-generation. Not cross-mode. Orthogonal quantity.

2. **JointThinking (arXiv:2508.03363, Aug 2025).**
   Similarity: pairs Think and NoThink passes on the same prompt.
   Differentiation: disagreement is `Con(q) = 1 iff A(r_t) = A(r_n)` — extracted *final-answer text* match after two *full generations*. No hidden states, no layers, no KL. Compute > 100x heavier than CTT's 2 prefill passes.

3. **AdaptThink (arXiv:2505.13417, May 2025, THU-KEG).**
   Similarity: binary Think-vs-NoThink gate per query.
   Differentiation: RL-trained policy on *query text only*; never consumes a cross-mode delta. Requires SFT+RL on 1.5B. CTT is training-free.

4. **To Think or Not to Think: Reasoning Router (HF blog, Mohseni).**
   Similarity: same task (per-query think/no_think gate).
   Differentiation: trained Qwen3-0.6B / mmBERT-small classifier on query text. Explicitly "no divergence comparisons." The direct query-only counterpart CTT displaces.

5. **LLM Router: Prefill is All You Need (arXiv:2603.20895, NVIDIA).**
   Similarity: uses prefill activations pre-generation.
   Differentiation: single-pass prefill → PCA → SharedTrunkNet, routing *across different models*. Not cross-mode in one model. No KL.

6. **R2R (arXiv:2505.21600).**
   Similarity: token-level routing using pass divergence.
   Differentiation: small-vs-large-*model* next-token disagreement, verifier-annotated, learned router. Not cross-mode in one model.

7. **Dynamic Early Exit (arXiv:2504.15895); BAEE (arXiv:2604.06613).**
   Both monitor *within a single Think decode* (Wait-token transitions; free-continuation polling / PSC). No parallel NoThink pass, no layer-wise KL.

8. **Contrastive Thinking Decoding (OpenReview czozyUMx2M); Thinking by Subtraction (arXiv:2602.18232).**
   Intra-mode contrasts (think vs noisy-think, or vs confidence-masked reference). Not think-vs-nothink cross-mode.

9. **"LLM Already Knows" / "LLMs Encode Difficulty" (arXiv:2509.12886; 2510.18147; 2511.03808).**
   Single-pass linear probes on hidden states for difficulty. Trained classifiers, not parameter-free; no cross-mode KL.

10. **Think-at-Hard (arXiv:2511.08577); ThinkRouter (arXiv:2602.11683).**
    Gate at token-step from current single-pass confidence. No parallel NoThink pass.

## Exhaustive Search Log

- "think nothink divergence KL routing LLM layer-wise" → HF reasoning-router (query-only classifier), Think Deep Not Just Long (within-pass JSD). No cross-mode layer-wise router.
- "cross-mode probing LLM hidden state" → ICR Probe, "Reasoning Models Know When They're Right," Between-the-Layers. All single-pass probes.
- "AdaptThink 2505.13417 features" → confirmed query-only RL, no cross-mode feature.
- "JointThinking 2508.03363 layer-level activation" → confirmed answer-text match `A(r_t)=A(r_n)`; no activations touched.
- "ThinkRouter 2602.11683 / Think-at-Hard 2511.08577 / Dynamic Early Exit 2504.15895" → all single-pass confidence gates; no cross-mode.
- "representation-based difficulty prediction mode-contrast" → LLM Already Knows, LLMs Encode Difficulty — single-mode hidden-state probes, trained.
- "BAEE Detection-Extraction Gap 2604.06613" → free-continuation polling in single Think pass, no cross-mode activation.
- "mechanistic interpretability hybrid reasoning Qwen3 layer-wise" → Thought Anchors study, Qwen3 tech report. No cross-mode KL inference work.
- "linear probes on thinking state" → 2504.05419 and ACL-2025 ICR Probe — both single-pass.
- "contrastive decoding think nothink" → CTD (think-vs-noisy-think intra-mode), Thinking by Subtraction (confidence-masked reference). Neither think-vs-nothink.
- "peak layer divergence middle-layer retrieval generation" → 2510.02091 characterizes shallow=retrieval, mid=reasoning — supports CTT's anchor but proposes no signal.
- "deep thinking settling layer JSD" → Think Deep Not Just Long (Google, Feb 2026) closest — within-pass layer JSD, not cross-mode.
- "Demystifying Hybrid Thinking 2510.12680" → training-level study; no activations, no routing signal.
- "Cross-Model Disagreement 2603.25450" → cross-*model* perplexity/entropy on answer, not cross-*mode* layer-wise.
- "coupling tax tomography LLM" → no hits for exact term; Transformer Block Coupling (Jacobian metric) is unrelated.

## Remaining Risks

- **DTR scoop (LOW-MODERATE).** Reviewer may argue CTT is "cross-mode DTR." Rebuttal: (a) DTR is post-hoc on sampled outputs; CTT is pre-generation. (b) DTR = intra-pass intermediate-vs-final (logit lens); CTT = matched-layer-vs-matched-layer across two passes with different mode tokens. (c) DTR answers "how much layer-wise refinement happened?"; CTT answers "how much does the `<think>` scaffold shift mid-layer state?" Share only the JSD/KL family.
- **JointThinking conflation (LOW).** Both read "pair Think+NoThink on same query." Rebuttal: JointThinking = two *full generations* + text-level answer match; CTT = two *1-token prefill* passes. >100x compute difference. No activations in JointThinking.
- **MI workshop noise.** ArXiv coverage through Apr 2026 is clean; NeurIPS-MI / ICML-MI 2025-2026 workshop papers could contain an overlap we missed. Residual risk only.
- **"Parameter-free" framing.** Reviewer may call "1 threshold = 1 parameter." Use "no learned parameters, calibration-only" (mirrors NoThinking / AdaptThink terminology).
- **No blog/tweet preemption found** — only HF reasoning-router blog exists and it explicitly does not use activations.

## Recommendation

**GO** with minor positioning revisions:

1. **Title reframe**: emphasize the paired cross-mode layer-trajectory. Candidate: "Coupling-Tax Tomography: Paired-Mode Layer-Wise KL as a Training-Free Think/NoThink Router."
2. **Pilot additions**:
   - Baseline DTR (Think Deep Not Just Long) — predicted to lose because DTR requires sampling responses first while CTT is pre-generation.
   - Baseline JointThinking — same direction but 100x compute; selling point is matching its routing AUC at <1% cost.
   - Baseline AdaptThink gate on the same items — show CTT matches without RL fine-tune.
3. **Mechanistic evidence plot**: KL-per-layer curve split by (nothink-correct/think-correct) vs (nothink-correct/think-wrong). If mid-layer peak is visibly larger for coupling-tax queries, the "coupling theorem" framing holds; else pivot to "cross-mode KL trajectory is the new primitive" and drop mid-layer specificity.
4. **Fallback if pilot AUC < 0.70**: subtract a "nothink-vs-nothink" self-KL control to isolate mode-specific component. This control is itself novel and defensible.

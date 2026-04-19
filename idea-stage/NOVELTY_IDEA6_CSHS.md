# Novelty Check: Idea 6 CSHS

## Verdict
**WEAKLY NOVEL (with significant preemption risk)**

The specific mechanism — orthogonal Procrustes on 100 (source, target) activation pairs, then injecting `λ·W·h_8B` into the target residual stream at inference for the first 128 tokens, applied cross-scale same-family (8B→27B) for reasoning — has not been published as a single unified method. However, every ingredient is published, and two papers come uncomfortably close: **Patchscopes** (cross-model same-family hidden-state patching with learned affine maps) and **Atlas-Alignment** (orthogonal-Procrustes-based steering across LLMs). The framing must emphasize CSHS's query-specific live activations and reasoning-accuracy objective to avoid being scooped.

## Closest Published Work

- **Ghandeharioun et al., "Patchscopes"** (arXiv:2401.06102, ICML 2024). Does cross-model patching between same-family different-scale LLMs (Vicuna 7B→13B, patching early layers) using **learned affine maps** (Tuned-Lens-style). Similarity: exactly the cross-scale same-family hidden-state injection primitive. Differentiation: purpose is **interpretability / decoding representations into natural language**, NOT improving reasoning accuracy. They explicitly frame it as "using a more capable model to explain representations of a smaller model." No Procrustes; no GSM8K; no first-K-token steering; no λ scalar calibration.

- **Puri et al., "Atlas-Alignment: Making Interpretability Transferable Across Language Models"** (arXiv:2510.27413, Oct 2025). Uses **orthogonal Procrustes** to translate between two LLM latent spaces, then steers target generation via translated concept vectors. Similarity: Procrustes + steering across LLMs is their recipe. Differentiation: (a) steering source is a **precomputed concept vector from a labelled atlas**, not the source model's **live hidden state on the current query**; (b) direction is subject-model ←→ atlas-model, not cross-scale same-family cascade; (c) tasks are safety/concept steering, not math reasoning; (d) all experiments within Llama 3.1 8B variants — no 8B→27B.

- **Liu et al., "Tuning Language Models by Proxy" / Proxy-Tuning** (arXiv:2401.08565, ICLR 2024). Small model shifts large model's **logits**, not hidden states. Differentiation: logit-space, requires a tuned+untuned small-model pair (difference vector), different mechanism entirely.

- **Dutta et al., "Activation Space Interventions Can Be Transferred Between Large Language Models"** (arXiv:2503.04429, ICML 2025). Closest in "cross-model intervention transfer" framing. Uses **autoencoders** (not Procrustes), tests safety/refusal/backdoors (not reasoning), same-family different-scale Llama/Qwen/Gemma. Differentiation: autoencoder vs. closed-form Procrustes, safety vs. reasoning, full-activation replacement vs. residual-additive first-K steering.

- **Pres et al., "Transferring Linear Features Across Language Models with Model Stitching"** (arXiv:2506.06609, NeurIPS 2025). Affine residual-stream maps between different-scale LLMs. Differentiation: used for **SAE/probe/steering-vector transfer** (interpretability-ops reuse), not for per-query inference steering of target reasoning.

## Exhaustive Search Log

1. `"Procrustes alignment hidden states two LLMs inference-time steering"` → AlignEZ, ODESteer, PITA — all same-model steering. No cross-model-Procrustes hit.
2. `"cross-model activation transfer inference small guide large hidden state"` → Dutta 2503.04429 (autoencoder, safety), LLM Modules 2502.08213 (trained cross-attention).
3. `Atlas-Alignment 2510.27413` → Procrustes YES + steering YES, but source=atlas concept vector, task=safety. **High risk, distinct mechanism.**
4. `Dutta 2503.04429` → Autoencoders (affine tested & rejected), same-family scale transfer, safety tasks, no Procrustes.
5. `"orthogonal Procrustes" residual stream cross-scale inference` → Atlas-Alignment + "When Embedding Models Meet" 2510.13406 (retrieval interoperability, not steering).
6. `stitching cross-scale Bansal` → Bansal 2021 (vision), Pres 2506.06609 (LLM residual affine, but for SAE/probe transfer — not per-query steering).
7. `Patchscopes cross-model` → 7B→13B with learned affine maps, purpose is **interpretability decoding**, not target-accuracy optimization.
8. `Speculative Thinking 2504.12329` → Large→small via token-level delegation. No hidden-state transfer, opposite direction.
9. `Proxy-Tuning 2401.08565` → Logit-space, not hidden state; needs tuned+untuned small-model pair.
10. `Relative representations Moschella 2209.15430 / Maiorca 2311.00664` → Vision + classification stitching; not generative LLM reasoning.
11. `Merullo 2209.15162` → Image→LLM single linear projection soft-prompt; LLM→LLM unaddressed.
12. `Collab 2503.21720 / Co-LLM 2403.03870` → Token-level multi-model, not hidden-state.
13. `ASM / LatentSeek / AdaRAS 2601.19847` → All same-model reasoning steering.
14. `KV Cache Steering 2507.08799` → Same-model only.

## Remaining Risks

**High-priority rebuttal hazards:**
1. **Patchscopes scoop.** "This is Patchscopes applied to reasoning." Rebuttal: (a) Patchscopes is for **interpretability** (decode small-model state via big model); we optimize **target accuracy**; (b) we use closed-form Procrustes vs. their learned affine; (c) additive first-K steering vs. full replacement; (d) we calibrate λ.
2. **Atlas-Alignment scoop.** "This is Atlas-Alignment's steering." Rebuttal: Atlas source = precomputed **atlas concept vector** (offline); CSHS source = **live hidden state on current query** (online). Atlas = "steer toward concept X"; CSHS = "condition on small model's query-specific plan." Task domain and direction also differ.
3. **Dutta (ICML 2025).** "They already showed cross-model intervention transfer." Rebuttal: autoencoders (not Procrustes), safety (not reasoning), static refusal vectors (not live query signal).
4. **Proxy-Tuning.** "Small-guides-large is proxy-tuning." Rebuttal: Proxy-Tuning is logit-space with tuned+untuned pair; CSHS is residual-stream with a single untuned forward pass — orthogonal mechanisms.
5. **Concurrent scoop.** Given Atlas-Alignment (Oct 2025) + Pres-Stitching (Jun 2025) occupy this space, concurrent ArXiv within 90 days cannot be excluded. Mitigation: lean on reasoning-accuracy + coupling-tax framing as our unique hook.

**Subtle risks:** Moschella 2022 "relative representations" reviewers may treat Procrustes-between-networks as folklore; frame the novelty around the *application* (cross-scale cascade for reasoning), not the primitive.

## Recommendation

**GO with REVISE on positioning.** The core mechanism is sufficiently novel in its *combination* (cross-scale + Procrustes + live-query + first-K-token + reasoning-accuracy objective), but the individual ingredients are well-known. Positioning must lead with:

1. **Mechanism differentiator**: "CSHS is the first method to use the *source model's live hidden state on the current query* as a steering source across scale, closed-form via Procrustes, applied additively for the first K tokens only." Contrast explicitly with Patchscopes (inspection-purpose, affine, not Procrustes), Atlas-Alignment (concept-atlas source, not live query), and Proxy-Tuning (logit-space).
2. **Empirical differentiator**: must show CSHS beats (a) Patchscopes-style affine-map injection, (b) Proxy-Tuning, (c) no-steering 27B-alone at matched 27B budget. Any of these missing = reviewer ammo.
3. **Theoretical hook**: coupling-tax absorption framing (small model plans, big model executes) is unique to our paper and converts CSHS from a "trick" to a principled answer to the scale-coupled tax. Keep this framing central.
4. **Pilot must include**: (i) ablation vs. learned affine map (isolate Procrustes contribution), (ii) λ sweep showing optimum ≠ 0 and ≠ 1, (iii) first-K vs. all-token steering (isolate the first-128-token design), (iv) 8B→27B cross-scale vs. 8B→8B same-scale (isolate cross-scale benefit).

**Do NOT claim** "first cross-model steering" or "first Procrustes-between-LLMs" — both are already published. **DO claim** "first query-conditional plan-steering across scale for reasoning, using closed-form Procrustes and first-K residual injection."

If 2-GPU-h pilot shows <1.5pp gain or requires λ-tuning per-query, **DOWNGRADE to ABANDON** — the preemption risk is too high to defend a marginal result.

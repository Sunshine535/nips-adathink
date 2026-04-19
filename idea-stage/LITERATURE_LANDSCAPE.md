# External Literature Landscape — Thinking Tax / Coupling Tax Direction

Phase 1 external scan, ≤2500 words. Date: 2026-04-19. Scope: published/on-arXiv work 2024-01 to 2026-04, no internal repo material.

## §1 Landscape (8 directions)

### D1. Test-time compute for reasoning LLMs (foundation)
- **Wei et al. 2022** "Chain-of-Thought Prompting" (2201.11903). Founding CoT paper. Established that longer reasoning improves accuracy at moderate budgets.
- **Snell et al. 2024** "Scaling Test-Time Compute Optimally" (2408.03314). Shows test-time compute can outperform scaling model params; introduces PRM + adaptive-update categorization. Difficulty-aware budget allocation is their central finding.
- **Muennighoff et al. 2025** "s1: Simple Test-Time Scaling" (2501.19393). Introduces **budget forcing** — append "Wait" to extend, or truncate with `</think>` to shorten. 1k-sample SFT + forcing matches o1-preview. The canonical "controlled CoT length" baseline every follow-up compares to.
- **Guo et al. 2025** "DeepSeek-R1" (2501.12948). RL-trained long-CoT recipe. Sets up the distillation corpus used by most efficient-reasoning follow-ups.

### D2. NoThink-style / fixed-budget crossover (MOST OVERLAPPING)
- **Ma, He, Snell, Griggs, Min, Zaharia 2025** "Reasoning Models Can Be Effective Without Thinking" (2504.09858, Apr 2025). Core overlap with our Thinking Tax finding. Prefills a dummy `<|beginning_of_thinking|>Okay, I think I have finished thinking.<|end_of_thinking|>` block on DeepSeek-R1-Distill-Qwen-32B. Shows NoThinking ≥ Thinking at budgets ≤ ~3000 tokens on AIME24/25, AMC23, OlympiadBench, MiniF2F, ProofNet, LiveCodeBench. Key result: 51.3 vs 28.9 on AMC23 at 700 tokens. **Parallel scaling** (best-of-N with self-certainty / majority) gives 9× lower latency than Thinking. Their §3.2 footnote tests 7B / 14B / 32B — "both exhibit similar behavior." Appendix B.2 only has AIME numbers for smaller scales.
- **Wu et al. 2025** "Thinking with NoThinking Calibration / JointThinking" (2508.03363). Generates Thinking + NoThinking answers in parallel, triggers extra Thinking only on disagreement. ICL paradigm, not training-based.
- **Zhang et al. 2025** "AdaptThink" (2505.13417). Trains R1-distill-1.5B with RL to choose Think vs NoThink per query; -53% tokens, +2.4% accuracy on math.
- **Wang et al. 2025** "CoThink" (2505.22017). Instruct model drafts outline, reasoning model fills in. 22% token reduction.

### D3. Adaptive / budget-aware thinking
- **Budget Guidance** (2506.13752). Gamma-distribution predictor over remaining length, CDF-guided decoding.
- **Elastic Reasoning** (2505.05315). Splits Thinking-phase and Solution-phase budgets independently; GRPO-trained for truncation robustness.
- **DAST** (2503.04472); **SelfBudgeter** (2505.11274); **DART** (2511.01170); **DiffAdapt** (2510.19669); **Think in Blocks** (2508.15507) — all RL/SFT for per-query budget prediction.
- **"Increasing the Thinking Budget is Not All You Need"** (2512.19585, Dec 2025). Argues budget scaling is non-monotonic; SC dominates.
- **Plan-and-Budget** (2505.16122). BAM analytic model for sub-question budgeting.

### D4. Latent / implicit reasoning (orthogonal)
- **Coconut** (2412.06769); **CODI** (2502.21074); **Heima** (2501.19201); **Quiet-STaR / Fast Quiet-STaR** (2403.09629, 2505.17746).
- **ThinkRouter** (2602.11683) — routes to latent vs discrete per-step by confidence.
- **Think-at-Hard** (2511.08577) — latent iteration only at hard tokens.

### D5. Confidence-based routing / early stopping
- **Dynamic Early Exit** (2504.15895). Exits at transition points; -31–43% tokens, +1.7–5.7% accuracy.
- **Think Just Enough** (Shannon entropy; 2510.08146). 25-50% savings.
- **DeepConf** (2508.15260). Lowest-group-confidence early stop + confidence-weighted aggregation. 99.9% AIME25 with -84.7% tokens on Qwen3 / GPT-OSS.
- **CISC** (2502.06233); **SeerSC** — confidence-weighted SC, 46% sample reduction.
- **BAEE / Detection–Extraction Gap** (2604.06613, Apr 2026). Critical recent — see §4.

### D6. Inverse scaling / overthinking
- **"Inverse Scaling in Test-Time Compute"** (2507.14417). Claude / o-series: longer thinking → worse performance. Five failure modes. Model-**length** inverse scaling, not model-size.
- **"Illusion of Thinking" / rebuttal** (2506.06941; 2506.09250). Reasoning collapse at Tower-of-Hanoi N=7 — argued to be context truncation.
- **"Mirage of Test-Time Scaling"** (2506.04210); **OptimalThinkingBench** (2508.13141); **Knowledge-Intensive TTS ineffective** (2509.06861); **"When More Is Less"** (2502.07266).

### D7. Chain compression / distillation
- **TokenSkip** (2502.12067); **Chain of Draft** (2502.18600, 7.6% of tokens); **Sketch-of-Thought** (2503.05179); **L1** (2503.04697); **O1-Pruner** (2501.12570); **LightThinker** (2502.15589); **Compressed CoT** (2412.13171).
- **Token Complexity** (Lee et al. 2503.01141). Theoretical minimum CoT length per problem.

### D8. Parallel scaling / verifier-guided
- **Large Language Monkeys** (2407.21787); **ParaThinker** (2509.04475, +12.3% / +7.5%, 7.1% latency overhead).
- **ThinkPRM** (2504.16828); **BiPRM** (2508.01682); **R-PRM** (2503.21295); **Early Rejection Partial Reward** (2508.01969, 9× FLOPs reduction).
- **Self-certainty best-of-N** (2502.18581) — verifier-free selector used by NoThinking.
- **Latency-Aware TTS** (2505.19634, 2509.09864).

## §2 Recurring limitations (what the field has NOT solved)

1. **No decomposition of the budget–accuracy curve into interpretable terms.** Everyone shows empirical crossover plots; no one derives `Acc_think(b) = F_L(b)·α_c + (1-F_L(b))·α_t` (natural-stop CDF × completed accuracy + truncation-penalty). This is the analytic lever you have that they don't.
2. **Model-scale axis is under-explored.** NoThinking (2504.09858) explicitly scopes to 32B primary; 7B/14B get only AIME numbers in appendix. Inverse Scaling (2507.14417) tests model length, not size × length. **No paper jointly sweeps (model-size × budget) to show the tax grows with scale.** Your 2.8× scaling of the tax 8B→27B is unclaimed.
3. **Qwen3-hybrid thinking axis is green-field.** Most NoThink-style work uses R1-Distill models; Qwen3's native hybrid mode (2505.09388) is tested only in DeepConf (2508.15260) and JointThinking (2508.03363), neither of which does a clean Think vs NoThink fixed-budget sweep with scale.
4. **Decoupled answer extraction is narrow.** BAEE (2604.06613) does free-continuation polling; Elastic Reasoning (2505.05315) trains for truncation robustness. **No paper feeds a Thinking-mode truncated chain as a prefix into the *same model's NoThink mode* to extract the answer.** This Stage-3 trick in the internal repo is externally uncovered.
5. **IRIS-style entropy stopping consistently underdelivers.** "Think Just Enough" (2510.08146) claims 25-50% savings; DeepConf (2508.15260) is the only scheme matching NoThinking-level savings. No published explanation of why entropy-only stopping fails — your 0/200 null result has no counterpart.
6. **No information-theoretic lower bound on fixed-budget accuracy.** Lee et al. (2503.01141) define "token complexity" heuristically. No Fano-type or rate-distortion derivation linking budget `b` to achievable accuracy under a chain-length prior.
7. **Inverse scaling work is fragmented.** Papadimitriou (2507.14417) shows length-inverse-scaling; "Illusion of Thinking" (2506.06941) shows complexity-collapse. Nobody has tied them to the Thinking-mode *structural truncation risk* the NoThink decomposition suggests.
8. **Every claim of "tax" in the literature is a different thing.** "Safety Tax" (2503.00555) is alignment-vs-reasoning; "Price of a Second Thought" (2505.22017) is a qualitative survey. None is structural.

## §3 Unexplored intersections (structural gaps)

The intersections below are **joint structural white-space** — not covered by any single paper from §1.

- **(scale × budget × mode)** joint sweep with inverse-scaling decomposition. Nobody has (Qwen3 8B/14B/32B) × (nothink/think) × (b ∈ [128, 8192]) × (GSM8K/MATH-500/BBH) with a mechanistic decomposition. Your internal work already has this.
- **F_L(b) as first-class object.** The empirical chain-length CDF under natural stopping is never published or used as a budget-allocator oracle. It gives a closed-form prediction of where the crossover happens.
- **Stage-3 decoupled extraction** (your Stage-3 technique: feed Thinking-mode prefix truncated at b into the *same model's NoThink mode*; NoThink produces the answer). BAEE uses free continuations; Elastic Reasoning trains for truncation; nobody does the cheap in-model mode-switch trick.
- **Inverse scaling of the coupling itself** — the tax grows with scale because F_L(b) worsens faster than α_c improves. This would convert a descriptive inverse-scaling observation into a predictive structural law.
- **Fano / rate-distortion bound** for chain length vs accuracy. Given a question distribution with entropy H and a chain-length prior with CDF F, derive a budget-accuracy upper bound and show Thinking mode loses by a quantifiable gap.
- **Dual-pass speculative NoThink→Think.** Speculative Thinking (Yang et al. 2504.12329) goes small→large. Nobody has done a NoThink→Think speculative cascade where NoThink proposes and Think verifies only on disagreement (JointThinking does ICL version but not speculative drafting).

## §4 Preemption risk assessment

Ranked by severity.

**RED — direct claim overlap with our core result:**

- **arXiv 2504.09858** — Ma, He, Snell, Griggs, Min, Zaharia, "Reasoning Models Can Be Effective Without Thinking" (Apr 2025, UC Berkeley + AI2 / Databricks). *Risk: very high.* They already own "NoThinking ≥ Thinking at controlled token budget on math/code/theorem-proving" — our cleanest claim is no longer novel in isolation. Our differentiators that remain unpublished: (a) Qwen3 hybrid-mode testing across 8B/14B/27B; (b) inverse-scaling quantification of the tax with model size; (c) analytical F_L(b)·α_c + (1-F_L(b))·α_t decomposition; (d) Stage-3 in-model decoupled answer extraction.

**ORANGE — claims an adjacent piece of our story:**

- **arXiv 2604.06613** — Wang, Zhu, "The Detection-Extraction Gap" (Apr 8 2026, U Chicago + Imperial). *Risk: high for Stage-3.* Explicitly evaluates Qwen3-8B / 32B (Think and NoThink), on MATH-500, GPQA-Diamond, HumanEval. Reports 52-88% of CoT tokens come after the answer is recoverable. BAEE uses free-continuation polling + PSC threshold; 70-78% token reduction, +5.8 pp on Think models. **They stop short of the exact feed-truncated-thinking-into-NoThink trick** but they own the phenomenon name and the empirical premise. This is the paper most likely to scoop decoupled-extraction.
- **arXiv 2507.14417** — Gema, Hosking, et al., "Inverse Scaling in Test-Time Compute" (Jul 2025). *Risk: medium.* They own the *length-inverse-scaling* label on Claude/o-series; they do not own *model-size-inverse-scaling of the thinking-nothink gap*. Our 2.8× scaling (8B → 27B) in GSM8K tax is a different claim, but any reviewer will ask for the contrast.
- **arXiv 2505.13417** — AdaptThink (May 2025). *Risk: medium.* Learns a Think/NoThink gate via RL. Our TOWN cascade would be claimed to be "the training-free deployment of AdaptThink's insight" unless we emphasize the analytic decomposition.
- **arXiv 2512.19585** — "Increasing the Thinking Budget is Not All You Need" (Dec 2025). *Risk: low-medium.* 4-page paper; qualitative claim that SC beats raw budget scaling. Overlaps in spirit but no analytical framework.

**YELLOW — overlapping framing but different mechanism:**

- **2508.03363** JointThinking; **2505.22017** CoThink; **2510.07364** "Base Models Know How to Reason"; **2510.06052** MixReasoning; **2506.08343** "NoWait"; **2508.15260** DeepConf; **2508.12140** Medical Thinking Budget. Each overlaps in one dimension (adaptive routing, dual-model, mode-switching, tax-on-one-benchmark) but none combines scale × mode × budget × decomposition.

**GREEN — not preemption, but sets expectations:**

- **2503.16419** "Stop Overthinking" (TMLR 2025 survey). Canonical survey; our work will be slotted into §6 of any follow-up survey of this paper.
- **2507.02076** "Reasoning on a Budget" (survey). Same.

**Bottom line on novelty.** The bare "nothink beats thinking at fixed budget" claim is preempted by 2504.09858. What remains unclaimed and defensible:
1. **Scale-coupled tax decomposition** `Acc_think(b) = F_L(b)·α_c + (1-F_L(b))·α_t` as a predictive law across 8B–27B.
2. **Inverse scaling of the tax with model size** (quantitative, not just qualitative).
3. **Natural-stop CDF F_L(b) as an oracle** for per-query budget allocation.
4. **Stage-3 in-model decoupled extraction** (Thinking-prefix → same-model NoThink mode).
5. **Explanation of why IRIS-entropy stopping fails** while BAEE's free-continuation polling succeeds.
6. **Information-theoretic lower bound** (Fano/rate-distortion) linking `b`, chain entropy, and Acc.

## §5 Top 5 unclaimed-frontier areas (best-paper candidates)

Ranked by novelty × depth × fit to our coupling-tax foundation.

**F1. The Coupling Tax as a closed-form structural law.**
Derive `Acc_think(b) = F_L(b)·α_c + (1-F_L(b))·α_t` with F_L the empirical chain-length CDF under natural stopping, measure F, α_c, α_t for each (model, benchmark, mode), and show the decomposition predicts the crossover with Thinking. Combine with a Fano bound: `1 - Acc_think(b) ≥ H(Y|X) / log|Y| - I(Y; CoT_{:b}) / log|Y|`. Convert descriptive inverse-scaling of the tax into a predictive structural result. Nothing in §1 does this.

**F2. Natural-stop oracle budget allocator.**
If F_L(b) is the chain-length CDF, then `b*(q) = F_L^{-1}(1-ε)` per query is the oracle budget. Learn an approximator of F_L from the prompt alone (no full rollout). Demonstrate it matches the oracle within 1pp while cutting tokens. This is the mechanistic explanation of why "learned allocator" (the internal attempt) saturates at 46.6% — the upper bound is the F_L-oracle; anything above that requires a decomposition shift, not better allocation.

**F3. Stage-3 Decoupled Extraction as a phase-transition phenomenon.**
Show that truncated-Thinking-prefix → same-model-NoThink-mode recovers 61pp on hard queries *because* the Thinking prefix encodes the commitment distribution (detection gap of BAEE 2604.06613) and the NoThink mode applies the answer-format prior cleanly. Frame as a structural property of hybrid-mode models (Qwen3 specifically). Connect to the "detection-extraction gap" phenomenon but with a *mode-switch extractor* rather than free continuation.

**F4. Inverse-Scaling Law of the Thinking Tax.**
Quantitatively characterize `Tax(s) = α_c - Acc_think(b)` as a function of model size s, and show `Tax(s) ≈ γ · log(s)` (or similar) on Qwen3 8B/14B/27B and a second family (DeepSeek-R1-Distill 1.5B/7B/32B/70B). Explain mechanistically via F_L(b; s) shift (larger models produce longer chains, F_L gets more right-heavy → higher truncation probability at fixed b). This is a scaling law paper with an inverse sign.

**F5. Speculative NoThink-drafted Thinking cascade (training-free).**
NoThink proposes cheap answer; Thinking verifies only when: (a) answer-entropy across k NoThink drafts > τ, or (b) self-certainty falls below threshold. Builds on Speculative Thinking (2504.12329) but inverts the direction (cheap → expensive only on disagreement). Pair with the F_L(b)-predicted budget for the fallback Thinking pass. This is the deployable artifact following from F1–F4.

## §6 Surprising negative results / unresolved puzzles from 2025-2026

1. **LiveCodeBench is a NoThink-killer** (2504.09858 §3.3, Figure 6). Thinking beats NoThinking even at low budget — their explanation is "disabling thinking box doesn't significantly reduce tokens." Unresolved: what structural property of coding separates it from math? Potential gap for us: maybe Thinking chains for code encode state (variables, type constraints) that the NoThink mode cannot reconstruct without the `<think>` block.
2. **F_L-diversity asymmetry** (NoThinking Table 1). NoThinking's answer-entropy std is consistently lower than Thinking's; the authors "hypothesize" this explains pass@k gains but don't formalize. Our `F_L(b)·α_c + (1-F_L(b))·α_t` decomposition predicts exactly this.
3. **Commitment-dynamics p-value flip** (2604.06613). In Think mode, model size has *no* effect on commitment point (p=0.76); in NoThink mode, scale matters (p<0.001). This is a puzzle — Thinking seems to "wash out" scale benefits, exactly matching our tax-grows-with-scale observation but never connected.
4. **Entropy-only stopping fails in deployed settings** (IRIS-style). "Think Just Enough" (2510.08146) only claims 25-50% savings; DeepConf's much higher 84.7% requires *group confidence*, not sequence entropy. No published paper diagnoses why single-sequence entropy is a poor stop signal. Our 0/200 IRIS trigger rate is *concordant* with DeepConf's need for group signal, but nobody has published this observation.
5. **"Illusion of the Illusion of Thinking"** (2506.09250). Reasoning collapse at N=7 Tower-of-Hanoi is attributable to context-length truncation, not reasoning failure. This implies Thinking mode failures at high budget are *output-formatting* failures, not *reasoning* failures — which is exactly what Stage-3 decoupled extraction exploits.
6. **"Base Models Know How to Reason, Thinking Models Learn When"** (2510.07364). Claims 91% of the thinking-model performance gap is recoverable by steering only 12% of tokens. Strong implicit support for our coupling view but they frame it as "bottom-up activation"; we can reframe as "thinking boxes waste 88% of computation that was never needed."

---

**File written: `/home/tarkoy/nips/nips-adathink/idea-stage/LITERATURE_LANDSCAPE.md`.**

Biggest gap: **nobody has published a closed-form decomposition `Acc_think(b) = F_L(b)·α_c + (1-F_L(b))·α_t` linking natural-stop chain-length CDF to budgeted accuracy, nor shown that this tax scales inversely with model size**; this is the structural NeurIPS-level story remaining after NoThinking (2504.09858) took the basic nothink-beats-thinking claim.

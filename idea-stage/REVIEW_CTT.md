# AC Review — Coupling-Tax Tomography (CTT)

**Reviewer role**: Senior Area Chair, NeurIPS 2026
**Decision brief**: Stress test before 40 GPU-h pilot commit
**Reviewed**: CTT one-liner + theory anchor + internal evidence + preemption set

---

## 1. Score: **5.5 / 10** (Borderline Weak Accept — conditional on pilot)

The idea is genuinely clever and theoretically grounded, and the paired-mode cross-layer KL appears to be an unclaimed primitive. But the method leans on two fragile assumptions (that the KL signal is *mode-specific* rather than *style-specific*, and that a "mid-layer peak" is universal across scales) which, if violated, collapse the contribution to a noisy re-skin of existing routers. Current internal evidence (coupling-tax phenomenon, 27B GSM8K Δ=10.5pp) motivates the *need* for a router, not the *efficacy* of CTT specifically. Best-paper potential exists only if the mechanistic story holds crisply on ≥3 scales.

---

## 2. Top 3 Strengths

1. **Theoretically principled, not a post-hoc regularity.** CTT is the *only* signal in the competing set (AdaptThink, JointThinking, DTR, BAEE, R2R) that is directly derived from a validated quantitative decomposition: `Acc_think(b) = F_L(b)·α_c + (1-F_L(b))·α_t`. Reviewers who reward mechanism-grounded methods will reward this.
2. **Efficient by design.** 2 prefill passes (no generation) places CTT at ~1% of JointThinking's cost while claiming overlapping routing targets. This is a defensible efficiency-frontier story independent of accuracy gains.
3. **Empirically meaningful target.** 27B GSM8K b=4096 shows nothink beating think by 10.5pp with 22/23 discordant samples favoring nothink (p<1e-5). This is a large, real, McNemar-confirmed gap — genuinely worth routing on, unlike microscopic thinking-tax effects elsewhere.

---

## 3. Top 5 Weaknesses (descending severity)

**W1. Confound: think-token prompt injection, not cross-mode retrieval/generation divergence.**
- **Why it matters**: The two forward passes differ by the `<think>` scaffold tokens in the prompt. KL at the answer position may primarily reflect *prompt-conditional stylistic drift* — identical to what you would see if you injected any structurally distinctive few-shot exemplars. If so, the signal is not specific to coupling tax; it is just prompt-sensitivity, which predicts nothing about accuracy.
- **Minimum fix**: A *length-matched and style-matched* control prompt (e.g., a null-scaffold "think tag" with blank content, or a distractor role prompt). Compute KL(think‖control) and KL(control‖nothink). If the CTT routing AUC drops below either control's AUC, the contribution is effectively dead.

**W2. "Mid-layer peak" is asserted, not validated as scale-invariant.**
- **Why it matters**: Qwen3-8B (36 layers), Qwen3-14B (40), Qwen3-27B (likely ~48-64) have different depth compositions. "Middle layer" is not a universal coordinate; retrieval-vs-generation transitions could occur at relative depths 0.45 in 8B but 0.6 in 27B (or be multimodal). A threshold on "mid-layer KL peak" that does not generalize reduces CTT to a hyperparameter-tuned bespoke router per scale — exactly the weakness AdaptThink avoids.
- **Minimum fix**: Report KL-per-relative-depth curves for all three scales, computed *on the same GSM8K prompts*. If peaks do not align within ±0.05 of relative depth, reframe from "mid-layer" to "aggregate trajectory" and drop the mechanistic claim.

**W3. Anti-correlation / sign-reversal risk, grounded in precedent.**
- **Why it matters**: You already report that IRIS entropy stopping was *anti-correlated* with correctness (0/200 triggered usefully). Cross-mode divergence signals have an analogous risk: a large mid-layer KL could equally plausibly indicate *thinking is productively injecting useful context the nothink pass lacks* — i.e., the opposite routing rule. Without a confusion-matrix-level validation on discordant samples (the 22/23 from 27B GSM8K b=4096), the direction of the threshold is a coin flip.
- **Minimum fix**: On discordant pairs (think-wrong ∧ nothink-correct) vs (think-correct ∧ nothink-wrong), plot KL(think, nothink) densities. Require >0.15 AUC separation on this split, *with sign reported*, before any pilot scaling.

**W4. "Parameter-free" is a framing trick; robustness to threshold is unaudited.**
- **Why it matters**: "1 calibration threshold" is de facto a learned parameter from held-out data. If the threshold is benchmark-specific (GSM8K ≠ MATH-500 ≠ BBH), you have a learned router with calibration cost equal to or exceeding AdaptThink's training set. Reviewers will pattern-match to "no free lunch" and dock novelty.
- **Minimum fix**: A single fixed threshold calibrated on GSM8K must transfer to MATH-500 and BBH with ≤2pp AUC drop. Otherwise, reposition as "calibration-light routing" and quantify calibration budget vs AdaptThink.

**W5. 2 prefill passes ≠ free; batching and memory claims are under-stated.**
- **Why it matters**: At 27B with long prompts, two prefill passes at inference time are real cost. The efficiency story ("<1% of JointThinking") assumes no generation. But compared to a query-only text classifier (HF reasoning-router baseline at 0.6B params), CTT is 10-100× heavier per routed query. A harsh reviewer will demand compute-matched comparison, not just accuracy.
- **Minimum fix**: Report Pareto frontier (accuracy vs wall-clock-ms-per-query) against: (a) query-only classifier, (b) JointThinking, (c) AdaptThink inference, (d) always-think, (e) always-nothink. CTT must dominate some region, not just exist.

---

## 4. Predicted Author Rebuttals & My AC Responses

| Author rebuttal | AC response |
|---|---|
| "CTT beats AdaptThink AUC at zero training cost." | *Conditionally accepted.* Only credible if the threshold transfers cross-dataset AND the mid-layer peak replicates at 8B/14B/27B. Otherwise it's a single-dataset trick. |
| "The coupling-tax theorem grounds CTT mechanistically." | *Partly accepted.* The decomposition identifies *that* a tax exists; it does not predict that mid-layer KL specifically measures it. Reviewer will demand an ablation showing KL-at-peak-layer correlates with α_c − α_t at the *query* level (not aggregate). |
| "Our 27B GSM8K evidence (22/23 discordant) is overwhelming." | *Rejected as standalone.* This motivates routing; it does not prove CTT routes correctly on those 22 samples. I will specifically ask: "What fraction of the 22 samples does CTT correctly send to nothink? Report per-sample." |
| "JointThinking is 100× costlier." | *Accepted as efficiency point, not as novelty point.* JointThinking's routing is exact (answer-match); CTT's is approximate. The question is whether CTT trades accuracy for efficiency *better than JointThinking's answer-match on truncated generations*. Demand a truncated-JointThinking baseline (e.g., 32-token peek). |
| "Think Deep, Not Just Long (DTR) is intra-pass; CTT is cross-pass." | *Accepted on formal novelty.* But DTR's methodology (layer-wise JSD, threshold calibration, answer-position focus) covers 80% of CTT's implementation surface. A reviewer may ask CTT to reproduce DTR as a strong baseline within a single ablation table. |

---

## 5. Failure Scenarios (ranked by probability)

1. **Stylistic-flip confound wins (≈35%)**: KL is dominated by the `<think>` prompt bias on token distributions, not by retrieval/generation phase divergence. Diagnosed by W1's control prompt. Fix: pivot to "nothink-vs-nothink self-KL normalization" (your own fallback — defensible).
2. **Mid-layer is benchmark-specific (≈25%)**: Peak layer differs between GSM8K and MATH-500 by >10% relative depth. Fix: reframe as "global KL-trajectory signal" (e.g., integrated KL, or learned 1-layer linear probe over all layers — but that re-introduces training).
3. **27B layer structure is qualitatively different (≈20%)**: 27B's Grouped-Query-Attention/MoE (if applicable) creates layer-wise KL dynamics that do not match 8B's. Risk is high because you have no 14B data yet as an interpolant.
4. **Sign reversal on coupling-tax queries (≈15%)**: KL is high on queries where think is correct, not on discordant think-wrong/nothink-correct pairs. The IRIS precedent (0/200 triggered) makes this non-trivial. Fix: sign-aware threshold, but this admits the signal is learned, not principled.
5. **No peak exists (≈5%)**: KL is approximately monotonic with depth. The "tomography" branding dies. Fallback to a one-number summary (e.g., final-layer KL) but then CTT ≈ cross-mode DTR at final layer — existing work.

---

## 6. Minimum Viable Paper (MVP) — hard requirements

1. **Three scales, one budget, one benchmark.** Qwen3-8B/14B/27B on GSM8K at b=4096. For each, (a) KL-per-layer curves split by discordant-pair outcome, (b) CTT routing AUC, (c) per-sample routing table for McNemar-discordant pairs.
2. **Cross-benchmark transfer.** Threshold calibrated on GSM8K, evaluated unchanged on MATH-500 and at least one BBH task. AUC must hold within 2pp.
3. **Stylistic-flip control (W1).** At minimum one null-scaffold prompt and report its AUC as a lower bound.
4. **Baseline suite.** DTR (intra-pass JSD), JointThinking (answer-match), AdaptThink (RL gate), query-only classifier, always-think, always-nothink. CTT must Pareto-dominate at least one non-trivial frontier region.
5. **Compute accounting.** Milliseconds/query, FLOPs, memory peak for all baselines on the same hardware.
6. **Sign-of-effect audit.** Explicit evidence that threshold direction is robust across scales and benchmarks (no per-dataset sign flip).

If the pilot cannot deliver MVP items 1–3, **abort**. 40 GPU-h is insufficient to rescue a wrong-sign or scale-collapsing signal post-hoc.

---

## 7. Stretch Experiments for Best-Paper Push

1. **Mechanistic plot**: overlay KL-per-layer with a retrieval-vs-generation probe (e.g., attention-head activation on key entities) to visually demonstrate the coupling-tax transition. This is the "money figure."
2. **Distillation**: train a 0.6B student to predict CTT's routing decision from prompt text only. If it recovers ≥90% of CTT's AUC, you shipped a deployable router *and* a scientific signal. Rare combination.
3. **Cross-family validation**: DeepSeek-R1-Distill-Llama-8B and one Mistral/Llama hybrid-thinker. If CTT transfers, the "coupling tax" concept generalizes beyond Qwen3.
4. **Scaling law for α_c, α_t**: plot α_c − α_t vs model scale to predict when CTT stops helping (e.g., at 70B, gap may close). This is the "theory meets empirics" angle reviewers love.
5. **Token-budget interaction**: demonstrate CTT's routing gain as a function of b ∈ {512, 1024, 2048, 4096}. Coupling tax should peak at some budget — showing this non-monotonicity empirically ties back to theory.

---

## 8. Ablations a Harsh Reviewer Will Demand

- **Cross-mode control**: think vs think-with-distractor-system-prompt. If KL routing still works, coupling-tax framing is wrong.
- **Layer-wise decomposition**: single-layer AUC curves for each layer, plus mean-AUC, plus learned-weighting upper bound. Argues for / against "peak" vs "trajectory."
- **Threshold-sensitivity analysis**: AUC as a function of threshold quantile, with bootstrap CI.
- **Answer-position extraction**: last-prompt-token vs first-generated-token vs average over next-k. Methodological robustness.
- **Ensemble vs single**: top-3 layer KL mean vs argmax-peak. Prevents a reviewer from arguing "peak is cherry-picked."
- **KL vs JSD vs L2 vs cosine**: metric ablation. Usually boring but bullet-proofs against "KL is asymmetric so you inflated the signal."

---

## 9. Alternative Framings if CTT Fails

- **Coupling-Tax Index (CTI)**: drop the router framing. Instead, publish `KL_mid` as a *per-query difficulty-under-think score* correlated with α_c − α_t. Positions CTT as a *measurement tool* for the coupling-tax phenomenon rather than a deployment system. Lower ambition, higher certainty of acceptance.
- **Budget-conditioned CTT**: route b to 256 or 4096 instead of mode-gate. The coupling tax appears specifically at long budgets, so this is mechanistically adjacent and opens a broader story than binary mode choice.
- **CTT-as-telemetry**: propose cross-mode KL as a debugging tool for hybrid-thinking LLM developers, with a benchmark suite. Reframes from algorithm paper to infrastructure paper (appropriate for D&B track).
- **Contrastive distillation**: use CTT scores as training signal for a small router (ties stretch #2 to a full method).

---

## 10. Final Verdict

- **Best-paper candidate?** **Maybe.** The mechanistic framing is genuinely novel and the 27B evidence is strong; but two fragile assumptions (mode-specificity of KL, scale-invariance of mid-layer peak) gate everything. Best-paper requires *both* assumptions surviving + ≥1 stretch experiment (mechanistic plot or distillation).
- **Before 40 GPU-h commit, do**:
  1. A **4 GPU-h mini-pilot** on 27B GSM8K b=4096 *discordant 22/23 samples only*: compute layer-wise KL + null-scaffold control. If CTT AUC on discordant pairs < 0.70 or the null control is within 0.10 AUC, abort and pivot to CTI framing (§9).
  2. Pre-register the threshold-selection protocol and sign convention *before* seeing 14B/27B results. Prevents post-hoc tuning that reviewers will smell.
  3. Allocate the full 40 GPU-h only if mini-pilot confirms the discordant-pair signal AND the null control passes. Otherwise, fall back to Stage-3 decoupled extraction (+61pp) with a honest "simple but correct" framing — which is an easier Weak Accept than a collapsed CTT.

**Bottom line**: Clever idea, real phenomenon, plausible path to acceptance, narrow path to best-paper. Do the mini-pilot. Do not skip the null-scaffold control. Do not trust any aggregate AUC without the per-discordant-sample table.

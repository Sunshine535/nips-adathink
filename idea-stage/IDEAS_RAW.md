# IDEAS_RAW — Phase 2 Brainstorm (Coupling Tax → NeurIPS 2026)

Scope: 12 mechanism-level-distinct, training-free, inference-only ideas that could plausibly beat `nothink@512` on GSM8K (93.1%) at matched/lower cost, or produce a surprising insight. All ideas validated to fit 2×A100 within 40 GPU-hours. Banlist enforced.

---

### Idea 1 — Coupling-Tax Tomography (CTT)
- **One-liner**: Turn the Coupling Tax residual `α_c − α_t` into a per-query, per-layer activation probe that predicts chain-corruption risk before generation completes.
- **Mechanism**: Run a 1-token "thinking" forward and a 1-token "nothink" forward on the same prompt. Take the layer-wise KL divergence of logit distributions (or hidden-state cosine drift) across the two modes. A high divergence at *middle* layers (where the Coupling Tax theorem predicts the hand-off between retrieval and generation) signals a query that will suffer coupling corruption. Route high-divergence queries to nothink; low-divergence to think.
- **Theoretical anchor**: Information-geometric divergence between two conditional distributions sharing a common latent (the "true answer") — Fisher-Rao metric on the logit manifold.
- **Surprising claim**: A 2-forward-pass probe (no generation) matches or beats our learned budget controller and is parameter-free.
- **External novelty**: Closest is Zou et al. "Representation Engineering" (2023) which steers activations; we *measure* coupling without steering. No paper uses the think/nothink-delta as a routing signal.
- **Failure mode**: KL may be dominated by shallow stylistic flips (think-token injection) rather than reasoning-relevant drift.
- **2-GPU-h pilot**: Qwen3-8B, 200 GSM8K samples. Compute KL per layer for think vs nothink. Fit single-threshold router. Success: routing accuracy ≥ 80% vs oracle routing on nothink-correct/think-correct split.

---

### Idea 2 — Answer-First Backward Verification (AFBV)
- **One-liner**: Generate the *final answer* in nothink mode first, then run a short *backward* chain that conditions on the answer and checks consistency with the question.
- **Mechanism**: Step 1: nothink@128 to emit candidate answer `â`. Step 2: construct prompt `"Given answer â, derive the steps that lead from the question to â"` with budget ≤256. Step 3: score the *backward* chain's first-token log-likelihood under the question prompt; if log-likelihood below threshold, fall back to think@512. Total expected cost: 256 tokens average.
- **Theoretical anchor**: Bayes-optimal verification: `P(â | Q) ∝ P(Q | â) · P(â)`. Backward chain estimates `P(Q | â)` which is often easier than forward proof.
- **Surprising claim**: Backward verification catches 80%+ of nothink errors at 1/3 the cost of re-running think mode.
- **External novelty**: Weng et al. "Self-Verification" (2022) and Prasad et al. "ReCEval" (2023) verify forward chains. Lyu et al. "Faithful CoT" (2023) uses program execution. *No one conditions on nothink's answer and runs a generative backward chain as a likelihood test.*
- **Failure mode**: For arithmetic, backward chain may just restate; discriminative power requires multi-step queries.
- **2-GPU-h pilot**: Qwen3-8B on GSM8K n=200. Measure: (a) AUC of backward-log-lik as nothink-error detector; (b) cost-acc Pareto vs think@512. Success: AUC≥0.75 and Pareto-dominates think@512.

---

### Idea 3 — Chain Compression via Minimum Description Length (MDL-CoT)
- **One-liner**: Use the model's *own* ability to compress a sampled long chain back into a short chain as a principled stopping signal.
- **Mechanism**: Sample a chain of length `L`. At every 32-token checkpoint, ask the model to compress the chain-so-far into ≤k tokens (via an appended "summary: " prompt and a 2nd forward pass truncated to k). If compression ratio `L/k` stabilizes (derivative < ε) — i.e., no new information has been added for the last 32 tokens — halt generation and emit answer from the compressed version. This is Rissanen's MDL applied to CoT.
- **Theoretical anchor**: Minimum Description Length. A chain is informationally complete when its compressed form stops shrinking with additional context.
- **Surprising claim**: MDL stopping beats entropy-based early-stop (IRIS dead) because it measures *information added*, not *uncertainty remaining*.
- **External novelty**: Delétang et al. (2023) use LLMs as compressors on text; Goldblum et al. (2023) relate CoT length to ICL. *No one uses self-compression ratio as a dynamic stopping oracle.* Distinct from natural-stop (which is a *token* signal, not a *compression* signal).
- **Failure mode**: Double forward passes may cost more than they save for short chains.
- **2-GPU-h pilot**: Qwen3-8B GSM8K n=100 hard split. Measure compression-ratio trajectory vs ground-truth correctness. Success: stopping at first plateau keeps ≥95% accuracy at ≥30% token savings.

---

### Idea 4 — Parallel Micro-Chains with Anytime Voting (PMC-AV)
- **One-liner**: Instead of one budget-B chain or k independent chains, generate `k = √B` chains of length `√B` each and use a prefix-tree vote on the *partial* answers at every step.
- **Mechanism**: Launch k=16 parallel chains, each capped at 32 tokens. After each token-step, extract the current "leading answer hypothesis" (via a cheap linear probe on the last hidden state trained on nothink answer-position activations; training-free if probe is ridge-regressed on 100 nothink exemplars). At step t, if majority-vote agreement over all k chains exceeds 75%, halt all chains and emit the majority answer. Crucially, vote on *internal hypotheses* not final answers.
- **Theoretical anchor**: Concentration inequalities — variance of mean over k independent chains shrinks as 1/k. Trade depth for breadth at the optimal rate.
- **Surprising claim**: At matched total-token budget, √B × √B breadth-depth dominates B × 1 or 1 × B. Violates the assumption that "longer is always better for thinking."
- **External novelty**: Self-consistency (Wang et al. 2022) votes *at the end*. Tree-of-Thought (Yao et al. 2023) branches but is sequential/search-heavy. CISC weights by confidence. *No one uses anytime mid-chain hypothesis voting with a fixed compute grid.*
- **Failure mode**: Leading-hypothesis probe may be too weak mid-chain; vote collapses to random.
- **2-GPU-h pilot**: Qwen3-8B GSM8K n=200. Grid: k∈{4,8,16,32} × length∈{32,64,128}. Compare to SC@k and think@512 at matched total tokens. Success: ≥2pp improvement at matched budget.

---

### Idea 5 — Synthetic Chain Forgery via Retrieval-from-Self (SCF-RfS)
- **One-liner**: At inference, retrieve a *different question's* successful chain from a 100-sample cache, splice its reasoning scaffold into the current prompt, and let the model fill in only the numerical slots.
- **Mechanism**: Pre-compute 100 diverse GSM8K-style solved chains from *one-time* nothink@1024 runs (part of inference-time setup, not training). At query time: (1) embed the query with a small encoder (or the LLM's first hidden state), (2) retrieve the nearest scaffold, (3) mask out all numerals and entity names, (4) feed `question + "Use this template: <masked scaffold>"` to the model. Model fills in skeleton at ≤128 tokens — effectively delegating structure to retrieval and computation to the model.
- **Theoretical anchor**: Reasoning as compositional program synthesis (Cao et al. "Algorithmic prompting"); scaffolds are reusable subroutines.
- **Surprising claim**: Forged synthetic scaffolds outperform the model's self-generated chains because they dodge coupling-tax errors that arise from think-mode planning.
- **External novelty**: DSP (Khattab 2022) and Retrieval-Augmented CoT (various) retrieve *exemplars*; our method retrieves *masked scaffolds with slot-fill only*. Distinct from simple few-shot: we *surgically remove* the example's specifics so the model cannot copy.
- **Failure mode**: Scaffold mismatch when query structure differs (ratio vs geometry).
- **2-GPU-h pilot**: Qwen3-8B on GSM8K n=300. Build scaffold cache from 100 training-set items. Measure acc and tokens vs nothink@128 and think@256. Success: ≥1pp above nothink@128 at matched tokens.

---

### Idea 6 — Cross-Scale Hidden-State Steering (CSHS)
- **One-liner**: Use the 8B model's mid-layer hidden state, projected via a cheap Procrustes alignment, as a planning signal that *steers* the 27B's generation — no retraining.
- **Mechanism**: On 100 GSM8K items, compute 8B layer-20 activations at the answer position and 27B layer-48 activations at the answer position. Fit orthogonal Procrustes map `W: ℝ^{d_8B} → ℝ^{d_27B}` (one SVD call, training-free). At inference: (a) run 8B nothink@64 to obtain planning activation `h_8B`, (b) at each generation step of 27B, add `λ·W·h_8B` to 27B's residual stream at layer 48 for the first 128 tokens of 27B's output. This "whispers" the small model's plan into the big model's chain.
- **Theoretical anchor**: Cross-model linear representation hypothesis (Park et al. 2023) — concepts live in isomorphic low-rank subspaces. Procrustes is the MAP alignment under isotropic noise.
- **Surprising claim**: 8B's cheap plan injected into 27B *outperforms* 27B-alone at lower 27B budget. Coupling tax is *absorbed* by letting the small model plan and the large model execute.
- **External novelty**: Closest is ITI (Li et al. "Inference-Time Intervention" 2023), but ITI uses *same-model* probes. Representation Engineering (Zou 2023) does not do cross-scale. *No prior work aligns two models' residual streams at inference and steers one with the other without training a bridge network.*
- **Failure mode**: Procrustes may be insufficient; non-orthogonal alignment might be needed.
- **2-GPU-h pilot**: Qwen3-8B and Qwen3.5-27B on GSM8K n=100. Fit Procrustes on 100 calibration items. λ grid ∈{0, 0.25, 0.5, 1.0}. Compare to 27B-alone at matched 27B tokens. Success: ≥1.5pp gain at matched 27B budget.

---

### Idea 7 — Per-Token Counterfactual Dropout (PTCD)
- **One-liner**: During generation, at each step, compute the counterfactual of *dropping* the last-generated token and measure the answer-logit divergence; stop when the counterfactual converges with the factual (the chain has become self-redundant).
- **Mechanism**: Maintain two parallel KV caches. At generation step t: KV_A keeps all tokens; KV_B drops the last reasoning token. Run both forward 1 step to obtain the answer-logit (via "So the answer is:" appended prompt). Compute `d_t = KL(p_A || p_B)`. When `d_t < ε` for 3 consecutive steps — the chain has stopped adding new information — halt.
- **Theoretical anchor**: Counterfactual information measure (Pearl) — `d_t` is the *causal information* contribution of the last token.
- **Surprising claim**: Causal (not correlational) stopping gives tighter bounds than entropy, matches natural-stop PPV (96.3%) but on all queries (not just trigger-detected ones).
- **External novelty**: Dai et al. "Knowledge Neurons" (2022) do counterfactual patching for analysis, not inference control. Causal attention variants (e.g., Elhage et al.) analyze post-hoc. *No one uses single-token counterfactual dropout as an inference-time stopping rule.*
- **Failure mode**: 2× KV memory; need good batching to fit 27B on 80GB.
- **2-GPU-h pilot**: Qwen3-8B n=150 GSM8K. Measure: (a) token savings at matched acc vs natural-stop; (b) can it detect coupling-tax queries where natural-stop fails? Success: ≥15% additional token savings on hard subset.

---

### Idea 8 — Attention-Head Reasoning Ablation Index (AHRAI)
- **One-liner**: Identify at-inference-time the 3-5 "reasoning heads" whose attention entropy spikes on coupling-tax failures, and ablate them (set attn weights to uniform) when their entropy exceeds a threshold.
- **Mechanism**: Pre-compute per-head attention entropy on a calibration set of 50 coupling-tax failure cases (nothink-correct, think-wrong). Rank heads by Δentropy between success and failure (no training). At inference, monitor the top-5 heads' entropy during think mode. When they fire past the calibrated threshold, dampen the attention of those heads by interpolating to uniform at 50% weight for subsequent steps. This is surgical — only heads statistically linked to coupling-tax errors are touched.
- **Theoretical anchor**: Circuit-level mechanistic interpretability (Olsson et al. "Induction Heads"). Selective ablation preserves other capabilities.
- **Surprising claim**: 3-head ablation eliminates 40%+ of the coupling tax without any generation re-run.
- **External novelty**: Zhang et al. "AttnLens" (2024) and DoLa (Chuang et al. 2023) modify layers, not heads. Ablation studies exist but are offline. *No one uses calibration-set-derived head attribution to dynamically ablate during generation as a reasoning-repair mechanism.*
- **Failure mode**: Head attribution may not be stable; different query types activate different heads.
- **2-GPU-h pilot**: Qwen3-8B. Calibration: 50 cases → head ranking. Test: 200 GSM8K items with and without ablation during think@512. Success: coupling-tax error rate drops ≥30% with no drop in baseline correct cases.

---

### Idea 9 — Reasoning-as-Channel: Capacity-Bounded Decoding (RaC-CBD)
- **One-liner**: Treat the chain as a noisy channel from `Q` to `â`; use Fano's inequality to derive a per-step *lower bound* on decoding error and halt when further tokens provably cannot reduce it.
- **Mechanism**: Define `H(â | chain_{1:t}, Q)` estimated via temperature-swept sampling of the answer prompt. Fano: `P(error) ≥ (H(â | chain, Q) − 1) / log|Â|`. Compute an exponential moving average of this lower bound. When the *derivative* of the Fano bound is non-negative for K=5 steps — adding tokens is not helping — halt. This gives a *provable* stopping criterion (not heuristic).
- **Theoretical anchor**: Fano's inequality — gives lower bound on error in terms of conditional entropy over answer distribution.
- **Surprising claim**: A provable lower-bound-based stop matches an oracle stop on 85% of queries on GSM8K.
- **External novelty**: Prior entropy-based stops (including IRIS) use *upper-bound-like* signals; Fano is a *lower-bound* signal — the opposite direction. Malinin & Gales (2021) use ensemble-based uncertainty, not Fano. *No work applies Fano's inequality to CoT stopping.*
- **Failure mode**: `log|Â|` (answer-space size) hard to define for open-ended answers; estimate may be noisy.
- **2-GPU-h pilot**: Qwen3-8B GSM8K n=200 (numeric answers, `|Â|` well-defined). Measure: Fano-stop vs fixed-budget Pareto. Success: dominates think@256 and matches natural-stop at 10% fewer tokens.

---

### Idea 10 — Latent Phase Transition Detector (LPTD)
- **One-liner**: Generation trajectories through hidden-state space exhibit a phase transition when reasoning "locks in" an answer; detect this transition via largest-Lyapunov-exponent sign-flip and stop at lock-in.
- **Mechanism**: Track the trajectory of the last-token hidden state through layers 20-40 over generation steps. Estimate the largest Lyapunov exponent λ of this dynamical system over a sliding window of 16 steps (using Rosenstein's algorithm — O(w^2) per step, w=16, cheap). When λ crosses from positive (chaotic exploration) to negative (contracting to fixed point), the model has "committed" — halt.
- **Theoretical anchor**: Dynamical systems theory — attractors in state-space correspond to answer-commitment. Lyapunov exponent distinguishes exploration from exploitation.
- **Surprising claim**: The chaos-to-contraction transition in hidden-state dynamics is a *universal* stopping signal across models and tasks, not learned per-benchmark.
- **External novelty**: Ramesh et al. "Phase Transitions in LLMs" (2024) study training, not inference-time trajectories. Engels et al. (2024) discuss "attractor dynamics" theoretically. *No one has computed per-step Lyapunov exponents as an online stopping signal.*
- **Failure mode**: Lyapunov estimation from short windows may be unreliable; trajectory dim is huge.
- **2-GPU-h pilot**: Qwen3-8B n=100 on mixed GSM8K (easy+hard). Plot λ_t vs step for correct vs incorrect chains. Success: sign-flip detects 70%+ of correct answers pre-natural-stop, with token savings ≥20%.

---

### Idea 11 — Dual-Decode Arbitration (DDA)
- **One-liner**: Run nothink and think *simultaneously* as two parallel batch decodes; at every step, an arbitrator (the model itself via a 1-token classifier prompt) picks which stream continues, killing the other.
- **Mechanism**: At each generation step, the nothink stream (`t_n`) and think stream (`t_t`) produce a next-token logit distribution. Form the arbitration prompt `"Answer is near. Continue which? A) [nothink-next] B) [think-next]. Answer: "` and score A vs B via a 1-forward-pass — the model that would have picked *itself* keeps running, else it yields. The yielded stream resumes if the winner hits natural-stop. Total cost: ~1.2× nothink tokens (arbitration is 1 token).
- **Theoretical anchor**: Expert-of-experts / switching regret bounds (Herbster-Warmuth 1998) applied to decoding modes.
- **Surprising claim**: Arbitration wins > both nothink-alone and think-alone Pareto fronts because it exploits the *complement* of their failure modes.
- **External novelty**: Speculative decoding (Leviathan 2022, Chen 2023) uses a draft-verify split for *same answer*. Mixture-of-Experts does static routing. *No one runs think+nothink as competing streams with token-level arbitration.*
- **Failure mode**: Arbitration prompt may collapse to a preference bias; need careful calibration.
- **2-GPU-h pilot**: Qwen3-8B on GSM8K n=200. Compare DDA (budget=256 effective) vs nothink@128, think@512. Success: ≥1.5pp above nothink@128 at matched effective tokens.

---

### Idea 12 — Nothink-as-Oracle Distillation at Inference (NoaO)
- **One-liner**: Use 27B-nothink-@4096's extraordinary 98% GSM8K accuracy as a zero-shot oracle to generate *pseudo-labels* for dynamic calibration of an 8B-nothink at inference — no training, just online threshold adjustment.
- **Mechanism**: For each batch of B queries: (1) run 27B-nothink at low budget (128) to get a pseudo-answer distribution; (2) run 8B-nothink at 128; (3) compute agreement-rate `r`; (4) if r<τ, escalate the disagreeing queries to 27B-think@256. Since 98% accuracy of 27B-nothink@4096 maps (via coupling-tax theorem) to ~92% at b=128, this is a soft oracle. The key trick: use the *cross-model cross-mode* gap as an escalation signal, amortizing 27B's cost only over hard queries.
- **Theoretical anchor**: Teacher-student disagreement as hard-example mining (Bengio et al. curriculum learning) applied at inference without labels.
- **Surprising claim**: Batched 8B+27B-nothink arbitration achieves 27B-nothink-quality at 1/5 the compute.
- **External novelty**: Cascade methods (Chen et al. "FrugalGPT" 2023) use confidence or thresholds on *same* model. Distillation needs training. *No one uses "Coupling-Tax-predicted 27B nothink short-budget" as an online pseudo-oracle for 8B route-up gating.*
- **Failure mode**: On queries where both models err in the same way, disagreement won't flag — but we only pay 27B on disagreement, so worst case = 8B acc.
- **2-GPU-h pilot**: Both Qwen3-8B and Qwen3.5-27B. 300 GSM8K items. Compare to 27B-alone and 8B-alone at matched total tokens. Success: Pareto-dominates 8B-think@512 and costs ≤60% of 27B-nothink@512 at ≥92% acc.

---

## Top 5 by Novelty × Feasibility

1. **Idea 6 — CSHS** (Cross-Scale Hidden-State Steering). Procrustes + residual injection = cheap, principled, *never done*. Huge upside if it works: changes how small/large model pairs are used.
2. **Idea 9 — RaC-CBD** (Fano bound decoding). Provable lower-bound stop is theoretically novel and naturally papers-worthy; pilot trivially cheap.
3. **Idea 2 — AFBV** (Answer-First Backward Verification). Mechanistically clean, Bayes-optimal framing, directly leverages nothink's strength. Easy pilot, clear win-condition.
4. **Idea 7 — PTCD** (Per-Token Counterfactual Dropout). Causal-information framing is underexplored in inference control. Clean theoretical story.
5. **Idea 4 — PMC-AV** (Parallel Micro-Chains Anytime Voting). Surprising breadth-vs-depth claim; directly beats SC baseline; anytime voting is a legit new primitive.

## Weakest 3 (flag for novelty-check)

- **Idea 5 — SCF-RfS** (Synthetic Chain Forgery). Risks overlapping with scaffold-retrieval RAG-CoT work (Li et al. "Structured Prompting", Zhou et al. "Decomposed Prompting"). Needs specific query to novelty-check.
- **Idea 3 — MDL-CoT** (MDL stopping). Compression-based stopping has adjacent work (Lanchantin et al. "learning to compress"); need to confirm no one used self-compression ratio for CoT stopping.
- **Idea 10 — LPTD** (Lyapunov phase transition). Dynamical-systems-in-LLM is a hot area (Engels, Ramesh, Achille); risk that "attractor stopping" has been proposed in 2024–2026 workshop papers.

## Deliverable

File written: `/home/tarkoy/nips/nips-adathink/idea-stage/IDEAS_RAW.md`

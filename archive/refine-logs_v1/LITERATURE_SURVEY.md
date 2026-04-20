# Literature Survey: "The Thinking Tax: When Chain-of-Thought Costs More Than It Saves"

**Prepared for:** NeurIPS 2026 Submission  
**Date:** 2026-04-08  
**Scope:** 2022--2026, focused on test-time compute, CoT efficiency, adaptive reasoning, overthinking, and cascade methods  

---

## Table of Contents

1. [Foundational Work: CoT and Test-Time Compute](#1-foundational-work)
2. [Test-Time Compute Scaling](#2-test-time-compute-scaling)
3. [Reasoning Models (o1, R1, Qwen3)](#3-reasoning-models)
4. [Overthinking and Compute Waste](#4-overthinking-and-compute-waste)
5. [Budget-Aware and Efficient Reasoning](#5-budget-aware-and-efficient-reasoning)
6. [Adaptive/Selective Reasoning and Cascade Methods](#6-adaptive-selective-reasoning)
7. [Reasoning Distillation](#7-reasoning-distillation)
8. [Dual-Process Theory (System 1 / System 2)](#8-dual-process-theory)
9. [Surveys](#9-surveys)
10. [Summary: Positioning "The Thinking Tax"](#10-positioning)

---

## 1. Foundational Work: CoT and Test-Time Compute {#1-foundational-work}

### 1.1 Wei et al. (2022) — Chain-of-Thought Prompting Elicits Reasoning in LLMs

| Field | Value |
|-------|-------|
| **Authors** | Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V. Le, Denny Zhou |
| **Venue** | NeurIPS 2022 |
| **arXiv** | [2201.11903](https://arxiv.org/abs/2201.11903) |
| **Key finding** | Few-shot CoT prompting ("Let's think step by step") dramatically improves LLM performance on arithmetic, commonsense, and symbolic reasoning tasks. Performance gains emerge at scale (~100B+ parameters). |
| **Relation to Thinking Tax** | **The paper that started it all.** Our work shows the other side: under fixed token budgets, CoT is a *liability* because the reasoning tokens crowd out answer tokens. Wei et al. assumed unbounded generation; we impose budget constraints that reveal CoT's hidden cost. |
| **Already cited** | Yes (`wei2022chain`) |

### 1.2 Wang et al. (2023) — Self-Consistency Improves Chain of Thought Reasoning

| Field | Value |
|-------|-------|
| **Authors** | Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou |
| **Venue** | ICLR 2023 |
| **arXiv** | [2203.11171](https://arxiv.org/abs/2203.11171) |
| **Key finding** | Sampling multiple diverse reasoning paths and majority-voting the final answer significantly outperforms greedy CoT decoding on GSM8K, ARC, StrategyQA. |
| **Relation to Thinking Tax** | Self-consistency (SC) multiplies the token budget by the sample count. Our SC@8/16 experiments show that under fixed total budget, SC with thinking mode is *worse* than nothink SC because each sample wastes tokens on truncated reasoning. |
| **Already cited** | Yes (`wang2023selfconsistency`) |

### 1.3 Yao et al. (2023) — Tree of Thoughts

| Field | Value |
|-------|-------|
| **Authors** | Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas Griffiths, Yuan Cao, Karthik Narasimhan |
| **Venue** | NeurIPS 2023 |
| **arXiv** | [2305.10601](https://arxiv.org/abs/2305.10601) |
| **Key finding** | Generalizes CoT to a tree structure with search (BFS/DFS) and evaluation, enabling deliberate problem-solving. |
| **Relation to Thinking Tax** | Represents the extreme of test-time compute: multiple branching CoT paths. Our budget-constrained setting shows that even *single* CoT paths are too expensive when budgets are tight, making tree-based approaches even more impractical under constraints. |
| **Already cited** | Yes (`yao2023tree`) |

### 1.4 Lightman et al. (2024) — Let's Verify Step by Step

| Field | Value |
|-------|-------|
| **Authors** | Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe |
| **Venue** | ICLR 2024 |
| **arXiv** | [2305.20050](https://arxiv.org/abs/2305.20050) |
| **Key finding** | Process Reward Models (PRMs) that evaluate each reasoning step outperform Outcome Reward Models (ORMs) for math verification. Released PRM800K dataset. |
| **Relation to Thinking Tax** | PRMs assume reasoning steps *exist* to evaluate. Under tight budgets, thinking-mode chains are truncated mid-step, making step-level verification unreliable. Our nothink mode sidesteps the need for process verification entirely. |
| **Already cited** | Yes (`lightman2023prm`) |

---

## 2. Test-Time Compute Scaling {#2-test-time-compute-scaling}

### 2.1 Snell et al. (2024) — Scaling LLM Test-Time Compute Optimally ⭐

| Field | Value |
|-------|-------|
| **Authors** | Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar |
| **Venue** | arXiv (Aug 2024); reviewed at ICLR 2025 |
| **arXiv** | [2408.03314](https://arxiv.org/abs/2408.03314) |
| **Key finding** | Compute-optimal test-time scaling can make a small model outperform a 14x larger model. Two mechanisms: (1) search via process reward models (best-of-N, tree search) and (2) iterative revision. The optimal strategy depends on problem difficulty. |
| **Relation to Thinking Tax** | **This is our primary foil.** Snell et al. show test-time compute scaling *works* — but they study the **compute-optimal frontier** (unlimited budget, allocate wisely). We study the **budget-constrained regime** where thinking mode *wastes* the finite budget. Our findings are complementary: when budgets are generous, think more; when tight, think less. |
| **Already cited** | Yes (`snell2024scaling`) |

### 2.2 Muennighoff et al. (2025) — s1: Simple Test-Time Scaling

| Field | Value |
|-------|-------|
| **Authors** | Niklas Muennighoff, Zitong Yang, Weijia Shi, Xian Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, Tatsunori Hashimoto |
| **Venue** | arXiv (Jan 2025) |
| **arXiv** | [2501.19393](https://arxiv.org/abs/2501.19393) |
| **Key finding** | Fine-tuned Qwen2.5-32B on ~1,000 curated reasoning traces (s1K dataset distilled from Gemini). Introduced **budget forcing**: suppress end-of-thinking token + append "Wait" to force longer reasoning, or force early stop. Achieved competitive results on competition math. |
| **Relation to Thinking Tax** | **Directly relevant.** s1 shows budget forcing can *improve* performance by forcing more thinking. We show the opposite: under output-token budgets, forcing thinking *hurts* because the model can't emit the answer. s1 controls *thinking length*; we control *total output length* (think + answer combined), revealing a fundamentally different dynamic. |
| **Already cited** | Yes (`muennighoff2025s1`) |

### 2.3 Brown et al. (2024) — Large Language Monkeys: Scaling Inference Compute with Repeated Sampling

| Field | Value |
|-------|-------|
| **Authors** | Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini |
| **Venue** | arXiv (Jul 2024) |
| **arXiv** | [2407.21787](https://arxiv.org/abs/2407.21787) |
| **Key finding** | Coverage (probability of at least one correct answer) scales predictably with number of samples. Repeated sampling is a simple, effective inference-time scaling strategy — but requires a verifier to identify correct samples. |
| **Relation to Thinking Tax** | Repeated sampling multiplies total token cost. Under budget constraints, nothink mode allows *more samples* per budget unit, making it the better base for sampling-based strategies. Our SC experiments demonstrate this. |
| **Already cited** | No — **recommend adding** |

### 2.4 Setlur et al. (2025) — Reasoning Strategy Optimization: A Pareto Framework

| Field | Value |
|-------|-------|
| **Authors** | Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, Aviral Kumar |
| **Venue** | arXiv (Mar 2025) |
| **arXiv** | [2503.04474](https://arxiv.org/abs/2503.04474) |
| **Key finding** | Proposes a Pareto framework for test-time compute that jointly optimizes accuracy and compute cost. Maps out the efficiency frontier across different reasoning strategies. |
| **Relation to Thinking Tax** | **Strongly complementary.** We show that thinking mode is *below* the Pareto frontier at all tested budgets ≤2048 — nothink mode dominates. Their framework can formalize our finding: TOWN operates *on* the Pareto frontier. |
| **Already cited** | Yes (`nvidia2025pareto`) |

---

## 3. Reasoning Models {#3-reasoning-models}

### 3.1 OpenAI (2024) — Learning to Reason with LLMs (o1)

| Field | Value |
|-------|-------|
| **Authors** | OpenAI |
| **Venue** | OpenAI Blog (Sep 2024) |
| **Key finding** | o1 uses extended internal chain-of-thought at inference time, trained via RL. Performance scales with test-time compute. Dramatic improvements on math, coding, science benchmarks. Thinking tokens are hidden from users. |
| **Relation to Thinking Tax** | The canonical example of "thinking mode." Our paper provides empirical evidence that when total output budget is fixed, the o1 paradigm (long internal CoT) is counterproductive for models that expose thinking tokens and share the output budget between reasoning and answer. |
| **Already cited** | Yes (`openai2024o1`) |

### 3.2 DeepSeek-AI (2025) — DeepSeek-R1: Incentivizing Reasoning via RL

| Field | Value |
|-------|-------|
| **Authors** | DeepSeek-AI (Guo, Yang, Zhang, et al.) |
| **Venue** | arXiv (Jan 2025) |
| **arXiv** | [2501.12948](https://arxiv.org/abs/2501.12948) |
| **Key finding** | Purely RL-trained R1-Zero develops emergent reasoning behaviors (self-verification, reflection, long CoT). R1 (RL + cold-start SFT) matches o1 on math/coding. Distilled versions (1.5B--70B) also strong. Open-sourced. |
| **Relation to Thinking Tax** | We use DeepSeek-R1-Distill-Llama-8B as a cross-model-family validation. R1's training explicitly encourages *longer* reasoning chains via RL, making it especially susceptible to the thinking tax under budget constraints. |
| **Already cited** | Yes (`deepseekr1`) |

### 3.3 Qwen Team (2024) — QwQ: Reflect Deeply on the Boundaries of the Unknown

| Field | Value |
|-------|-------|
| **Authors** | Qwen Team (Alibaba) |
| **Venue** | Qwen Blog (Nov 2024) |
| **Key finding** | QwQ-32B-Preview: reasoning model with explicit `<think>` tags, extended CoT, competitive with o1-mini on math benchmarks. |
| **Relation to Thinking Tax** | Predecessor to Qwen3's hybrid thinking mode. QwQ established the `<think>` tag convention used by Qwen3, which is our primary experimental model. |
| **Already cited** | Yes (`qwq2024`) |

### 3.4 Qwen Team (2025) — Qwen3 Technical Report

| Field | Value |
|-------|-------|
| **Authors** | Qwen Team (Alibaba) |
| **Venue** | arXiv (May 2025) |
| **arXiv** | [2505.09388](https://arxiv.org/abs/2505.09388) |
| **Key finding** | Qwen3 series (0.6B--235B) with hybrid thinking architecture: `enable_thinking=True` (thinking mode) vs `enable_thinking=False` (nothink mode) toggled per request. Single model handles both modes. |
| **Relation to Thinking Tax** | **Our primary experimental model.** The Qwen3 architecture with its explicit toggle is the ideal testbed for the thinking tax: same weights, same training, two modes. We show nothink mode dominates at all budgets ≤2048 across 8B, 9B, and 27B variants. |
| **Already cited** | Yes (`qwen3_2025`) |

---

## 4. Overthinking and Compute Waste {#4-overthinking-and-compute-waste}

### 4.1 Chen et al. (2024) — Do Not Think That Much for 2+3=? On the Overthinking of o1-Like LLMs ⭐

| Field | Value |
|-------|-------|
| **Authors** | Xingyu Chen, Jiahao Xu, Tian Tian, et al. |
| **Venue** | arXiv (Dec 2024); ICLR 2025 Workshop ME-FoMo |
| **arXiv** | [2412.21187](https://arxiv.org/abs/2412.21187) |
| **Key finding** | Defines **Loss of Efficiency (LoE)** — overthinking where models over-analyze simple problems. Proposes: (1) self-training on shortest correct reasoning traces, (2) budget-forced inference based on difficulty. On QwQ-32B-Preview: +1.3% accuracy with 48.6% token reduction across 5 math benchmarks. |
| **Relation to Thinking Tax** | **Closest prior work.** They identify the same problem (wasteful thinking) and propose budget-forced solutions. However, they study *thinking length within thinking mode*. We go further: we show that **switching to nothink mode entirely** is better than any budget-forced thinking under tight constraints. Our TOWN cascade subsumes their approach. |
| **Already cited** | Yes (`chen2024overthinking`) |

### 4.2 Gor et al. (2025) — Reasoning Models Can Be Effective Without Thinking ⭐

| Field | Value |
|-------|-------|
| **Authors** | Maharshi Gor, Jena D. Hwang, Faeze Brahman, Arman Cohan, Tushar Khot |
| **Venue** | arXiv (Mar 2025) |
| **arXiv** | [2503.02508](https://arxiv.org/abs/2503.02508) |
| **Key finding** | DeepSeek-R1 with a "No-Thinking" prompt (bypassing `<think>` block) matches or outperforms full thinking on most benchmarks *except* hard math/coding. Budget forcing can cut thinking time by 40% with minimal loss. R1 without thinking still outperforms its base model (V3). |
| **Relation to Thinking Tax** | **Most closely related concurrent work.** Their "No-Thinking" finding on R1 parallels our "nothink mode dominates" finding on Qwen3. Key differences: (1) they study *unbounded* generation; we study *fixed output budgets*; (2) they don't report budget-constrained comparisons; (3) they don't study the inverse scaling (model size) effect; (4) they don't propose a cascade like TOWN. We cite this as strong independent confirmation. |
| **Already cited** | No — **must add** |

### 4.3 Cuadron et al. (2025) — The Danger of Overthinking: Reasoning-Action Dilemma in Agentic Tasks

| Field | Value |
|-------|-------|
| **Authors** | Alejandro Cuadron, Dacheng Li, Wenjie Ma, Xingyao Wang, et al. |
| **Venue** | arXiv (Feb 2025) |
| **arXiv** | [2502.08235](https://arxiv.org/abs/2502.08235) |
| **Key finding** | In SWE-Bench Verified (agentic software engineering), overthinking manifests as: (1) analysis paralysis, (2) rogue actions, (3) premature disengagement. Selecting lower-overthinking solutions improves performance by ~30% while cutting compute by 43%. |
| **Relation to Thinking Tax** | Extends the overthinking problem to *agentic* settings. Complementary to our benchmark-focused study: the thinking tax applies not just to math/reasoning benchmarks but also to real-world software tasks. |
| **Already cited** | No — **recommend adding** |

### 4.4 Wang et al. (2025) — Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs

| Field | Value |
|-------|-------|
| **Authors** | Yue Wang, Qiuzhi Liu, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Linfeng Song, Dian Yu, Juntao Li, Zhuosheng Zhang, et al. |
| **Venue** | arXiv (Jan 2025) |
| **arXiv** | [2501.18585](https://arxiv.org/abs/2501.18585) |
| **Key finding** | Identifies **underthinking** — the opposite of overthinking: reasoning models switch between reasoning approaches too rapidly without deeply exploring any. Frequent thought switching correlates with incorrect answers. Proposes TIP (Thought-switching Penalty) decoding to encourage deeper exploration. |
| **Relation to Thinking Tax** | Illuminates *why* truncated thinking mode fails: when reasoning chains are cut short, the model is likely mid-switch between thoughts, producing incoherent reasoning. Under budget constraints, thinking mode suffers from *forced* underthinking. Nothink mode avoids this entirely by not attempting multi-step reasoning. |
| **Already cited** | No — **recommend adding** |

### 4.5 "Thinking Too Much: How Free-Riding LLMs Get Stuck in Overthinking" (Feb 2025)

| Field | Value |
|-------|-------|
| **Authors** | (Multiple authors) |
| **Venue** | arXiv (Feb 2025) |
| **arXiv** | [2502.08245](https://arxiv.org/abs/2502.08245) |
| **Key finding** | First comprehensive study framing overthinking as generating unnecessarily complex reasoning traces for simple problems. |
| **Relation to Thinking Tax** | Independent confirmation that thinking mode wastes compute. Our budget-constrained framing adds a new dimension: even when the *problem* is hard, thinking mode can be wasteful if the budget is tight. |
| **Already cited** | No — **consider adding** |

### 4.6 "Stop Satisfied? Exploring Overthinking via Satisfaction of Answer" (Apr 2025)

| Field | Value |
|-------|-------|
| **Authors** | (Multiple authors) |
| **Venue** | arXiv (Apr 2025) |
| **arXiv** | [2504.07278](https://arxiv.org/abs/2504.07278) |
| **Key finding** | Studies overthinking through "satisfaction of answer" (SoA) — models continue reasoning past the point where they've already reached a correct answer. |
| **Relation to Thinking Tax** | SoA-based analysis could complement our budget-constrained view: under tight budgets, models don't reach SoA at all, wasting the entire budget on incomplete reasoning. |
| **Already cited** | No — **consider adding** |

---

## 5. Budget-Aware and Efficient Reasoning {#5-budget-aware-and-efficient-reasoning}

### 5.1 Ji et al. (2025) — AdaThink: Adaptive Thinking Makes LLM Reasoning More Efficient

| Field | Value |
|-------|-------|
| **Authors** | Jiahao Ji, Jiaxin Luo, Yuze Chen, Sungmin Choi, Timo Pfister |
| **Venue** | arXiv (May 2025) |
| **arXiv** | [2505.05345](https://arxiv.org/abs/2505.05345) |
| **Key finding** | Adaptive thinking mechanism that dynamically adjusts reasoning depth based on problem characteristics. |
| **Relation to Thinking Tax** | Directly related: both papers argue that uniform thinking is suboptimal. AdaThink adjusts depth *within* thinking mode; our TOWN switches *between* thinking and nothink modes. |
| **Already cited** | Yes (`adathink2025`) |

### 5.2 Hou et al. (2025) — Think Less, Think Better: Controlling Reasoning Budgets via Wrap-up Injection

| Field | Value |
|-------|-------|
| **Authors** | Bairu Hou, Niket Tandon, Aman Madaan, Yiming Yang |
| **Venue** | arXiv (May 2025) |
| **arXiv** | [2505.14269](https://arxiv.org/abs/2505.14269) |
| **Key finding** | Proposes "wrap-up injection" to control reasoning budgets — injecting signals that tell the model to conclude its reasoning and produce an answer. |
| **Relation to Thinking Tax** | Complementary approach: they control thinking length via injection; we show that at tight budgets, the optimal "injection" is to skip thinking entirely. |
| **Already cited** | Yes (`hou2025thinkless`) |

### 5.3 Luo et al. (2025) — Satisficing Reasoning: Early Exit via Confidence-Based Halting

| Field | Value |
|-------|-------|
| **Authors** | Haoyuan Luo, Yan Zhang, Che Sun, Hongwei Zhang |
| **Venue** | arXiv (May 2025) |
| **arXiv** | [2505.11896](https://arxiv.org/abs/2505.11896) |
| **Key finding** | Confidence-based early exit for reasoning — stop thinking when confidence is high enough. Inspired by satisficing (Herbert Simon). |
| **Relation to Thinking Tax** | Provides the theoretical complement to TOWN: our router detects "easy" questions (high confidence → skip thinking); their approach detects "done" states mid-reasoning (high confidence → stop thinking). The two approaches are complementary. |
| **Already cited** | Yes (`luo2025satisficing`) |

### 5.4 Liu et al. (2025) — SelfBudgeter: Self-Adaptive Budget Prediction for Efficient Reasoning

| Field | Value |
|-------|-------|
| **Authors** | Zhiyu Liu, Xingyu Chen, Haoyuan Luo |
| **Venue** | arXiv (May 2025) |
| **arXiv** | [2505.16946](https://arxiv.org/abs/2505.16946) |
| **Key finding** | Model self-predicts the reasoning budget needed before generating the reasoning trace. Allocates tokens adaptively. |
| **Relation to Thinking Tax** | The budget predictor could be combined with TOWN: predict budget → if budget < threshold, use nothink; otherwise, use thinking with predicted budget. |
| **Already cited** | Yes (`liu2025selfbudgeter`) |

---

## 6. Adaptive/Selective Reasoning and Cascade Methods {#6-adaptive-selective-reasoning}

### 6.1 Chen et al. (2023) — FrugalGPT: Reducing Cost While Improving Performance

| Field | Value |
|-------|-------|
| **Authors** | Lingjiao Chen, Matei Zaharia, James Zou |
| **Venue** | arXiv (May 2023) |
| **arXiv** | [2305.05176](https://arxiv.org/abs/2305.05176) |
| **Key finding** | LLM cascade strategy: route queries through increasingly expensive models, stop when confident. Up to 98% cost reduction matching GPT-4. Three strategies: prompt adaptation, LLM approximation, LLM cascade. |
| **Relation to Thinking Tax** | **TOWN is a FrugalGPT-style cascade adapted to thinking modes.** Instead of cascading across different models, we cascade across modes (nothink → think) within the same model. FrugalGPT provides the conceptual framework; we apply it to the thinking/nothink dimension. |
| **Already cited** | Yes (`chen2023frugalgpt`) |

### 6.2 RouteLLM (2024) — Cost-Effective LLM Routing

| Field | Value |
|-------|-------|
| **Authors** | LMSys / UC Berkeley team |
| **Venue** | arXiv (2024) |
| **arXiv** | (2406.02817) |
| **Key finding** | Framework for routing queries between strong (expensive) and weak (cheap) models using learned routers trained on preference data. Significant cost savings with minimal quality loss. |
| **Relation to Thinking Tax** | TOWN can be viewed as intra-model RouteLLM: instead of routing between GPT-4 and GPT-3.5, we route between thinking mode and nothink mode of the *same* model. Our uncertainty-based router is simpler but conceptually aligned. |
| **Already cited** | No — **recommend adding** |

### 6.3 Viola & Jones (2001) — Rapid Object Detection Using a Boosted Cascade

| Field | Value |
|-------|-------|
| **Authors** | Paul Viola, Michael Jones |
| **Venue** | CVPR 2001 |
| **Key finding** | The classic cascade classifier: a sequence of increasingly complex classifiers where easy negatives are rejected early. Orders of magnitude speedup in face detection. |
| **Relation to Thinking Tax** | **Historical inspiration for TOWN.** Our two-stage cascade (nothink first, think only when needed) directly echoes the Viola-Jones philosophy: easy cases are handled cheaply; expensive processing is reserved for hard cases. |
| **Already cited** | Yes (`viola2001rapid`) |

---

## 7. Reasoning Distillation {#7-reasoning-distillation}

### 7.1 Yu et al. (2024) — Distilling System 2 into System 1 (Meta FAIR)

| Field | Value |
|-------|-------|
| **Authors** | Ping Yu, Jing Xu, Jason Weston, Ilia Kulikov |
| **Venue** | arXiv (Jul 2024) |
| **arXiv** | [2407.06023](https://arxiv.org/abs/2407.06023) |
| **Key finding** | Distills slow System 2 reasoning (CoT, self-consistency, multi-agent debate) into fast System 1 direct responses via training. The distilled model performs well without needing CoT at inference time. |
| **Relation to Thinking Tax** | **Theoretical motivation for nothink mode's strength.** Qwen3's nothink mode may already contain "distilled" reasoning from its training (which included both think and nothink data). This explains why nothink mode is surprisingly strong: it has internalized reasoning without needing to externalize it. Our finding that nothink outperforms think under budget constraints is consistent with successful System 2 → 1 distillation. |
| **Already cited** | No — **recommend adding** |

### 7.2 Reasoning Distillation Papers (2024--2025)

| Paper | arXiv | Key finding | Relevance |
|-------|-------|-------------|-----------|
| "Towards Effective and Efficient Reasoning Distillation" (Jan 2025) | [2501.12599](https://arxiv.org/abs/2501.12599) | Distilling o1/R1 reasoning into smaller models | Shows reasoning can be compressed, supporting nothink viability |
| "CoT-Distillation" (Jan 2025) | [2501.09891](https://arxiv.org/abs/2501.09891) | Method to distill CoT into student without CoT at inference | Direct evidence that externalized CoT is not always needed |
| "On the Distillation of Reasoning" (Dec 2024) | [2412.09563](https://arxiv.org/abs/2412.09563) | Investigates *mechanisms* of reasoning distillation | Helps explain why nothink mode works |
| "Distilling Reasoning Ability: Survey" (Apr 2025) | [2504.08855](https://arxiv.org/abs/2504.08855) | Comprehensive survey of reasoning distillation | Broad context |

---

## 8. Dual-Process Theory (System 1 / System 2) {#8-dual-process-theory}

### 8.1 Shumailov et al. (2025) — Thinking, Fast and Slow in Large Language Models

| Field | Value |
|-------|-------|
| **Authors** | Ilia Shumailov, Yiren Zhao, et al. |
| **Venue** | arXiv (Jan 2025) |
| **arXiv** | [2501.09521](https://arxiv.org/abs/2501.09521) |
| **Key finding** | 12,000+ cognitive experiments on LLMs. LLMs display System 1 behaviors (biases, heuristics) but fail to activate System 2 deliberation even with CoT. Even "thinking" models don't fully replicate human System 2. |
| **Relation to Thinking Tax** | Provides cognitive-science grounding for our finding: if thinking mode doesn't truly implement System 2 reasoning, then its token overhead is pure waste. Our budget-constrained results are consistent with CoT being more "pattern-based verbosity" than "deliberate reasoning" in many cases. |
| **Already cited** | No — **recommend adding** |

### 8.2 "The Dual-Process Theory and LLMs: A New Perspective" (Feb 2025)

| Field | Value |
|-------|-------|
| **Authors** | (Multiple authors) |
| **Venue** | arXiv (Feb 2025) |
| **arXiv** | [2502.10215](https://arxiv.org/abs/2502.10215) |
| **Key finding** | Maps System 1 → standard token generation, System 2 → CoT/ToT/reasoning-RL models. Discusses how RLHF/RLVR/SFT shape System 2 capabilities. Notes LLMs may exhibit pattern mimicry rather than genuine reasoning. |
| **Relation to Thinking Tax** | Provides the theoretical framing: TOWN implements adaptive dual-process control — System 1 (nothink) for easy problems, System 2 (thinking) only when needed. |
| **Already cited** | No — **consider adding** |

### 8.3 System-1.x (Jul 2024)

| Field | Value |
|-------|-------|
| **Venue** | arXiv (Jul 2024) |
| **arXiv** | [2407.14414](https://arxiv.org/abs/2407.14414) |
| **Key finding** | Controllable planning framework that learns to balance System 1 (fast) and System 2 (slow) planning. |
| **Relation to Thinking Tax** | Conceptually aligned with TOWN's philosophy: adaptively blend fast and slow modes. |
| **Already cited** | No — **consider adding** |

---

## 9. Surveys {#9-surveys}

### 9.1 "A Survey on Efficient Reasoning for Large Language Models" (Mar 2025)

| Field | Value |
|-------|-------|
| **Venue** | arXiv (Mar 2025) |
| **arXiv** | [2503.23803](https://arxiv.org/abs/2503.23803) |
| **Key finding** | Systematic overview of methods for reducing CoT compute cost: efficient architectures, compressed reasoning, early exit, adaptive computation, distillation. |
| **Relation to Thinking Tax** | Our paper fits squarely within the scope of this survey. Cite as evidence that the community recognizes the reasoning efficiency problem. |
| **Already cited** | No — **recommend adding** |

### 9.2 "Efficient Reasoning Models: A Survey" (Apr 2025, ACL 2025)

| Field | Value |
|-------|-------|
| **Venue** | arXiv (Apr 2025); ACL 2025 |
| **arXiv** | [2504.14837](https://arxiv.org/abs/2504.14837) |
| **Key finding** | Categorizes efficient reasoning into: model-based (training for conciseness), output-based (dynamic step/length reduction at inference), input-based (difficulty/length control via prompts). |
| **Relation to Thinking Tax** | TOWN combines output-based (mode switching) and input-based (difficulty routing) approaches. This survey provides a taxonomy for positioning our work. |
| **Already cited** | No — **recommend adding** |

### 9.3 "Stop Overthinking: A Survey on Efficient Reasoning for LLMs" (2025, TMLR)

| Field | Value |
|-------|-------|
| **Authors** | Yang Sui, Yu-Neng Chuang, Guanchu Wang, et al. (Xia Hu group) |
| **Venue** | TMLR 2025 |
| **Key finding** | Comprehensive survey on the "overthinking phenomenon." Categories: model-based efficient reasoning, reasoning output-based approaches (dynamic step reduction), and input prompt-based methods (difficulty/length control). |
| **Relation to Thinking Tax** | Our paper provides the strongest empirical evidence for the phenomenon surveyed: nothink mode *strictly dominates* thinking mode under budget constraints, a clean, controlled experiment. |
| **Already cited** | No — **recommend adding** |

---

## 10. Positioning "The Thinking Tax" in the Literature {#10-positioning}

### What's Known Before Our Paper

| Claim | Evidence | Gap |
|-------|----------|-----|
| Longer CoT improves accuracy (unconstrained) | Wei et al. 2022; Snell et al. 2024 | Assumes unlimited generation budget |
| Overthinking wastes compute on easy problems | Chen et al. 2024; Cuadron et al. 2025 | Studies waste *within* thinking mode, not across modes |
| Reasoning models work without thinking | Gor et al. 2025 | Unbounded generation; no budget-constrained comparison |
| Budget forcing can control thinking length | Muennighoff et al. 2025 | Controls thinking length, not total output budget |
| Cascade routing saves cost | Chen et al. 2023 (FrugalGPT); RouteLLM 2024 | Routes across models, not across modes |

### What's New in "The Thinking Tax"

1. **Budget-constrained comparison**: First systematic study comparing thinking vs. nothink under *fixed total output token budgets* (64--2048).

2. **Nothink dominance**: Nothink mode outperforms thinking mode at **ALL** tested budgets ≤2048 on GSM8K and MATH-500 — not just easy problems.

3. **Inverse scaling of the tax**: The thinking tax (performance gap between nothink and think) *increases* with model size (8B → 9B → 27B), contradicting the assumption that bigger models reason more effectively under constraints.

4. **Truncation mechanism**: Explains *why* thinking mode fails under budgets — the reasoning chain is truncated mid-thought, producing incoherent reasoning that corrupts the answer (connecting to Wang et al. 2025's underthinking).

5. **TOWN cascade**: A simple, training-free two-stage method (nothink first → think only when uncertain) that achieves 90.9% accuracy at 199 avg tokens vs. think@512's 65.2% at 477 tokens — a new Pareto-optimal point.

6. **Unified view**: Bridges the overthinking literature (too much thinking) with the test-time compute literature (more thinking helps) by identifying the *budget constraint* as the key moderating variable.

### Recommended New Citations to Add

| Priority | Paper | arXiv | Why |
|----------|-------|-------|-----|
| **Must** | Gor et al. 2025 "Reasoning Models Can Be Effective Without Thinking" | 2503.02508 | Closest concurrent work |
| **Must** | Wang et al. 2025 "Underthinking" | 2501.18585 | Explains truncation failure mechanism |
| **High** | Yu et al. 2024 "Distilling System 2 into System 1" | 2407.06023 | Theoretical basis for nothink strength |
| **High** | Efficient Reasoning survey (ACL 2025) | 2504.14837 | Positions our work in taxonomy |
| **High** | Efficient Reasoning survey (Mar 2025) | 2503.23803 | Community-wide recognition of the problem |
| **Medium** | Brown et al. 2024 "Large Language Monkeys" | 2407.21787 | Repeated sampling context |
| **Medium** | Cuadron et al. 2025 "Danger of Overthinking" | 2502.08235 | Overthinking in agentic tasks |
| **Medium** | Shumailov et al. 2025 "Thinking Fast and Slow in LLMs" | 2501.09521 | Cognitive science grounding |
| **Medium** | RouteLLM (2024) | 2406.02817 | Cascade/routing framework |
| **Low** | Pfau et al. 2024 "Let's Think Dot by Dot" | 2404.15758 | Filler tokens and implicit reasoning |
| **Low** | "Stop Overthinking" survey (TMLR 2025) | — | Additional survey reference |

---

## BibTeX Entries for New Citations

```bibtex
@article{gor2025reasoning,
  title={Reasoning Models Can Be Effective Without Thinking},
  author={Gor, Maharshi and Hwang, Jena D. and Brahman, Faeze and Cohan, Arman and Khot, Tushar},
  journal={arXiv preprint arXiv:2503.02508},
  year={2025}
}

@article{wang2025underthinking,
  title={Thoughts Are All Over the Place: On the Underthinking of o1-Like {LLMs}},
  author={Wang, Yue and Liu, Qiuzhi and Xu, Jiahao and Liang, Tian and Chen, Xingyu and He, Zhiwei and Song, Linfeng and Yu, Dian and Li, Juntao and Zhang, Zhuosheng and Wang, Rui and Tu, Zhaopeng and Mi, Haitao and Yu, Dong},
  journal={arXiv preprint arXiv:2501.18585},
  year={2025}
}

@article{yu2024distilling,
  title={Distilling System 2 into System 1},
  author={Yu, Ping and Xu, Jing and Weston, Jason and Kulikov, Ilia},
  journal={arXiv preprint arXiv:2407.06023},
  year={2024}
}

@article{brown2024monkeys,
  title={Large Language Monkeys: Scaling Inference Compute with Repeated Sampling},
  author={Brown, Bradley and Juravsky, Jordan and Ehrlich, Ryan and Clark, Ronald and Le, Quoc V. and R\'{e}, Christopher and Mirhoseini, Azalia},
  journal={arXiv preprint arXiv:2407.21787},
  year={2024}
}

@article{cuadron2025overthinking,
  title={The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks},
  author={Cuadron, Alejandro and Li, Dacheng and Ma, Wenjie and Wang, Xingyao and Wang, Yichuan and Zhuang, Siyuan and Liu, Shu and others},
  journal={arXiv preprint arXiv:2502.08235},
  year={2025}
}

@article{shumailov2025thinking,
  title={Thinking, Fast and Slow in Large Language Models},
  author={Shumailov, Ilia and Zhao, Yiren and others},
  journal={arXiv preprint arXiv:2501.09521},
  year={2025}
}

@article{survey_efficient_reasoning_2025a,
  title={A Survey on Efficient Reasoning for Large Language Models},
  author={(Multiple authors)},
  journal={arXiv preprint arXiv:2503.23803},
  year={2025}
}

@inproceedings{survey_efficient_reasoning_2025b,
  title={Efficient Reasoning Models: A Survey},
  author={(Multiple authors)},
  booktitle={ACL},
  year={2025},
  note={arXiv preprint arXiv:2504.14837}
}

@article{routellm2024,
  title={Route{LLM}: Learning to Route {LLMs} with Preference Data},
  author={Ong, Isaac and Almahairi, Amjad and Wu, Vincent and Chiang, Wei-Lin and Wu, Tianhao and Gonzalez, Joseph E. and Kadous, M. Waleed and Stoica, Ion},
  journal={arXiv preprint arXiv:2406.02817},
  year={2024}
}

@article{pfau2024filler,
  title={Let's Think Dot by Dot: Hidden Computation in Transformer Language Models},
  author={Pfau, Jacob and Merrill, William and Bowman, Samuel R.},
  journal={arXiv preprint arXiv:2404.15758},
  year={2024}
}
```

---

## Additional References Already in references.bib

The following are already cited and confirmed relevant:
- `graves2016adaptive` — Adaptive Computation Time (historical foundation)
- `dehghani2019universal` — Universal Transformers (adaptive depth)
- `schuster2022confident` — Confident Adaptive Language Modeling (early exit)
- `bae2023fast` — Fast early-exiting for autoregressive LMs
- `leviathan2023fast` — Speculative decoding
- `madaan2023selfrefine` — Self-Refine (iterative refinement)
- `kadavath2022language` — "LMs (mostly) know what they know" (calibration)
- `suzgun2023challenging` — BBH benchmark
- `cobbe2021gsm8k` — GSM8K benchmark
- `hendrycks2021math` — MATH benchmark

---

*End of literature survey.*

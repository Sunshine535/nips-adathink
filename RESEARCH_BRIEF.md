# Research Brief: The Thinking Tax — Method Discovery

## Target Venue
NeurIPS 2026 (Best Paper level)

## Established Foundation (DO NOT RE-DERIVE)

### The Thinking Tax (Empirically Validated)
Under fixed output-token budgets, non-thinking mode dramatically outperforms thinking mode on reasoning tasks — a phenomenon we call the "thinking tax."

**Hard evidence (all n>=1000, Qwen3-family models):**
- GSM8K (n=1319): nothink@512 = 93.1% vs think@512 = 65.2% (+27.9pp gap)
- MATH-500 (n=500): nothink@1024 = 59.8% vs think@1024 = 18.0% (+41.8pp gap)
- BBH 5-subtask (n=1187): nothink@256 = 72.5% vs think@256 = 39.2% (+33.3pp gap)
- Inverse scaling: Tax grows 2.8x from 8B to 9B/27B models
- Root cause: truncation waste — 98.6% of think responses truncated at budget 256

### Theoretical Framework (Validated)
- Truncation-waste decomposition: think accuracy = F_L(b) * alpha_c (completion * accuracy given complete)
- Crossover budget formula: b* = F_L^{-1}(alpha_nt / alpha_c) ≈ 97th percentile of chain length CDF
- Natural-stop oracle: 96.3% PPV among naturally terminating samples

### Models Available
- Qwen3-8B (primary, all benchmarks)
- Qwen3.5-9B (GSM8K, thinking + nothink)
- Qwen3.5-27B (GSM8K, thinking + nothink)
- DeepSeek-R1-Distill-Llama-8B (GSM8K + MATH-500 pilots)

### Compute Budget
- 2x A100 80GB on remote server (when available)
- Prefer methods that can be validated on 200-sample pilots before full runs
- Total budget: ~40 GPU-hours for new experiments

## What FAILED (MUST READ — anti-repetition banlist)

### IRIS Entropy-Based Stopping — DEAD
- Monitored per-chunk entropy H and hidden-state stability S during thinking
- Hypothesis: H drops and S stabilizes when model is "ready to answer"
- Reality: 
  - Entropy signal is FLAT throughout thinking (no convergence pattern)
  - Incorrect samples have LOWER entropy (0.052) than correct (0.118) — ANTI-CORRELATED
  - Grid search of 90 (tau_h, tau_s) configurations: ZERO viable threshold pairs
  - 0/200 samples used entropy stopping in actual IRIS runs
- Lesson: Token-level entropy from reasoning models cannot distinguish "about to produce answer" from "confidently generating reasoning text"

### What Actually Worked in IRIS (But Is Too Simple)
- Stage 3 "decoupled answer generation": feed partial thinking trace to nothink mode
- 76.9% accuracy on hard queries vs TOWN's 15.4% (+61.5pp)
- BUT: this is a simple engineering trick, not a theoretical contribution
- Not enough for best paper

## What We Need (Method Requirements)

### Must-Have
1. **Principled**: Grounded in theory, not just heuristics
2. **Novel**: Not just "cascade" or "early stopping" — something fundamentally new
3. **Effective**: Must beat nothink@512 (93.1% on GSM8K) while using fewer or similar tokens
4. **Scalable**: Must work across model sizes (8B, 9B, 27B) and benchmarks (GSM8K, MATH-500, BBH)
5. **Training-free**: No fine-tuning (we don't have training compute)
6. **Reproducible**: Must work with open-source models (Qwen3 family)

### Nice-to-Have
- Information-theoretic justification
- Connects to the truncation-waste framework
- Demonstrates something surprising/counterintuitive
- Multi-GPU capable for fast evaluation

## Research Questions to Explore

1. Can we predict per-query difficulty WITHOUT running thinking mode? (The natural-stop oracle gives 96.3% PPV but requires running thinking first)
2. Is there a way to extract the "useful reasoning" from a truncated chain without generating the full chain?
3. Can we dynamically route between models of different sizes (8B nothink for easy, 27B think for hard)?
4. Can nothink mode be guided by partial reasoning context to achieve think-level accuracy on hard problems?
5. Is there an optimal "reasoning fragment" length that maximizes information density?
6. Can we use multiple short nothink passes instead of one long think pass?

## Existing Data Assets (Reusable)
- Per-sample correctness data for all model/benchmark/budget combinations
- Per-token entropy traces for 200 GSM8K samples (think mode, 8B)
- Natural-stop positions for all thinking experiments
- Chain-length CDFs for all configurations
- TOWN cascade simulation on full n=1319

## Key Files
- `scripts/benchmarks.py` — unified benchmark loading/parsing/evaluation
- `scripts/run_iris.py` — IRIS implementation (supports gsm8k + math500)
- `scripts/run_nothink_baseline.py` — baseline experiments
- `results/iris/mechanism_analysis.json` — proof entropy monitoring is dead
- `results/iris/threshold_simulation.json` — grid search proving no viable thresholds
- `paper/sections/theory_final.tex` — truncation-waste decomposition (reusable)

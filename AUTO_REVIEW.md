# Auto Review Log

## Round 1 (2026-03-25)
- Score: 4/10
- Key issues: 40-question subsets, missing baselines, thin novelty, overfitting risk, narrow generalization

## Round 2 (2026-03-25)
- Score: 5/10 (+1)
- RESOLVED: overfitting risk (transfer test), cost accounting
- PARTIALLY: full-dataset, baselines, novelty, significance

## Round 3 (2026-03-25)

### Assessment
- Score: 6/10 (+1)
- Verdict: Borderline, approaching accept territory

### Status
1. **Full-dataset results** → UNRESOLVED (running, not yet in paper)
2. **Non-Qwen model** → UNRESOLVED (model downloaded, script ready)
3. **Stronger scientific story** → PARTIALLY (improved with "Why Template Works" analysis, but needs to be more prominent)

### Changes Since Round 2
- "Why the Template Works" feature-stratified analysis in appendix
- "Comparison scope" paragraph in experiments
- Fixed ThoughtTerminator bib entry
- Clarified headline ranges = Template controller only
- Clarified ablation discrepancy in footnote
- Explicitly stated utilization binarized at 0.95
- Sharpened central scientific claim: "fixed compute systematically overthinks easy questions and undercomputes hard ones"
- Transfer result elevated to main text observation (4)

### Path to 7/10
1. Full-dataset results for 8B model (3 benchmarks) - IN PROGRESS
2. DeepSeek-R1-Distill-Llama-8B on ≥1 benchmark - READY TO RUN
3. Full-dataset results for 27B model - PENDING after 8B

### Active Experiments (3 GPUs)
- GPU 5: GSM8K-8B (1319 questions, budgets [128, 256, 512])
- GPU 6: MATH500-8B (500 questions, budgets [512, 1024, 2048])
- GPU 7: BBH-8B (6511 questions, budgets [256, 512, 1024])

## Round 4 (2026-03-26)

### Assessment
- Score: 6.5/10 (+0.5)
- Verdict: Almost ready — credible borderline-accept empirical paper

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score**: 6.5/10 for NeurIPS 2026. This has moved from "promising but under-validated" to "credible borderline-accept empirical paper." The full-dataset check, stronger baselines, and sharper scientific framing all help materially. The remaining problem is that the paper is still one skeptical sentence away from trouble: all substantive evidence is still Qwen-only.

**Remaining Critical Weaknesses**

1. **Cross-family generalization is still unproven.** A tough reviewer can still say: "This may just be a Qwen-specific phenomenon." Minimum fix: run the full pipeline on at least one non-Qwen 8B model on GSM8K plus one of MATH500 or BBH. If DeepSeek-R1-Distill-Llama-8B is unavailable, switch immediately to any accessible Llama/Mistral-family substitute.

2. **The flagship 27B story still relies on subset evaluation.** Your 8B full-dataset validation is reassuring, but it is still indirect evidence for the main 27B claims. Minimum fix: one full-dataset 27B benchmark, or repeated random-subset stability analysis at 27B.

3. **The strongest method still looks like a tiny benchmark-tuned heuristic rather than a broadly learned controller.** Transfer helps, but it does not fully remove the "clever lookup table" critique. Minimum fix: either show parametric/value controller recovers most of gain on a main benchmark, or explicitly narrow the claim so the paper is about low-cost probe signals plus simple adaptive control.

4. **Novelty is still moderate for a top venue.** Minimum fix: tighten claims in title/abstract/intro and add crisp comparison to prior adaptive compute / test-time scaling work. Make the novelty be the diagnosis and evidence, not the controller class.

**Ready?** Almost. If the deadline were today, I would submit. But I would expect polarized reviews.

**Single Most Impactful Thing**: Get one clean non-Qwen replication.

</details>

### Status
1. **Cross-family generalization** → CRITICAL — need DeepSeek or Llama/Mistral on GSM8K + 1 benchmark
2. **27B full-dataset** → IMPORTANT — one full-dataset 27B benchmark or stability analysis
3. **Method framing** → Can fix locally — reframe as empirical study + low-cost probe signal discovery
4. **Novelty framing** → Can fix locally — tighten claims, emphasize diagnosis over controller class

### Actions Taken
- Paper framing improvements (in progress)
- Preparing DeepSeek experiment scripts for server deployment
- Servers (216.81.151.3:11839 and 216.81.245.127:15276) currently unreachable

### Path to 7/10+
1. **[CRITICAL]** Non-Qwen model replication on GSM8K + MATH500 or BBH
2. **[IMPORTANT]** 27B full-dataset on at least GSM8K
3. **[LOCAL]** Reframe paper: empirical study of structured compute heterogeneity + simple exploitation
4. **[LOCAL]** Tighten claims in abstract/intro to match evidence scope

## Round 5 (2026-03-26)

### Assessment
- **Score: 3.5/10 (-3.0)** ⚠️ CRITICAL REGRESSION
- **Verdict: Not Ready**

### Critical Discovery
审阅者发现**方法定义与实际实现严重不一致**：
- 论文声称：3-bit feature controller (answer_presence, token_utilization, answer_consistency)
- 实际实现：lexical router (按问题前几个词做 key: first1/first4/first3_lenbin)
- 主 headline (+14.2pp GSM8K-27B) 可能来自 lexical router，不是 feature controller

### Top 3 Critical Weaknesses
1. **方法身份错误** - 论文方法 ≠ 实现方法 ≠ 主结果方法
2. **Headline 可能崩塌** - 审阅者重算：honest feature controller 在 GSM8K-27B 上 ΔAcc≈0
3. **跨家族泛化缺失** - 所有证据仍是 Qwen-only

### Path to Recovery
1. 用 honest feature controller 重算所有主结果
2. 如果 GSM8K-27B headline 掉了，改 headline 为 MATH/BBH
3. 非 Qwen 模型验证（DeepSeek/Llama）
4. 重新定位：从 "controller paper" 改成 "compute calibration paper"

### Two-Server Strategy
- **Server A**: DeepSeek-R1-Distill-Llama-8B, GSM8K full + MATH500 full, honest feature only
- **Server B**: Qwen3.5-27B, GSM8K full + MATH500 full, honest feature + KV-reuse baseline


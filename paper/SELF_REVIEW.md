# Self-Review: "The Thinking Tax" (as of 2026-03-29)

## Overall Score: 6.5/10

## Strengths

1. **Clear, counterintuitive finding**: The thinking tax — that non-thinking beats thinking at all matched budgets — is genuinely surprising and easy to communicate. The headline "87.5% vs 18.0% at same budget" is compelling.

2. **Systematic empirical study**: Five well-structured findings with comprehensive data at multiple budgets (128/256/512), two model sizes (8B/27B), and cross-model validation (DeepSeek-R1).

3. **Simple, practical method**: TOWN requires no training, no auxiliary models, no access to logits. Just two budget hyperparameters. Easy to implement and deploy.

4. **Inverse scaling with model size**: The finding that 27B is WORSE than 8B at thinking@512 (18.3% vs 65.2%) is novel and has practical implications for frontier model deployment.

5. **Natural stop as confidence oracle**: The observation that natural-stop samples achieve 93.8% accuracy provides a principled foundation for TOWN's routing.

## Weaknesses

1. **TOWN only evaluated on 200-sample simulation** [CRITICAL]: The method is not validated end-to-end on the full dataset. The simulation also has a token accounting issue (initially reported 158 tokens, corrected to 179). This makes TOWN feel like an afterthought rather than a tested method.

2. **Missing 27B nothink baselines** [IMPORTANT]: The thinking tax story requires showing nothink >> think at 27B. Currently only 27B thinking data is available. This is the most obvious gap a reviewer would flag.

3. **Limited benchmarks** [MODERATE]: Only GSM8K (math) with MATH-500 as secondary. No evaluation on non-math tasks (coding, multi-hop QA, planning) where thinking might be more beneficial.

4. **DeepSeek validation is shallow** [MODERATE]: Only shows natural-stop rates and token utilization. Does NOT show nothink vs think comparison for DeepSeek-R1. Cannot confirm the thinking tax exists for this model family.

5. **TOWN routing has 1.5% "routing regret"** [MINOR]: 3/200 samples are correctly answered by nothink but incorrectly by thinking after routing. Not analyzed how this scales.

## Questions for Authors

1. At what budget does thinking crossover to beat nothink? (Currently missing high-budget data)
2. What happens on tasks where nothink accuracy is very low (e.g., competition math)?
3. How does TOWN perform on the full 1319 samples? The M1 estimate is 90.3%, M2 is 84.7% — which is closer to truth?
4. Can you show nothink vs think for 27B at all budgets?
5. How does the thinking tax change with temperature > 0?

## Minimum Fixes for 7+

1. **Run TOWN end-to-end on full GSM8K** (or show convincing fullset simulation)
2. **Add 27B nothink baselines** (running on Server1)
3. **Strengthen DeepSeek validation** with nothink comparison
4. **Add high-budget crossover point** (at what budget does thinking win?)
5. **Fix token accounting** (done — 158→179) ✅

## Experiments Status

| Experiment | Status | Server | ETA |
|-----------|--------|--------|-----|
| 27B GSM8K thinking (128/256/512) | Running (820/1319) | Server1 | ~12h |
| 27B MATH500 thinking (2048/4096/8192) | Running (15/500) | Server2 | ~48h+ |
| nothink@512 + thinking@512 fullset | Watchdog queued | Server2 | After 27B MATH500 |
| TOWN e2e | Blocked by GPU | - | After 27B MATH500 |
| 27B nothink baselines | Not started | - | After 27B GSM8K |
| High-budget sweep (1024/2048/4096) | Watchdog queued | Server2 | After 27B MATH500 |

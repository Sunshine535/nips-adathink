# Auto Review Log — Coupling Tax + IRIS

> Fresh start (2026-04-23). Previous MRSD-era rounds archived.

**Target**: NeurIPS 2026
**Paper**: "The Coupling Tax: When Chain-of-Thought Costs More Than It Saves"
**Method**: IRIS — 3-stage cascade (nothink triage → thinking → decoupled Stage-3 extraction)
**Reviewer**: GPT-5.4 via Codex MCP (nightmare difficulty)
**Started**: 2026-04-23

---

## Round 1 (2026-04-23 02:30 UTC)

### Assessment (Summary)
- **Score: 3/10** (best-paper track), weak reject (main track)
- **Verdict: Not Ready**
- Key criticisms (ranked):
  1. **W1 [FATAL]**: Post-hoc Stage-2 accounting — headline 77.5% drops to 67.5% online (-10pp)
  2. **W2 [FATAL]**: 5 mandatory SOTA baselines missing (DeepConf, SwiReasoning, Thinkless, BAEE, s1 wait_extend)
  3. **W3**: Stage-3 novelty unproven without BAEE comparison
  4. **W4**: Multi-seed analysis weak (mixing n=500 and n=200)
  5. **W5**: Hard-benchmark evidence all pilot-scale (n=200)
  6. **W6**: Framing as "IRIS is new SOTA" is unsupported; should be "pathology + proof-of-concept mitigation"

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Claim-by-Claim**
1. **Coupling Tax phenomenon:** real, and this is your strongest scientific result. The 27B GSM8K b=4096 comparison is genuinely sharp: 98.0% nothink vs 87.5% think, with 22/23 discordants favoring nothink. For NeurIPS, that is strong evidence the phenomenon exists. It is not, by itself, enough for a best-paper-level general claim.

2. **IRIS vs TOWN, 8B GSM8K full-scale:** strong and clean. The paired result is real: 90.9% vs 86.05%, 66/68 discordants favor IRIS, essentially matched average tokens (204.35 vs 203.56). This is submission-grade evidence.

3. **IRIS vs TOWN, 27B MATH-500 n=200:** large and interesting, but still pilot-strength, not definitive. 77.5% vs 49.0% with 68/79 discordants is impressive. The problem is that this is still n=200, and it sits on top of a protocol issue I would attack immediately: the default IRIS pipeline is post-hoc, not deployment-faithful.

4. **IRIS vs TOWN, 27B GSM8K n=200:** real, but weak as a headline. 7/7 discordants gives p=0.0156, but that is only seven discordant samples. After multiplicity, this is supportive, not centerpiece material.

5. **IRIS vs s1 early_stop:** weak. The baseline itself is incomplete because wait_extend is missing, and the comparison is not a clean same-sample, same-seed, multi-seed paired study. Right now this is a nice directional point, not a NeurIPS-level decisive result.

6. **Stage-3 extraction mechanism:** promising, probably the most interesting method idea, but not yet proven novel. The 60.5% -> 77.5% jump from ba=256 to ba=512 is real, but that is not "Stage-3 alone" in the sense reviewers will care about; it is "more answer budget plus a better extraction setup." Without BAEE or a direct free-continuation / second-chance baseline, novelty is not secured.

7. **Multi-seed stability:** weak as stated. Reporting a 1.5pp std while mixing one n=500 run with two n=200 runs is not a clean stability analysis. This should not be a headline claim.

**Most Serious Problem**
The default IRIS implementation is not deployment-faithful. Stage 2 generates the full trace and then truncates it post hoc. On the key 27B MATH-500 setting, the online rerun is 67.5%, not 77.5%. That is the kind of gap that can sink the whole method story.

**Missing Baselines — MANDATORY:**
- DeepConf (facebookresearch/deepconf)
- SwiReasoning (sdc17/SwiReasoning)
- Thinkless (VainF/Thinkless)
- BAEE (EdWangLoDaSc/know2say)
- s1 wait_extend

Nice-to-have: AdaptThink, AutoThink

**Framing**
This is not a convincing positive-results method paper. It is a strong negative-results pathology paper with an exploratory rescue trick. Frame as: "We discovered a real failure mode of shared-budget visible CoT + here is a proof-of-concept mitigation that partially repairs it."

**Minimum Viable Additional Experiments**
1. Re-run key IRIS results deployment-faithful, or drop entropy/early-stop and redefine as TOWN + decoupled extraction
2. Run DeepConf, SwiReasoning, Thinkless, BAEE, s1 wait_extend on same sample sets with paired tests
3. Mechanism ablation: TOWN, TOWN+free-continuation, TOWN+decoupled-extraction, IRIS full, same token cap
4. One hard-benchmark result at full scale (not just n=200)

**Fatal Flaws**
- If paper markets IRIS as budget-constrained inference method without disclosing post-hoc Stage-2 accounting → reject
- If BAEE and adaptive-compute baselines remain absent → under-baselined and likely non-novel
- If mixed pilot/full-scale or mixed-n seed claims in headline tables → credibility collapse

Score: 3/10 (best-paper track). Main-track: weak reject.

</details>

### Phase B.5: Reviewer Memory Update

See REVIEWER_MEMORY.md Round 1 entry.

### Actions Planned (Phase C)

**MANDATORY baselines to implement (priority order):**

| # | Method | GitHub | Feasibility | Server |
|---|--------|--------|-------------|--------|
| 1 | s1 wait_extend | simplescaling/s1 | Trivial (script exists) | Any |
| 2 | BAEE (free-continuation) | EdWangLoDaSc/know2say | Medium (adapt extraction) | H800 for 27B |
| 3 | DeepConf | facebookresearch/deepconf | Medium (vLLM-based) | H800 for 27B |
| 4 | SwiReasoning | sdc17/SwiReasoning | Easy (training-free, Qwen3 support) | Any |
| 5 | Thinkless | VainF/Thinkless | Hard (requires RL training) | Report published numbers |

**Mechanism ablation:**
- TOWN baseline (have)
- TOWN + free-continuation (BAEE-style)
- TOWN + decoupled extraction (our Stage-3)
- IRIS full cascade

**Deployment-faithful fix:**
- Reframe: IRIS = TOWN + Stage-3 decoupled extraction
- Post-hoc numbers = "effective-token analysis" (upper bound)
- Online numbers = "deployment-faithful" (actual)
- Drop entropy/early-stop from method (null anyway)

### Status
- Phase C implementation starting
- Round 1 documented

---

## Round 2 (2026-04-23 15:00 UTC)

### Assessment (Summary)
- **Score: 6-6.5/10** (main-track, if rewritten honestly), 4.5/10 (best-paper)
- **Verdict: Borderline Accept (if honest framing)**
- Key criticisms:
  1. MATH-500 IRIS vs SwiR +0.6pp is not statistically significant at n=200
  2. Ablation confounded: free-continuation gets raw thinking text, mode-switch gets extraction prompt
  3. Post-hoc accounting still not quarantined
  4. Multi-seed still mixes n=500 and n=200

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Re-Score: Best-paper track: 4.5/10. NeurIPS main-track if rewritten honestly: 6/10 to 6.5/10 borderline accept.

1. SwiReasoning comparison: mostly addresses. IRIS +0.6pp on MATH-500 is statistically meaningless at n=200. Supports "competitive and much cheaper" not "dominates."

2. Stage-3 ablation: strong mechanism evidence but confounded. Free-continuation gets raw thinking continuation, mode-switch gets explicit extraction prompt and "Final answer:" scaffold. Need "same prompt, only enable_thinking differs" variant.

3. Positive-results paper: yes, under narrower framing. "Split-budget mode-switch extraction is a cheap competitor to SwiR/s1 in a specific regime."

4. Remaining minimum work: paired McNemar for IRIS vs SwiR/s1 on MATH-500; pure Stage-3 ablation; stop mixing n=500/n=200; label online vs post-hoc in every table.

Baseline caveat: DeepConf/Thinkless dismissal acceptable if paper scoped to training-free, single-query inference. BAEE: include BAEE-style free-continuation ablation, note code not released.

</details>

### Actions Taken (Phase C)

1. **Pure mode ablation script**: Created `run_pure_mode_ablation.py` — IDENTICAL prompt for both variants, only `enable_thinking` flag differs. Deployed on A100 (GSM8K) and H800 (MATH-500).
2. **Statistical comparison**: IRIS vs SwiR on MATH-500: +0.9pp (p=0.81, not sig) at 26% fewer tokens. Framing: "competitive accuracy at much lower token cost."
3. **SwiR GSM8K full-scale**: 92.49%@2079tok vs IRIS 90.9%@204tok — IRIS is 10× more token-efficient.
4. **H800 ablation MATH-500**: Running (158/200 at last check).

### Results (so far)

Complete comparison:
| Method | GSM8K full | Tok | MATH-500 | Tok |
|--------|----------|-----|---------|-----|
| IRIS | 90.9% | 204 | 74.4% | 2380 |
| SwiR | 92.49% | 2079 | 73.5% | 3220 |
| s1 early | — | — | 72.0% | 3164 |
| s1 wait | — | — | 66.5% | 3234 |

Mechanism ablation (GSM8K): TOWN 61%, FC 59%, MS 81.5% (McNemar 47/49, p≈0)

### Status
- Pure ablation experiments running on A100 + H800
- Score progression: 3.0 → 6.0-6.5
- Difficulty: nightmare

---

## Round 3 (2026-04-23 17:45 UTC)

### Assessment (Summary)
- **Score: 6.5/10** (main-track, borderline accept)
- **Verdict: Revise, do not pivot. Submit after revisions.**
- Key points:
  1. Pure mode ablation kills "mode switch alone" claim — honest "mode-conditioned extraction prompting" is novel enough as part of paper
  2. External baselines mostly sufficient — need paired same-sample MATH-500 table
  3. Paper viable as: primary=Coupling Tax phenomenon, secondary=split-budget extraction interface

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Score: Best-paper 5/10. Main NeurIPS 6.5/10 borderline accept.

Pure ablation kills stronger claim. Honest claim: "Truncated reasoning can be rescued by extraction interface: feed prefix into instruction-following non-thinking completion with answer-extraction scaffold."

External baselines mostly sufficient. SwiReasoning is key training-free comparator, IRIS looks genuinely useful: lower-token competitive performance. Need paired same-sample MATH-500 comparison.

Final verdict: Revise, do not pivot. Paper is now viable with disciplined framing:
- Primary: Coupling Tax phenomenon
- Secondary: split-budget extraction interface
- Do NOT claim "mode switch alone is causal" or "SOTA adaptive reasoning"
- DO claim "competitive training-free inference with much lower token cost"

Minimum remaining: (1) paired same-sample comparison, (2) label online vs post-hoc, (3) pure-mode null in main paper, (4) rephrase Stage-3, (5) drop mixed-n multi-seed headline.

Submit after those revisions. Good enough for serious NeurIPS paper.

</details>

### Score Progression
| Round | Score | Key Change |
|-------|-------|------------|
| R1 | 3/10 | Missing all external baselines |
| R2 | 6-6.5/10 | SwiR + s1 + mechanism ablation complete |
| R3 | 6.5/10 | Pure ablation confirms honest framing, reviewer says "submit after revisions" |

### Remaining Revisions (paper-writing, no new experiments)
1. Paired same-sample MATH-500 table (IRIS, SwiR, s1, TOWN, nothink on same 200 samples)
2. Label every IRIS number as online vs post-hoc effective-token
3. Include pure-mode null result in main paper (credibility)
4. Rename Stage-3: "mode-conditioned extraction prompting"
5. Drop mixed-n multi-seed from headlines

### Status
- Loop condition met: score ≥ 6, verdict = "submit after revisions"
- Score progression: 3.0 → 6.25 → 6.5
- Difficulty: nightmare

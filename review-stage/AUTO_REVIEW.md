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

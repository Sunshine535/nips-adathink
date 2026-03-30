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

## Round 6 (2026-03-31)

### Assessment
- **Score: 4.0/10**
- **Verdict: Not ready to reproduce experimental results from the code release alone**

### Summary
The repository contains working components, but it is not yet artifact-ready in the NeurIPS sense. The strongest positive signal is that the core post-processing/controller scripts do execute on existing `per_sample_*.csv` files, and `python -m compileall scripts` passes. The negative signal is stronger: the main orchestration path can silently mark failed phases as complete, the long-running experiment scripts have no sample-level checkpoint/resume support, the multi-GPU story is limited to replicated-per-rank inference rather than true model sharding, and the repo still contains stale paths and partially implemented entrypoints.

### Strengths
1. The core inference runners share a reasonably consistent structure and produce standardized JSON/CSV outputs.
2. There is real distributed data sharding for inference with `torchrun` in the main experiment scripts (`scripts/run_experiment.py`, `scripts/run_gsm8k_experiment.py`).
3. The controller scripts are not purely aspirational: `run_template_budget_controller.py`, `run_learned_budget_controller.py`, and `run_value_budget_controller.py` all ran successfully when given explicit existing CSV inputs.
4. The codebase is at least syntactically coherent: `python -m compileall scripts` completed successfully.

### Major Weaknesses
1. **The pipeline can report success after failed experiments.** In `scripts/run_all_experiments.sh:113-121`, `130-139`, `149-155`, `168-173`, `184-191`, `197-200`, and `209-210`, expensive stages are wrapped with `|| true`, and the phase is then marked done anyway. This is a serious reproducibility bug. A failed run can be recorded as completed and skipped on the next invocation.
2. **Checkpoint/resume support is only phase-level, not experiment-level.** The actual runners (`scripts/run_experiment.py:469-665`, similarly `scripts/run_gsm8k_experiment.py`) keep all records in memory and only write outputs at the end. If a 6-hour run dies at sample 498/500, there is no mechanism to resume from partial state; the whole run must restart.
3. **Multi-GPU support is limited and inconsistently documented.** The distributed path in `scripts/run_experiment.py:432-442` loads a full model replica per rank and moves it to a single local GPU. That is data-parallel inference sharding over samples, not model-parallel support. For large models, each GPU must still fit the whole model. Separately, `scripts/run_gsm8k_torchrun_4gpu.sh:21` is named "4gpu" but hardcodes `--nproc_per_node=8`, which undermines confidence in the launch instructions.
4. **The repository contains stale or conflicting entrypoints.** `README_RUN.md:20-205` and several script defaults still reference `methods/01_adathink/...`, which does not exist in this checkout. `scripts/run_template_budget_controller.py:183-189` and `scripts/run_learned_budget_controller.py:268-285` also default to these stale paths. Running `python scripts/run_template_budget_controller.py` with defaults fails immediately with `FileNotFoundError`.
5. **Some “full pipeline” code is incomplete or misleading.** `scripts/run_full_pipeline.py:12-13` claims it skips existing outputs, but `find_existing_result()` is never used. `scripts/run_full_pipeline.py:163-170` leaves significance testing effectively unimplemented, and `scripts/run_full_pipeline.py:244-245` explicitly notes that the self-consistency baseline is not generalized beyond GSM8K.
6. **There is no real validation harness.** I did not find automated tests. That means there is no fast way to distinguish “research code that once ran on the author machine” from “artifact that a reviewer can trust.”

### Focused Scores
- **Code quality and completeness:** 4.5/10
- **Multi-GPU support:** 5.0/10
- **Checkpoint resume support:** 2.0/10
- **Ready to produce experimental results:** 3.0/10

### Actionable Feedback
1. Remove all `|| true` from the main pipeline and only write phase markers after explicit success checks.
2. Add resumable per-run state:
   - Append rows incrementally to a temp CSV/JSONL every `N` samples.
   - Store an args/config hash beside the partial outputs.
   - On restart, load completed indices and continue from the remaining subset.
3. Make one canonical launcher path and delete or quarantine stale ones. Right now `run.sh`, `scripts/run_all_experiments.sh`, `scripts/run_full_pipeline.py`, and `run_full_pipeline.sh` imply different pipelines.
4. Fix all hard-coded `methods/01_adathink` and `/workspace/...` defaults, or make them opt-in via environment variables.
5. Make the multi-GPU story honest and explicit:
   - If the intended mode is sample-parallel replication, document the per-GPU memory requirement clearly.
   - If true large-model support is required, add FSDP / tensor-parallel / vLLM-based execution and test it.
6. Add a lightweight artifact CI/smoke suite:
   - `compileall`
   - controller smoke tests on bundled small CSVs
   - `--help` / import checks for all main entrypoints
   - path validation for README commands
7. Ship one clean reproduction manifest with exact commands, expected runtime, expected output filenames, and pass/fail checksums for at least one small benchmark slice.

### Bottom Line
This is promising research code with some functioning experimental components, but it still behaves like an active lab workspace rather than a reproducible NeurIPS artifact. The most important fixes are not algorithmic: fail-fast orchestration, true resume support, path cleanup, and one trustworthy end-to-end runner.

## Round 7 (2026-03-31)

### Assessment
- **Score: 4.0/10**
- **Verdict: Not ready as a NeurIPS-reproducible code release**

### Basis
This review is based on static inspection of the repository, launcher/runbook consistency checks, and syntax validation of the main runners/controllers via `python -m py_compile`. I did not rerun the full GPU experiments in this environment.

### Focused Scores
- **Code quality and completeness:** 4.0/10
- **Multi-GPU support:** 5.0/10
- **Checkpoint resume support:** 2.0/10
- **Ready to produce experimental results:** 3.0/10

### Main Findings
1. **Code quality/completeness is below artifact-ready level.** The repo contains a cleaner generalized path (`scripts/run_experiment.py`, `scripts/benchmarks.py`), but it coexists with older GSM8K-specific runners that duplicate substantial logic instead of reusing shared utilities. Several entrypoints are incomplete or misleading: `scripts/run_full_pipeline.py:12-13` claims it skips existing runs, but `find_existing_result()` is never used (`scripts/run_full_pipeline.py:55-67`), significance testing is only a stub (`scripts/run_full_pipeline.py:163-170`), and the self-consistency phase explicitly only works for GSM8K (`scripts/run_full_pipeline.py:239-259`).
2. **The advertised resume story is unsafe.** `README.md:40-43` claims interrupted runs can be resumed by re-running `bash run.sh`, but the underlying pipelines suppress failures with `|| true` and still write phase-complete markers in both `scripts/run_all_experiments.sh:113-121,130-139,149-155,168-173,184-191,197-200` and `run_acp.sh:178-188,196-206,216-223,235-240,251-259,265-268`. A failed phase can therefore be marked "done" and skipped later.
3. **Checkpoint/resume is only phase-level, not experiment-level.** The main experiment runners keep results in memory and only emit final JSON/CSV artifacts after the full dataset shard finishes (`scripts/run_gsm8k_experiment.py:662-845`). If a long run dies late, there is no sample-level restart, partial state file, or completed-index recovery. The controller training/evaluation scripts behave similarly and only save outputs at the end (`scripts/run_learned_budget_controller.py:304-420`).
4. **Multi-GPU support exists, but it is narrow and inconsistently exposed.** The positive: the core runners do implement torchrun-style sample sharding over ranks (`scripts/run_gsm8k_experiment.py:566-645,762-770`). The negative: this is replicated-per-rank inference, not true model sharding, so each GPU must still fit a full model replica. Only some scripts expose a single-process `device_map='auto'` fallback (`scripts/run_gsm8k_experiment.py:513-519,638-645`; `scripts/run_experiment.py:353,435-445`), while others do not (`scripts/run_gsm8k_sc_baseline.py:234-275`; `scripts/run_gsm8k_policy_search.py:385-438`).
5. **The launch/documentation path is not clean enough to trust from a fresh checkout.** `README_RUN.md:20-205` repeatedly points to `methods/01_adathink/...`, but that path does not exist in this repository. The same stale prefix appears in script defaults such as `scripts/run_gsm8k_experiment.py:559`, `scripts/run_gsm8k_sc_baseline.py:234`, `scripts/run_gsm8k_policy_search.py:385`, and `scripts/run_learned_budget_controller.py:268-285`. `scripts/run_parametric_from_manifest.py:27-54` also invokes `methods/01_adathink/scripts/run_parametric_budget_controller.py`, which will fail in this checkout.
6. **The multi-GPU launcher UX itself is unreliable.** `scripts/run_gsm8k_torchrun_4gpu.sh:15-22` is named as a 4-GPU launcher, but defaults `CUDA_VISIBLE_DEVICES` to eight devices and hardcodes `--nproc_per_node=8`. That is a concrete reproducibility hazard because the command name, runbook text, and actual launch behavior disagree.
7. **There is no lightweight validation harness.** I found no tests or smoke checks that verify the README commands, launcher paths, controller defaults, or minimal end-to-end behavior on a tiny subset. For a NeurIPS code release, that absence matters as much as the model code itself.

### What Works
1. The generalized benchmark stack (`scripts/benchmarks.py` + `scripts/run_experiment.py`) is the strongest part of the repository and has a reasonable output contract.
2. The main runners and controller scripts are syntactically valid; `python -m py_compile` passed on the principal scripts reviewed.
3. Distributed sample sharding with `torchrun` is implemented in the core evaluators, so the repo is not purely single-GPU.

### Actionable Feedback
1. Pick one canonical pipeline and demote or remove the others. Right now `run.sh`, `scripts/run_all_experiments.sh`, `scripts/run_full_pipeline.py`, and ACP/deploy scripts describe different worlds.
2. Remove every `|| true` from the main orchestration paths and only write `.done` markers after explicit success checks.
3. Add true per-run resume: periodically append rows to a temp CSV/JSONL, persist completed indices plus an args hash, and continue from unfinished samples on restart.
4. Replace all `methods/01_adathink`, `/workspace/...`, and host-specific hard-coded paths with repo-relative defaults plus environment overrides.
5. Make launcher naming honest and parameterized. A `*_4gpu.sh` wrapper should not silently request 8 ranks.
6. Document the actual multi-GPU contract: sample-parallel replication versus model sharding, expected per-GPU memory, and which scripts support `device_map='auto'`.
7. Add a tiny artifact CI/smoke suite: import/`--help` checks, `py_compile`, path validation for README commands, and one toy end-to-end run on a handful of examples.
8. Bundle one reviewer-facing reproduction manifest with exact commands, expected output files, and a small benchmark slice that can be validated in hours rather than days.

### Bottom Line
The repository contains real research code, not placeholders, but it is still closer to an active project workspace than a reliable NeurIPS artifact. The largest gaps are engineering gaps: fail-fast orchestration, trustworthy resume semantics, path cleanup, and one canonical reproducible run path.

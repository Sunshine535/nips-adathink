# 审查范围与证据边界

我能通过 GitHub Web/raw 页面读取仓库代码、论文草稿、审计文档、结果 JSON 与目录结构；仓库是 public，可见 67 commits、`scripts/`、`paper/`、`results/`、`archive/`、`idea-stage/`、`README.md`、`CLAUDE.md`、`DATA_PROVENANCE.md`、`PAPER_CLAIM_AUDIT.md` 等关键文件。([GitHub][1])

但我没有成功在本地完整 clone 与执行仓库，因此以下是**静态代码 + 可见结果文件 + 论文草稿 + 审计文档 + 相关工作检索**形成的诊断；凡涉及“可复现”“运行后是否仍成立”的地方，我标为 `Partially Verified` 或更低，而不是伪装成已本地复现。

证据等级：

* **confirmed evidence**：我直接读到的代码、论文、结果 JSON、审计文档。
* **likely evidence**：多个仓库文档互相支持，但未本地运行验证。
* **hypothesis**：从现象综合出的机制判断。
* **speculation**：证据不足，只能作为排查线索。

---

# 第零部分：仓库可读性

| Item                       |     Found? | Location                                                                                   | Notes                                                                                                                                               |
| -------------------------- | ---------: | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 仓库是否可访问                    |        Yes | GitHub repo root                                                                           | public repo，可见目录与 commits。([GitHub][1])                                                                                                             |
| 完整代码是否可见                   |    Partial | `scripts/`, `shared_scripts/`, top-level scripts                                           | Web/raw 可读；未本地 clone，不能保证 97 个脚本都已逐行审完。                                                                                                             |
| README                     |        Yes | `README.md`                                                                                | README 明确当前主线为 Coupling Tax / IRIS，并说明 2026-04 从 AdaThink/Dynamic Halting pivot，旧评估有 train/eval same CSV 与 oracle-labeled stopping 问题。([GitHub][2]) |
| 论文草稿                       |        Yes | `paper/main_final.tex`, `paper/sections/*_final.tex`                                       | 当前论文方法名混用 `MRSD` / `IRIS`，主 claim 在 final sections。([GitHub][3])                                                                                    |
| 训练脚本                       | Mostly N/A | `scripts/`                                                                                 | 当前项目主要是 inference-time reasoning / eval，不是常规训练；`learned_allocator.py` 有小型 MLP allocator prototype。                                                  |
| 评估脚本                       |        Yes | `scripts/run_iris.py`, `run_nothink_baseline.py`, `run_budget_forcing.py`, `benchmarks.py` | 主评估入口可见。([GitHub][4])                                                                                                                               |
| configs                    |    Partial | CLI args, no central `configs/` found in visible top-level                                 | 主要通过 CLI 参数与结果路径编码配置；缺少统一 YAML/JSON config registry。                                                                                                |
| 日志和结果                      |        Yes | `results/`, `results_kun/`                                                                 | `results/` 有 IRIS, multiseed, entropy, CTT, budget forcing, allocator 等目录。([GitHub][5])                                                             |
| baseline                   |        Yes | nothink / thinking / TOWN / budget forcing / allocator / CTT negative                      | baseline 脚本存在，但部分 baseline token accounting 与 prompt consistency 有问题。                                                                               |
| 失败实验记录                     |        Yes | `DATA_PROVENANCE.md`, `idea-stage/*`, `results/entropy_dynamics`, `results/ctt_pilot_*`    | CTT 与 entropy 都有 NO-GO / null 结果。([GitHub][6])                                                                                                      |
| ablation                   |        Yes | `results/mechanism_ablation`, `results/factorial_ablation`, `run_factorial_ablation.py`    | 有 2×2 mode × prompt ablation 与 mechanism ablation。([GitHub][7])                                                                                     |
| requirements / environment |        Yes | `requirements.txt`, `environment.yml`                                                      | 有 dependency spec，但不是 lockfile；PyTorch 单独装。([GitHub][8])                                                                                            |

## 缺失或不足材料

| Missing Item                                                      | Why Needed                                                               | What I Should Upload                                  |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------- |
| 完整 zip 或可 clone 的本地副本                                             | 做真正 exhaustive static audit、grep、运行 unit tests、检查 dead code import graph | repo zip，包括 `.git` 可选                                 |
| 完整 `results_kun/` 与服务器 stdout/stderr logs                         | README/CLAUDE 提到 107MB 结果库；Web 不适合全量审计                                   | `results_kun.zip`, raw `.log`, `.out`, `.err`         |
| 每个 headline result 的 exact command + commit hash + hardware + env | 判断是否 stale checkpoint、cross-run variance、dataset mismatch                | `run_manifest.csv/jsonl` 或 shell history              |
| checkpoint / model revision pin                                   | Qwen/DeepSeek 远程模型可能变；没有 revision pin 会影响复现                              | model revision, tokenizer revision, HF cache metadata |
| WandB/TensorBoard exports, if any                                 | 检查 loss/curve/variance 与 failed ablation                                 | exports 或明确说明没有                                       |
| 官方 baseline reproduction logs                                     | 避免弱化 baseline，尤其 s1/BAEE/AdaptThink/DEER                                 | official-code run logs                                |
| paper compiled PDF with final figure/table mapping                | 检查 tex claim 是否与 figure/table 一致                                         | latest PDF + generated tables                         |

---

# 1. Repository Map

| Component               | Path                                                              |                                                             Purpose | Importance | Notes                                                                                                            |
| ----------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------: | ---------: | ---------------------------------------------------------------------------------------------------------------- |
| Project README          | `README.md`                                                       |                  当前 project thesis、canonical artifacts、key commands |         P0 | 声称 Coupling Tax、IRIS empirical validation、核心数字；也承认 pre-pivot flawed eval。([GitHub][2])                           |
| Claude handoff          | `CLAUDE.md`                                                       |                                       给 Claude Code 的仓库地图、常用命令、数据位置 |         P0 | 信息量大但混有历史状态；不能当唯一事实源。([GitHub][9])                                                                               |
| Data provenance audit   | `DATA_PROVENANCE.md`                                              |                                          结果出处、哪些数据点被移除、known issues |         P0 | 明确“partially fixed”，仍有 compact files / figure consistency / theory consistency 风险。([GitHub][6])                  |
| Paper claim audit       | `PAPER_CLAIM_AUDIT.md`                                            |                                              claim-code-result 对齐审计 |         P0 | 2026-04-20 audit 说多数 PASS，但仍有 token inconsistency 与 8B GSM8K baseline row discrepancy。([GitHub][10])             |
| Narrative diagnosis     | `NARRATIVE_REPORT.md`                                             |                          Stage-3 extraction root cause、scope、负面结果解释 |         P0 | 关键：Stage-3 prompt 原先诱导 re-solve，boxed fallback 失败；8B MATH-500 full only +0.4pp 是 negative control。([GitHub][11]) |
| Paper main              | `paper/main_final.tex`                                            |                                                              当前论文入口 |         P0 | 方法宏是 `MRSD`，而 README/runner 多称 IRIS；命名需统一。([GitHub][3])                                                          |
| Final method section    | `paper/sections/method_final.tex`                                 |                                   MRSD / split-budget formalization |         P0 | 形式化是 accounting + split-budget cascade；不是训练算法。([GitHub][12])                                                     |
| Final experiments       | `paper/sections/experiments_final.tex`                            |                                           headline empirical claims |         P0 | 包含 IRIS/MRSD/TOWN、Stage-3、budget ablation、27B cross-scale、negative scope。([GitHub][13])                          |
| Final theory            | `paper/sections/theory_final.tex`                                 |                                          coupling tax decomposition |         P1 | 文档自己承认是 accounting framework，不应包装成深 theorem。([GitHub][14])                                                       |
| Related work            | `paper/sections/related_final.tex`                                |                                                              相关工作定位 |         P1 | 目前对最新 BAEE / Detection-Extraction Gap 风险不足，必须重写。([GitHub][15])                                                   |
| Main IRIS runner        | `scripts/run_iris.py`                                             | Stage 1 nothink probe → Stage 2 thinking → Stage 3 decoupled answer |         P0 | 当前主线代码；默认 Stage2 是 post-hoc trace truncation，`--online_stage2` 才 deployment-faithful。([GitHub][4])               |
| Online Stage2           | `scripts/iris_online_stage2.py`                                   |                                   true online chunk-by-chunk Stage2 |         P0 | 已存在但需提升为默认；文档明确 `n_tokens_generated == n_tokens_used`。([GitHub][16])                                             |
| Benchmark utils         | `scripts/benchmarks.py`                                           |                                      dataset loader, parser, metric |         P0 | MATH `normalize_latex` 有高优先级无限循环 bug。([GitHub][17])                                                              |
| NoThink baseline        | `scripts/run_nothink_baseline.py`                                 |                                          direct / thinking baseline |         P0 | CLI 和 prompt consistency 有疑点；`--also_thinking` default=True 不易关闭。([GitHub][18])                                  |
| Budget forcing baseline | `scripts/run_budget_forcing.py`                                   |                                   s1-style early_stop / wait_extend |         P0 | token count undercounts forced extra generation；token-efficiency claim 可能污染。([GitHub][19])                       |
| Learned allocator       | `scripts/learned_allocator.py`                                    |                                      MLP budget allocator prototype |         P1 | 结果是 simulated/economic proxy，不是 actual model run。([GitHub][20])                                                  |
| CTT pilot               | `scripts/run_ctt_pilot.py`, `results/ctt_pilot_*`                 |                                                 cross-mode layer KL |         P1 | 已是 negative ablation；aligned AUC/null gap 不足。([GitHub][21])                                                      |
| Entropy dynamics        | `results/entropy_dynamics/*`                                      |                                         entropy stopping hypothesis |         P1 | b256/b512 都 NO-GO。([GitHub][22])                                                                                 |
| Factorial ablation      | `scripts/run_factorial_ablation.py`, `results/factorial_ablation` |                                      mode × prompt causal isolation |         P1 | 对 Stage-3 机制解释重要，但不能单独构成主方法。([GitHub][23])                                                                       |
| IRIS improved results   | `results/iris_improved_20260417/*`                                |                        improved prompt / answer budget / retry runs |         P0 | 27B MATH n50 JSON 明确 acc=0.8, avg_tokens=3529.26；更大 headline 需要继续对 raw result 对齐。([GitHub][24])                  |
| Multiseed               | `results/multiseed_20260419/multiseed_summary.json`               |                                                  stability evidence |         P0 | 8B MATH-500 seeds 42/123/456，mean 0.7413, std 0.0152；sample sizes unequal。([GitHub][25])                         |
| Archive                 | `archive/`                                                        |                                             pre-pivot / stale lines |         P1 | 应保留为 historical negative evidence，不应用于 positive claim。                                                           |
| Idea-stage docs         | `idea-stage/*`                                                    |                      experiment plan, novelty scan, pivot rationale |         P1 | `FINAL_PROPOSAL.md` 明确 CTT null、post-hoc limitation、P0 online rewrite。([GitHub][26])                             |

## 仓库当前试图解决的问题

仓库当前问题不是“让模型想得更久”，而是：**在固定 output-token budget 下，thinking mode 的 reasoning tokens 与 final answer tokens 共享预算，导致 answer emission 被截断；non-thinking mode 反而在低预算更强；因此需要识别何时直接答、何时 thinking、何时把 reasoning 与 answering 分离。** 论文称这个现象为 **Coupling Tax**，方法称 **IRIS/MRSD split-budget cascade**。([GitHub][2])

当前已有方法核心假设：

1. natural stop 是 cheap confidence / routing signal；
2. thinking trace 即使没有完整 final answer，也可能包含足够信息；
3. 对 hit-budget / truncated traces，用 non-thinking / extraction prompt 做 Stage-3 answer generation 可以回收答案；
4. 在高 truncation regime，split budget 比 TOWN / thinking-only 更好。

当前主要入口：

* 主方法：`scripts/run_iris.py`
* deployment-faithful Stage2：`scripts/iris_online_stage2.py`
* benchmark / metric：`scripts/benchmarks.py`
* baseline：`scripts/run_nothink_baseline.py`, `scripts/run_budget_forcing.py`, `run_town_sample` inside `run_iris.py`
* results：`results/iris_*`, `results/multiseed_*`, `results/entropy_dynamics`, `results/ctt_pilot_*`
* paper claims：`paper/main_final.tex`, `paper/sections/*_final.tex`, top-level audits

---

# 2. Result Reliability Audit

| Result ID | Result Name                           | Dataset          | Metric                     |                                                            Claimed Value |                                                                             Logged Value | Config                             | Seed       | Command                               | Checkpoint    | Status                | Reliability                                    | Issue                                                                     |
| --------- | ------------------------------------- | ---------------- | -------------------------- | -----------------------------------------------------------------------: | ---------------------------------------------------------------------------------------: | ---------------------------------- | ---------- | ------------------------------------- | ------------- | --------------------- | ---------------------------------------------- | ------------------------------------------------------------------------- |
| R01       | 27B GSM8K NoThink vs Think            | GSM8K            | accuracy                   |                                                             98.0 vs 87.5 |                                                      audit/README only in inspected docs | b4096 claimed                      | unclear    | Missing exact command                 | no checkpoint | Partially Verified    | medium                                         | Strong claim, but exact command/raw JSON not inspected here.              |
| R02       | 8B GSM8K IRIS vs TOWN                 | GSM8K            | accuracy                   | README says IRIS 90.9 vs TOWN 86.0; audit notes baseline row discrepancy |                                                              Partially visible in audits | b? / ba? mixed                     | 42?        | Missing exact command                 | no checkpoint | Partially Verified    | medium                                         | `PAPER_CLAIM_AUDIT` flags 89/90/93 baseline ambiguity.([GitHub][10])      |
| R03       | 27B MATH-500 improved IRIS n50        | MATH-500         | accuracy, tokens           |                                   80.0 on n50; larger headline elsewhere |                                        JSON: acc 0.8, avg_tokens 3529.26, stage 10/11/29 | b1=512,b2=4096,ba=512              | 42         | inferred from JSON path               | none          | Verified              | medium                                         | n50 verified; not sufficient for full claim alone.([GitHub][27])          |
| R04       | 27B MATH-500 headline IRIS/TOWN       | MATH-500         | accuracy                   |                                 README/paper claims strong IRIS/TOWN gap |                                                           raw n200 not fully parsed here | b4096/ba512 likely                 | 42         | Missing exact command                 | none          | Partially Verified    | medium                                         | Strong signal but needs exact manifest, same-sample TOWN, online rerun.   |
| R05       | 8B MATH-500 multiseed IRIS            | MATH-500         | accuracy std               |                                                  mean 0.7413, std 0.0152 |                                JSON: seeds 42 n500=0.744, 123 n200=0.725, 456 n200=0.755 | likely b4096                       | 42/123/456 | Missing exact commands                | none          | Verified              | high for logged JSON; medium for general claim | Unequal n across seeds; still useful stability evidence.([GitHub][25])    |
| R06       | 8B MATH-500 improved prompt full n500 | MATH-500         | accuracy delta             |                                                      +0.4pp in narrative |                                                           docs/results directory visible | b4096/ba512                        | 42         | Missing exact command                 | none          | Partially Verified    | medium                                         | Important negative control: Stage-3 not universal.([GitHub][11])          |
| R07       | Entropy stopping NO-GO                | GSM8K / Qwen3-8B | entropy drop / correctness |                                                                    NO-GO | b512: acc .635, natural stop .475, decision NO-GO; b256: acc .2, natural stop .02, NO-GO | b256,b512                          | 42         | `collect_entropy_dynamics.py` in meta | none          | Verified              | high                                           | Rules out entropy-only mechanism.([GitHub][22])                           |
| R08       | CTT pilot null                        | GSM8K 27B        | AUC / null gap             |                                                             CTT negative |            aligned: mid_max_auc .535, mid_mean_auc .565, null .509, gap .026, pass false | n=200                              | 42         | script usage visible                  | none          | Verified              | high                                           | Cross-mode KL is not a useful router.([GitHub][28])                       |
| R09       | Budget forcing                        | MATH-500         | accuracy/tokens            |                                      optional positive/negative baseline |                                              result file exists; code undercounts tokens | b2048/b4096                        | 42         | script visible                        | none          | Possibly Contaminated | low / unusable for token claims                | Extra forced answer / wait tokens not counted.([GitHub][19])              |
| R10       | Learned allocator                     | MATH-500 8B      | simulated savings          |                                                   46.56% learned savings |                                                                            JSON verified | train/test split on same benchmark | 42         | script visible                        | none          | Partially Verified    | low for empirical claim                        | Simulated proxy, not actual run with predicted budgets.([GitHub][20])     |
| R11       | Alpha curve fit                       | MATH/GSM?        | RMSE/prediction            |                                                           theory support |                                                              train RMSE ~0 from 3 points | b={128,256,512}                    | unclear    | missing                               | none          | Partially Verified    | low                                            | Degenerate fit; held-out support insufficient.([GitHub][29])              |
| R12       | Pre-pivot AdaThink/Dynamic Halting    | older            | mixed                      |                                                                 obsolete |                                                                  README says flawed eval | old configs                        | mixed      | old                                   | none          | Possibly Contaminated | unusable                                       | train/eval same CSV + oracle-labeled stopping; archive only.([GitHub][2]) |

---

# 3. 代码正确性审查：Suspected Bug Table

| Priority | File                                                         | Function/Class                                   | Code Region                    | Suspicion                                                                                                   | Evidence                                                                                                 | How to Verify                                                                                                         | Proposed Fix for Claude Code                                                                                                         | Expected Effect                                              | Confidence                               |                       |      |
| -------: | ------------------------------------------------------------ | ------------------------------------------------ | ------------------------------ | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- | --------------------- | ---- |
|       P0 | `scripts/benchmarks.py`                                      | `normalize_latex`                                | whitespace normalization       | `while " " in s: s = s.replace(" ", " ")` 是无限循环，只要字符串含普通空格就 hang                                            | 直接读到代码。([GitHub][17])                                                                                    | `python - <<'PY'\nfrom scripts.benchmarks import normalize_latex\nprint(normalize_latex('6 - 5i'))\nPY` 应在 fix 前 hang | 改成 `s = re.sub(r"\\s+", "", s)` 或明确只压缩多空格：`while "  " in s`；补 unit test                                                              | 防止 MATH metric silent hang / missing samples                 | high                                     |                       |      |
|       P0 | `scripts/run_budget_forcing.py`                              | `generate_with_forcing`                          | return token count             | early_stop 追加 STOP 后最多 32 tokens，wait_extend 继续生成 remaining tokens，但函数只返回 `gen_len`                         | 代码显示 extra generation 后仍 `return text, gen_len`。([GitHub][19])                                           | 构造 fake tokenizer/model 或 run n=2，比较 decoded token length vs logged tokens                                            | 返回 `gen_len + forced_len + injected_token_len`，并分字段记录 `initial_tokens`, `forced_tokens`, `injected_tokens`, `total_generated_tokens` | 修正 budget forcing token baseline；旧 token-efficiency claim 降级 | high                                     |                       |      |
|       P0 | `scripts/run_iris.py`                                        | `generate_adaptive_thinking` / `run_iris_sample` | Stage2 accounting              | 默认一次生成完整 `max_think_tokens`，再 post-hoc 找 stop；`tokens_total` 用 `n_tokens_used`，elapsed 用完整生成时间              | 代码注释明确 “generate all at once… analyze trace”，`run_iris.py` 有 `--online_stage2` 才真实。([GitHub][4])         | 同一 n=5 对比 default vs `--online_stage2`，检查 `n_tokens_generated == n_tokens_used`                                       | 主线强制 `--online_stage2`；post-hoc runner 改名为 analysis-only                                                                             | 使 wall-clock/FLOPs/token-saving claims 合法                    | high                                     |                       |      |
|       P0 | `scripts/run_iris.py`                                        | Stage1 accept                                    | `if not hit_budget_s1: accept` | Stage1 只根据 natural stop 接受，无 answer validity / verifier；paper failure analysis也说 Stage0 false accept 是显著错误源 | paper reports Stage0 false accept contributes error class; code confirms no verifier gate.([GitHub][13]) | 统计 accepted Stage1 中 wrong rate / invalid parse rate                                                                  | 新增 `acceptance_verifier` / `valid_final_answer` gate；不通过则 escalate                                                                   | 降低 easy-gate false positives                                 | high                                     |                       |      |
|       P1 | `scripts/run_nothink_baseline.py`                            | argparse                                         | `--also_thinking`              | `action="store_true", default=True` 导致默认永远跑 thinking，无法通过 CLI 关闭                                            | 代码可见。([GitHub][18])                                                                                      | `python scripts/run_nothink_baseline.py --help` + dry parse                                                           | 改成 mutually exclusive `--also_thinking/--no_also_thinking`，默认显式写入 meta                                                               | 减少 baseline 混乱                                               | high                                     |                       |      |
|       P1 | `scripts/run_nothink_baseline.py` vs `scripts/benchmarks.py` | loaders/prompts                                  | MATH/GSM loaders               | GSM8K 使用 `openai/gsm8k` vs `gsm8k`；MATH prompt/parser不同                                                     | 代码中多套 loader/prompt。([GitHub][18])                                                                       | same seed 输出 sample IDs/hash 对齐                                                                                       | 所有脚本改用 `benchmarks.py` canonical loader + sample id manifest                                                                         | 消除 split/order/version mismatch                              | medium                                   |                       |      |
|       P1 | `scripts/learned_allocator.py`                               | train/eval                                       | allocator result               | 训练/测试来自同一 MATH-500 pool，且评估是 oracle-log simulation，不是真跑模型                                                   | 代码和 JSON 显示 400/100 split、simulated savings。([GitHub][20])                                               | 将 predicted budgets 实际 run n=50，对比 simulated tokens/accuracy                                                          | 改名为 `offline_allocator_analysis.py`；实际执行版另写                                                                                          | 防止把 proxy claim 当 empirical runtime result                   | high                                     |                       |      |
|       P1 | `scripts/run_ctt_pilot.py`                                   | `prefill_hidden_states`                          | device                         | `next(model.parameters()).device` 对 `device_map=auto` sharding 可能不稳                                         | 代码可见。([GitHub][21])                                                                                      | 在 multi-GPU auto map 上 n=1 smoke                                                                                      | 使用统一 `model_input_device` helper                                                                                                     | 降低 sharded model failure                                     | medium                                   |                       |      |
|       P1 | `scripts/run_iris.py` / paper                                | naming                                           | IRIS vs MRSD                   | README/runner 叫 IRIS，paper macro 叫 MRSD/fullmethod；主线叙事混乱                                                   | paper main/method/experiments可见。([GitHub][3])                                                            | grep `IRIS                                                                                                            | MRSD                                                                                                                                 | Mrsd`                                                        | 统一名称：旧 IRIS 留作 ablation，新方法单名 `RCV-IRIS` | 降低 reviewer confusion | high |
|       P2 | `scripts/run_iris.py`, `run_factorial_ablation.py`           | Stage3 prompt                                    | `The final answer is \boxed{`  | prompt 直接打开 `\boxed{`，部分结果出现 `ANSWER`/empty/`-`，说明 extraction prompt 仍有 formatting hazard                   | n50 raw shows multiple `pred="ANSWER"` / fallback cases。([GitHub][27])                                   | prompt variant ablation on n=50 with parser source distribution                                                       | 不强制裸 open brace；使用 natural boxed suffix + strict parser + retry accounting                                                           | 减少 parser-induced artifacts                                  | medium                                   |                       |      |
|       P2 | result logging                                               | all runners                                      | meta                           | 多个 JSON 有 model/seed，但缺 full command, git commit, HF revision, env hash, sample IDs                         | inspected JSON meta incomplete。([GitHub][27])                                                            | inspect JSON schema                                                                                                   | 新增 `run_manifest.jsonl` + `sample_manifest.json`                                                                                     | 提升 reproducibility                                           | high                                     |                       |      |

---

# 4. Claim-Code-Result Matrix

| Claim                                                             | Source File                                                       | Implementation File                                       | Result Evidence                        | Status                | Problem                                                                             | Confidence  |
| ----------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------- | -------------------------------------- | --------------------- | ----------------------------------------------------------------------------------- | ----------- |
| Coupling Tax exists under fixed output budget                     | README, `paper/sections/theory_final.tex`, `conclusion_final.tex` | `run_nothink_baseline.py`, `run_iris.py`, `benchmarks.py` | 27B GSM8K claim; MATH/GSM paper tables | Partially Supported   | Need full manifest + official baseline alignment                                    | medium      |
| Natural stop is high-PPV routing signal                           | `conclusion_final.tex`, `method_final.tex`                        | `run_iris.py` Stage1/TOWN                                 | stage distribution / paper claims      | Partially Supported   | Natural stop alone causes false accepts; should not be sole gate                    | medium-high |
| IRIS/MRSD split-budget beats TOWN on hard/high-truncation samples | `experiments_final.tex`, README                                   | `run_iris.py` Stage3                                      | 27B MATH n50 JSON, multiseed MATH      | Partially Supported   | Default post-hoc Stage2 and token accounting weaken deployment claim                | medium      |
| Stage-3 decoupled answer generation is key                        | `NARRATIVE_REPORT.md`, `experiments_final.tex`                    | `generate_decoupled_answer`, `run_factorial_ablation.py`  | prompt/root-cause notes, stage3 table  | Partially Supported   | It works in high-coupling regime but not universal; cannot be final mechanism alone | medium      |
| Entropy/stability stopping is useful                              | earlier IRIS naming / code                                        | `generate_adaptive_thinking`                              | entropy NO-GO                          | Contradicted          | Entropy-only is empirically weak                                                    | high        |
| CTT internal KL can detect tax                                    | idea-stage CTT docs                                               | `run_ctt_pilot.py`                                        | aligned AUC .535/.565, null gap .026   | Contradicted          | Should be archived as negative ablation                                             | high        |
| Learned allocator saves tokens                                    | README / result                                                   | `learned_allocator.py`                                    | MLP JSON                               | Partially Supported   | Only simulated oracle-log replay, not actual inference                              | high        |
| Budget forcing is fair baseline                                   | paper optional                                                    | `run_budget_forcing.py`                                   | code + result file                     | Possibly Contaminated | token undercount means token comparison invalid until fixed                         | high        |
| Theorem verified / decomposition predictive                       | README, theory                                                    | `fit_alpha_curves.py`, alpha JSON                         | alpha curve JSON                       | Partially Supported   | curve fit degenerate with few points; claim should be weak/accounting               | medium      |
| General across architectures                                      | paper / README                                                    | DeepSeek scripts/results                                  | limited                                | Unclear               | DeepSeek prompt-based nothink may not instantiate true mode separation              | medium      |
| Current results are submittable                                   | `PAPER_CLAIM_AUDIT.md`                                            | all                                                       | audit PASS with flagged issues         | Partially Supported   | Audit itself flags stale/token inconsistencies                                      | medium      |

---

# 5. Phenomenon Ledger

| ID  | Observation                                                                            | Type      | Where Found                            | Setting                            | Metric                  | Compared To             | Reliability     | What It Suggests                                                    | What It Rules Out                            | Confidence |
| --- | -------------------------------------------------------------------------------------- | --------- | -------------------------------------- | ---------------------------------- | ----------------------- | ----------------------- | --------------- | ------------------------------------------------------------------- | -------------------------------------------- | ---------- |
| P01 | NoThink can beat Thinking at low/fixed budgets                                         | Positive  | README/paper                           | 27B GSM8K, Qwen family             | accuracy                | thinking@same budget    | medium          | explicit thinking consumes answer budget; mode choice matters       | “thinking always helps”                      | medium     |
| P02 | Split-budget Stage3 helps when Stage2 hits budget                                      | Positive  | n50 JSON, paper/narrative              | 27B MATH high B2                   | acc/stage               | TOWN/truncated thinking | medium          | reasoning prefix can contain useful partial computation             | pure nothink-only as final path              | medium     |
| P03 | Gains concentrate in high-coupling-tax regime                                          | Mixed     | `NARRATIVE_REPORT.md`                  | 27B MATH strong; 8B MATH full weak | delta pp                | baseline                | high            | method needs regime detection                                       | universal Stage3 claim                       | high       |
| P04 | 8B MATH-500 full n500 improved Stage3 only +0.4pp                                      | Negative  | `NARRATIVE_REPORT.md`                  | Qwen3-8B MATH                      | acc                     | prompt variants         | medium          | extraction prompt not enough when recoverability margin low         | “prompt fix solves method”                   | high       |
| P05 | Multiseed IRIS MATH stable around 74%                                                  | Positive  | `multiseed_summary.json`               | Qwen3-8B MATH                      | acc std                 | seeds                   | high for logged | some robust signal exists                                           | all positives are seed luck                  | medium     |
| P06 | Entropy drop/stability NO-GO                                                           | Negative  | entropy results                        | Qwen3-8B GSM8K b256/b512           | entropy/correctness     | threshold hypothesis    | high            | hidden-token entropy is wrong signal                                | entropy-only adaptive halting                | high       |
| P07 | CTT KL nearly null                                                                     | Negative  | CTT aligned JSON                       | 27B GSM8K                          | AUC/gap                 | null scaffold           | high            | internal cross-mode KL is not cheap route criterion                 | CTT as main mechanism                        | high       |
| P08 | Stage0 false accepts are meaningful error source                                       | Negative  | paper failure analysis + code          | MATH/GSM cascade                   | error class             | accept-on-natural-stop  | medium-high     | need acceptance verifier                                            | natural stop as sufficient correctness proof | high       |
| P09 | Stage3 prompt/boxed parsing materially changes results                                 | Mixed     | narrative + raw n50                    | MATH                               | boxed/fallback rate     | old prompt              | high            | answer extraction is a mechanism and a confound                     | raw accuracy without parser-source audit     | high       |
| P10 | Online Stage2 exists but not default                                                   | Anomalous | `iris_online_stage2.py`, `run_iris.py` | all                                | token generated vs used | default post-hoc        | high            | deployment claim blocked until online default                       | wall-clock claim from post-hoc               | high       |
| P11 | Budget forcing baseline undercounts tokens                                             | Anomalous | code                                   | all budget forcing                 | avg_tokens              | IRIS/TOWN               | high            | token comparisons need canonical counter                            | old BF token-efficiency                      | high       |
| P12 | Learned allocator captures coarse difficulty but is simulated                          | Mixed     | code + JSON                            | MATH-500                           | savings proxy           | oracle replay           | high            | question features may help routing                                  | allocator as proved inference method         | high       |
| P13 | Alpha curve fit too thin                                                               | Unstable  | alpha JSON                             | few budgets                        | RMSE/predict            | held-out                | medium          | theory useful as diagnostic, not headline                           | precise law claim                            | medium     |
| P14 | DeepSeek prompt-based nothink may not behave as separate mode                          | Mixed     | conclusion/limitations                 | cross-arch                         | qualitative             | Qwen true mode          | medium          | architecture/mode separation is necessary                           | universal model-family claim                 | medium     |
| P15 | Existing related work increasingly overlaps NoThink/adaptive/early-exit/extraction gap | Anomalous | literature + external search           | 2025–2026                          | novelty risk            | current claims          | high            | paper must pivot to missing mechanism, not “NoThink beats thinking” | novelty from NoThink alone                   | high       |

---

# 6. Design Constraints

| Constraint ID | Derived From Observation | Constraint Type    | Meaning                                                      | Implication for New Method                                           | Confidence |
| ------------- | ------------------------ | ------------------ | ------------------------------------------------------------ | -------------------------------------------------------------------- | ---------- |
| C01           | P01/P02                  | Must Preserve      | split reasoning vs answering has real value under truncation | Keep split-budget as component, not final explanation                | high       |
| C02           | P03/P04                  | Must Generalize    | method only helps when recoverability margin exists          | Add query-/prefix-conditioned recoverability gate                    | high       |
| C03           | P08                      | Must Fix           | natural stop has false accepts                               | Add acceptance verifier before Stage0 accept                         | high       |
| C04           | P06                      | Must Avoid         | entropy/stability alone is wrong signal                      | Do not base main method on entropy thresholds                        | high       |
| C05           | P07                      | Must Avoid         | CTT KL is not predictive                                     | Archive CTT as negative evidence                                     | high       |
| C06           | P09                      | Must Control       | prompt/parser artifacts can manufacture gains                | Log parser source, boxed rate, fallback rate, retry rate             | high       |
| C07           | P10/P11                  | Must Stabilize     | token accounting can invalidate conclusions                  | Enforce generated/used token equality and count all forced tokens    | high       |
| C08           | P12                      | Must Test          | difficulty routing may help but only if actual inference     | Transform allocator into calibrated gate, not simulated claim        | medium     |
| C09           | P15                      | Must Differentiate | related work preempts NoThink/adaptive/early-exit claims     | Main novelty must be recoverability-calibrated mode/budget routing   | high       |
| C10           | R02/R03                  | Must Control       | same-sample comparisons needed                               | Always evaluate A/B/C on identical sample IDs, seed, hardware        | high       |
| C11           | P03/P14                  | Must Not Claim     | not universal across all models/tasks                        | Scope to models with genuine think/nothink modes and high truncation | high       |
| C12           | all                      | Must Test          | prove new mechanism, not old positive fragment               | Mandatory A/B/C: Existing Best Fragment, New w/o mechanism, Full New | high       |

---

# 7. Negative-to-Insight Analysis

| Negative Observation                | Failed Assumption                            | Why the Assumption Failed                                                            | What Mechanism Is Missing        | New Design Requirement                                |
| ----------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------- | ----------------------------------------------------- |
| 8B MATH full only +0.4pp            | Stage3 extraction universally improves       | Extraction only helps if prefix contains recoverable answer and prompt can elicit it | Recoverability estimator         | Gate Stage3 by prefix sufficiency / extraction margin |
| Entropy dynamics NO-GO              | entropy drop signals correctness             | Entropy is local token uncertainty, not answer-level sufficiency                     | Answer-level semantic confidence | Use answer agreement/verifier, not entropy-only       |
| CTT null                            | think/nothink hidden KL encodes failure      | Cross-mode logit-lens distance does not align with correctness labels                | Task-conditioned verifier        | Archive internal KL; use behavioral probes            |
| Stage0 false accept                 | natural stop = correct                       | Natural completion can be confidently wrong or parseable but wrong                   | Acceptance/rejection verifier    | Verify direct answer before accepting                 |
| Budget forcing token undercount     | baseline accounting fair                     | extra answer/wait generation changes token budget                                    | Unified token ledger             | Count injected and generated tokens separately        |
| Post-hoc Stage2                     | truncation accounting = compute saved        | generated trace already paid compute                                                 | Online controller                | Make online Stage2 mandatory                          |
| Learned allocator simulated success | offline oracle replay = deployable allocator | predicted budget not actually run; same benchmark split                              | Calibrated deployable router     | Run actual predicted budgets on held-out samples      |
| Prompt/boxed improvements           | method gain = reasoning mechanism            | formatting fixes recover parser failures but may not solve reasoning                 | Parser-source control            | Report boxed/fallback/retry and use prompt ablation   |
| DeepSeek mode issue                 | prompt can create genuine nothink mode       | some models do not expose distinct inference mode                                    | Mode-separation test             | Add precondition diagnostic for mode separability     |
| Related work overlap                | current claim is novel enough                | NoThink/adaptive/BAEE papers already close                                           | Mechanism-level differentiation  | Compare to official NoThinking/AdaptThink/DEER/BAEE   |

---

# 8. Method Synthesis Table

| Evidence Fragment                  | Source in Repo                       | What It Reveals                                         | Generalized Principle                          | Use in New Method?  | How to Transform It                                    |
| ---------------------------------- | ------------------------------------ | ------------------------------------------------------- | ---------------------------------------------- | ------------------- | ------------------------------------------------------ |
| Stage3 decoupled answer generation | `run_iris.py`, `NARRATIVE_REPORT.md` | mode switch can recover answer from truncated reasoning | answer emission should be separately budgeted  | Yes                 | Keep as action, not unconditional fallback             |
| Natural stop triage                | `run_iris.py`, paper                 | cheap signal of completion                              | completion evidence is useful but insufficient | Yes                 | Wrap with acceptance verifier                          |
| Entropy/stability                  | `run_iris.py`, entropy results       | token-level signal weak                                 | local uncertainty ≠ correctness                | Ablation only       | Remove from main gate; log only                        |
| CTT KL                             | `run_ctt_pilot.py`, CTT result       | internal mode distance unhelpful                        | hidden probes need strong labels               | Historical negative | Archive; cite as failed mechanism                      |
| Learned allocator                  | `learned_allocator.py`               | difficulty/routing may be predictable                   | per-query adaptive compute is needed           | Transform           | Use only as dev-calibrated router if actual run passes |
| Prompt/boxed fix                   | `NARRATIVE_REPORT.md`                | output format can dominate MATH scores                  | extraction quality is a separate variable      | Yes, controlled     | Standardize extraction prompt and log parser source    |
| Online Stage2 module               | `iris_online_stage2.py`              | deployment-faithful accounting is implementable         | real compute savings require online control    | Yes                 | Make mandatory for main method                         |
| Budget forcing                     | `run_budget_forcing.py`              | closest simple baseline                                 | reviewer-required baseline                     | Baseline only       | Fix token count and compare                            |
| Multiseed result                   | `multiseed_summary.json`             | positive not pure seed luck                             | stability must be measured                     | Yes                 | Require 3–5 seed CI before claim                       |
| Negative 8B MATH                   | narrative                            | method not universal                                    | need applicability gate                        | Yes                 | Add “do not extract if low recoverability”             |

---

# 9. Missing Mechanism Diagnosis

1. **Missing Mechanism Name:**
   **Recoverability-Calibrated Acceptance and Extraction Control**

2. **One-Sentence Diagnosis:**
   当前 IRIS/MRSD 已经发现“reasoning 与 answering 应分预算”，但缺少一个明确机制来判断**何时直接接受、何时继续 thinking、何时 Stage3 extraction 是可恢复的、何时 extraction 只是 prompt/parser trick 或会失败**。

3. **Evidence From Positive Results:**
   Stage3 在 27B MATH 和高 truncation 条件下能回收答案，说明 truncated reasoning prefix 有时包含可用信息。n50 27B MATH raw JSON 里大量 Stage3 after budget exhausted 成功，支持“prefix recoverability”存在。([GitHub][27])

4. **Evidence From Negative Results:**
   8B MATH full +0.4pp、entropy NO-GO、CTT null 说明“只要提取”“只看 entropy”“只看 internal KL”都不是核心机制。([GitHub][11])

5. **Evidence From Unstable Results:**
   multiseed 稳定性尚可，但 sample sizes unequal；baseline row/token inconsistencies 说明结论对 evaluation hygiene 敏感。([GitHub][25])

6. **Evidence From Failed Ablations:**
   CTT AUC/null gap 不过线；entropy b256/b512 都 NO-GO；这些失败说明缺的不是“更好的 hidden scalar”，而是 answer-level recoverability control。

7. **Why Existing Method Cannot Solve It:**
   `run_iris.py` 的 Stage1 接受条件是 not hit budget；Stage3 触发主要由 Stage2 是否 hit budget/entropy/natural stop 决定，而不是由“该 prefix 是否可恢复、forced extraction 是否可靠、direct answer 是否可信”决定。

8. **Why Simple Tuning Cannot Solve It:**
   调 `b1/b2/b_answer/tau_h/tau_s` 只能移动触发边界，不能识别 false accept、low-recoverability prefix、prompt-induced extraction shift，也不能修复 post-hoc accounting。

9. **Why Existing Best Positive Fragment Is Insufficient:**
   “Stage3 extraction + improved prompt”解释不了 8B MATH negative control、entropy/CTT null、Stage0 false accept，也难以和 2026 BAEE / detection-extraction gap 工作区分。

10. **What New Mechanism Must Do:**
    对每个样本在线估计：
    `acceptability(Stage0 answer)`, `recoverability(prefix)`, `extractability(prefix under low-shift prompt)`, `expected gain vs extra cost`，再选择 action。

11. **Confidence:**
    **medium-high** for missing mechanism; **medium** for proposed implementation success, because new method尚未跑实验。

---

# 10. New MAIN METHOD PATH

## 唯一推荐主线：RCV-IRIS

1. **Method Name Placeholder:**
   **RCV-IRIS: Recoverability-Calibrated Verifier IRIS**

2. **One-Sentence Core Idea:**
   把现有 IRIS 从“natural-stop / hit-budget 触发的 split-budget cascade”升级为**在线、可审计、由 acceptance verifier 与 prefix recoverability verifier 控制的 mode/budget/action policy**。

3. **Core Missing Mechanism It Adds:**
   Recoverability-calibrated control：显式判断一个答案/前缀是否值得接受、继续、提取或放弃。

4. **What Phenomena It Explains:**

   * NoThink 低预算强：Stage0 可接受样本多。
   * Stage3 高 truncation 有效：部分 prefix recoverable。
   * 8B MATH gain 小：recoverability / extractability margin 不足。
   * entropy/CTT 失败：token-level/internal scalar 不是 answer-level sufficiency。
   * Stage0 false accept：缺 acceptance gate。

5. **What Negative Results It Fixes:**
   不再把 entropy/CTT 当主机制；不再无条件 Stage3；不再自然停就接受；不再 post-hoc 报 deployment saving。

6. **What Existing Positive Signals It Generalizes:**
   Stage3 positive signal 被解释为“在高 recoverability + 高 truncation 条件下，answer budget decoupling 有正 utility”。

7. **Why Existing Best Path Is Not Enough:**
   现有最好路径是“improved Stage3 prompt + larger answer budget + retry”。这仍是 extraction fragment，不知道什么时候该用、什么时候会失败，也无法解释 negative controls。

8. **Core Mechanism:**
   对每个 query 维护在线状态 `h_t = {stage, tokens_used, natural_stop, parse_validity, answer_agreement, prefix_recoverability, extractor_margin}`，通过 verifier/gate 决定动作：

   * `ACCEPT_STAGE0`
   * `THINK_MORE`
   * `EXTRACT_STAGE3`
   * `FALLBACK_TOWN`
   * `ABSTAIN_LOG_FAILURE` for analysis

9. **New Objective / Loss:**
   推理时 policy maximizes calibrated utility：

   [
   a^*(q,h_t)=\arg\max_{a \in \mathcal{A}}
   \widehat{P}(\text{correct}\mid q,h_t,a)

   * \lambda \cdot \widehat{T}(a)
   * \eta \cdot \widehat{R}_{\text{false-accept}}(a)
     ]

   可选 dev calibration loss：

   [
   L_{\text{cal}} =
   \mathrm{CE}(g_0(q), y_{\text{stage0-correct}})

   * \mathrm{CE}(g_r(q,z_t), y_{\text{stage3-success}})
   * \beta \cdot \mathrm{ECE}(g_0,g_r)
   * \lambda \cdot \mathrm{token_cost}
     ]

10. **New Architecture or Module:**
    不是模型训练架构；是 inference-time controller modules：

    * `acceptance_verifier.py`
    * `recoverability_probe.py`
    * `rcv_policy.py`
    * `run_rcv_iris.py`
    * `run_manifest.py`

11. **New Training Procedure:**
    默认 training-free heuristic + dev calibration。若训练 gate，只允许用 held-out dev split，不允许在 test 上调 threshold。

12. **New Evaluation Protocol:**
    同一样本、同一硬件、同一 parser、同一 token counter，对比：
    A. Existing Best Positive Fragment Only
    B. New MAIN METHOD Without New Mechanism
    C. Full RCV-IRIS

13. **What Existing Components It Reuses:**
    `benchmarks.py` after fix, `iris_online_stage2.py`, Stage3 extraction code after prompt/accounting rewrite, TOWN baseline logic, result aggregation.

14. **What Existing Components It Deletes:**
    不直接删除旧实验；删除/禁用 main path 中 entropy/CTT gating。

15. **What Existing Components It Rewrites:**
    `run_iris.py` main path、budget forcing token accounting、baseline loader/prompt/manifest、paper claims.

16. **What Existing Components It Keeps Only as Ablation:**
    entropy stopping, CTT KL, Stage3-only, natural-stop-only, learned allocator.

17. **What Existing Components It Keeps Only as Baseline:**
    nothink, thinking, TOWN, budget forcing, existing best IRIS fragment.

18. **Why This Is Not Merely the Existing Best Path:**
    Existing path asks “does Stage3 help after truncation?” RCV-IRIS asks “is this specific answer/prefix recoverable enough to justify accept/extract/continue under cost?” That is a different decision variable.

19. **Why This Could Produce Real Positive Results:**
    It targets the two largest unresolved error modes: false accepts and bad extractions. It preserves the high-regime split-budget advantage while avoiding low-margin Stage3.

20. **Why This Is Mechanism-Level Different from Prior Work:**
    It is not merely NoThinking, not merely budget forcing, not merely early exit, not merely BAEE free continuation. The differentiator must be: **mode-aware, split-budget, recoverability-calibrated routing across nothink / thinking / extraction actions with unified token accounting**.

21. **Main Risk:**
    BAEE / Detection-Extraction Gap may subsume the recoverability idea; if official BAEE dominates with comparable cost, novelty collapses unless RCV-IRIS shows mode-aware split-budget advantage.

22. **Minimal Falsification Experiment:**
    On 100 MATH-500 hard/high-truncation samples and 100 GSM8K samples, Full RCV-IRIS must beat both:

    * Existing Best Fragment Only
    * RCV-IRIS without verifier gates
      while reducing Stage0 false accept and not increasing token cost unfairly.

23. **Confidence:**
    **medium**. Mechanism diagnosis is strong; empirical success remains unproven.

---

# 11. Formal Method Description

## Problem Setup

Given question (q), model (M) with thinking and non-thinking modes, fixed or soft token budget (B), choose an inference policy (\pi) that maximizes accuracy under cost:

[
\max_\pi \mathbb{E}[\mathbf{1}{\hat{y}*\pi(q)=y} - \lambda T*\pi(q)]
]

Existing IRIS decomposes budget into (B_1) nothink probe, (B_r) reasoning, (B_a) answer extraction. RCV-IRIS adds calibrated decisions between stages.

## Existing Method Failure

Existing IRIS routes mostly by natural stop / hit budget / entropy. That fails when:

* natural stop is wrong;
* prefix is not recoverable;
* forced extraction shifts model into wrong intermediate answer;
* post-hoc token savings do not equal compute savings.

## New Insight

The real object to estimate is not “entropy low?” but:

[
\Delta_{\text{extract}}(q,z_t)
==============================

## P(\text{Stage3 correct}\mid q,z_t)

## P(\text{TOWN/truncated correct}\mid q,z_t)

\lambda C_{\text{Stage3}}
]

and:

[
R_{\text{accept}}(q,a_0)
========================

P(\text{Stage0 answer wrong}\mid q,a_0)
]

## Algorithm: RCV-IRIS

**Input:** question (q), model (M), tokenizer, budgets (B_1,B_r,B_a), thresholds (\tau_0,\tau_r), token cost (\lambda)
**Output:** prediction (\hat{y}), full audit trace

Steps:

1. Run Stage0 in non-thinking mode with budget (B_1).
2. Parse answer (a_0); compute `valid_final`, `format_quality`, `self_agreement` or cheap verifier score (g_0(q,a_0)).
3. If natural stop and (g_0 \ge \tau_0), accept; otherwise escalate.
4. Run online Stage2 in thinking mode chunk-by-chunk using `iris_online_stage2.py`.
5. At checkpoints or budget exhaustion, compute prefix recoverability score (g_r(q,z_t)):

   * extraction probe validity;
   * strict/soft extraction agreement;
   * optional free-continuation mini probe if affordable;
   * parser-source confidence.
6. If natural Stage2 final answer is valid and verified, accept.
7. If truncated and (g_r \ge \tau_r), run Stage3 decoupled extraction with (B_a).
8. If (g_r < \tau_r) and remaining budget exists, continue thinking or move to next budget rung.
9. If still low, fallback to TOWN/truncated output but log as low-recoverability failure.
10. Write full manifest: sample id, model revision, token fields, verifier decisions, parser source, command, commit.

## Objective

[
L_{\text{total}} =
L_{\text{accept}}
+
L_{\text{recover}}
+
\beta L_{\text{calibration}}
+
\lambda L_{\text{cost}}
]

where:

* (L_{\text{accept}}): penalizes accepting wrong Stage0 answers; maps to Stage0 false accept phenomenon.
* (L_{\text{recover}}): predicts Stage3 success from prefix; maps to 8B MATH negative and 27B MATH positive.
* (L_{\text{calibration}}): prevents threshold overfitting / cherry-pick.
* (L_{\text{cost}}): ensures token accounting fairness.

If no training is used, these become logged decision scores and dev-tuned thresholds rather than gradient-trained losses.

## Required Logging

Every sample must log:

* `stage0_pred`, `stage0_valid`, `stage0_verifier_score`, `stage0_accept`
* `stage2_tokens_generated`, `stage2_tokens_used`, `online=True`
* `prefix_recoverability_score`
* `extractor_prompt_variant`
* `stage3_pred_source`: boxed / final / fallback / invalid
* `retry_used`
* `forced_tokens`, `injected_tokens`, `total_generated_tokens`
* `decision_action`
* `counterfactuals`: existing-best-fragment output, no-verifier output, full RCV output

## Required Ablations

1. Existing Best Positive Fragment Only.
2. Online IRIS without acceptance/recoverability verifier.
3. RCV with Stage0 verifier only.
4. RCV with prefix recoverability only.
5. Full RCV-IRIS.
6. Full RCV but forced extraction prompt removed.
7. Full RCV with BAEE-style free continuation probe, if implemented.
8. Same token budget, same samples.

---

# 12. Related Work and Novelty Risk

| Paper                                                             | Year / Venue        | Code                   | Mechanism                                                                                    | Why Close                                                           | Difference from New MAIN METHOD                                                                                   | Novelty Risk | Required Differentiation Experiment                                                        |
| ----------------------------------------------------------------- | ------------------- | ---------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------ |
| Reasoning Models Can Be Effective Without Thinking                | 2025 arXiv          | paper links code/media | NoThinking, parallel scaling, verifier/best-of-N aggregation                                 | Directly preempts “nothink beats thinking under low budget”         | RCV-IRIS uses NoThink only as one action inside mode/budget controller                                            | high         | Compare against NoThinking pass@k / best-of-N at matched latency/tokens.([arXiv][30])      |
| s1: Simple test-time scaling                                      | 2025 arXiv          | official GitHub linked | budget forcing via early termination / Wait extension                                        | Your repo has `run_budget_forcing.py`; reviewers will require it    | RCV routes among modes and split answer budget; not just force reasoning length                                   | medium-high  | Fix BF token accounting, run official or faithful BF baseline.([arXiv][31])                |
| Scaling LLM Test-Time Compute Optimally                           | 2024 arXiv          | code/media links       | compute-optimal allocation by prompt difficulty                                              | Adaptive per-prompt compute is core overlap                         | RCV’s novelty must be mode-aware recoverability + split answer budget                                             | medium       | Compare to compute-optimal best-of-N / verifier allocation.([arXiv][32])                   |
| AdaptThink                                                        | 2025 arXiv / EMNLP  | official code linked   | RL teaches model when to Think vs NoThink                                                    | Very close to adaptive mode selection                               | RCV is training-free/dev-calibrated and adds prefix recoverability/extraction action                              | high         | Matched model/task comparison vs AdaptThink official checkpoint/code.([arXiv][33])         |
| Dynamic Early Exit in Reasoning Models / DEER                     | 2025 arXiv          | paper links            | confidence at reasoning transition points                                                    | Close to early exit/control                                         | RCV controls accept/extract/continue across modes; DEER exits within thinking                                     | medium-high  | Same benchmark: DEER vs RCV, log token/accuracy Pareto.([arXiv][34])                       |
| Detection–Extraction Gap / BAEE                                   | 2026 arXiv          | official code linked   | prefix self-consistency/free continuations detect recoverability; forced extraction can fail | Extremely close to “recoverability” mechanism; highest novelty risk | RCV must avoid claiming forced extraction alone; must show mode-aware split-budget routing adds value beyond BAEE | very high    | Official BAEE baseline; RCV must beat BAEE or show complementary regime/cost.([arXiv][35]) |
| Let Me Think! Long CoT can be worth exponentially many short ones | NeurIPS 2025 poster | supplementary          | sequential vs parallel scaling theory                                                        | Challenges “skip thinking” overgeneralization                       | RCV scopes to budget-constrained/high-truncation regimes, not anti-CoT broadly                                    | medium       | Include tasks where long sequential CoT wins; report failure modes.([OpenReview][36])      |
| Current repo’s CTT / entropy attempts                             | internal            | repo code              | hidden KL / entropy stopping                                                                 | Mechanism-near negative baselines                                   | RCV explicitly abandons them as primary signal                                                                    | low          | Keep as negative ablation, not novelty claim.                                              |

**Novelty verdict:**
当前 paper 若仍主打 “NoThink beats Think / natural stop + Stage3 extraction” 风险很高。2026 BAEE 尤其危险，因为它明确指出 forced extraction can fail and recoverability should be measured by free continuations,且提出 Black-box Adaptive Early Exit。RCV-IRIS 还能成立的唯一新颖空间是：**mode-aware split-budget routing + acceptance control + recoverability-calibrated extraction/continuation policy**。如果实验只显示 Stage3 prompt 更好，novelty 不足。

---

# 13. Keep / Delete / Rewrite / Archive Plan

| Item                               | Type         | File / Directory / Claim / Experiment     | Current Role               | Problem Under New MAIN PATH | Action                                    | Reason                        |
| ---------------------------------- | ------------ | ----------------------------------------- | -------------------------- | --------------------------- | ----------------------------------------- | ----------------------------- |
| Benchmark parser                   | code         | `scripts/benchmarks.py`                   | metric core                | infinite-loop bug           | REWRITE                                   | P0 correctness                |
| Online Stage2                      | code         | `scripts/iris_online_stage2.py`           | deployment-faithful Stage2 | not default                 | MERGE INTO NEW METHOD                     | Required for honest compute   |
| Post-hoc Stage2                    | code         | `run_iris.py` default                     | analysis convenience       | invalid deployment saving   | KEEP ONLY AS ABLATION                     | Mark analysis-only            |
| Stage3 extraction                  | code         | `generate_decoupled_answer`               | positive fragment          | no recoverability control   | MERGE INTO NEW METHOD                     | Keep as controlled action     |
| Entropy gating                     | code/claim   | `tau_h`, `tau_s` main logic               | old IRIS mechanism         | NO-GO                       | KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE | log only                      |
| CTT                                | code/results | `run_ctt_pilot.py`, `results/ctt_pilot_*` | failed route signal        | null                        | ARCHIVE                                   | negative ablation             |
| Learned allocator                  | code/result  | `learned_allocator.py`, result JSON       | simulated routing          | not actual inference        | KEEP ONLY AS ABLATION                     | Rename offline analysis       |
| Budget forcing                     | baseline     | `run_budget_forcing.py`                   | baseline                   | token undercount            | REWRITE                                   | Must be fair                  |
| NoThink baseline                   | baseline     | `run_nothink_baseline.py`                 | baseline                   | CLI/prompt mismatch         | REWRITE                                   | Canonical baseline            |
| TOWN baseline                      | baseline     | inside `run_iris.py`                      | baseline                   | tied to main script         | KEEP ONLY AS BASELINE                     | modularize                    |
| Old pre-pivot files                | docs/results | `archive/`, old AdaThink docs             | history                    | contaminated eval           | ARCHIVE                                   | never cite as positive        |
| Result tables                      | results      | `results/*`                               | evidence                   | mixed reliability           | FREEZE                                    | add reliability manifest      |
| Paper claim “theorem verified”     | claim        | README/paper                              | theory claim               | overstrong                  | REWRITE                                   | call accounting/decomposition |
| Paper claim “Stage3 key universal” | claim        | experiments/conclusion                    | contribution               | contradicted by 8B MATH     | REWRITE                                   | scope to high recoverability  |
| Paper related work                 | paper        | `related_final.tex`                       | positioning                | misses BAEE risk            | REWRITE                                   | required for integrity        |
| `PAPER_CLAIM_AUDIT.md`             | audit        | top-level                                 | claim hygiene              | useful but incomplete       | KEEP                                      | update after RCV              |
| `DATA_PROVENANCE.md`               | audit        | top-level                                 | provenance                 | useful                      | KEEP                                      | expand with manifest          |
| `CLAUDE.md`                        | handoff      | top-level                                 | commands/context           | stale mixed                 | REWRITE                                   | align with RCV plan           |
| `NARRATIVE_REPORT.md`              | docs         | top-level                                 | diagnosis                  | valuable                    | KEEP                                      | becomes historical rationale  |
| RCV modules                        | new code     | `scripts/rcv_*`                           | new method                 | absent                      | REWRITE / ADD                             | main path                     |

---

# 14. Claude Code Implementation Plan

## Task 1: Freeze and label old evidence

**Purpose:** Prevent old positive fragments and contaminated results from silently driving new claims.
**Which Phenomenon / Constraint It Addresses:** P10/P11/C07/C12.
**Why It Supports New MAIN METHOD PATH:** RCV needs clean A/B/C comparison.
**Files to Inspect:** `results/`, `DATA_PROVENANCE.md`, `PAPER_CLAIM_AUDIT.md`, `README.md`.
**Files to Edit:** add `results/RESULT_RELIABILITY_LEDGER.md`, update `DATA_PROVENANCE.md`.
**Files to Delete / Archive:** none.
**Functions / Classes:** none.
**Exact Change:** Create a ledger tagging each result dir as `verified_log`, `partial`, `possibly_contaminated`, `archive_only`; mark budget forcing token results and post-hoc Stage2 savings as contaminated until fixed.
**Do Not Change:** raw result JSONs.
**Verification Command:** `python scripts/check_result_ledger.py --results results --ledger results/RESULT_RELIABILITY_LEDGER.md` after adding script.
**Expected Result:** every result dir has a reliability tag.
**Failure Means:** missing/untracked result dirs.
**Rollback Condition:** ledger omits or mutates raw data.
**Priority:** P0.
**Confidence:** high.

## Task 2: Fix benchmark parser and add metric tests

**Purpose:** Remove metric hang/corruption.
**Which Phenomenon / Constraint It Addresses:** C06/C07.
**Files to Inspect:** `scripts/benchmarks.py`.
**Files to Edit:** `scripts/benchmarks.py`, add `tests/test_benchmarks.py`.
**Exact Change:** Replace whitespace loop; add tests for `"6 - 5i"`, `"(3, \\frac{\\pi}{2})"`, `"x=5"`, nested boxed fractions.
**Do Not Change:** correctness semantics beyond bug fix unless tests expose issue.
**Verification Command:** `python -m pytest tests/test_benchmarks.py -q`.
**Expected Result:** tests pass under 2 seconds.
**Failure Means:** metric unreliable; stop.
**Rollback Condition:** accuracy on a fixed parser fixture changes without explanation.
**Priority:** P0.
**Confidence:** high.

## Task 3: Canonicalize dataset/sample manifests

**Purpose:** Ensure all A/B/C run on identical samples.
**Files to Inspect:** `scripts/benchmarks.py`, `run_iris.py`, `run_nothink_baseline.py`, `run_budget_forcing.py`.
**Files to Edit:** `scripts/benchmarks.py`, add `scripts/make_sample_manifest.py`.
**Exact Change:** All runners accept `--sample_manifest`; manifest stores benchmark, split, HF dataset id, sample index, question hash, gold hash, seed.
**Do Not Change:** benchmark definitions.
**Verification Command:** `python scripts/make_sample_manifest.py --benchmark math500 --n_samples 10 --seed 42 --output /tmp/math10.json && python scripts/make_sample_manifest.py --benchmark math500 --n_samples 10 --seed 42 --output /tmp/math10b.json && diff /tmp/math10.json /tmp/math10b.json`.
**Expected Result:** identical manifests.
**Failure Means:** seed/order not reproducible.
**Priority:** P0.
**Confidence:** high.

## Task 4: Fix budget forcing token accounting

**Purpose:** Make baseline fair.
**Files to Inspect/Edit:** `scripts/run_budget_forcing.py`.
**Exact Change:** Track `initial_generated_tokens`, `injected_prompt_tokens`, `forced_generated_tokens`, `total_generated_tokens`; report both output-only and injection-inclusive counts.
**Do Not Change:** budget forcing algorithm otherwise.
**Verification Command:** `python scripts/run_budget_forcing.py --model Qwen/Qwen3-8B --benchmark math500 --n_samples 2 --budget 128 --variant early_stop --seed 42 --output_dir results/smoke/bforce_fixed`.
**Expected Result:** per-sample has token fields; `total_generated_tokens >= initial_generated_tokens`.
**Failure Means:** baseline token claims invalid.
**Priority:** P0.
**Confidence:** high.

## Task 5: Make online Stage2 mandatory for main method

**Purpose:** Remove post-hoc compute savings from main path.
**Files:** `scripts/run_iris.py`, `scripts/iris_online_stage2.py`.
**Exact Change:** Add `--stage2_mode {online,posthoc_analysis}` default `online`; require explicit `posthoc_analysis` flag for old behavior; write `online=True` in meta.
**Do Not Change:** old post-hoc function, except labeling.
**Verification Command:** `python scripts/run_iris.py --model Qwen/Qwen3-8B --benchmark math500 --n_samples 2 --b1 64 --b2_max 128 --b_answer 64 --stage2_mode online --output_dir results/smoke/iris_online`.
**Expected Result:** `stage2.tokens_generated == stage2.tokens_used` for all Stage2 samples.
**Failure Means:** cannot make deployment claim.
**Priority:** P0.
**Confidence:** high.

## Task 6: Implement RCV verifier signals

**Purpose:** Add missing mechanism.
**Files to Add:** `scripts/rcv_signals.py`, `tests/test_rcv_signals.py`.
**Exact Change:** Implement:

* `answer_validity_score(text, benchmark)`
* `stage0_acceptance_features(question, pred, raw_text, parse_source)`
* `prefix_recoverability_features(question, prefix, extraction_outputs)`
* `extractor_margin(strict_pred, soft_pred, parse_sources)`
  No model calls yet; pure feature layer.
  **Verification Command:** `python -m pytest tests/test_rcv_signals.py -q`.
  **Expected Result:** deterministic features.
  **Failure Means:** cannot audit verifier.
  **Priority:** P0.
  **Confidence:** medium-high.

## Task 7: Implement cheap acceptance verifier

**Purpose:** Reduce Stage0 false accepts.
**Files to Add/Edit:** `scripts/rcv_verifier.py`, `scripts/run_rcv_iris.py`.
**Exact Change:** Add optional model-based verifier prompt with strict token cap; default uses deterministic validity + self-agreement if model verifier disabled.
**Do Not Change:** baseline Stage0 behavior.
**Verification Command:** `python scripts/run_rcv_iris.py --model Qwen/Qwen3-8B --benchmark gsm8k --n_samples 5 --b1 128 --b2_max 256 --b_answer 64 --enable_stage0_verifier --output_dir results/smoke/rcv_stage0`.
**Expected Result:** per-sample `stage0_verifier_score`, `stage0_accept_decision`.
**Failure Means:** cannot diagnose false accepts.
**Priority:** P0.
**Confidence:** medium.

## Task 8: Implement prefix recoverability gate

**Purpose:** Decide whether Stage3 is justified.
**Files:** `scripts/rcv_verifier.py`, `scripts/run_rcv_iris.py`.
**Exact Change:** On Stage2 truncation, run strict and soft extraction probes within `B_a_probe`; compute agreement and parse validity; optional free-continuation mini-probe.
**Do Not Change:** final Stage3 extraction budget accounting.
**Verification Command:** `python scripts/run_rcv_iris.py --model Qwen/Qwen3-8B --benchmark math500 --n_samples 5 --b1 128 --b2_max 512 --b_answer 128 --enable_recoverability_gate --output_dir results/smoke/rcv_recover`.
**Expected Result:** per-sample `recoverability_score`, `extractor_margin`, `decision_action`.
**Failure Means:** new mechanism not measurable.
**Priority:** P0.
**Confidence:** medium.

## Task 9: Add A/B/C experiment harness

**Purpose:** Prove not just old fragment.
**Files to Add:** `scripts/run_rcv_ablation_suite.py`.
**Exact Change:** Run same manifest through:
A. existing best fragment only
B. RCV without verifier gates
C. full RCV-IRIS
and aggregate with paired stats.
**Verification Command:** `python scripts/run_rcv_ablation_suite.py --model Qwen/Qwen3-8B --benchmark math500 --n_samples 10 --seed 42 --output_dir results/smoke/abc`.
**Expected Result:** one table with same sample IDs across A/B/C.
**Failure Means:** cannot validate method.
**Priority:** P0.
**Confidence:** high.

## Task 10: Add paper-claim guard

**Purpose:** Prevent overclaiming.
**Files:** `PAPER_CLAIM_AUDIT.md`, `paper/sections/*_final.tex`.
**Exact Change:** Do not update positive claims until A/B/C minimal experiments pass. Add TODO blocks for RCV thesis.
**Verification Command:** `grep -R "SOTA\\|state-of-the-art\\|guarantee\\|universal" paper/sections README.md`.
**Expected Result:** no unsupported SOTA/universal claims.
**Failure Means:** academic-integrity risk.
**Priority:** P1.
**Confidence:** high.

---

# 15. Minimal Verification Experiments

| Priority | Experiment                               | Hypothesis                          | Command                                                                                                                                                  | Config    | Dataset          | Seeds | Metric                   | Success Criterion                                | Failure Interpretation              |
| -------: | ---------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------------- | ----- | ------------------------ | ------------------------------------------------ | ----------------------------------- |
|       P0 | smoke test                               | code imports/run                    | `python scripts/run_rcv_iris.py --model Qwen/Qwen3-8B --benchmark gsm8k --n_samples 2 --b1 64 --b2_max 128 --b_answer 64 --output_dir results/smoke/rcv` | tiny      | GSM8K            | 42    | no crash                 | JSON with manifest                               | implementation broken               |
|       P0 | data sanity                              | same samples across methods         | `python scripts/make_sample_manifest.py --benchmark math500 --n_samples 20 --seed 42 --output results/smoke/math20.json`                                 | manifest  | MATH-500         | 42    | hash match               | A/B/C same IDs                                   | invalid comparison                  |
|       P0 | metric sanity                            | parser not hanging                  | `python -m pytest tests/test_benchmarks.py -q`                                                                                                           | unit      | fixtures         | n/a   | pass                     | all pass                                         | stop                                |
|       P0 | one-batch overfit / deterministic replay | same input same output under greedy | run same n=3 twice                                                                                                                                       | temp=0    | GSM8K            | 42    | exact JSON pred equality | deterministic or logged nondet                   | hardware/model nondeterminism       |
|       P0 | checkpoint/loading check                 | model/tokenizer revision logged     | smoke command                                                                                                                                            | Qwen3-8B  | GSM8K            | 42    | meta fields              | revision/hash present                            | reproducibility gap                 |
|       P0 | reproduce current negative entropy       | entropy not main signal             | existing entropy script or static result                                                                                                                 | b256/b512 | GSM8K            | 42    | NO-GO                    | matches NO-GO                                    | if positive, re-evaluate constraint |
|       P0 | reproduce CTT null                       | CTT not main signal                 | `python scripts/run_ctt_pilot.py ... --n_samples 20`                                                                                                     | tiny      | GSM8K            | 42    | AUC                      | no strong AUC                                    | if strong, inspect sampling         |
|       P0 | reproduce current best fragment          | old Stage3 signal exists            | `python scripts/run_iris.py ... --stage2_mode online --run_town`                                                                                         | old best  | MATH-500         | 42    | acc/tokens               | within ±2pp on n small not expected; sanity only | old result not stable               |
|       P0 | new mechanism activation check           | verifier changes decisions          | `python scripts/run_rcv_iris.py ... --enable_stage0_verifier --enable_recoverability_gate`                                                               | RCV       | MATH-500         | 42    | gate rate                | nonzero accept/reject/extract decisions          | mechanism inactive                  |
|       P0 | Existing Best Positive Fragment Only     | old fragment alone baseline         | `python scripts/run_rcv_ablation_suite.py --arm existing_fragment`                                                                                       | A         | MATH/GSM         | 42    | acc/tokens               | baseline logged                                  | cannot compare                      |
|       P0 | New MAIN Without New Mechanism           | online IRIS no verifier             | `python scripts/run_rcv_ablation_suite.py --arm rcv_no_gate`                                                                                             | B         | MATH/GSM         | 42    | acc/tokens               | comparable to IRIS                               | if better than full, gate harmful   |
|       P0 | Full New MAIN METHOD                     | RCV improves                        | `python scripts/run_rcv_ablation_suite.py --arm full_rcv`                                                                                                | C         | MATH/GSM         | 42    | acc/tokens               | beats A and B paired                             | if not, stop/pivot                  |
|       P1 | Stage0 verifier ablation                 | reduces false accepts               | suite with `--disable_stage0_verifier`                                                                                                                   | ablation  | GSM8K/MATH       | 42    | false accept rate        | lower false accepts no big acc loss              | verifier useless                    |
|       P1 | recoverability gate ablation             | avoids bad Stage3                   | suite with `--disable_recoverability_gate`                                                                                                               | ablation  | MATH hard subset | 42    | Stage3 success rate      | higher precision Stage3                          | gate not predictive                 |
|       P1 | small baseline comparison                | fair baseline                       | fixed nothink/think/TOWN/BF                                                                                                                              | unified   | MATH/GSM         | 42    | paired acc/tokens        | no weak baseline                                 | baseline gap                        |
|       P1 | multi-seed stability                     | not seed pick                       | suite n=100 seeds 42/123/456                                                                                                                             | small     | MATH/GSM         | 3     | mean/std/CI              | full RCV mean > A/B                              | unstable                            |
|       P1 | expansion gate                           | scale to n=500 only if small passes | suite n=500                                                                                                                                              | full      | MATH-500         | 42    | acc CI                   | CI lower > baseline                              | no full claim                       |
|       P1 | official BF reproduction                 | baseline integrity                  | official/faithful s1 BF                                                                                                                                  | matched   | MATH/GSM         | 42    | acc/tokens               | included                                         | reviewer risk                       |
|       P1 | official BAEE comparison                 | novelty                             | BAEE official code                                                                                                                                       | matched   | MATH-500         | 42    | Pareto                   | RCV complementary/beats in regime                | novelty collapse                    |
|       P2 | robustness/generalization                | not MATH-only                       | BBH / code if supported                                                                                                                                  | fixed     | BBH              | 42    | acc/tokens               | no hidden negative                               | scope narrow                        |
|       P2 | statistical significance                 | claim validity                      | paired bootstrap/McNemar                                                                                                                                 | final     | full             | ≥3    | CI/p                     | significant vs A/B                               | no strong claim                     |

---

# 16. Baseline and SOTA Plan

| Baseline                             | Why Required                    | Official Code                 | Dataset                 | Metric                         | Reproduction Requirement                                 | Fairness Risk                                                                    |
| ------------------------------------ | ------------------------------- | ----------------------------- | ----------------------- | ------------------------------ | -------------------------------------------------------- | -------------------------------------------------------------------------------- |
| NoThinking direct                    | closest simple baseline         | yes for NoThinking paper      | GSM8K/MATH/BBH          | acc/tokens/latency             | same model/budget                                        | current repo already overlaps; must not claim novelty over it without comparison |
| Thinking fixed budget                | basic baseline                  | n/a                           | all                     | acc/tokens                     | same prompt/parser                                       | token budget must include answer tokens                                          |
| TOWN                                 | existing repo baseline          | internal                      | all                     | acc/tokens                     | same sample manifest                                     | must not use weaker prompt than IRIS                                             |
| Existing Best IRIS Fragment          | prove new mechanism             | internal                      | all                     | acc/tokens                     | frozen exact config                                      | cannot let new method just inherit it                                            |
| Budget forcing / s1-style            | reviewer-required               | official s1 exists            | MATH/GSM                | acc/tokens                     | fixed token counter first                                | current script undercounts tokens                                                |
| AdaptThink                           | adaptive mode selection overlap | official code linked          | math datasets           | acc/length                     | official checkpoint or faithful implementation           | training vs training-free fairness                                               |
| DEER                                 | dynamic early exit overlap      | paper code if available       | MATH/GSM/GPQA           | acc/tokens                     | same model if possible                                   | requires token/logprob/markers                                                   |
| BAEE                                 | highest novelty-risk baseline   | official code linked by paper | MATH-500/GPQA/HumanEval | acc/serial reduction/API calls | official implementation                                  | may dominate RCV; must compare                                                   |
| Self-consistency / best-of-N NoThink | low-latency parallel competitor | standard                      | MATH/GSM                | acc/tokens/latency             | same total compute                                       | serial vs parallel latency accounting                                            |
| Official task SOTA                   | claim boundary                  | task-specific                 | each benchmark          | acc                            | only if reproducible or clearly paper-reported reference | cannot claim SOTA unless fair                                                    |

---

# 17. Paper Thesis Reconstruction

1. **New Paper Thesis:**
   Budget-constrained reasoning failures are not solved by “think less” or “extract harder” alone; they require recoverability-calibrated routing across non-thinking, thinking, and answer-extraction modes.

2. **Main Technical Contribution:**
   RCV-IRIS: an online inference-time controller that separates acceptance, reasoning, and extraction decisions using calibrated answer/prefix recoverability signals.

3. **Main Empirical Claim:**
   If experiments pass: Full RCV-IRIS improves the accuracy–token Pareto frontier over existing IRIS fragment, TOWN, budget forcing, and NoThinking baselines on high-truncation regimes, while preserving or improving stability.

4. **What Previous Failures Taught Us:**
   Entropy, hidden KL, and unconditional Stage3 are insufficient; the missing variable is recoverability.

5. **What We Should Not Claim:**

   * universal superiority over thinking;
   * SOTA;
   * entropy/stability is the mechanism;
   * CTT is predictive;
   * wall-clock/FLOPs savings from post-hoc traces;
   * Stage3 works on all datasets/models.

6. **What We Can Claim If Experiments Pass:**
   RCV-IRIS provides evidence that answer-level recoverability control is a better routing principle than natural-stop-only or entropy-only control.

7. **Required Baselines:**
   NoThinking, Thinking, TOWN, Budget Forcing, Existing Best Fragment, BAEE, DEER/AdaptThink if feasible.

8. **Required Ablations:**
   verifier off, recoverability gate off, Stage3 off, online vs post-hoc analysis-only, prompt variants, token counter.

9. **Required Robustness Tests:**
   3+ seeds, full MATH-500, GSM8K, BBH subset, cross-model Qwen 8B/27B, optional DeepSeek only if true mode separability passes.

10. **Reviewer Likely Objections:**
    “This is BAEE/AdaptThink/NoThinking renamed”; “baseline weak”; “post-hoc token accounting”; “prompt/parser artifact”; “negative results hidden.”

11. **How New MAIN METHOD Answers Them:**
    official baselines, A/B/C mechanism ablation, parser-source logging, online accounting, negative ablation retention.

12. **What Would Make This NeurIPS-Strong:**
    clear mechanism + fair official baselines + strong paired gains + negative results honestly scoped.

13. **What Would Make This Rejected:**
    only improved Stage3 prompt, missing BAEE comparison, token/accounting bugs unresolved, overclaiming.

14. **What Would Be Required for Oral-Level Strength:**
    RCV-IRIS consistently beats BAEE/DEER/AdaptThink-like baselines in a clearly defined regime, with strong mechanistic analysis of recoverability.

15. **What Would Be Required for Best-Paper-Level Strength:**
    A general theory + broad empirical law showing when reasoning state is recoverable/extractable across model families/tasks, with robust open-source implementation and official baseline domination. This is far beyond current evidence.

---

# 18. Reviewer Risk Assessment

| Risk                       | Why Reviewer May Object             | Evidence Needed            | How New MAIN METHOD Addresses It                           | Remaining Weakness                    |
| -------------------------- | ----------------------------------- | -------------------------- | ---------------------------------------------------------- | ------------------------------------- |
| novelty risk               | BAEE/AdaptThink/NoThinking close    | official comparisons       | positions as recoverability-calibrated mode/budget routing | BAEE may still dominate               |
| incremental risk           | Stage3 prompt tweak looks minor     | A/B/C ablation             | proves verifier/recoverability gate adds value             | only if effect size clear             |
| baseline weakness          | budget forcing bug, prompt mismatch | fixed baseline logs        | canonical token counter/manifest                           | official code still needed            |
| reproducibility            | no exact commands/revisions         | run manifest               | add manifest schema                                        | old results remain partial            |
| cherry-picking             | many result dirs                    | reliability ledger         | freeze all results, report negatives                       | must enforce in paper                 |
| negative hiding            | entropy/CTT failed                  | include negative ablations | archive as evidence                                        | space pressure in paper               |
| overclaiming               | “theorem verified”, “universal”     | claim audit                | rewrite thesis                                             | discipline required                   |
| unclear mechanism          | old mechanism mixed                 | mechanism logging          | recoverability scores                                      | verifier may be noisy                 |
| ablation insufficiency     | Stage3-only may drive               | A/B/C required             | paired same-sample ablation                                | compute cost                          |
| dataset limitation         | math-heavy                          | BBH/code/GPQA              | scope high-truncation math first                           | broad generalization unproven         |
| compute unfairness         | serial/parallel/token mismatch      | token/latency/FLOPs logs   | count all calls/tokens                                     | API-call baselines tricky             |
| implementation reliability | parser/token bugs                   | unit tests                 | P0 fixes first                                             | cannot rescue old contaminated claims |
| related work omission      | BAEE recent                         | rewritten related          | cite/compare BAEE                                          | high novelty risk remains             |

---

# 19. Final Decision

## 1. One-Sentence Verdict

从所有正面、负面、不稳定和失败现象推导出的唯一主路线是：**把 IRIS/MRSD 改造成 RCV-IRIS，即一个在线、可审计、由 acceptance verifier 和 prefix recoverability verifier 控制的 split-budget mode-routing 方法。**

## 2. Current Most Likely Root Cause

当前失败最可能不是单一 code bug，而是：

* **missing mechanism**：缺 recoverability-calibrated decision control；
* **evaluation/accounting bug**：post-hoc Stage2 与 budget forcing token undercount；
* **method assumption failure**：natural stop / entropy / CTT 不能作为充分路由信号；
* **novelty issue**：NoThinking、AdaptThink、DEER、BAEE 已覆盖大量相邻空间。

## 3. Why This Is Not Just the Existing Best Path

Existing best path = improved Stage3 extraction prompt + answer budget/retry。
RCV-IRIS = 对每个样本先估计 direct answer acceptability，再估计 prefix recoverability，再决定 accept / continue / extract / fallback。它改变的是**决策变量**，不是把 Stage3 prompt 调得更保守。

## 4. Phenomena Explained

RCV-IRIS 同时解释：

* NoThinking 低预算正面；
* Stage3 在 27B MATH 高 truncation 正面；
* 8B MATH full weak/negative；
* entropy/CTT null；
* Stage0 false accepts；
* prompt/parser sensitivity；
* post-hoc vs online accounting gap。

## 5. Mechanism Missing in Current Method

**Answer/prefix recoverability control**：当前没有判断“这个答案是否可信”“这个 reasoning prefix 是否足够”“forced extraction 是否会产生 distribution shift”。

## 6. New Mechanism

**Recoverability-Calibrated Verifier Gate**：

* Stage0 acceptance verifier；
* Stage2 online prefix recoverability gate；
* Stage3 extraction only when expected utility positive；
* unified token/call accounting；
* paired counterfactual logging。

## 7. What to Delete / Archive / Rewrite

* **ARCHIVE:** CTT as negative, pre-pivot AdaThink/Dynamic Halting, entropy main gating.
* **REWRITE:** `benchmarks.py`, `run_budget_forcing.py`, `run_iris.py`, `run_nothink_baseline.py`, paper related/claims.
* **KEEP AS BASELINE:** NoThink, Think, TOWN, Budget Forcing after fix, Existing Best Fragment.
* **MERGE INTO NEW:** online Stage2, controlled Stage3 extraction, manifest logging.

## 8. First Five Claude Code Tasks

1. Freeze old result reliability ledger.
2. Fix `benchmarks.py::normalize_latex` and add metric tests.
3. Add sample manifest and canonical loaders.
4. Fix `run_budget_forcing.py` token accounting.
5. Make online Stage2 mandatory for main method.

## 9. Minimal Experiments

最小队列：

1. smoke + parser sanity；
2. same-sample manifest；
3. reproduce entropy/CTT negatives；
4. reproduce existing best positive fragment；
5. A/B/C:

   * Existing Best Positive Fragment Only；
   * New Method Without New Mechanism；
   * Full RCV-IRIS；
6. Stage0 verifier ablation；
7. recoverability gate ablation；
8. 3-seed small stability；
9. official BAEE/budget forcing comparison gate。

## 10. Continue / Stop / Pivot Criteria

**Continue** if Full RCV-IRIS beats A and B on paired same-sample tests, reduces Stage0 false accepts, and does not rely on token/accounting artifacts.

**Stop** if Full RCV-IRIS does not improve over Existing Best Fragment Only on n=100 MATH hard subset and n=100 GSM8K, or gains vanish after parser/token fixes.

**Pivot** if official BAEE dominates RCV-IRIS at matched cost; then the only viable pivot is “mode-aware BAEE / split-budget BAEE,” not old IRIS.

## 11. NeurIPS-Level Gap

当前距离 strong NeurIPS 还缺：

* official BAEE/DEER/AdaptThink/budget forcing comparison；
* clean online full-scale runs；
* 3–5 seed confidence intervals；
* paper thesis rewrite；
* exact reproducibility manifest；
* negative results integrated, not hidden。

## 12. Oral / Best Paper Gap

Oral-level 需要清晰机制 + broad regimes + official baselines beaten。
Best-paper-level 需要更一般的 recoverability/extractability theory across model families/tasks。当前证据远未到这一层。

## 13. Confidence

**Overall confidence: medium.**

原因：现象诊断很强，尤其 entropy/CTT/Stage0/budget-accounting 指向同一缺失机制；但 RCV-IRIS 尚未实现和跑 A/B/C，因此 empirical success 只能是 evidence-backed hypothesis，不能当已证明结论。

---

# 20. Final Claude Code Instruction

```text
Claude Code, execute the following plan.
You must implement the New MAIN METHOD PATH defined in the GPT-5.5 Pro diagnosis report: RCV-IRIS, Recoverability-Calibrated Verifier IRIS.

Do not invent a different method.
Do not optimize for superficial positive results.
Do not weaken baselines.
Do not delete negative results silently.
Do not change metrics or datasets unless explicitly instructed.
Do not rewrite unrelated files.

Your tasks are:

1. Freeze and label all old results.
   - Add results/RESULT_RELIABILITY_LEDGER.md.
   - Mark post-hoc IRIS token-saving results as analysis-only.
   - Mark budget forcing token-efficiency results as contaminated until token accounting is fixed.
   - Mark CTT and entropy experiments as historical negative evidence.
   - Do not modify raw result JSONs.

2. Fix benchmark correctness first.
   - Inspect scripts/benchmarks.py.
   - Fix normalize_latex so it cannot infinite-loop on strings with spaces.
   - Add tests/test_benchmarks.py for MATH parser cases:
     "6 - 5i", "(3, \\frac{\\pi}{2})", "x=5", nested boxed fractions, fallback invalid answers.
   - Run: python -m pytest tests/test_benchmarks.py -q.
   - Stop if tests fail.

3. Add canonical sample manifests.
   - Add scripts/make_sample_manifest.py.
   - All future runners must accept --sample_manifest.
   - Manifest must include benchmark, split, dataset id, sample index, question hash, gold hash, seed.
   - Verify deterministic manifests with diff.

4. Fix scripts/run_budget_forcing.py.
   - Count all generated tokens, including early_stop forced answer tokens and wait_extend continuation tokens.
   - Log initial_generated_tokens, injected_prompt_tokens, forced_generated_tokens, total_generated_tokens.
   - Do not change the algorithm except accounting.
   - Run a 2-sample smoke test and save logs.

5. Rewrite IRIS execution mode.
   - Make online Stage2 the default for main-method runs.
   - Keep post-hoc Stage2 only as --stage2_mode posthoc_analysis.
   - Ensure n_tokens_generated == n_tokens_used for online Stage2 samples.
   - Do not use post-hoc token savings in main claims.

6. Implement RCV signal modules.
   - Add scripts/rcv_signals.py.
   - Add deterministic features:
     answer_validity_score,
     stage0_acceptance_features,
     prefix_recoverability_features,
     extractor_margin.
   - Add unit tests.
   - Do not call the model inside feature tests.

7. Implement RCV verifier and policy.
   - Add scripts/rcv_verifier.py and scripts/run_rcv_iris.py.
   - Stage0 must not accept solely because it naturally stopped.
   - Stage0 acceptance requires answer validity plus verifier/consistency threshold.
   - Stage2 must run online.
   - On Stage2 truncation, compute recoverability score before Stage3.
   - Stage3 extraction only runs when expected utility is positive or when ablation config forces it.
   - Log every decision action.

8. Add A/B/C ablation harness.
   - Add scripts/run_rcv_ablation_suite.py.
   - It must run the same sample manifest through:
     A. Existing Best Positive Fragment Only,
     B. New MAIN METHOD Without New Mechanism,
     C. Full RCV-IRIS.
   - Save paired per-sample outputs and aggregate paired stats.
   - Do not proceed to full benchmark until the small A/B/C run passes.

9. Run minimal verification experiments.
   - smoke test,
   - data sanity,
   - metric sanity,
   - checkpoint/model-revision logging check,
   - current negative entropy/CTT reproduction or static confirmation,
   - existing best fragment reproduction,
   - new mechanism activation check,
   - A/B/C on n=10,
   - then A/B/C on n=100 if n=10 passes.

10. Update paper claims only after minimal experiments pass.
   - Do not claim SOTA.
   - Do not claim universal superiority over thinking.
   - Do not claim entropy or CTT is the mechanism.
   - Do not cite contaminated results as support.
   - Rewrite related work to include NoThinking, s1 budget forcing, AdaptThink, DEER, and BAEE / Detection-Extraction Gap.

For every task:
- make the smallest necessary change;
- show the diff;
- run the specified verification command;
- save logs;
- report failures;
- stop if verification fails;
- do not proceed to full benchmark until minimal tests pass.

At the end, output:
- files changed;
- files archived;
- configs added;
- commands run;
- logs;
- result table;
- failed checks;
- unresolved issues;
- whether Full New MAIN METHOD beats:
  A. Existing Best Positive Fragment Only,
  B. New MAIN METHOD Without New Mechanism,
  C. Full New MAIN METHOD.
```

[1]: https://github.com/Sunshine535/nips-adathink "GitHub - Sunshine535/nips-adathink: AdaThink: Adaptive Test-Time Compute Control for Reasoning LLMs (NeurIPS 2026) · GitHub"
[2]: https://github.com/Sunshine535/nips-adathink/raw/main/README.md "raw.githubusercontent.com"
[3]: https://github.com/Sunshine535/nips-adathink/raw/main/paper/main_final.tex "raw.githubusercontent.com"
[4]: https://github.com/Sunshine535/nips-adathink/raw/main/scripts/run_iris.py "raw.githubusercontent.com"
[5]: https://github.com/Sunshine535/nips-adathink/tree/main/results "nips-adathink/results at main · Sunshine535/nips-adathink · GitHub"
[6]: https://github.com/Sunshine535/nips-adathink/raw/main/DATA_PROVENANCE.md "raw.githubusercontent.com"
[7]: https://github.com/Sunshine535/nips-adathink/tree/main/results/mechanism_ablation "https://github.com/Sunshine535/nips-adathink/tree/main/results/mechanism_ablation"
[8]: https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/requirements.txt "https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/requirements.txt"
[9]: https://github.com/Sunshine535/nips-adathink/raw/main/CLAUDE.md "raw.githubusercontent.com"
[10]: https://github.com/Sunshine535/nips-adathink/raw/main/PAPER_CLAIM_AUDIT.md "raw.githubusercontent.com"
[11]: https://github.com/Sunshine535/nips-adathink/raw/main/NARRATIVE_REPORT.md "raw.githubusercontent.com"
[12]: https://github.com/Sunshine535/nips-adathink/raw/main/paper/sections/method_final.tex "raw.githubusercontent.com"
[13]: https://github.com/Sunshine535/nips-adathink/raw/main/paper/sections/experiments_final.tex "raw.githubusercontent.com"
[14]: https://github.com/Sunshine535/nips-adathink/raw/main/paper/sections/theory_final.tex "raw.githubusercontent.com"
[15]: https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/paper/sections/related_final.tex "https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/paper/sections/related_final.tex"
[16]: https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/scripts/iris_online_stage2.py "https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/scripts/iris_online_stage2.py"
[17]: https://github.com/Sunshine535/nips-adathink/raw/main/scripts/benchmarks.py "https://github.com/Sunshine535/nips-adathink/raw/main/scripts/benchmarks.py"
[18]: https://github.com/Sunshine535/nips-adathink/raw/main/scripts/run_nothink_baseline.py "raw.githubusercontent.com"
[19]: https://github.com/Sunshine535/nips-adathink/raw/main/scripts/run_budget_forcing.py "https://github.com/Sunshine535/nips-adathink/raw/main/scripts/run_budget_forcing.py"
[20]: https://github.com/Sunshine535/nips-adathink/raw/main/scripts/learned_allocator.py "raw.githubusercontent.com"
[21]: https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/scripts/run_ctt_pilot.py "https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/scripts/run_ctt_pilot.py"
[22]: https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/results/entropy_dynamics/go_no_go_summary_Qwen3_8B_b512.json "https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/results/entropy_dynamics/go_no_go_summary_Qwen3_8B_b512.json"
[23]: https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/scripts/run_factorial_ablation.py "https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/scripts/run_factorial_ablation.py"
[24]: https://github.com/Sunshine535/nips-adathink/tree/main/results/iris_improved_20260417/27b_math500_b4096_ba512 "https://github.com/Sunshine535/nips-adathink/tree/main/results/iris_improved_20260417/27b_math500_b4096_ba512"
[25]: https://github.com/Sunshine535/nips-adathink/raw/main/results/multiseed_20260419/multiseed_summary.json "raw.githubusercontent.com"
[26]: https://github.com/Sunshine535/nips-adathink/raw/main/idea-stage/FINAL_PROPOSAL.md "raw.githubusercontent.com"
[27]: https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/results/iris_improved_20260417/27b_math500_b4096_ba512/iris_27b_math500_improved_final.json "https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/results/iris_improved_20260417/27b_math500_b4096_ba512/iris_27b_math500_improved_final.json"
[28]: https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/results/ctt_pilot_27b_gsm8k/analysis_aligned.json "https://raw.githubusercontent.com/Sunshine535/nips-adathink/main/results/ctt_pilot_27b_gsm8k/analysis_aligned.json"
[29]: https://github.com/Sunshine535/nips-adathink/raw/main/results/analysis/alpha_curve_fit.json "raw.githubusercontent.com"
[30]: https://arxiv.org/abs/2504.09858 "https://arxiv.org/abs/2504.09858"
[31]: https://arxiv.org/abs/2501.19393 "https://arxiv.org/abs/2501.19393"
[32]: https://arxiv.org/abs/2408.03314 "https://arxiv.org/abs/2408.03314"
[33]: https://arxiv.org/abs/2505.13417 "https://arxiv.org/abs/2505.13417"
[34]: https://arxiv.org/abs/2504.15895 "https://arxiv.org/abs/2504.15895"
[35]: https://arxiv.org/abs/2604.06613 "https://arxiv.org/abs/2604.06613"
[36]: https://openreview.net/forum?id=GuvQJGgbLm "https://openreview.net/forum?id=GuvQJGgbLm"

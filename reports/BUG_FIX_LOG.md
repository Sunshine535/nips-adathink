# Bug Fix Log

## Bug Fix: benchmarks.py normalize_latex (GPT-5.5 suspected infinite loop)

Files changed: NONE
Reason: GPT-5.5 reported `while " " in s` (single space) as infinite loop. Actual code is `while "  " in s` (double space) — correct multi-space compression. No bug exists.
Evidence: Line 148: `while "  " in s:` with two spaces
Verification: Inspected source code directly
Before: N/A (no bug)
After: N/A
Remaining risk: None for this specific issue

## Bug Fix: run_budget_forcing.py token undercount

Files changed: `scripts/run_budget_forcing.py`
Reason: `generate_with_forcing()` returns only initial `gen_len`, ignoring forced/extended tokens
Evidence: Line 95 `return text, gen_len` — extra tokens from early_stop (up to 32) or wait_extend (up to remaining) not counted
Change: Added `total_generated` computation that includes forced/extended tokens; returns `total_generated` instead of `gen_len`
Verification command: Compare old vs new token counts on same samples
Before: Token counts excluded forced tokens → underestimated baseline cost
After: Token counts include all generated tokens
Remaining risk: Old result files (`bforce_wait_extend.json`, `bforce_early_stop*.json`) have underestimated tokens. Must note in paper.

## Bug Fix: run_nothink_baseline.py --also_thinking always True

Files changed: `scripts/run_nothink_baseline.py`
Reason: `action="store_true", default=True` makes flag always True, CLI cannot disable
Evidence: Line 215
Change: Changed `default=True` to `default=False`
Verification: `python scripts/run_nothink_baseline.py --help` should show default=False
Before: --also_thinking always enabled, baseline always ran thinking too
After: --also_thinking defaults to False (nothink only), explicit flag enables thinking
Remaining risk: Old baseline results may have included thinking mode unexpectedly. Check meta fields in old JSONs.

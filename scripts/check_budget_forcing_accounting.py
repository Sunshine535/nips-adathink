#!/usr/bin/env python3
"""Audit budget forcing result files for field-level token accounting.

Checks whether V2 field-level fields exist per sample.
If missing, marks result as partial_total_token_only.

Usage:
    python3 scripts/check_budget_forcing_accounting.py results/budget_forcing_v2/*.json
"""
import json, sys, glob

REQUIRED_FIELDS = [
    "initial_generated_tokens",
    "forced_generated_tokens",
    "extended_generated_tokens",
    "injected_prompt_tokens",
    "input_prompt_tokens_initial",
    "input_prompt_tokens_forced",
    "total_output_generated_tokens",
    "tokens",
]


def audit_file(path):
    with open(path) as f:
        d = json.load(f)

    meta = d.get("meta", {})
    schema = meta.get("schema_version")
    token_status = meta.get("token_count_status")
    samples = d.get("per_sample", [])

    print(f"## {path}")
    print(f"Schema version: {schema}")
    print(f"Token count status: {token_status}")
    print(f"Samples: {len(samples)}")

    if not samples:
        print("**EMPTY** — no samples to audit")
        print()
        return

    s0 = samples[0]
    present = [f for f in REQUIRED_FIELDS if f in s0]
    missing = [f for f in REQUIRED_FIELDS if f not in s0]

    print(f"Fields present: {present}")
    print(f"Fields missing: {missing}")

    if missing:
        print(f"\n**VERDICT: partial_total_token_only** — missing {len(missing)} V2 fields")
        if "tokens" in s0:
            print(f"'tokens' field IS present — can use for total output token count")
            # Check consistency
            totals = [s.get("tokens", 0) for s in samples]
            print(f"Token range: min={min(totals)}, max={max(totals)}, mean={sum(totals)/len(totals):.0f}")
        else:
            print("**CRITICAL**: 'tokens' field also missing — cannot verify any token count")
    else:
        # Verify sum consistency
        errors = 0
        for i, s in enumerate(samples):
            expected_total = s["initial_generated_tokens"] + s["forced_generated_tokens"] + s["extended_generated_tokens"]
            actual_total = s["total_output_generated_tokens"]
            if abs(expected_total - actual_total) > 0:
                errors += 1
                if errors <= 3:
                    print(f"  MISMATCH sample {i}: {expected_total} != {actual_total}")
            if s["tokens"] != s["total_output_generated_tokens"]:
                print(f"  ALIAS MISMATCH sample {i}: tokens={s['tokens']} != total={s['total_output_generated_tokens']}")
        if errors == 0:
            print(f"\n**VERDICT: field_level_verified** — all {len(REQUIRED_FIELDS)} fields present, sums consistent")
        else:
            print(f"\n**VERDICT: field_level_inconsistent** — {errors} sum mismatches")

    # Summary stats
    summary = d.get("summary", {})
    print(f"\nAccuracy: {summary.get('accuracy', 'N/A')}")
    print(f"Avg tokens: {summary.get('avg_tokens', 'N/A')}")
    if "avg_initial_tokens" in summary:
        print(f"Avg initial: {summary['avg_initial_tokens']:.1f}")
        print(f"Avg forced: {summary.get('avg_forced_tokens', 0):.1f}")
        print(f"Avg extended: {summary.get('avg_extended_tokens', 0):.1f}")
    print()


def main():
    files = []
    for pattern in sys.argv[1:]:
        files.extend(sorted(glob.glob(pattern)) if "*" in pattern else [pattern])
    if not files:
        print("Usage: python3 scripts/check_budget_forcing_accounting.py results/budget_forcing_v2/*.json")
        sys.exit(1)
    print("# Budget Forcing V2 Accounting Audit\n")
    for f in files:
        audit_file(f)


if __name__ == "__main__":
    main()

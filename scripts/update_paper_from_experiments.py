#!/usr/bin/env python3
"""
Update paper LaTeX tables with new experiment results.

Reads results from:
  1. Server2 crossover experiments (results/crossover/ or results_kun/crossover/)
  2. Server1 27B nothink experiments (results/fulltest_27b_nothink/ or results_kun/fulltest_27b_nothink/)

Updates:
  - paper/sections/experiments_final.tex: Crossover table (Table 5)
  - results/paper_figures/table_model_size_scaling.tex: Add 27B nothink rows

Usage:
    python scripts/update_paper_from_experiments.py --check        # dry run
    python scripts/update_paper_from_experiments.py --update       # apply changes
    python scripts/update_paper_from_experiments.py --sync-server  # rsync from servers first
"""

import argparse
import json
import os
import re
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
RESULTS_KUN = ROOT / "results_kun"


def load_crossover_data():
    """Load crossover experiment results."""
    data = {}  # {(mode, budget): {'accuracy': float, 'avg_tokens': float, ...}}

    # Check multiple possible locations
    dirs = [
        RESULTS / "crossover",
        RESULTS_KUN / "crossover",
    ]

    for d in dirs:
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            try:
                with open(f) as fh:
                    j = json.load(fh)
                # Extract mode and budget from filename or contents
                if 'results' in j:
                    for key, val in j['results'].items():
                        # key like "nothink_512" or "thinking_1024"
                        parts = key.rsplit('_', 1)
                        if len(parts) == 2:
                            mode, budget = parts[0], int(parts[1])
                            data[(mode, budget)] = val
                elif 'accuracy' in j:
                    # Single-result file
                    mode = 'nothink' if 'nothink' in f.name else 'thinking'
                    budget = None
                    for b in [128, 256, 512, 1024, 2048, 4096]:
                        if str(b) in f.name:
                            budget = b
                            break
                    if budget:
                        data[(mode, budget)] = j
            except (json.JSONDecodeError, Exception) as e:
                print(f"  Warning: Could not parse {f}: {e}")

    # Also load from known data sources
    known_files = [
        RESULTS_KUN / "nothink_baseline_fullset_complete.json",
    ]
    for f in known_files:
        if f.exists():
            with open(f) as fh:
                j = json.load(fh)
            if 'results' in j:
                for key, val in j['results'].items():
                    parts = key.rsplit('_', 1)
                    if len(parts) == 2:
                        mode, budget_str = parts
                        try:
                            budget = int(budget_str)
                            if (mode, budget) not in data:
                                data[(mode, budget)] = val
                        except ValueError:
                            pass

    return data


def load_27b_nothink_data():
    """Load 27B nothink experiment results."""
    data = {}

    dirs = [
        RESULTS / "fulltest_27b_nothink",
        RESULTS_KUN / "fulltest_27b_nothink",
    ]

    for d in dirs:
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            try:
                with open(f) as fh:
                    j = json.load(fh)
                if 'results' in j:
                    for key, val in j['results'].items():
                        parts = key.rsplit('_', 1)
                        if len(parts) == 2:
                            mode, budget_str = parts
                            try:
                                budget = int(budget_str)
                                data[(mode, budget)] = val
                            except ValueError:
                                pass
            except Exception as e:
                print(f"  Warning: Could not parse {f}: {e}")

    return data


def format_pct(val, bold=False):
    """Format percentage for LaTeX."""
    s = f"{val*100:.1f}\\%"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def update_crossover_table(data, dry_run=True):
    """Update the crossover table in experiments_final.tex."""
    tex_file = PAPER / "sections" / "experiments_final.tex"
    if not tex_file.exists():
        print(f"  ERROR: {tex_file} not found")
        return False

    content = tex_file.read_text()

    # Known data points (already in paper)
    known = {
        ('nothink', 128): 0.508,
        ('nothink', 256): 0.875,
        ('thinking', 128): 0.030,
        ('thinking', 256): 0.180,
        ('thinking', 512): 0.652,
    }

    # Merge with new data
    for (mode, budget), val in data.items():
        if isinstance(val, dict):
            acc = val.get('accuracy', val.get('acc'))
            if acc is not None:
                if acc > 1:
                    acc = acc / 100  # Convert percentage
                known[(mode, budget)] = acc
        elif isinstance(val, (int, float)):
            known[(mode, budget)] = val if val <= 1 else val / 100

    # Build new table rows
    budgets = [128, 256, 512, 1024, 2048, 4096]
    new_rows = []
    crossover_budget = None

    for b in budgets:
        nt = known.get(('nothink', b))
        th = known.get(('thinking', b))

        nt_str = format_pct(nt) if nt is not None else "\\textbf{TBD}"
        th_str = format_pct(th) if th is not None else "\\textbf{TBD}"

        if nt is not None and th is not None:
            gap = (nt - th) * 100
            gap_str = f"$+${gap:.1f}pp" if gap > 0 else f"$-${abs(gap):.1f}pp"

            # Check crossover
            if gap < 0 and crossover_budget is None:
                crossover_budget = b
        else:
            gap_str = "\\textbf{TBD}"

        new_rows.append(f"{b} & {nt_str} & {th_str} & {gap_str} \\\\")

    # Build replacement table body
    new_body = "\n".join(new_rows)

    # Count TBDs
    tbd_count = new_body.count("TBD")

    print(f"\n  Crossover table update:")
    print(f"  TBD entries remaining: {tbd_count}")
    for row in new_rows:
        print(f"    {row}")
    if crossover_budget:
        print(f"  🎯 Crossover point: budget={crossover_budget}")

    if not dry_run and tbd_count < 12:  # Only update if we have some new data
        # Replace the table body using regex
        pattern = r'(\\midrule\n)(.*?)(\\bottomrule)'

        # Find the crossover table specifically
        crossover_start = content.find('\\label{tab:crossover}')
        if crossover_start > 0:
            # Find the midrule after this label
            midrule_pos = content.find('\\midrule', crossover_start)
            bottomrule_pos = content.find('\\bottomrule', crossover_start)

            if midrule_pos > 0 and bottomrule_pos > 0:
                old_body = content[midrule_pos + len('\\midrule\n'):bottomrule_pos]
                new_content = content[:midrule_pos + len('\\midrule\n')] + \
                             new_body + "\n" + \
                             content[bottomrule_pos:]
                tex_file.write_text(new_content)
                print(f"  ✅ Updated {tex_file}")

                # Also remove placeholder text if all TBDs are gone
                if tbd_count == 0:
                    new_content = new_content.replace(
                        "\\textbf{[Placeholder---data filling in as experiments complete.]}",
                        ""
                    )
                    tex_file.write_text(new_content)

                return True

    return False


def check_status():
    """Print status of all data availability."""
    print("=" * 60)
    print("EXPERIMENT DATA STATUS")
    print("=" * 60)

    # Crossover data
    print("\n📊 Crossover Data (8B nothink high-budget):")
    crossover = load_crossover_data()
    if crossover:
        for (mode, budget), val in sorted(crossover.items()):
            acc = val.get('accuracy', '?') if isinstance(val, dict) else val
            print(f"  {mode}@{budget}: acc={acc}")
    else:
        print("  ❌ No crossover data found")

    # 27B nothink
    print("\n📊 27B Nothink Data:")
    nothink_27b = load_27b_nothink_data()
    if nothink_27b:
        for (mode, budget), val in sorted(nothink_27b.items()):
            acc = val.get('accuracy', '?') if isinstance(val, dict) else val
            print(f"  {mode}@{budget}: acc={acc}")
    else:
        print("  ❌ No 27B nothink data found")

    # What's still needed
    print("\n📋 What's Still Needed:")
    needed_crossover = []
    for b in [512, 1024, 2048, 4096]:
        if ('nothink', b) not in crossover:
            needed_crossover.append(f"nothink@{b}")
    for b in [1024, 2048, 4096]:
        if ('thinking', b) not in crossover:
            needed_crossover.append(f"thinking@{b}")

    if needed_crossover:
        print(f"  Crossover: {', '.join(needed_crossover)}")
    else:
        print("  ✅ Crossover data complete!")

    needed_27b = []
    for b in [128, 256, 512]:
        if ('nothink', b) not in nothink_27b:
            needed_27b.append(f"27B nothink@{b}")

    if needed_27b:
        print(f"  27B Nothink: {', '.join(needed_27b)}")
    else:
        print("  ✅ 27B nothink data complete!")

    print()


def sync_from_servers():
    """Rsync results from both servers."""
    servers = [
        {
            'name': 'Server2',
            'host': 'root@216.81.245.127',
            'port': '15276',
            'remote_dir': '/workspace/nips-adathink/results/crossover/',
            'local_dir': str(RESULTS_KUN / 'crossover/'),
        },
        {
            'name': 'Server1',
            'host': 'root@216.81.151.3',
            'port': '11839',
            'remote_dir': '/workspace/nips-adathink/results/fulltest_27b_nothink/',
            'local_dir': str(RESULTS_KUN / 'fulltest_27b_nothink/'),
        },
    ]

    for s in servers:
        print(f"\n🔄 Syncing from {s['name']}...")
        os.makedirs(s['local_dir'], exist_ok=True)
        cmd = [
            'rsync', '-avz', '--progress',
            '-e', f"ssh -p {s['port']} -i ~/.ssh/kun_ed25519 -o ConnectTimeout=10",
            f"{s['host']}:{s['remote_dir']}",
            s['local_dir'],
        ]
        try:
            subprocess.run(cmd, check=True, timeout=60)
            print(f"  ✅ Synced {s['name']}")
        except subprocess.TimeoutExpired:
            print(f"  ⚠️ Timeout syncing {s['name']}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to sync {s['name']}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Update paper with experiment results")
    parser.add_argument('--check', action='store_true', help='Check data status (dry run)')
    parser.add_argument('--update', action='store_true', help='Apply updates to paper')
    parser.add_argument('--sync-server', action='store_true', help='Rsync from servers first')
    args = parser.parse_args()

    if args.sync_server:
        sync_from_servers()

    if args.check or not (args.update or args.sync_server):
        check_status()

    if args.update:
        crossover = load_crossover_data()
        update_crossover_table(crossover, dry_run=False)


if __name__ == '__main__':
    main()

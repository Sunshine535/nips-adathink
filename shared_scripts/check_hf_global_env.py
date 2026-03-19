#!/usr/bin/env python3
"""Check whether HF mirror/cache envs are globally active."""

import os
from pathlib import Path


def grep_env_in_file(path: Path, keys):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    hits = []
    for i, line in enumerate(lines, 1):
        if any(k in line for k in keys):
            hits.append((i, line))
    return hits


keys = ["HF_ENDPOINT", "XDG_CACHE_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE"]

print("[Current process env]")
for k in keys:
    print(f"{k}={os.environ.get(k)}")

print("\n[Shell startup files]")
for p in [
    Path.home() / ".bashrc",
    Path.home() / ".profile",
    Path.home() / ".bash_profile",
    Path.home() / ".zshrc",
    Path("/etc/environment"),
]:
    hits = grep_env_in_file(p, keys)
    if hits:
        print(f"{p}:")
        for ln, txt in hits:
            print(f"  {ln}: {txt}")

try:
    from huggingface_hub import constants

    print("\n[huggingface_hub resolved]")
    print("ENDPOINT=", constants.ENDPOINT)
    print("HF_HOME=", constants.HF_HOME)
    print("HF_HUB_CACHE=", constants.HF_HUB_CACHE)
except Exception as e:
    print("\n[huggingface_hub resolved]")
    print("huggingface_hub not installed in this interpreter:", e)

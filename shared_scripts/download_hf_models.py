#!/usr/bin/env python3
"""Download all experiment-related Hugging Face models with grouped selection.

Usage examples:
  python scripts/download_hf_models.py --group all
  python scripts/download_hf_models.py --group qwen3_low_cost qwen35_main
  python scripts/download_hf_models.py --group qwen35_large --dry-run
  python scripts/download_hf_models.py --group embedding reranker --dry-run
"""

import argparse
import os
import threading
import time
from pathlib import Path

# Ensure defaults are set before importing huggingface_hub.
DEFAULT_XDG_CACHE_HOME = "/mnt/ae22ef98-e02a-4dfa-acf4-a307e8941cd9/nwh"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
os.environ.setdefault("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
os.environ.setdefault("HF_HOME", str(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface"))

from huggingface_hub import HfApi, snapshot_download


QWEN3_LOW_COST_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

QWEN35_MAIN_MODELS = [
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
]

QWEN35_LARGE_MODELS = [
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
]

MODEL_GROUPS = {
    "qwen3_low_cost": QWEN3_LOW_COST_MODELS,
    "qwen35_main": QWEN35_MAIN_MODELS,
    "qwen35_large": QWEN35_LARGE_MODELS,
    "reasoning": [
        *QWEN3_LOW_COST_MODELS,
        *QWEN35_MAIN_MODELS,
        "Qwen/QwQ-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
    "math": [
        "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-35B-A3B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ],
    "base": [
        "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-14B-Base",
        "Qwen/Qwen3.5-35B-A3B-Base",
        "mistralai/Mistral-7B-v0.3",
    ],
    "embedding": [
        "BAAI/bge-m3",
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-large-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "jinaai/jina-embeddings-v3",
    ],
    "reranker": [
        "BAAI/bge-reranker-v2-m3",
        "jinaai/jina-reranker-v2-base-multilingual",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ],
    "verifier": [
        "cross-encoder/nli-deberta-v3-base",
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    ],
}

DEFAULT_ALL_GROUPS = [
    "qwen3_low_cost",
    "qwen35_main",
    "reasoning",
    "math",
    "base",
    "embedding",
    "reranker",
    "verifier",
]


def dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for u in units:
        if value < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(value)}{u}"
            return f"{value:.2f}{u}"
        value /= 1024.0
    return f"{num_bytes}B"


def hf_hub_root(cache_dir: str | None) -> Path:
    if cache_dir:
        return Path(cache_dir)
    hf_home = Path(os.environ.get("HF_HOME", str(Path(DEFAULT_XDG_CACHE_HOME) / "huggingface")))
    return hf_home / "hub"


def repo_cache_dir(repo_id: str, cache_dir: str | None) -> Path:
    return hf_hub_root(cache_dir) / f"models--{repo_id.replace('/', '--')}"


class RepoProgressMonitor:
    """Lightweight progress hints while snapshot_download is running."""

    def __init__(self, repo_id: str, cache_dir: str | None, interval_s: int = 15):
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.interval_s = max(3, interval_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_signature = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _collect(self):
        repo_dir = repo_cache_dir(self.repo_id, self.cache_dir)
        blobs_dir = repo_dir / "blobs"
        snapshots_dir = repo_dir / "snapshots"

        incomplete_files = list(blobs_dir.glob("*.incomplete")) if blobs_dir.exists() else []
        incomplete_n = len(incomplete_files)
        incomplete_size = 0
        latest_mtime = 0.0
        for p in incomplete_files:
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            incomplete_size += st.st_size
            latest_mtime = max(latest_mtime, st.st_mtime)

        done_files = 0
        if snapshots_dir.exists():
            for snap in snapshots_dir.glob("*"):
                if not snap.is_dir():
                    continue
                done_files += len(list(snap.glob("model.safetensors*")))

        return incomplete_n, incomplete_size, int(latest_mtime), done_files

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            incomplete_n, incomplete_size, latest_mtime, done_files = self._collect()
            signature = (incomplete_n, incomplete_size, latest_mtime, done_files)
            if signature != self._last_signature:
                print(
                    f"[Progress] {self.repo_id}: "
                    f"incomplete={incomplete_n}, "
                    f"incomplete_size={human_size(incomplete_size)}, "
                    f"done_shards={done_files}",
                    flush=True,
                )
                self._last_signature = signature


def resolve_groups(group_names):
    if "all" in group_names:
        group_names = DEFAULT_ALL_GROUPS

    models = []
    for g in group_names:
        if g not in MODEL_GROUPS:
            raise ValueError(f"Unknown group: {g}")
        models.extend(MODEL_GROUPS[g])

    return dedupe_keep_order(models)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        nargs="+",
        default=["all"],
        help=(
            "Groups to download: all/" + "/".join(MODEL_GROUPS.keys())
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print models, do not download")
    parser.add_argument("--validate-only", action="store_true", help="Validate model existence, no download")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache dir override. By default uses HF_HOME/hub from env.",
    )
    parser.add_argument("--revision", default=None, help="Optional fixed revision/commit for all models")
    parser.add_argument("--max-workers", type=int, default=8, help="Download workers per model")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=15,
        help="Seconds between repo-level progress prints while downloading.",
    )
    parser.add_argument("--no-progress-monitor", action="store_true", help="Disable repo-level progress prints")
    return parser.parse_args()


def main():
    args = parse_args()
    models = resolve_groups(args.group)

    print("[Config]")
    print("HF_ENDPOINT=", os.environ.get("HF_ENDPOINT"))
    print("XDG_CACHE_HOME=", os.environ.get("XDG_CACHE_HOME"))
    print("HF_HOME=", os.environ.get("HF_HOME"))
    print("cache_dir=", args.cache_dir)
    print()

    print("[Models]")
    for i, m in enumerate(models, 1):
        print(f"{i:02d}. {m}")
    print()

    api = HfApi(endpoint=os.environ.get("HF_ENDPOINT"))
    print("[Validate]")
    ok_models = []
    for m in models:
        try:
            info = api.model_info(m)
            sha = (info.sha or "")[:8]
            print(f"OK   {m}  sha={sha}")
            ok_models.append(m)
        except Exception as e:
            print(f"FAIL {m}  error={type(e).__name__}: {str(e).splitlines()[0]}")
    print(f"\nValidated: {len(ok_models)}/{len(models)}")

    if args.dry_run or args.validate_only:
        return 0

    print("\n[Download]")
    for m in ok_models:
        print(f"Downloading: {m}")
        monitor = None
        if not args.no_progress_monitor:
            monitor = RepoProgressMonitor(m, args.cache_dir, interval_s=args.progress_interval)
            monitor.start()
        try:
            snapshot_download(
                repo_id=m,
                revision=args.revision,
                cache_dir=args.cache_dir,
                max_workers=args.max_workers,
            )
        finally:
            if monitor is not None:
                monitor.stop()
        print(f"Done: {m}")

    print("\nAll downloads completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

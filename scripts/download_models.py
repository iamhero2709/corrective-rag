"""
Download all required models with progress bars.
Usage: python scripts/download_models.py [--models embed,nli,gen]
"""

import argparse
import os
import sys
from pathlib import Path

MODELS = {
    "embed": ("BAAI/bge-small-en-v1.5", "~130MB — sentence embedding"),
    "nli": ("cross-encoder/nli-deberta-v3-xsmall", "~280MB — NLI cross-encoder"),
    "gen": ("Qwen/Qwen2.5-1.5B-Instruct", "~3GB — text generation (CPU-friendly)"),
}


def download(repo_id: str, cache_dir: str, desc: str):
    print(f"\n{desc}")
    print(f"  repo: {repo_id}")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
    print(f"  done")


def main():
    parser = argparse.ArgumentParser(description="Download RAG models")
    parser.add_argument("--models", default="all",
                        help="Comma-separated: embed,nli,gen (default: all)")
    parser.add_argument("--cache-dir",
                        default=os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
                        help="HF cache directory")
    args = parser.parse_args()

    selected = MODELS.keys() if args.models == "all" else args.models.split(",")
    total_size = sum(int(m[1].split("~")[1].split("MB")[0]) for k, m in MODELS.items()
                     if k in selected and "MB" in m[1])

    print(f"Downloading {len(selected)} model(s)")
    print(f"Cache dir: {args.cache_dir}")
    print()

    for key in selected:
        if key not in MODELS:
            print(f"Unknown model: {key}  (choices: {', '.join(MODELS)})", file=sys.stderr)
            sys.exit(1)
        repo, desc = MODELS[key]
        download(repo, args.cache_dir, desc)

    print("\nAll models downloaded.")
    print("Next: run `python -m spacy download en_core_web_sm` for the spacy model.")


if __name__ == "__main__":
    main()

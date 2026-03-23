#!/usr/bin/env python3
"""
Launch the full v2 matrix (all models × protocols × optional K × seeds × 8ch and/or 22ch).

Example (dry run — prints commands only):
  python -m reproduction_benchmark_v2.orchestrate_v2 --dry_run

Example (execute):
  python -m reproduction_benchmark_v2.orchestrate_v2 --data "C:/path/to/BCI2a"

Uses subprocess to invoke run_benchmark.py so each run is isolated.
"""
import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reproduction_benchmark_v2.config import (
    MODELS_FULL,
    SEEDS,
    K_PER_CLASS_GRID,
    DEFAULT_BCI2A_PATH,
    DEFAULT_RESULTS_DIR,
)


def main():
    p = argparse.ArgumentParser(description="Orchestrate full reproduction v2 grid")
    p.add_argument("--data", type=str, default=str(DEFAULT_BCI2A_PATH))
    p.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--channels", type=str, default="both", choices=["8", "22", "both"])
    p.add_argument("--models", type=str, nargs="*", default=None, help="Subset of models; default all")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    models = args.models if args.models else MODELS_FULL
    chans = ["8", "22"] if args.channels == "both" else [args.channels]

    cmd_base = [
        sys.executable,
        "-m",
        "reproduction_benchmark_v2.run_benchmark",
        "--data",
        args.data,
        "--results_dir",
        args.results_dir,
        "--run_all_protocols",
        "--run_all_k",
        "--run_all_seeds",
    ]
    if args.epochs is not None:
        cmd_base += ["--epochs", str(args.epochs)]

    n = 0
    for ch in chans:
        for m in models:
            cmd = cmd_base + ["--model", m, "--channels", ch]
            n += 1
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=str(_ROOT), check=False)
    print(f"# Total invocations: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()

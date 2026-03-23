#!/usr/bin/env python3
"""
Reproduction benchmark v2 — BCI IV-2a, protocols W / L / F (no TTA).

Run from the repository root (e.g. Thesis/file_ver2):
  python -m reproduction_benchmark_v2.run_benchmark --help
"""
import json
import os
import sys
import csv
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reproduction_benchmark_v2.config import (
    DEFAULT_BCI2A_PATH,
    DEFAULT_RESULTS_DIR,
    K_PER_CLASS_GRID,
    MATRIX_8CH,
    MATRIX_22CH,
    SEEDS,
    DEFAULT_EPOCHS,
    TIME_REL_START_SEC,
    TIME_REL_END_SEC,
    LEGACY_TIME_REL_START_SEC,
    LEGACY_TIME_REL_END_SEC,
)
from reproduction_benchmark_v2.data_loader import describe_window
from reproduction_benchmark_v2.protocols import protocol_W, protocol_L, protocol_F


def _normalize_path(p):
    if not p:
        return p
    p = os.path.expanduser(str(p))
    if os.path.sep == "/" and len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/").lstrip("/")
        return os.path.normpath(f"/mnt/{drive}/{rest}")
    return os.path.normpath(p.replace("\\", os.path.sep))


def _is_allowed(ch_label, protocol, model_name):
    matrix = MATRIX_8CH if ch_label == "8ch" else MATRIX_22CH
    if protocol not in matrix:
        return False
    return model_name in matrix[protocol]


def run_protocol(protocol, data_path, n_channels, model_name, seed, rel_start, rel_end, k_per_class=None, epochs=None, batch_size=64):
    epochs = epochs if epochs is not None else DEFAULT_EPOCHS
    if protocol == "W":
        return protocol_W(data_path, n_channels, model_name, seed, rel_start, rel_end, epochs, batch_size)
    if protocol == "L":
        return protocol_L(data_path, n_channels, model_name, seed, rel_start, rel_end, epochs, batch_size)
    if protocol == "F":
        k = k_per_class if k_per_class is not None else K_PER_CLASS_GRID[0]
        return protocol_F(data_path, n_channels, model_name, seed, k, rel_start, rel_end, epochs, batch_size)
    raise ValueError(f"Unknown protocol: {protocol}")


def build_run_metadata(
    *,
    data_path,
    results_dir,
    ch_label,
    protocol,
    model_name,
    seed,
    rel_start_sec,
    rel_end_sec,
    k_per_class,
    epochs,
    batch_size,
    extra=None,
):
    meta = {
        "benchmark": "reproduction_benchmark_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "BCI Competition IV-2a",
        "data_path": str(data_path),
        "results_root": str(results_dir),
        "channels": ch_label,
        "n_channels": 8 if ch_label == "8ch" else 22,
        "protocol": protocol,
        "model": model_name,
        "seed": seed,
        "preprocessing": {
            "loader": "preprocess.load_BCI2a_data (artifact trials included, same as v1 benchmark)",
            "standardization": "Per-channel StandardScaler fit on train split only; applied to train/val/test (no leakage from test into scaler fit).",
            "train_val_split": "80/20 stratified from training pool (protocol-specific; see protocols.py).",
        },
        "time_window": describe_window(rel_start_sec, rel_end_sec),
        "training": {
            "max_epochs": epochs,
            "early_stopping": "start_epoch=100, patience=80 (via main_TrainValTest._make_early_stopping when available)",
            "batch_size": batch_size,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
        },
        "protocol_F_k_per_class": k_per_class if protocol == "F" else None,
        "notes": [
            "conformer uses ShallowConvNet placeholder (same as reproduction_benchmark v1).",
            "TTA protocol is not part of v2.",
        ],
    }
    if extra:
        meta["extra"] = extra
    return meta


def save_results(
    results_dir,
    ch_label,
    protocol,
    model_name,
    seed,
    results,
    metadata,
    k_per_class=None,
):
    subdir = results_dir / "bci2a" / "accuracy" / ch_label / protocol / model_name / f"seed_{seed}"
    if k_per_class is not None and protocol == "F":
        subdir = subdir / f"K{k_per_class}"
    subdir.mkdir(parents=True, exist_ok=True)

    rows = [{k: v for k, v in r.items() if k not in ("confusion_matrix", "history")} for r in results]
    subjectwise_path = subdir / "subjectwise.csv"
    with open(subjectwise_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "trialAcc", "macroF1", "kappa", "ITR"])
        w.writeheader()
        w.writerows(rows)

    for r in results:
        if "confusion_matrix" in r:
            cm = r["confusion_matrix"]
            subj = r.get("subject", "unknown")
            cm_path = subdir / f"confusion_{subj}.csv"
            np.savetxt(cm_path, cm, fmt="%d", delimiter=",")

    try:
        from training_utils import save_training_curves
        for r in results:
            if "history" in r and r["history"]:
                subj = r.get("subject", "unknown")
                save_training_curves(r["history"], str(subdir), f"subject_{subj}")
    except Exception:
        pass

    accs = [r["trialAcc"] for r in results]
    f1s = [r["macroF1"] for r in results]
    summary = {
        "mean_trialAcc": float(np.mean(accs)),
        "std_trialAcc": float(np.std(accs)),
        "median_trialAcc": float(np.median(accs)),
        "iqr_trialAcc": float(np.percentile(accs, 75) - np.percentile(accs, 25)),
        "mean_macroF1": float(np.mean(f1s)),
        "std_macroF1": float(np.std(f1s)),
    }
    summary_path = subdir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    metadata["summary"] = summary
    with open(subdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    readme = subdir / "RUN_INFO.md"
    with open(readme, "w", encoding="utf-8") as f:
        f.write("# Run metadata (reproduction v2)\n\n")
        f.write("See `run_metadata.json` for full machine-readable parameters.\n\n")
        f.write("## Quick facts\n\n")
        f.write(f"- **Model:** {metadata['model']}\n")
        f.write(f"- **Protocol:** {metadata['protocol']}\n")
        f.write(f"- **Channels:** {metadata['channels']}\n")
        f.write(f"- **Seed:** {metadata['seed']}\n")
        tw = metadata["time_window"]
        f.write(f"- **Time (within 1.5–6 s BNCI crop):** {tw['time_relative_to_segment_start_sec'][0]}–{tw['time_relative_to_segment_start_sec'][1]} s → {tw['n_samples']} samples\n")
        f.write(f"- **Mean trial accuracy:** {summary['mean_trialAcc']:.4f}\n")

    return subdir, summary


def main():
    parser = argparse.ArgumentParser(description="BCI IV-2a reproduction benchmark v2")
    parser.add_argument("--data", type=str, default=str(DEFAULT_BCI2A_PATH), help="Path to BCI IV-2a folder")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--protocol", type=str, default=None, choices=["W", "L", "F"],
                        help="Ignored if --run_all_protocols (then runs W, L, F).")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["eegnetv4", "shallow", "deep4", "conformer", "fbcsp_lda", "db_atcnet", "atcnet"],
    )
    parser.add_argument("--channels", type=str, default="8", choices=["8", "22"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k_per_class", type=int, default=None, help="Protocol F only")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--run_all_protocols", action="store_true", help="Run W, L, F")
    parser.add_argument("--run_all_k", action="store_true", help="Protocol F: K=1,5,10,20")
    parser.add_argument("--run_all_seeds", action="store_true", help=f"Run seeds {SEEDS}")
    parser.add_argument(
        "--time_start",
        type=float,
        default=None,
        help="Seconds relative to start of BNCI 1.5–6 s segment (default: v2 default 0.5)",
    )
    parser.add_argument(
        "--time_end",
        type=float,
        default=None,
        help="Seconds relative to segment start (default: v2 default 4.5). Use 0 and 4.5 for full 1125-sample crop.",
    )
    args = parser.parse_args()

    data_path = _normalize_path(args.data)
    if not os.path.isdir(data_path):
        print(f"ERROR: --data must be an existing directory. Got: {data_path}")
        sys.exit(1)

    results_dir = Path(_normalize_path(args.results_dir)) if args.results_dir else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    rel_start = TIME_REL_START_SEC if args.time_start is None else args.time_start
    rel_end = TIME_REL_END_SEC if args.time_end is None else args.time_end

    n_channels = int(args.channels)
    ch_label = f"{n_channels}ch"
    if args.run_all_protocols:
        protocols_to_run = ["W", "L", "F"]
    elif args.protocol is not None:
        protocols_to_run = [args.protocol]
    else:
        parser.error("Provide --protocol or use --run_all_protocols")
    k_list = K_PER_CLASS_GRID if args.run_all_k else [args.k_per_class or K_PER_CLASS_GRID[0]]
    epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS
    seeds_to_run = SEEDS if args.run_all_seeds else [args.seed]

    for seed in seeds_to_run:
        for protocol in protocols_to_run:
            if not _is_allowed(ch_label, protocol, args.model):
                print(f"[SKIP] {ch_label}/{protocol}/{args.model} not in matrix")
                continue
            if protocol == "F":
                for k in k_list:
                    print(f"[{protocol}] model={args.model} channels={n_channels} seed={seed} K={k} time={rel_start}-{rel_end}s")
                    results = run_protocol(protocol, data_path, n_channels, args.model, seed, rel_start, rel_end, k, epochs, args.batch_size)
                    meta = build_run_metadata(
                        data_path=data_path,
                        results_dir=results_dir,
                        ch_label=ch_label,
                        protocol=protocol,
                        model_name=args.model,
                        seed=seed,
                        rel_start_sec=rel_start,
                        rel_end_sec=rel_end,
                        k_per_class=k,
                        epochs=epochs,
                        batch_size=args.batch_size,
                    )
                    subdir, summary = save_results(results_dir, ch_label, protocol, args.model, seed, results, meta, k)
                    print(f" mean_trialAcc: {summary['mean_trialAcc']:.4f} ± {summary['std_trialAcc']:.4f}")
                    print(f" -> {subdir}")
            else:
                print(f"[{protocol}] model={args.model} channels={n_channels} seed={seed} time={rel_start}-{rel_end}s")
                results = run_protocol(protocol, data_path, n_channels, args.model, seed, rel_start, rel_end, None, epochs, args.batch_size)
                meta = build_run_metadata(
                    data_path=data_path,
                    results_dir=results_dir,
                    ch_label=ch_label,
                    protocol=protocol,
                    model_name=args.model,
                    seed=seed,
                    rel_start_sec=rel_start,
                    rel_end_sec=rel_end,
                    k_per_class=None,
                    epochs=epochs,
                    batch_size=args.batch_size,
                )
                subdir, summary = save_results(results_dir, ch_label, protocol, args.model, seed, results, meta)
                print(f" mean_trialAcc: {summary['mean_trialAcc']:.4f} ± {summary['std_trialAcc']:.4f}")
                print(f" -> {subdir}")

    print("Done.")


if __name__ == "__main__":
    main()

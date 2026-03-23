# Reproduction benchmark v2

Standalone repo layout: outputs go to **`../results/`** by default (under `file_ver2/results/`).

## What’s new vs v1

| Item | v1 | v2 |
|------|----|----|
| Results folder | `results_reproduction/` | `results_reproduction_v2/` |
| Time window | Implicit 1.5–6 s (1125 samples) | **Default 0.5–4.5 s** within that crop (1000 samples); override with flags |
| Protocols | W, L, F, TTA | **W, L, F only** |
| Seeds (default) | 0–4 | **0, 1** (override with `--run_all_seeds` using config `SEEDS`) |
| 22ch matrix | Limited models | **Same full model list as 8ch** |
| Metadata | CSV only | **`run_metadata.json` + `RUN_INFO.md` per run folder** |
| ATCNet | Not in registry | **`atcnet`** added |

## Protocols

- **W** — Within-subject: train session 1, test session 2 (per subject).
- **L** — LOSO: train on other subjects’ session 1, test target session 2.
- **F** — Few-shot: same as L but merge **K** labelled trials per class from target’s session 1 into the training pool; **K ∈ {1,5,10,20}**.

“Finetuning” in the sense of **K-shot calibration** is **protocol F** with `--run_all_k`.  
This is **not** the separate “fine-tune 20% of session 1” protocol from `run_experiment_plan_v3.py` (that would be another script).

## Time window semantics

Data are loaded with `preprocess.load_BCI2a_data` (standard **1.5–6.0 s** epoch, 1125 samples), then **sliced** along time:

- **Default v2:** `--time_start 0.5 --time_end 4.5` → indices `[125:1125)` = **1000 samples** (4.0 s at 250 Hz).
- **Full legacy crop:** `--time_start 0 --time_end 4.5` → **1125 samples** (same effective window as v1).

## Run (from `file_ver1`)

```bash
python -m reproduction_benchmark_v2.run_benchmark --data "PATH/TO/BCI2a" --results_dir "PATH/TO/results_reproduction_v2" --protocol L --model db_atcnet --channels 22 --seed 0
```

Full sweep (one model, all protocols W+L+F, all K, seeds 0 and 1):

```bash
python -m reproduction_benchmark_v2.run_benchmark --data "..." --protocol W --model eegnetv4 --channels 8 --run_all_protocols --run_all_seeds --run_all_k
```

(Use `--protocol W` with `--run_all_protocols` — the `W` value is ignored when `run_all_protocols` is set; all of W, L, F run.)

## Outputs

Each leaf directory contains:

- `subjectwise.csv`, `summary.csv`, `confusion_*.csv`
- `run_metadata.json` — full parameters
- `RUN_INFO.md` — short human-readable summary

## DB-ATCNet vs ATCNet

See **`VERIFY_DB_ATCNET_VS_ATCNET.md`**.

## `conformer`

Still uses **ShallowConvNet** as a placeholder (same as v1). Documented in `run_metadata.json` notes.

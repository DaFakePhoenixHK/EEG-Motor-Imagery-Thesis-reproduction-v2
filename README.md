# BCI IV-2a reproduction benchmark (standalone)

Self-contained **Python** project for reproducing **W / L / F** protocols on **BCI Competition IV-2a** (4-class motor imagery).  
This is a clean copy of `reproduction_benchmark_v2` plus minimal dependencies (no full `file_ver1` thesis tree).

## Layout

| Path | Contents |
|------|----------|
| `reproduction_benchmark_v2/` | CLI, protocols, data loader, model registry |
| `models.py`, `attention_models.py` | CNN / ATCNet / DB-ATCNet (Apache-2.0, Altaheri et al.) |
| `preprocess.py` | `load_BCI2a_data` |
| `training_utils.py` | Early stopping + training curve plots |
| `reproduction_benchmark/fbcsp_lda.py` | FBCSP + LDA baseline |
| `data/bci2a/` | **Put your dataset here** (see `data/README.md`) |
| `results/` | Default output (gitignored) |

## Setup

**New PC with nothing installed?** See **[SETUP_NEW_PC.md](SETUP_NEW_PC.md)** (Windows checklist: Git, Python, venv, pip, data).

```bash
cd file_ver2
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Copy BCI IV-2a `.mat` files into `data/bci2a/` (or pass `--data`).

## Run

From the **`file_ver2`** directory:

```bash
python -m reproduction_benchmark_v2.run_benchmark --data "path/to/bci2a" --protocol L --model eegnetv4 --channels 8 --seed 0
```

Full sweep (protocols W+L+F, all K, seeds 0+1):

```bash
python -m reproduction_benchmark_v2.run_benchmark --data "path/to/bci2a" --run_all_protocols --run_all_k --run_all_seeds --model eegnetv4 --channels 8
```

See `reproduction_benchmark_v2/README.md` and `VERIFY_DB_ATCNET_VS_ATCNET.md`.

## GitHub

Remote: **<https://github.com/DaFakePhoenixHK/EEG-Motor-Imagery-Thesis-reproduction-v2>**

```bash
git remote add origin https://github.com/DaFakePhoenixHK/EEG-Motor-Imagery-Thesis-reproduction-v2.git
git push -u origin main
```

## License

Model code (`models.py`, `attention_models.py`, `preprocess.py`) retains the **Apache-2.0** notice from the original EEG-ATCNet / KSU distribution.  
Other files in this repo are provided for research reproduction; cite the original papers when using the architectures.

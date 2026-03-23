# Setup on a brand-new PC (Windows)

Use this checklist when **nothing is installed** yet. Assumes **Windows 10/11**.

---

## 1. Git (to clone the repo)

1. Download **Git for Windows**: https://git-scm.com/download/win  
2. Run the installer (defaults are fine; optional: choose your editor).  
3. Restart the terminal, then verify:

```powershell
git --version
```

---

## 2. Python 3.10 or 3.11 (recommended)

TensorFlow 2.x works well with **64-bit** Python **3.10** or **3.11**.

1. Download from https://www.python.org/downloads/  
2. During install, check **“Add python.exe to PATH”**.  
3. Verify:

```powershell
python --version
pip --version
```

If `python` is not found, try `py -3.11` (or `py -3.10`) instead of `python`.

---

## 3. Clone the project

```powershell
cd $HOME\Desktop
git clone https://github.com/DaFakePhoenixHK/EEG-Motor-Imagery-Thesis-reproduction-v2.git
cd EEG-Motor-Imagery-Thesis-reproduction-v2
```

---

## 4. Virtual environment (keeps packages isolated)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

---

## 5. Install Python dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

This installs **TensorFlow**, NumPy, SciPy, scikit-learn, matplotlib, MNE, etc.

**First run** may download large wheels; wait until it finishes without errors.

---

## 6. GPU (optional — only if you have an NVIDIA GPU)

- Default `pip install tensorflow` uses **CPU**; training still works, but slower.
- For **GPU**, see the official guide (CUDA/cuDNN versions must match your TF version):  
  https://www.tensorflow.org/install/pip  
- If you skip this, you are on **CPU-only** — fine for testing; long runs will be slower.

---

## 7. Dataset (BCI Competition IV-2a)

The repo **does not** include `.mat` files (too large for Git).

1. Obtain **BCI Competition IV-2a** (same files you used before).  
2. Put them under:

```text
EEG-Motor-Imagery-Thesis-reproduction-v2\data\bci2a\
```

So you have paths like `data\bci2a\A01T.mat`, `A01E.mat`, … (or your usual BNCI layout).

3. Or pass any folder that contains those files:

```powershell
python -m reproduction_benchmark_v2.run_benchmark --data "D:\path\to\your\bci2a" --protocol L --model eegnetv4 --channels 8 --seed 0
```

---

## 8. Smoke test

With venv **activated** and **inside the repo folder**:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
python -m reproduction_benchmark_v2.run_benchmark --help
```

---

## 9. Run one experiment (example)

```powershell
python -m reproduction_benchmark_v2.run_benchmark --data ".\data\bci2a" --protocol L --model eegnetv4 --channels 8 --seed 0
```

Results go under `results\` by default.

---

## What you do **not** need for this project

- **Node.js**, **Java**, **Docker** — not required for this repo.  
- **MATLAB** — not required; data is read in Python (`scipy.io`).

---

## Troubleshooting

| Problem | What to try |
|--------|-------------|
| `pip` fails building a package | Use Python **3.10/3.11 64-bit**, upgrade pip, retry. |
| TensorFlow import error | Reinstall: `pip install "tensorflow>=2.9,<2.19"` |
| Out of memory while training | Use a smaller model (`eegnetv4`), fewer epochs in code, or a machine with more RAM; GPU helps. |
| Slow training | Expected on CPU; reduce scope (one protocol, one seed) for testing. |

---

## Minimal install summary

| Software | Role |
|----------|------|
| **Git** | Clone / update the repo |
| **Python 3.10/3.11 (64-bit)** | Runtime |
| **pip + requirements.txt** | All project libraries |

Everything else is optional (GPU) or data (`.mat` files you provide).

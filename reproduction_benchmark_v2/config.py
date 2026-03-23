"""
Reproduction benchmark v2 (BCI IV-2a): separate from results_reproduction/.
- Time window: default 0.5–4.5 s within the BNCI 1.5–6 s crop (see data_loader.py).
- Protocols: W, L, F only (no TTA). F runs K ∈ {1,5,10,20}.
- Seeds: 0, 1 by default.
"""
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
# Repository root (this folder is self-contained under e.g. Thesis/file_ver2)
PARENT_DIR = BENCH_DIR.parent
DEFAULT_BCI2A_PATH = PARENT_DIR / "data" / "bci2a"
DEFAULT_RESULTS_DIR = PARENT_DIR / "results"

# BCI IV-2a
N_SUBJECTS = 9
N_CLASSES = 4
FS = 250
N_CHANNELS_FULL = 22

# Raw BNCI crop from preprocess.load_BCI2a_data: 1.5–6.0 s → 1125 samples
FULL_CROP_SAMPLES = 1125

# Default v2 window: 0.5–4.5 s *relative to start of the 1.5–6 s segment*
# (i.e. keep samples [125:1125] → 1000 samples = 4.0 s)
TIME_REL_START_SEC = 0.5
TIME_REL_END_SEC = 4.5

# Legacy window: full 1.5–6 s segment (1125 samples)
LEGACY_TIME_REL_START_SEC = 0.0
LEGACY_TIME_REL_END_SEC = 4.5  # duration of segment = 1125/250

# 8-channel subset (BNCI 22ch order, 0-based)
EIGHT_CH_INDICES = [3, 7, 8, 9, 10, 11, 14, 16]

# Protocol F
K_PER_CLASS_GRID = [1, 5, 10, 20]

# Seeds (v2 default)
SEEDS = [0, 1]

# Training
DEFAULT_EPOCHS = 500
EARLY_STOP_START_EPOCH = 100
EARLY_STOP_PATIENCE = 80

# All models for full 8ch / 22ch matrix
MODELS_FULL = [
    "fbcsp_lda",
    "eegnetv4",
    "shallow",
    "deep4",
    "conformer",
    "atcnet",
    "db_atcnet",
]

MATRIX_8CH = {p: list(MODELS_FULL) for p in ("W", "L", "F")}
MATRIX_22CH = {p: list(MODELS_FULL) for p in ("W", "L", "F")}

# Old benchmark path (for importing fbcsp_lda)
LEGACY_BENCH_DIR = PARENT_DIR / "reproduction_benchmark"

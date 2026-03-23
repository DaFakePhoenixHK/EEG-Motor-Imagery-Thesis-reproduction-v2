"""
BCI IV-2a loader for reproduction v2.

Uses preprocess.load_BCI2a_data (same as v1): returns (N, 22, 1125) for the
standard 1.5–6.0 s epoch, then optionally crops a sub-window in *seconds
relative to the first sample of that segment*.

Default v2: 0.5–4.5 s → indices [125:1125) = 1000 samples (4.0 s at 250 Hz).
"""
import os
import sys
from pathlib import Path

import numpy as np

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from .config import (
    FS,
    EIGHT_CH_INDICES,
    TIME_REL_START_SEC,
    TIME_REL_END_SEC,
)


def _slice_time(X, rel_start_sec: float, rel_end_sec: float, fs: int = FS):
    """X: (N, C, T). Crop time axis to [rel_start_sec, rel_end_sec) relative to segment start."""
    s = int(round(rel_start_sec * fs))
    e = int(round(rel_end_sec * fs))
    if s < 0 or e > X.shape[-1] or s >= e:
        raise ValueError(
            f"Invalid time slice [{rel_start_sec}, {rel_end_sec}) s for T={X.shape[-1]} "
            f"(indices {s}:{e})"
        )
    return X[..., s:e].astype(np.float32)


def load_bci2a_raw(
    data_path,
    subject_id,
    session_train,
    n_channels=22,
    rel_start_sec: float | None = None,
    rel_end_sec: float | None = None,
):
    """
    Load one subject; optionally crop time.

    rel_start_sec / rel_end_sec:
      None → use config defaults (TIME_REL_START_SEC, TIME_REL_END_SEC).
      To load full 1.5–6 s (1125 samples): rel_start_sec=0, rel_end_sec=4.5.
    """
    from preprocess import load_BCI2a_data

    if rel_start_sec is None:
        rel_start_sec = TIME_REL_START_SEC
    if rel_end_sec is None:
        rel_end_sec = TIME_REL_END_SEC

    path = str(data_path).replace("\\", "/").rstrip("/") + "/"
    if not os.path.exists(os.path.join(path, f"A0{subject_id}T.mat")):
        path = os.path.join(path, f"s{subject_id}") + "/"
    X, y = load_BCI2a_data(path, subject_id, session_train)
    # X: (N, 22, 1125)
    if n_channels == 8:
        X = X[:, EIGHT_CH_INDICES, :]
    elif n_channels != 22:
        raise ValueError("n_channels must be 8 or 22")

    X = _slice_time(X, rel_start_sec, rel_end_sec, FS)
    return X, y.astype(np.int32)


def n_times_from_window(rel_start_sec: float, rel_end_sec: float, fs: int = FS) -> int:
    s = int(round(rel_start_sec * fs))
    e = int(round(rel_end_sec * fs))
    return e - s


def describe_window(rel_start_sec: float, rel_end_sec: float) -> dict:
    return {
        "epoch_source": "preprocess.load_BCI2a_data (1.5–6.0 s BNCI crop, 1125 samples before sub-crop)",
        "time_relative_to_segment_start_sec": [rel_start_sec, rel_end_sec],
        "duration_sec": rel_end_sec - rel_start_sec,
        "n_samples": n_times_from_window(rel_start_sec, rel_end_sec),
        "fs_hz": FS,
    }


def to_4d(X, n_channels):
    return X[:, np.newaxis, :, :]


def standardize_fit_apply(X_train, X_val, X_test, n_channels):
    from sklearn.preprocessing import StandardScaler

    scalers = []
    for j in range(n_channels):
        s = StandardScaler()
        s.fit(X_train[:, 0, j, :])
        scalers.append(s)
    for X in (X_train, X_val, X_test):
        for j in range(n_channels):
            X[:, 0, j, :] = scalers[j].transform(X[:, 0, j, :])
    return scalers

"""
Protocols W, L, F for reproduction v2 (no TTA).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from tensorflow.keras.utils import to_categorical

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from reproduction_benchmark_v2.config import DEFAULT_EPOCHS, EARLY_STOP_START_EPOCH, EARLY_STOP_PATIENCE
from reproduction_benchmark_v2.data_loader import (
    load_bci2a_raw,
    to_4d,
    standardize_fit_apply,
    n_times_from_window,
)


def _shuffle(X, y, seed):
    idx = np.arange(len(X))
    np.random.default_rng(seed).shuffle(idx)
    return X[idx], y[idx]


def _acc_f1_kappa(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    kap = float(cohen_kappa_score(y_true, y_pred))
    return acc, f1, kap


def _itr(n_classes, acc):
    import math
    if acc <= 0 or acc >= 1:
        return 0.0
    p = acc
    b = math.log2(n_classes) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n_classes - 1))
    return max(0, b)


def _fit_model(model, model_name, X_tr, y_tr_oh, X_val, y_val_oh, epochs, batch_size, seed):
    if model_name == "fbcsp_lda":
        from reproduction_benchmark.fbcsp_lda import FBCSP_LDA
        clf = FBCSP_LDA(n_classes=y_tr_oh.shape[1], random_state=seed)
        clf.fit(X_tr, np.argmax(y_tr_oh, axis=1))
        return clf, None
    if model is None:
        raise ValueError("Model is None")
    try:
        from training_utils import _make_early_stopping
        es = _make_early_stopping(start_epoch=EARLY_STOP_START_EPOCH, patience=EARLY_STOP_PATIENCE)
    except Exception:
        es = []
    callbacks = [es] if isinstance(es, (list, tuple)) and len(es) > 0 else ([es] if es else [])
    if not isinstance(callbacks, list):
        callbacks = [callbacks]
    history = model.fit(X_tr, y_tr_oh, validation_data=(X_val, y_val_oh), epochs=epochs,
                        batch_size=batch_size, callbacks=callbacks, verbose=1)
    return model, history


def _predict(model, model_name, X_te, n_classes):
    if model_name == "fbcsp_lda":
        return model.predict(X_te)
    return np.argmax(model.predict(X_te), axis=1)


def protocol_W(data_path, n_channels, model_name, seed, rel_start_sec, rel_end_sec, epochs=None, batch_size=64):
    from reproduction_benchmark_v2.models_registry import get_model
    n_times = n_times_from_window(rel_start_sec, rel_end_sec)
    n_classes = 4
    epochs = epochs or DEFAULT_EPOCHS
    results = []
    for sub_id in range(1, 10):
        X_tr, y_tr = load_bci2a_raw(data_path, sub_id, True, n_channels, rel_start_sec, rel_end_sec)
        X_te, y_te = load_bci2a_raw(data_path, sub_id, False, n_channels, rel_start_sec, rel_end_sec)
        X_tr, y_tr = _shuffle(X_tr, y_tr, seed)
        X_tr = to_4d(X_tr, n_channels)
        X_te = to_4d(X_te, n_channels)
        X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)
        standardize_fit_apply(X_tr, X_val, X_te, n_channels)
        y_tr_oh = to_categorical(y_tr, n_classes)
        y_val_oh = to_categorical(y_val, n_classes)
        model = get_model(model_name, n_channels, n_times, n_classes, seed)
        model, hist = _fit_model(model, model_name, X_tr, y_tr_oh, X_val, y_val_oh, epochs, batch_size, seed)
        y_pred = _predict(model, model_name, X_te, n_classes)
        acc, f1, kap = _acc_f1_kappa(y_te, y_pred)
        itr = _itr(n_classes, acc)
        cm = confusion_matrix(y_te, y_pred)
        res = {"subject": sub_id, "trialAcc": acc, "macroF1": f1, "kappa": kap, "ITR": itr, "confusion_matrix": cm}
        if hist is not None:
            res["history"] = hist.history
        results.append(res)
    return results


def protocol_L(data_path, n_channels, model_name, seed, rel_start_sec, rel_end_sec, epochs=None, batch_size=64):
    from reproduction_benchmark_v2.models_registry import get_model
    n_times = n_times_from_window(rel_start_sec, rel_end_sec)
    n_classes = 4
    epochs = epochs or DEFAULT_EPOCHS
    results = []
    for target_sub in range(1, 10):
        X_all_tr, y_all_tr = [], []
        for sub_id in range(1, 10):
            if sub_id == target_sub:
                continue
            X_tr, y_tr = load_bci2a_raw(data_path, sub_id, True, n_channels, rel_start_sec, rel_end_sec)
            X_all_tr.append(X_tr)
            y_all_tr.append(y_tr)
        X_all_tr = np.concatenate(X_all_tr, axis=0)
        y_all_tr = np.concatenate(y_all_tr, axis=0)
        X_all_tr, y_all_tr = _shuffle(X_all_tr, y_all_tr, seed)
        X_all_tr = to_4d(X_all_tr, n_channels)
        X_te, y_te = load_bci2a_raw(data_path, target_sub, False, n_channels, rel_start_sec, rel_end_sec)
        X_te = to_4d(X_te, n_channels)
        X_tr, X_val, y_tr, y_val = train_test_split(X_all_tr, y_all_tr, test_size=0.2, random_state=seed, stratify=y_all_tr)
        standardize_fit_apply(X_tr, X_val, X_te, n_channels)
        y_tr_oh = to_categorical(y_tr, n_classes)
        y_val_oh = to_categorical(y_val, n_classes)
        model = get_model(model_name, n_channels, n_times, n_classes, seed)
        model, hist = _fit_model(model, model_name, X_tr, y_tr_oh, X_val, y_val_oh, epochs, batch_size, seed)
        y_pred = _predict(model, model_name, X_te, n_classes)
        acc, f1, kap = _acc_f1_kappa(y_te, y_pred)
        itr = _itr(n_classes, acc)
        cm = confusion_matrix(y_te, y_pred)
        res = {"subject": target_sub, "trialAcc": acc, "macroF1": f1, "kappa": kap, "ITR": itr, "confusion_matrix": cm}
        if hist is not None:
            res["history"] = hist.history
        results.append(res)
    return results


def protocol_F(data_path, n_channels, model_name, seed, k_per_class, rel_start_sec, rel_end_sec, epochs=None, batch_size=64):
    from reproduction_benchmark_v2.models_registry import get_model
    n_times = n_times_from_window(rel_start_sec, rel_end_sec)
    n_classes = 4
    epochs = epochs or DEFAULT_EPOCHS
    results = []
    for target_sub in range(1, 10):
        X_all_tr, y_all_tr = [], []
        for sub_id in range(1, 10):
            if sub_id == target_sub:
                continue
            X_tr, y_tr = load_bci2a_raw(data_path, sub_id, True, n_channels, rel_start_sec, rel_end_sec)
            X_all_tr.append(X_tr)
            y_all_tr.append(y_tr)
        X_all_tr = np.concatenate(X_all_tr, axis=0)
        y_all_tr = np.concatenate(y_all_tr, axis=0)
        X_all_tr, y_all_tr = _shuffle(X_all_tr, y_all_tr, seed)
        X_all_tr = to_4d(X_all_tr, n_channels)
        X_cal, y_cal = load_bci2a_raw(data_path, target_sub, True, n_channels, rel_start_sec, rel_end_sec)
        X_te, y_te = load_bci2a_raw(data_path, target_sub, False, n_channels, rel_start_sec, rel_end_sec)
        X_cal = to_4d(X_cal, n_channels)
        X_te = to_4d(X_te, n_channels)
        cal_list = []
        for c in range(n_classes):
            idx = np.where(y_cal == c)[0]
            np.random.default_rng(seed).shuffle(idx)
            take = min(k_per_class, len(idx))
            cal_list.append(idx[:take])
        cal_idx = np.concatenate(cal_list)
        X_cal = X_cal[cal_idx]
        y_cal = y_cal[cal_idx]
        X_comb = np.concatenate([X_all_tr, X_cal], axis=0)
        y_comb = np.concatenate([y_all_tr, y_cal], axis=0)
        X_comb, y_comb = _shuffle(X_comb, y_comb, seed)
        X_tr, X_val, y_tr, y_val = train_test_split(X_comb, y_comb, test_size=0.2, random_state=seed, stratify=y_comb)
        standardize_fit_apply(X_tr, X_val, X_te, n_channels)
        y_tr_oh = to_categorical(y_tr, n_classes)
        y_val_oh = to_categorical(y_val, n_classes)
        model = get_model(model_name, n_channels, n_times, n_classes, seed)
        model, hist = _fit_model(model, model_name, X_tr, y_tr_oh, X_val, y_val_oh, epochs, batch_size, seed)
        y_pred = _predict(model, model_name, X_te, n_classes)
        acc, f1, kap = _acc_f1_kappa(y_te, y_pred)
        itr = _itr(n_classes, acc)
        cm = confusion_matrix(y_te, y_pred)
        res = {"subject": target_sub, "trialAcc": acc, "macroF1": f1, "kappa": kap, "ITR": itr, "confusion_matrix": cm}
        if hist is not None:
            res["history"] = hist.history
        results.append(res)
    return results

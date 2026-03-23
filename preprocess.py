"""
Copyright (C) 2022 King Saud University, Saudi Arabia
SPDX-License-Identifier: Apache-2.0

Author: Hamdi Altaheri
"""

import os
import numpy as np
import scipy.io as sio
from scipy import signal
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


# BCI2a preprocessing constants (used in load_BCI2a_data and for reporting)
BCI2A_FS = 250
BCI2A_N_CHANNELS = 22
BCI2A_WINDOW_LENGTH_RAW = 7 * 250   # samples extracted per trial (before cropping)
BCI2A_T1_SAMPLE = int(1.5 * 250)    # start of motor imagery segment (samples)
BCI2A_T2_SAMPLE = int(6.0 * 250)    # end of motor imagery segment (samples)
BCI2A_MI_START_SEC = 1.5
BCI2A_MI_END_SEC = 6.0
BCI2A_IN_SAMPLES = BCI2A_T2_SAMPLE - BCI2A_T1_SAMPLE  # 1125


def get_preprocessing_info_bci2a(used_preprocessed_path=None):
    """
    Return a dict describing BCI2a preprocessing (for reports).
    If used_preprocessed_path is set, notes that user-provided data was used instead.
    """
    if used_preprocessed_path:
        return {
            'source': 'user_provided',
            'preprocessed_path': used_preprocessed_path,
            'note': 'Data was loaded from user-provided preprocessed files; no internal preprocessing was applied.',
        }
    return {
        'dataset': 'BCI Competition IV-2a',
        'raw_data': {
            'session_train': 'AxxT.mat (session 1)',
            'session_test': 'AxxE.mat (session 2)',
            'sampling_rate_hz': BCI2A_FS,
            'n_channels': BCI2A_N_CHANNELS,
            'downsampling': 'None (data used at 250 Hz as provided)',
            'filtering': 'None (no bandpass/bandstop in this pipeline; raw EEG segments used)',
        },
        'trial_segment': {
            'extraction_window_seconds': 7.0,
            'extraction_window_samples': BCI2A_WINDOW_LENGTH_RAW,
            'motor_imagery_period_seconds': (BCI2A_MI_START_SEC, BCI2A_MI_END_SEC),
            'motor_imagery_period_description': '1.5 s to 6.0 s (relative to trial start); 4.5 s duration',
            'samples_after_crop': BCI2A_IN_SAMPLES,
        },
        'standardization': 'Per-channel StandardScaler. Fit on train portion only (80% of session 1 after split); transform train, val, and test. No leakage from val/test into scaler.',
        'shuffle': 'Shuffle train and test with random_state=42 before split.',
        'train_val_split': 'Split session 1 into 80% train / 20% val first, then standardize (fit on train only). Test = full session 2.',
    }


def load_BCI2a_data(data_path, subject, training, all_trials=True):
    """Load BCI Competition IV-2a subject-specific: session 1 = train, session 2 = test."""
    n_channels = 22
    n_tests = 6 * 48
    window_Length = 7 * 250
    fs = 250
    t1 = int(1.5 * fs)
    t2 = int(6 * fs)
    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))
    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]
        for trial in range(0, a_trial.size):
            if a_artifacts[trial] != 0 and not all_trials:
                continue
            # Handle numpy array/scalar conversion properly
            trial_start = int(a_trial[trial].item() if hasattr(a_trial[trial], 'item') else a_trial[trial])
            trial_end = trial_start + window_Length
            data_return[NO_valid_trial, :, :] = np.transpose(
                a_X[trial_start:trial_end, :22])
            class_val = a_y[trial].item() if hasattr(a_y[trial], 'item') else a_y[trial]
            class_return[NO_valid_trial] = int(class_val)
            NO_valid_trial += 1
    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return - 1).astype(int)
    return data_return, class_return


def load_BCI2a_data_cosupervisor_style(data_path, subject, training, all_trials=True):
    """
    Load BCI2a with cosupervisor-style preprocessing (128 Hz, 0.5-4.0 s epoch) BUT NO FILTERING.
    Matches cosupervisor's downsampling and time window, but keeps raw frequencies (no bandpass).
    
    Steps:
    1. Load raw data (250 Hz, 7 s window)
    2. Downsample to 128 Hz
    3. Extract epoch 0.5-4.0 s (relative to trial start, which is 2.0 s before cue)
       So: trial_start + 0.5s = cue - 1.5s, trial_start + 4.0s = cue + 2.0s
    4. NO bandpass filtering (raw frequencies preserved)
    
    Returns: (data_return, class_return) where data_return is (n_trials, n_channels, n_samples)
    with n_samples = 448 (3.5 s * 128 Hz).
    """
    n_channels = 22
    n_tests = 6 * 48
    fs_original = 250
    fs_target = 128
    # Extract 7-second window at original sampling rate
    window_Length_original = 7 * fs_original  # 1750 samples at 250 Hz
    # Time window: 0.5-4.0 s relative to trial start (trial start is 2.0 s before cue)
    t_start_sec = 0.5
    t_end_sec = 4.0
    # At original fs: samples from trial_start
    t1_original = int(t_start_sec * fs_original)  # 125 samples
    t2_original = int(t_end_sec * fs_original)   # 1000 samples
    # After downsampling: target samples
    t1_target = int(t_start_sec * fs_target)     # 64 samples
    t2_target = int(t_end_sec * fs_target)       # 512 samples
    n_samples_final = t2_target - t1_target      # 448 samples
    
    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, n_samples_final))
    NO_valid_trial = 0
    
    if training:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']
    
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]
        
        for trial in range(0, a_trial.size):
            if a_artifacts[trial] != 0 and not all_trials:
                continue
            
            trial_start = int(a_trial[trial].item() if hasattr(a_trial[trial], 'item') else a_trial[trial])
            trial_end = trial_start + window_Length_original
            
            # Extract 7-second window: (n_channels, 1750)
            trial_data = np.transpose(a_X[trial_start:trial_end, :22])  # (22, 1750)
            
            # Downsample each channel from 250 Hz to 128 Hz
            # scipy.signal.resample: new_length = old_length * (new_fs / old_fs)
            n_samples_target = int(trial_data.shape[1] * fs_target / fs_original)  # 1750 * 128/250 = 896
            trial_data_downsampled = np.zeros((n_channels, n_samples_target), dtype=np.float32)
            for ch in range(n_channels):
                trial_data_downsampled[ch, :] = signal.resample(trial_data[ch, :], n_samples_target)
            
            # Extract epoch 0.5-4.0 s (at 128 Hz: 64 to 512 samples)
            epoch_data = trial_data_downsampled[:, t1_target:t2_target]  # (22, 448)
            
            data_return[NO_valid_trial, :, :] = epoch_data
            class_val = a_y[trial].item() if hasattr(a_y[trial], 'item') else a_y[trial]
            class_return[NO_valid_trial] = int(class_val)
            NO_valid_trial += 1
    
    data_return = data_return[0:NO_valid_trial, :, :]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return - 1).astype(int)
    return data_return, class_return


def standardize_data(X_train, X_test, channels):
    for j in range(channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, 0, j, :])
        X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])
    return X_train, X_test


def standardize_fit_train_transform_train_val_test(X_train, X_val, X_test, channels):
    """Fit StandardScaler on train only, then transform train/val/test. Use after train/val split to avoid leakage."""
    for j in range(channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, 0, j, :])
        X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
        X_val[:, 0, j, :] = scaler.transform(X_val[:, 0, j, :])
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])
    return X_train, X_val, X_test


def standardize_fit_train_return_scalers(X_train, channels):
    """Fit StandardScaler per channel on X_train; return list of scalers for later use on val/test."""
    scalers = []
    for j in range(channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, 0, j, :])
        scalers.append(scaler)
    return scalers


def standardize_apply_scalers(X, scalers, channels):
    """Apply pre-fitted scalers to X (in-place). X shape (N, 1, channels, T)."""
    for j in range(channels):
        X[:, 0, j, :] = scalers[j].transform(X[:, 0, j, :])
    return X


def _infer_xy_from_npz(z):
    """
    Infer data (X) and labels (y) from an npz with unknown keys.
    Returns (X, y) or (None, None). X will be (n_trials, n_chans, n_times).
    """
    keys = list(z.keys())
    # Find array that looks like EEG data: 3D (n_trials, ..., ...)
    X_candidates = [(k, z[k]) for k in keys
                   if hasattr(z[k], 'shape') and z[k].ndim == 3 and z[k].shape[0] >= 10]
    if not X_candidates:
        return None, None
    # Prefer key names that suggest data
    for name in ('x', 'X', 'data', 'epochs'):
        for k, arr in X_candidates:
            if k == name or (isinstance(k, str) and name in k.lower()):
                X = np.asarray(arr)
                break
        else:
            continue
        break
    else:
        _, X = X_candidates[0]
        X = np.asarray(X)
    # Find labels: 1D (n_trials,) or 2D (n_trials, n_classes). Prefer key 'y' / 'labels' so we
    # don't mistake metadata (e.g. 'subject', 'session', 'trial') for labels.
    meta_like_keys = frozenset(('subject', 'subject_id', 'subjects', 'session', 'trial', 'dataset', 'file_id', 'is_rest'))
    y = None
    for preferred in ('y', 'Y', 'labels', 'label', 'target'):
        for k in keys:
            if k in meta_like_keys:
                continue
            if preferred != k and (not isinstance(k, str) or preferred.lower() not in k.lower()):
                continue
            arr = z[k]
            if not hasattr(arr, 'shape'):
                continue
            if arr.ndim == 1 and len(arr) == X.shape[0]:
                y = arr
                break
            if arr.ndim == 2 and arr.shape[0] == X.shape[0] and 1 <= arr.shape[1] <= 4:
                y = arr
                break
        if y is not None:
            break
    if y is None:
        for k in keys:
            if k in meta_like_keys:
                continue
            arr = z[k]
            if not hasattr(arr, 'shape'):
                continue
            if arr.ndim == 1 and len(arr) == X.shape[0]:
                y = arr
                break
            if arr.ndim == 2 and arr.shape[0] == X.shape[0] and 1 <= arr.shape[1] <= 4:
                y = arr
                break
    if y is None:
        return None, None
    # Ensure X is (n_trials, n_chans, n_times): EEG typically n_chans < n_times
    if X.shape[1] > X.shape[2]:
        X = np.transpose(X, (0, 2, 1))
    y = np.asarray(y).flatten() if np.asarray(y).ndim == 2 and np.asarray(y).shape[1] == 1 else np.asarray(y)
    if np.asarray(y).ndim == 2:
        y = np.argmax(y, axis=-1)
    y = y.astype(np.int32)
    if y.min() >= 1 and y.max() <= 4:
        y = (y - 1).astype(np.int32)
    return X.astype(np.float32), y


def _get_subject_mask_from_npz(z, n_trials):
    """
    Get 1D subject array from npz if present; return (array or None). Subject IDs typically 1-9.
    Cosupervisor bci2a.py (DatasetOutput) saves meta with keys e.g. 'subject' per trial;
    save_npz may store top-level 'subject' or nest under 'meta' (allow_pickle).
    """
    # 1) Top-level subject array (e.g. np.savez(..., subject=..., **meta))
    for key in ('subject', 'subject_id', 'subjects', 'sub', 'subject_index', 'subject_indices', 'subject_ids', 'meta_subject'):
        if key in z:
            arr = np.asarray(z[key]).flatten()
            if arr.shape[0] == n_trials:
                return arr
    # 2) Cosupervisor format: meta is object array (0-d or (1,)) holding dict with key 'subject'
    if 'meta' in z:
        meta = z['meta']
        try:
            if hasattr(meta, 'ndim'):
                if meta.ndim == 0:
                    meta = meta.item()
                elif meta.size == 1:
                    meta = meta.flat[0]  # e.g. shape (1,) -> unwrap to dict
            if isinstance(meta, dict) and 'subject' in meta:
                arr = np.asarray(meta['subject']).flatten()
                if arr.shape[0] == n_trials:
                    return arr
        except (ValueError, TypeError, AttributeError):
            pass
    return None


def load_bci2a_summary_npz(train_npz_path, test_npz_path, val_frac=0.2, random_state=42, subject_id=None):
    """
    Load cosupervisor-style BCI2a preprocessed data from two .npz files.
    Preprocess: fs=128 Hz (downsampled), bandpass 4-40 Hz, epoch 0.5-4 s -> 448 samples/trial.
    Data merges all subjects 1-9; if subject_id is set (1-9), filter to that subject only
    (npz must contain 'subject' or 'subject_id' or 'subjects' array).
    Returns (X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot), data_spec
    where data_spec = {'n_channels': int, 'in_samples': int, 'n_classes': int, 'fs': 128}.
    """
    train_npz_path = os.path.abspath(os.path.expanduser(train_npz_path))
    test_npz_path = os.path.abspath(os.path.expanduser(test_npz_path))
    if not os.path.isfile(train_npz_path) or not os.path.isfile(test_npz_path):
        return None, None
    try:
        with np.load(train_npz_path, allow_pickle=True) as z:
            X_tr, y_tr = _infer_xy_from_npz(z)
            sub_tr = _get_subject_mask_from_npz(z, X_tr.shape[0]) if X_tr is not None else None
        with np.load(test_npz_path, allow_pickle=True) as z:
            X_te, y_te = _infer_xy_from_npz(z)
            sub_te = _get_subject_mask_from_npz(z, X_te.shape[0]) if X_te is not None else None
    except Exception:
        return None, None
    if X_tr is None or X_te is None:
        return None, None
    # Filter by subject if requested (MI dataset merges all subjects; fixed-subject uses one)
    if subject_id is not None and sub_tr is not None and sub_te is not None:
        # Subject IDs in npz may be 1-based (1-9) or 0-based (0-8)
        sub_tr = np.asarray(sub_tr).flatten()
        sub_te = np.asarray(sub_te).flatten()
        mask_tr = (sub_tr == subject_id) | (sub_tr == subject_id - 1)
        mask_te = (sub_te == subject_id) | (sub_te == subject_id - 1)
        if np.any(mask_tr) and np.any(mask_te):
            X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
            X_te, y_te = X_te[mask_te], y_te[mask_te]
        # else: no trials for this subject, keep all data and warn below
    elif subject_id is not None and (sub_tr is None or sub_te is None):
        import warnings
        warnings.warn(
            "subject_id=%s was set but npz files do not contain a subject array (tried top-level "
            "'subject'/'subject_id'/... and 'meta' dict with 'subject'). Using all subjects in the npz." % (subject_id,),
            UserWarning,
            stacklevel=2,
        )
    # Shapes: X (n_trials, n_chans, n_times)
    n_ch = X_tr.shape[1]
    n_times = X_tr.shape[2]
    n_classes = int(max(y_tr.max(), y_te.max()) + 1)
    n_classes = max(n_classes, 4)
    # Train/val split
    X_tr, y_tr = shuffle(X_tr, y_tr, random_state=random_state)
    n_val = max(1, int(len(X_tr) * val_frac))
    X_val = X_tr[-n_val:]
    y_val = y_tr[-n_val:]
    X_train = X_tr[:-n_val]
    y_train = y_tr[:-n_val]
    # One-hot
    y_train_onehot = to_categorical(y_train, num_classes=n_classes)
    y_val_onehot = to_categorical(y_val, num_classes=n_classes)
    y_test_onehot = to_categorical(y_te, num_classes=n_classes)
    # (N, n_ch, n_times) -> (N, 1, n_ch, n_times)
    X_train = X_train[:, np.newaxis, :, :].astype(np.float32)
    X_val = X_val[:, np.newaxis, :, :].astype(np.float32)
    X_test = X_te[:, np.newaxis, :, :].astype(np.float32)
    # MI dataset: downsampled to 128 Hz, segment 0.5 s–4 s
    data_spec = {'n_channels': n_ch, 'in_samples': n_times, 'n_classes': n_classes, 'fs': 128}
    return (X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot), data_spec


def load_user_preprocessed(folder_path):
    """
    Load user-provided preprocessed arrays from a folder.
    Expects either:
      - preprocessed.npz with keys: X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot
      - or X_train.npy, y_train_onehot.npy, X_val.npy, y_val_onehot.npy, X_test.npy, y_test_onehot.npy
    Shapes: X_* (N, 1, n_channels, n_samples), y_* onehot (N, n_classes).
    Returns (X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot) or None if invalid.
    """
    folder_path = os.path.abspath(os.path.expanduser(folder_path))
    if not os.path.isdir(folder_path):
        return None
    npz_path = os.path.join(folder_path, 'preprocessed.npz')
    if os.path.isfile(npz_path):
        try:
            with np.load(npz_path, allow_pickle=False) as z:
                X_train = z['X_train']
                y_train_onehot = z['y_train_onehot']
                X_val = z['X_val']
                y_val_onehot = z['y_val_onehot']
                X_test = z['X_test']
                y_test_onehot = z['y_test_onehot']
            for a in (X_train, X_val, X_test):
                if a.ndim != 4 or a.shape[1] != 1:
                    return None
            return (X_train.astype(np.float32), y_train_onehot,
                    X_val.astype(np.float32), y_val_onehot,
                    X_test.astype(np.float32), y_test_onehot)
        except Exception:
            return None
    required = ['X_train.npy', 'y_train_onehot.npy', 'X_val.npy', 'y_val_onehot.npy', 'X_test.npy', 'y_test_onehot.npy']
    if not all(os.path.isfile(os.path.join(folder_path, f)) for f in required):
        return None
    try:
        X_train = np.load(os.path.join(folder_path, 'X_train.npy')).astype(np.float32)
        y_train_onehot = np.load(os.path.join(folder_path, 'y_train_onehot.npy'))
        X_val = np.load(os.path.join(folder_path, 'X_val.npy')).astype(np.float32)
        y_val_onehot = np.load(os.path.join(folder_path, 'y_val_onehot.npy'))
        X_test = np.load(os.path.join(folder_path, 'X_test.npy')).astype(np.float32)
        y_test_onehot = np.load(os.path.join(folder_path, 'y_test_onehot.npy'))
        if X_train.ndim != 4 or X_train.shape[1] != 1:
            return None
        return (X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot)
    except Exception:
        return None


def get_data(path, subject, dataset='BCI2a', classes_labels='all', LOSO=False, isStandard=True, isShuffle=True, preprocessing_style='original'):
    """
    Load BCI2a data with optional preprocessing styles.
    
    Args:
        preprocessing_style: 'original' (250 Hz, 1.5-6.0 s) or 'cosupervisor_no_filter' (128 Hz, 0.5-4.0 s, no bandpass)
    """
    if LOSO:
        raise NotImplementedError("LOSO loading not replicated here; use original repo.")
    if dataset == 'BCI2a':
        path = path.replace('\\', '/').rstrip('/') + '/'
        # Support flat folder: A01T.mat, A01E.mat directly in path (no s1/, s2/ subfolders)
        if os.path.exists(os.path.join(path, 'A01T.mat')):
            path_dir = path
        else:
            path_dir = path + 's{:}/'.format(subject + 1)
        
        if preprocessing_style == 'cosupervisor_no_filter':
            X_train, y_train = load_BCI2a_data_cosupervisor_style(path_dir, subject + 1, True)
            X_test, y_test = load_BCI2a_data_cosupervisor_style(path_dir, subject + 1, False)
        else:  # 'original'
            X_train, y_train = load_BCI2a_data(path_dir, subject + 1, True)
            X_test, y_test = load_BCI2a_data(path_dir, subject + 1, False)
    else:
        raise Exception("'{}' dataset is not supported in this replication.".format(dataset))
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)
    N_tr, N_ch, T = X_train.shape
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train)
    N_tr, N_ch, T = X_test.shape
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test)
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)
    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot

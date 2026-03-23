"""
FBCSP + LDA for BCI IV-2a (4-class, One-vs-Rest).
Filter Bank CSP: bandpass in multiple bands, CSP per band, concatenate features, LDA.
Uses MNE CSP + sklearn LDA. CSP fit only on training data (leakage-safe).

References:
 - https://github.com/orvindemsy/BCICIV2a-FBCSP
 - Ang et al. 2012 Frontiers: https://www.frontiersin.org/articles/10.3389/fnins.2012.00039
"""
import numpy as np

# Suppress MNE INFO messages ("Computing rank", "Estimating covariance", etc.)
try:
    import mne
    mne.set_log_level("WARNING")
except ImportError:
    pass
from scipy import signal as scipy_signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FBCSP_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36)]
FS = 250


def _bandpass(X, low, high, fs=FS, order=5):
    """X: (N, C, T). Returns (N, C, T)."""
    nyq = fs / 2
    low_n = max(0.01, low / nyq)
    high_n = min(0.99, high / nyq)
    b, a = scipy_signal.butter(order, [low_n, high_n], btype="band")
    out = np.zeros_like(X, dtype=np.float64)
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            out[i, c, :] = scipy_signal.filtfilt(b, a, X[i, c, :].astype(np.float64))
    return out


def _extract_csp_feat(X, csp_list, n_components=4, n_classes=4):
    """Transform using pre-fitted CSPs. X: (N,C,T). Filter per band then CSP transform."""
    from mne import create_info
    from mne.epochs import EpochsArray
    feats = []
    for band_idx, (low, high) in enumerate(FBCSP_BANDS):
        X_filt = _bandpass(X, low, high)
        info = create_info(X_filt.shape[1], FS, "eeg")
        epochs = EpochsArray(X_filt, info, tmin=0, verbose=False)
        data = epochs.get_data()
        band_csps = csp_list[band_idx]
        band_feat = []
        for csp in band_csps:
            if csp is None:
                band_feat.append(np.zeros((X.shape[0], n_components)))
            else:
                band_feat.append(csp.transform(data))
        feats.append(np.concatenate(band_feat, axis=1))
    return np.concatenate(feats, axis=1)


def _fit_csp_per_band(X_train, y_train, n_components=4, n_classes=4):
    """Fit CSP per band, One-vs-Rest. Returns list of lists of fitted CSPs."""
    try:
        from mne.decoding import CSP
        from mne import create_info
        from mne.epochs import EpochsArray
    except ImportError as e:
        import sys
        print(f"FBCSP+LDA: MNE not installed ({e}). Falling back to random predictions (~25%%). Install with: pip install mne", file=sys.stderr)
        return None
    info = create_info(X_train.shape[1], FS, "eeg")
    all_csps = []
    for low, high in FBCSP_BANDS:
        X_filt = _bandpass(X_train, low, high)
        epochs = EpochsArray(X_filt, info, tmin=0, verbose=False)
        data = epochs.get_data()
        band_csps = []
        for c in range(n_classes):
            y_bin = (y_train == c).astype(int)
            if np.sum(y_bin) < 2 or np.sum(1 - y_bin) < 2:
                band_csps.append(None)
                continue
            csp = CSP(n_components=n_components, transform_into="average_power", log=True)
            csp.fit(data, y_bin)
            band_csps.append(csp)
        all_csps.append(band_csps)
    return all_csps


class FBCSP_LDA:
    """FBCSP + LDA. Multi-class: One-vs-Rest. CSP fit only on train."""

    def __init__(self, n_classes=4, n_components=4, random_state=None):
        self.n_classes = n_classes
        self.n_components = n_components
        self.csp_list = None
        self.scaler = StandardScaler()
        self.lda = OneVsRestClassifier(LinearDiscriminantAnalysis())

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 4:
            X = X[:, 0, :, :]  # (N,1,C,T) -> (N,C,T)
        self.csp_list = _fit_csp_per_band(X, y, self.n_components, self.n_classes)
        if self.csp_list is None:
            import warnings
            warnings.warn(
                "FBCSP+LDA: MNE unavailable or CSP fit failed. Using random predictions (~25%%). Install MNE: pip install mne",
                UserWarning,
                stacklevel=2,
            )
            self._fallback = True
            return self
        self._fallback = False
        X_feat = _extract_csp_feat(X, self.csp_list, self.n_components, self.n_classes)
        X_feat = self.scaler.fit_transform(X_feat)
        self.lda.fit(X_feat, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 4:
            X = X[:, 0, :, :]
        if getattr(self, "_fallback", True):
            return np.random.randint(0, self.n_classes, X.shape[0])
        X_feat = _extract_csp_feat(X, self.csp_list, self.n_components, self.n_classes)
        X_feat = self.scaler.transform(X_feat)
        return self.lda.predict(X_feat)

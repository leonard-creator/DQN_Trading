"""Data loading, feature normalization and windowing utilities.

The loader is robust to the different CSV layouts in this repo (SundPGI, SAP,
iShares): it selects the feature columns *by name*, strips BOMs, and parses
thousands-separated / quoted Volume values. Normalization statistics are fit on
the TRAINING data only and reused for validation/test, which avoids the
window-local MinMax leakage the previous version had.
"""

import json
import numpy as np
import pandas as pd

# Feature columns fed to the network (selected by name, order preserved).
DEFAULT_FEATURES = ["Close", "Volume", "ROC12", "MFI14", "FVolatility", "SMA20_Rel", "SMA50_Rel"]

# Per-feature transform applied before the network sees it:
#   logreturn_z : log return of the level, then z-score (scale-free price signal)
#   log_z       : log1p(x), then z-score (compresses heavy-tailed volume)
#   z           : plain z-score using train-set mean/std
#   div100      : divide by 100 (bounded 0-100 oscillators -> ~0-1, no stats)

# adding moving average to better identify trends
FEATURE_TRANSFORMS = {
    "Close": "logreturn_z",
    "Volume": "log_z",
    "ROC12": "z",
    "MFI14": "div100",
    "FVolatility": "z",
    "SMA20_Rel": "z",
    "SMA50_Rel": "z",
}


def formatPrice(n):
    """Human-readable signed euro amount."""
    return ("-€" if n < 0 else "€") + "{0:.2f}".format(abs(n))


def load_ohlcv(path):
    """Read a stock CSV into a DataFrame with clean column names.

    `encoding='utf-8-sig'` drops the BOM present in SAP.csv, and
    `thousands=','` turns quoted values like "1,529,638" into real numbers.
    """
    # index_col=False stops pandas from silently promoting the first data column
    # to the row index when a row has more fields than the header. SAP.csv has a
    # trailing comma (11 fields vs 10 headers); without this every named column
    # would shift by one (Close would actually hold High, etc.).
    df = pd.read_csv(path, thousands=",", encoding="utf-8-sig", index_col=False)
    df.columns = [str(c).strip() for c in df.columns]
    # drop spurious empty / "Unnamed" trailing columns produced by the extra comma
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    return df


def load_features(key, features, test=False):
    """Load a dataset and return (DataFrame, raw_close_prices).

    Raises a clear error if a requested feature column is absent (e.g. the
    iShares file has no ROC12/MFI14/FVolatility columns).
    """
    path = ("test_data/" if test else "train_data/") + key
    df = load_ohlcv(path)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"{key} is missing feature columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    close = df["Close"].to_numpy(dtype=np.float64)
    return df, close


class FeatureScaler:
    """Fits per-feature normalization on training data and applies it elsewhere.

    Persisted to JSON next to the model so evaluation uses identical statistics.
    """

    def __init__(self, features):
        self.features = list(features)
        self.transforms = [FEATURE_TRANSFORMS.get(f, "z") for f in self.features]
        self.mean = {}
        self.std = {}

    @staticmethod
    def _pretransform(kind, x):
        """Apply the non-parametric part of a transform (before z-scoring)."""
        x = np.asarray(x, dtype=np.float64)
        if kind == "logreturn_z":
            lr = np.zeros_like(x)
            lr[1:] = np.log(x[1:] / x[:-1])   # first day has no prior -> 0
            return lr
        if kind == "log_z":
            return np.log1p(np.clip(x, 0, None))
        if kind == "div100":
            return x / 100.0
        return x  # "z" / unknown: leave raw, z-scored below

    def fit(self, df):
        """Compute mean/std for z-scored features using TRAIN data only."""
        for f, kind in zip(self.features, self.transforms):
            pt = self._pretransform(kind, df[f].to_numpy())
            if kind == "z" or kind.endswith("_z"):
                self.mean[f] = float(np.mean(pt))
                self.std[f] = float(np.std(pt) + 1e-8)   # guard against zero variance
        return self

    def transform(self, df):
        """Return a normalized (T, n_feat) float32 matrix."""
        cols = []
        for f, kind in zip(self.features, self.transforms):
            pt = self._pretransform(kind, df[f].to_numpy())
            if f in self.mean:                             # z-scored feature
                pt = (pt - self.mean[f]) / self.std[f]
            cols.append(pt.astype(np.float32))
        return np.stack(cols, axis=1)

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(
                {"features": self.features, "transforms": self.transforms,
                 "mean": self.mean, "std": self.std}, fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        s = cls(d["features"])
        s.transforms = d["transforms"]
        s.mean = d["mean"]
        s.std = d["std"]
        return s


def get_window(matrix, t, window):
    """Return the `window` feature rows ending at index `t` (inclusive).

    Left zero-pads at the start of the series so the shape is always
    (window, n_feat). This is the market half of the observation.
    """
    start = t - window + 1
    if start >= 0:
        block = matrix[start:t + 1]
    else:
        pad = np.zeros((-start, matrix.shape[1]), dtype=matrix.dtype)
        block = np.concatenate((pad, matrix[0:t + 1]), axis=0)
    return block.astype(np.float32)

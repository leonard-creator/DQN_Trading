"""Per-ticker daily price data: download, storage, loading and the test-period guard.

Storage layout (PROTOCOL §3)
----------------------------
    data/raw/<FILE>.csv     one file per ticker, FULL history, columns
                            Date, Open, High, Low, Close, Volume
    data/raw/MANIFEST.json  provenance: source, download time, rows, sha256

Files are written once and then treated as frozen inputs. Yahoo re-bases
adjusted prices whenever a dividend is paid, so re-downloading later gives
slightly different numbers. Keeping the first download (and its hash) is what
makes the baseline numbers reproducible (milestone M1).

Splits are never stored as separate files. They are cut by date at load time
(see harness/splits.py). This avoids the old problems of per-ticker cut dates
and zero-padded windows at the start of each validation file.

Test-period guard (PROTOCOL §9)
-------------------------------
`load_prices()` drops every bar on or after `splits.test_start` unless it is
given the unlock token. The only way to obtain the token is
`unlock_test_period()`, which writes `experiments/FINAL_TEST.lock` and refuses
if that file already exists. So the frozen test period can be read at most
once, and only on purpose (scripts/final_test.py --i-am-sure).
"""

import datetime as _dt
import hashlib
import json
import os

import numpy as np
import pandas as pd

from harness.config import repo_path

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
LOCK_FILE = repo_path("experiments", "FINAL_TEST.lock")


# ---------------------------------------------------------------------------
# file naming
# ---------------------------------------------------------------------------
def ticker_to_file(ticker):
    """Filesystem-safe name for a ticker: '^GDAXI' -> 'GDAXI', 'BRK.B' -> 'BRK_B'."""
    return ticker.replace("^", "").replace("/", "_").replace(".", "_")


def raw_path(ticker, raw_dir):
    base = raw_dir if os.path.isabs(raw_dir) else repo_path(raw_dir)
    return os.path.join(base, ticker_to_file(ticker) + ".csv")


# ---------------------------------------------------------------------------
# download (network; only used by scripts/download_universe.py)
# ---------------------------------------------------------------------------
def download_ticker(ticker, start, end_inclusive):
    """Download adjusted daily OHLCV for one ticker from Yahoo Finance.

    `end_inclusive` is the last date wanted. yfinance treats `end` as exclusive,
    so one day is added here.
    """
    import yfinance as yf   # imported lazily: only the download needs it

    end_excl = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(ticker, start=start, end=end_excl, interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):           # single ticker can still be MultiIndex
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().rename(columns={"Datetime": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return clean_prices(df[["Date"] + PRICE_COLUMNS])


def clean_prices(df):
    """Basic sanity cleaning applied to every price file.

    * sort by date, drop duplicate dates (keep the last value)
    * drop rows without a positive Close (log returns would break)
    * missing Volume -> 0 (indices like ^VIX have no volume)
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")
    df = df[np.isfinite(df["Close"]) & (df["Close"] > 0)]
    df["Volume"] = df["Volume"].fillna(0.0)
    return df.reset_index(drop=True)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(raw_dir, entries):
    """Merge `entries` ({ticker: info}) into data/raw/MANIFEST.json."""
    path = os.path.join(raw_dir if os.path.isabs(raw_dir) else repo_path(raw_dir), "MANIFEST.json")
    manifest = {}
    if os.path.exists(path):
        with open(path) as fh:
            manifest = json.load(fh)
    manifest.update(entries)
    with open(path, "w") as fh:
        json.dump(dict(sorted(manifest.items())), fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# loading (no network)
# ---------------------------------------------------------------------------
def load_raw(ticker, raw_dir):
    """Load one ticker's full stored history as a DataFrame indexed by Date.

    CSV is the default storage format. A Parquet file with the same stem is
    used instead if it exists (and pyarrow is installed). This keeps the loader
    pluggable without adding a dependency.
    """
    path = raw_path(ticker, raw_dir)
    pq = os.path.splitext(path)[0] + ".parquet"
    if os.path.exists(pq):
        df = pd.read_parquet(pq)
    elif os.path.exists(path):
        df = pd.read_csv(path)
    else:
        raise FileNotFoundError(
            f"No data for {ticker} at {path}. Run: python scripts/download_universe.py")
    df = clean_prices(df)
    return df.set_index("Date")[PRICE_COLUMNS]


class _TestUnlockToken:
    """Proof that the final-test lock was written. Only created by unlock_test_period()."""

    def __init__(self, lock_path):
        self.lock_path = lock_path


def unlock_test_period(info, lock_path=LOCK_FILE):
    """Write the final-test lock file and return the unlock token.

    Raises RuntimeError if the lock already exists: the frozen test period may
    be evaluated only once (PROTOCOL §9). `info` (dict) is stored in the lock
    file, e.g. config hash, git commit, timestamp.
    """
    if os.path.exists(lock_path):
        with open(lock_path) as fh:
            prior = fh.read()
        raise RuntimeError(
            "The frozen test period has already been evaluated. Lock file "
            f"{lock_path} exists:\n{prior}\nPROTOCOL §9 allows one evaluation only.")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    payload = dict(info, unlocked_at=_dt.datetime.now().isoformat(timespec="seconds"))
    # 'x' mode = fail if the file appeared in the meantime (no silent overwrite)
    with open(lock_path, "x") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return _TestUnlockToken(lock_path)


def load_prices(tickers, cfg, unlock=None):
    """Load several tickers, cut at the test-period boundary unless unlocked.

    Returns {ticker: DataFrame[Open, High, Low, Close, Volume] indexed by Date}.

    Without a valid `unlock` token every bar with Date >= splits.test_start is
    removed, so development code physically cannot see the frozen test period.
    """
    test_start = pd.Timestamp(cfg["splits"]["test_start"])
    allowed = isinstance(unlock, _TestUnlockToken) and os.path.exists(unlock.lock_path)
    if unlock is not None and not allowed:
        raise RuntimeError("Invalid test unlock token: use harness.data.unlock_test_period().")
    out = {}
    for t in tickers:
        df = load_raw(t, cfg["data"]["raw_dir"])
        if not allowed:
            df = df[df.index < test_start]
        else:
            df = df[df.index <= pd.Timestamp(cfg["splits"]["test_end"])]
        out[t] = df
    return out


# ---------------------------------------------------------------------------
# universe helpers
# ---------------------------------------------------------------------------
def all_tickers(cfg):
    """Every tradable ticker in the universe, in config order."""
    return [t for bucket in cfg["universe"]["buckets"].values() for t in bucket]


def ticker_sets(cfg):
    """The evaluation sets named in experiment configs.

    train     : universe minus the leave-assets-out set (used for training AND
                for development-time evaluation)
    leave_out : the 7 held-out tickers (M4 / M5 only)
    single    : the single-asset reference (^GDAXI)
    all       : everything
    plus any `extra_ticker_sets` of the experiment config
    """
    universe = all_tickers(cfg)
    leave = list(cfg["universe"]["leave_out"])
    sets = {
        "train": [t for t in universe if t not in leave],
        "leave_out": leave,
        "single": [cfg["universe"]["single_asset"]],
        "all": universe,
    }
    # Experiment configs may define extra evaluation sets, e.g. the 2-ETF
    # deployment portfolio of the neo-broker scenario (M4). They live in the
    # experiment YAML, not in protocol.yaml, so no existing hash changes.
    for name, tickers in (cfg.get("extra_ticker_sets") or {}).items():
        unknown = [t for t in tickers if t not in universe]
        if unknown:
            raise KeyError(f"extra ticker set '{name}' uses tickers outside the universe: {unknown}")
        if name in sets:
            raise KeyError(f"extra ticker set '{name}' would overwrite a PROTOCOL set")
        sets[name] = list(tickers)
    return sets


def bucket_of(cfg):
    """{ticker: bucket name}."""
    return {t: b for b, ts in cfg["universe"]["buckets"].items() for t in ts}

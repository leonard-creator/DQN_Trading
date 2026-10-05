"""Pull stock data from Yahoo Finance and prepare it for training.

Downloads OHLCV, computes the technical indicators the trainer expects
(ROC12, MFI14, FVolatility) so the output is directly usable, cleans it, and
writes a chronological train / validation / test split:

    train_data/<name>_train.csv
    train_data/<name>_val.csv
    test_data/<name>_test.csv

Example:
    python scrape_data.py AAPL --start 2015-01-01 --name AAPL
    python train.py AAPL_train.csv aapl --val-stock AAPL_val.csv --episodes 50
    python evaluate.py AAPL_test.csv aapl_best --test

Requires internet access and `pip install yfinance`.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# canonical output column order (matches the SundPGI layout the loader expects)
OUT_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "ROC12", "MFI14", "FVolatility", "SMA20_Rel", "SMA50_Rel"]


def compute_indicators(df, roc_n=12, mfi_n=14, vol_n=14):
    """Add ROC12, MFI14, FVolatility columns to an OHLCV DataFrame.

    * ROC12       = 12-period rate of change of Close, in %.
    * MFI14       = 14-period Money Flow Index (0-100 oscillator).
    * FVolatility = 14-period rolling std of daily log returns (~realized vol).

    These are computed from OHLCV so ANY Yahoo ticker works (unlike relying on
    pre-supplied indicator columns, which e.g. the iShares CSV lacks).
    """
    out = df.copy()
    close, high, low, vol = out["Close"], out["High"], out["Low"], out["Volume"]

    # Rate of change (%)
    out["ROC12"] = 100.0 * (close / close.shift(roc_n) - 1.0)

    # Money Flow Index
    typical = (high + low + close) / 3.0
    money_flow = typical * vol
    up = typical > typical.shift(1)
    pos_mf = money_flow.where(up, 0.0).rolling(mfi_n).sum()
    neg_mf = money_flow.where(~up, 0.0).rolling(mfi_n).sum()
    # where there is no negative flow the ratio is infinite -> MFI = 100
    mfr = pos_mf / neg_mf.replace(0.0, np.nan)
    out["MFI14"] = (100.0 - 100.0 / (1.0 + mfr)).fillna(100.0)

    # Realized volatility (rolling std of log returns)
    log_ret = np.log(close / close.shift(1))
    out["FVolatility"] = log_ret.rolling(vol_n).std()
    
    # We calculate Close / SMA. 
    # > 1 means price is above average (bullish), < 1 means below (bearish).
    out["SMA20_Rel"] = close / close.rolling(window=20).mean()
    out["SMA50_Rel"] = close / close.rolling(window=50).mean()

    return out


def clean(df):
    """Drop indicator warm-up rows and any remaining non-finite rows."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=OUT_COLUMNS[1:]).reset_index(drop=True)
    # drop rows with non-positive price/volume that would break log transforms
    df = df[(df["Close"] > 0) & (df["Volume"] >= 0)].reset_index(drop=True)
    return df


def chronological_split(df, train_frac, val_frac):
    """Split in time order (never shuffle a time series)."""
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test


def download(ticker, start, end, interval):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is required: `pip install yfinance` (and internet access).")
    # auto_adjust=True gives split/dividend-adjusted OHLC (clean returns)
    raw = yf.download(ticker, start=start, end=end, interval=interval,
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        sys.exit(f"No data returned for {ticker} ({start}..{end}). Check the ticker/dates.")
    # yfinance may return MultiIndex columns for a single ticker -> flatten
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()  # Date becomes a column
    raw = raw.rename(columns={"Datetime": "Date"})
    return raw[["Date", "Open", "High", "Low", "Close", "Volume"]]


def main():
    p = argparse.ArgumentParser(description="Scrape + prepare Yahoo Finance data for the DQN trader")
    p.add_argument("ticker", help="Yahoo Finance symbol, e.g. AAPL, ^GSPC, SAP.DE")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None, help="default: today")
    p.add_argument("--interval", default="1d")
    p.add_argument("--name", default=None, help="output basename (default: ticker, sanitized)")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)   # test = 1 - train - val
    args = p.parse_args()

    name = (args.name or args.ticker).replace("^", "").replace(".", "_").replace("/", "_")

    print(f"Downloading {args.ticker} ({args.start}..{args.end or 'today'}, {args.interval}) ...")
    raw = download(args.ticker, args.start, args.end, args.interval)
    print(f"  got {len(raw)} rows")

    df = clean(compute_indicators(raw))[OUT_COLUMNS]
    print(f"  {len(df)} rows after indicators + cleaning")
    if len(df) < 200:
        print("  WARNING: very few rows — consider an earlier --start.")

    train, val, test = chronological_split(df, args.train_frac, args.val_frac)
    os.makedirs("train_data", exist_ok=True)
    os.makedirs("test_data", exist_ok=True)
    outputs = [(f"train_data/{name}_train.csv", train),
               (f"train_data/{name}_val.csv", val),
               (f"test_data/{name}_test.csv", test)]
    for path, part in outputs:
        part.to_csv(path, index=False)
        d0, d1 = part["Date"].iloc[0], part["Date"].iloc[-1]
        print(f"  wrote {path:34s} {len(part):5d} rows  [{d0} .. {d1}]")

    print("\nReady to train:")
    print(f"  python train.py {name}_train.csv {name} --val-stock {name}_val.csv --episodes 50")
    print(f"  python evaluate.py {name}_test.csv {name}_best --test")


if __name__ == "__main__":
    main()

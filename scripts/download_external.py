"""OWNER-RUN downloads for the v2 programme (PROTOCOL Part II). Nothing calls this automatically.

Project rule: network access is triggered by the project owner only. This
script is never imported or scheduled by the code; run it yourself:

    python scripts/download_external.py --dry-run     # plan + size estimates, NO network
    python scripts/download_external.py q2            # Step 0b, check Q2      (~25 MB)
    python scripts/download_external.py french        # Step 4, long history   (~1-3 MB; +~6 MB with --industries 5 49)
    python scripts/download_external.py fred          # optional BAA10Y        (~0.3 MB)
    python scripts/download_external.py all

What it downloads (and where)
-----------------------------
q2 (PROTOCOL Part II §V2.3, Q2 "adjusted prices change with every new dividend"):
    data/audit/q2_<YYYYMMDD>/yahoo_adjusted/   fresh auto-adjusted daily bars for all 34 series
                                               (same call as data/raw, for a full-overlap comparison)
    data/audit/q2_<YYYYMMDD>/yahoo_raw/        UNADJUSTED closes + dividends + splits, 5 spot-check tickers
    data/audit/q2_<YYYYMMDD>/stooq/            second free source (stooq.com CSV), same 5 tickers
    The comparison itself (returns of data/raw vs fresh; 5 tickers x 50 dates vs
    the second source; flag |diff| > 5 bp) runs OFFLINE later in Step 0b.
french (Step 4, conditional):
    data/longhistory/french/   Kenneth French daily industry portfolios + daily Fama/French factors
fred (optional credit proxy, needs an owner decision before use):
    data/external/fred/BAA10Y.csv   Moody's Baa minus 10-year Treasury, daily from 1986

Safety
------
* Existing files are never overwritten (use --force to re-download).
* Every target folder gets MANIFEST.json: source URL, download time, bytes, sha256.
* The URLs below follow each provider's public file-naming scheme as known on
  2026-10-06. If one returns an error, the script reports it and continues;
  the file can then be fetched by hand from the provider's web page.
"""

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.config import load_config, repo_path                  # noqa: E402
from harness.data import all_tickers, ticker_to_file                # noqa: E402

# Pre-declared spot-check tickers for Q2: one per asset type, incl. the index.
SPOT_CHECK = ["SPY", "EFA", "TLT", "GLD", "^GDAXI"]
STOOQ_SYMBOL = {"SPY": "spy.us", "EFA": "efa.us", "TLT": "tlt.us", "GLD": "gld.us", "^GDAXI": "^dax"}
STOOQ_URL = "https://stooq.com/q/d/l/?s={sym}&i=d"

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/ftp/"
FRENCH_FILES = {
    "factors": "F-F_Research_Data_Factors_daily_CSV.zip",          # Mkt-RF, SMB, HML, RF; daily from 1926-07
    5: "5_Industry_Portfolios_daily_CSV.zip",                      # the page cited in PROTOCOL Part II
    10: "10_Industry_Portfolios_daily_CSV.zip",
    12: "12_Industry_Portfolios_daily_CSV.zip",
    49: "49_Industry_Portfolios_daily_CSV.zip",
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA10Y"

# Rough size estimates (compressed download): ~100 years x 252 days of rows.
ESTIMATES_MB = {
    "q2 yahoo_adjusted (34 series, 2000-2026)": 20.0,
    "q2 yahoo_raw (5 tickers + dividends/splits)": 3.0,
    "q2 stooq (5 tickers)": 1.5,
    "french factors daily": 0.4,
    "french 5 industries daily": 0.9,
    "french 10 / 12 industries daily (each)": 1.6,
    "french 49 industries daily": 6.0,
    "fred BAA10Y": 0.3,
}


# ---------------------------------------------------------------------------
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def write_manifest(folder, entries):
    path = os.path.join(folder, "MANIFEST.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(entries)
    with open(path, "w") as fh:
        json.dump(dict(sorted(old.items())), fh, indent=2)


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "research-download/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def save_bytes(folder, name, data, url, force):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    if os.path.exists(path) and not force:
        print(f"  skip {name} (exists; --force to replace)")
        return None
    with open(path, "wb") as fh:
        fh.write(data)
    entry = {"source": url, "bytes": len(data), "sha256": sha256_bytes(data),
             "downloaded_at": dt.datetime.now().isoformat(timespec="seconds")}
    write_manifest(folder, {name: entry})
    print(f"  ok   {name:45s} {len(data) / 1e6:6.2f} MB")
    return path


# ---------------------------------------------------------------------------
def download_q2(force):
    import pandas as pd
    import yfinance as yf
    from harness.data import download_ticker
    cfg = load_config()
    d = cfg["data"]
    root = repo_path("data", "audit", f"q2_{dt.date.today():%Y%m%d}")
    tickers = all_tickers(cfg) + list(cfg["universe"]["context"])

    print(f"[q2] fresh auto-adjusted Yahoo bars for {len(tickers)} series -> {root}/yahoo_adjusted")
    for t in tickers:
        try:
            df = download_ticker(t, d["download_start"], d["download_end"])
            buf = df.to_csv(index=False, date_format="%Y-%m-%d").encode()
            save_bytes(os.path.join(root, "yahoo_adjusted"), ticker_to_file(t) + ".csv", buf,
                       f"yfinance auto_adjust=True {t}", force)
        except Exception as exc:
            print(f"  FAIL {t}: {type(exc).__name__}: {exc}")
        time.sleep(0.5)

    print(f"[q2] unadjusted bars + dividends + splits for {SPOT_CHECK}")
    for t in SPOT_CHECK:
        try:
            raw = yf.download(t, start=d["download_start"], interval="1d", auto_adjust=False,
                              actions=True, progress=False, threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            buf = raw.reset_index().to_csv(index=False, date_format="%Y-%m-%d").encode()
            save_bytes(os.path.join(root, "yahoo_raw"), ticker_to_file(t) + ".csv", buf,
                       f"yfinance auto_adjust=False actions=True {t}", force)
        except Exception as exc:
            print(f"  FAIL {t}: {type(exc).__name__}: {exc}")
        time.sleep(0.5)

    print(f"[q2] second free source (stooq) for {SPOT_CHECK}")
    for t in SPOT_CHECK:
        url = STOOQ_URL.format(sym=STOOQ_SYMBOL[t])
        try:
            data = fetch(url)
            if not data.startswith(b"Date"):
                raise ValueError(f"unexpected response (first bytes: {data[:60]!r})")
            save_bytes(os.path.join(root, "stooq"), ticker_to_file(t) + ".csv", data, url, force)
        except Exception as exc:
            print(f"  FAIL {t} ({url}): {type(exc).__name__}: {exc}")
        time.sleep(1.0)


def download_french(industries, force):
    folder = repo_path("data", "longhistory", "french")
    for key in ["factors"] + list(industries):
        if key not in FRENCH_FILES:
            print(f"  skip unknown industry set {key}")
            continue
        name = FRENCH_FILES[key]
        url = FRENCH_BASE + name
        try:
            data = fetch(url)
            path = save_bytes(folder, name, data, url, force)
            if path:                                   # also extract the CSV next to the zip
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for member in z.namelist():
                        z.extract(member, folder)
                        print(f"       extracted {member}")
        except Exception as exc:
            print(f"  FAIL {name} ({url}): {type(exc).__name__}: {exc}")


def download_fred(force):
    folder = repo_path("data", "external", "fred")
    try:
        data = fetch(FRED_URL)
        save_bytes(folder, "BAA10Y.csv", data, FRED_URL, force)
    except Exception as exc:
        print(f"  FAIL BAA10Y ({FRED_URL}): {type(exc).__name__}: {exc}")


def dry_run(industries):
    cfg = load_config()
    n = len(all_tickers(cfg)) + len(cfg["universe"]["context"])
    print("DRY RUN (no network). Planned downloads and rough sizes:\n")
    print(f"q2      {n} Yahoo series (adjusted), 5 Yahoo series (raw + actions), 5 stooq series")
    print(f"french  factors + industries {list(industries)}  from {FRENCH_BASE}")
    print(f"fred    {FRED_URL}\n")
    for k, v in ESTIMATES_MB.items():
        print(f"  {k:48s} ~{v:5.1f} MB")
    total = 24.5 + 0.4 + sum({5: 0.9, 10: 1.6, 12: 1.6, 49: 6.0}.get(i, 0) for i in industries) + 0.3
    print(f"\n  total for 'all' with industries {list(industries)}: ~{total:.0f} MB "
          f"(~{n + 10 + len(industries) + 2} HTTP requests)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", nargs="?", choices=["q2", "french", "fred", "all"])
    p.add_argument("--dry-run", action="store_true", help="show the plan and sizes; no network")
    p.add_argument("--industries", type=int, nargs="+", default=[5],
                   help="French industry sets to fetch (default: 5, the set cited in the protocol)")
    p.add_argument("--force", action="store_true", help="re-download existing files")
    args = p.parse_args()
    if args.dry_run or args.what is None:
        dry_run(args.industries)
        return
    if args.what in ("q2", "all"):
        download_q2(args.force)
    if args.what in ("french", "all"):
        download_french(args.industries, args.force)
    if args.what in ("fred", "all"):
        download_fred(args.force)


if __name__ == "__main__":
    main()

"""Download the PROTOCOL universe (+ context series) into data/raw/.

    python scripts/download_universe.py            # only missing tickers
    python scripts/download_universe.py --force    # re-download everything

Each ticker is written ONCE as a CSV with its full daily history and recorded
in data/raw/MANIFEST.json (rows, first/last date, sha256, download time,
yfinance version). Existing files are never overwritten without --force,
because Yahoo re-bases adjusted prices over time and a silent re-download
would change every baseline number (see harness/data.py).
"""

import argparse
import datetime as dt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.config import load_config, repo_path            # noqa: E402
from harness.data import (all_tickers, download_ticker, file_sha256,  # noqa: E402
                          raw_path, write_manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="re-download tickers that already exist")
    p.add_argument("--tickers", nargs="*", help="subset to download (default: universe + context)")
    args = p.parse_args()

    cfg = load_config()
    d = cfg["data"]
    tickers = args.tickers or (all_tickers(cfg) + list(cfg["universe"]["context"]))
    raw_dir = repo_path(d["raw_dir"])
    os.makedirs(raw_dir, exist_ok=True)

    import yfinance as yf
    entries, failed = {}, []
    for t in tickers:
        path = raw_path(t, d["raw_dir"])
        if os.path.exists(path) and not args.force:
            print(f"  skip {t:7s} (exists)")
            continue
        try:
            df = download_ticker(t, d["download_start"], d["download_end"])
        except Exception as exc:                       # keep going, report at the end
            print(f"  FAIL {t:7s} {type(exc).__name__}: {exc}")
            failed.append(t)
            continue
        df.to_csv(path, index=False, date_format="%Y-%m-%d")
        entries[t] = {
            "file": os.path.relpath(path, repo_path()),
            "rows": int(len(df)),
            "first": df["Date"].iloc[0].strftime("%Y-%m-%d"),
            "last": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
            "sha256": file_sha256(path),
            "source": "yfinance auto_adjust=True interval=1d",
            "yfinance_version": yf.__version__,
            "downloaded_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        print(f"  ok   {t:7s} {len(df):5d} rows  {entries[t]['first']} .. {entries[t]['last']}")
        time.sleep(0.5)                                # be polite to the free endpoint

    if entries:
        print("manifest ->", write_manifest(d["raw_dir"], entries))
    if failed:
        sys.exit(f"{len(failed)} ticker(s) failed: {failed}")


if __name__ == "__main__":
    main()

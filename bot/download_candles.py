"""Prefetch candles into the local CSV cache (run when the T-Bank API is up).

get_candles only fetches ranges missing from the cache, so extending the
range backwards downloads just the gap before the earliest cached bar.

Usage (from bot/, with venv activated):
    python3 download_candles.py --from 2024-01-01 --to 2025-01-01
    python3 download_candles.py --from 2024-01-01 --to 2025-01-01 --timeframes 5min,15min,30min,1h
"""
import argparse
from datetime import datetime, timezone

from candles import load_all_candles
from config import TIMEFRAMES


def main():
    p = argparse.ArgumentParser(description="Prefetch candles into the local cache")
    p.add_argument("--from", dest="date_from", required=True, metavar="DATE", help="start date YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", required=True, metavar="DATE", help="end date YYYY-MM-DD")
    p.add_argument("--timeframes", default=",".join(TIMEFRAMES),
                   help="comma-separated timeframes (default: config TIMEFRAMES)")
    args = p.parse_args()

    from_dt = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = datetime.strptime(args.date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    load_all_candles(from_dt, to_dt, timeframes=args.timeframes.split(","))
    print("Done.")


if __name__ == "__main__":
    main()

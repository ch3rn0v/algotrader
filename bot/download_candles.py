"""Prefetch candles into the local CSV cache (run when the T-Bank API is up).

get_candles only fetches ranges missing from the cache, so extending the
range backwards downloads just the gap before the earliest cached bar.

Instruments can be given three ways:
  --tickers GAZP,VTBR,T   resolve each ticker to a FIGI via the API (TQBR board)
  --figis BBG...,BBG...   explicit FIGIs (no lookup)
  (neither)               fall back to the config ASSETS

Usage (from bot/, with venv activated):
    python3 download_candles.py --from 2024-01-01 --to 2025-01-01
    python3 download_candles.py --tickers GAZP,VTBR,T,LKOH,ROSN \
        --from 2024-01-01 --to 2026-01-01 --timeframes 5min,15min
    python3 download_candles.py --figis BBG004730RP0 --from 2024-01-01 --to 2025-01-01
"""
import argparse
from datetime import datetime, timezone

from candles import get_candles, load_all_candles
from config import TIMEFRAMES


def _resolve_tickers(tickers: list[str]) -> dict[str, str]:
    """Map each ticker to its TQBR FIGI via the API. Errors on 0 or >1 match."""
    from find_figi import find_figi  # imports tinkoff client; keep it lazy
    figis = {}
    for t in tickers:
        matches = find_figi(t)
        if not matches:
            raise SystemExit(f"No TQBR instrument found for ticker '{t}'. "
                             f"Run: python3 find_figi.py {t} --all")
        if len(matches) > 1:
            names = ", ".join(f"{m['ticker']}={m['figi']}" for m in matches)
            raise SystemExit(f"Ticker '{t}' is ambiguous on TQBR: {names}. Use --figis.")
        figis[t] = matches[0]["figi"]
        print(f"  {t} -> {matches[0]['figi']}  ({matches[0]['name']})")
    return figis


def main():
    p = argparse.ArgumentParser(description="Prefetch candles into the local cache")
    p.add_argument("--from", dest="date_from", required=True, metavar="DATE", help="start date YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", required=True, metavar="DATE", help="end date YYYY-MM-DD")
    p.add_argument("--timeframes", default=",".join(TIMEFRAMES),
                   help="comma-separated timeframes (default: config TIMEFRAMES)")
    p.add_argument("--tickers", default=None, help="comma-separated tickers to resolve to FIGIs and download")
    p.add_argument("--figis", default=None, help="comma-separated explicit FIGIs to download")
    args = p.parse_args()

    from_dt = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = datetime.strptime(args.date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    tfs = args.timeframes.split(",")

    if args.tickers or args.figis:
        figis = {}
        if args.tickers:
            print("Resolving tickers...")
            figis.update(_resolve_tickers(args.tickers.split(",")))
        for f in (args.figis.split(",") if args.figis else []):
            figis[f] = f
        for label, figi in figis.items():
            for tf in tfs:
                df = get_candles(figi, tf, from_dt, to_dt)
                print(f"  {label} {tf}: {len(df)} bars")
    else:
        load_all_candles(from_dt, to_dt, timeframes=tfs)
    print("Done.")


if __name__ == "__main__":
    main()

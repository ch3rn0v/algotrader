"""Look up instrument FIGIs by ticker via the T-Bank Invest API.

Usage (from repo root, with venv activated):
    python3 bot/find_figi.py SBERP
    python3 bot/find_figi.py PLZL
    python3 bot/find_figi.py IMOEX
"""

import argparse
import os
from pathlib import Path

from tinkoff.invest import Client

# Load .env so the script works standalone (same as candles.py does at import time).
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


def find_figi(ticker: str, class_code: str = "TQBR") -> list[dict]:
    """Return instruments matching the given ticker, filtered to class_code (default TQBR).

    TQBR is the MOEX main board T+2 regime for standard equities — the right
    choice for regular main-session trading of shares and preferred shares.
    Pass class_code=None to see all results unfiltered.
    """
    with Client(os.environ["TBANK_TOKEN"]) as client:
        response = client.instruments.find_instrument(query=ticker)
    results = [
        {
            "ticker": inst.ticker,
            "figi": inst.figi,
            "name": inst.name,
            "class_code": inst.class_code,
            "instrument_type": inst.instrument_type,
        }
        for inst in response.instruments
    ]
    if class_code is not None:
        results = [r for r in results if r["class_code"] == class_code]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Look up FIGI by ticker via T-Bank API")
    parser.add_argument("ticker", help="Ticker to search for (e.g. SBERP, PLZL, IMOEX)")
    parser.add_argument("--all", action="store_true", help="Show all class codes, not just TQBR")
    args = parser.parse_args()

    results = find_figi(args.ticker, class_code=None if args.all else "TQBR")
    if not results:
        print(f"No instruments found for '{args.ticker}' (class_code=TQBR). Try --all to see all results.")
    else:
        print(f"{'TICKER':<12} {'FIGI':<16} {'CLASS':<8} {'TYPE':<12} NAME")
        print("-" * 72)
        for r in results:
            print(f"{r['ticker']:<12} {r['figi']:<16} {r['class_code']:<8} {r['instrument_type']:<12} {r['name']}")

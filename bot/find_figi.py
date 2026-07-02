"""Look up instrument FIGIs by ticker, or verify a FIGI, via the T-Bank Invest API.

Usage (from repo root, with venv activated):
    python3 bot/find_figi.py SBERP
    python3 bot/find_figi.py IMOEX --all
    python3 bot/find_figi.py --verify BBG0047315Y7
"""

import argparse
import os

from tinkoff.invest import Client, InstrumentIdType
from tinkoff.invest.exceptions import RequestError

import config  # noqa: F401 — loads bot/.env into os.environ


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


def verify_figi(figi: str) -> dict | None:
    """Return instrument details for a given FIGI, or None if not found."""
    try:
        with Client(os.environ["TBANK_TOKEN"]) as client:
            response = client.instruments.get_instrument_by(
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
                id=figi,
            )
    except RequestError as e:
        if "not found" in str(e).lower():
            return None
        raise
    inst = response.instrument
    return {
        "ticker": inst.ticker,
        "figi": inst.figi,
        "name": inst.name,
        "class_code": inst.class_code,
        "instrument_type": inst.instrument_type,
        "currency": inst.currency,
        "exchange": inst.exchange,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Look up or verify instrument FIGIs via T-Bank API")
    parser.add_argument("ticker", nargs="?", help="Ticker to search for (e.g. SBERP, PLZL)")
    parser.add_argument("--verify", metavar="FIGI", help="Verify a FIGI and show its instrument details")
    parser.add_argument("--all", action="store_true", help="Show all class codes, not just TQBR")
    args = parser.parse_args()

    if args.verify:
        info = verify_figi(args.verify)
        if info is None:
            print(f"FIGI '{args.verify}' not found in T-Bank API.")
        else:
            print(f"FIGI:     {info['figi']}")
            print(f"Ticker:   {info['ticker']}")
            print(f"Name:     {info['name']}")
            print(f"Class:    {info['class_code']}")
            print(f"Type:     {info['instrument_type']}")
            print(f"Currency: {info['currency']}")
            print(f"Exchange: {info['exchange']}")
    elif args.ticker:
        results = find_figi(args.ticker, class_code=None if args.all else "TQBR")
        if not results:
            print(f"No instruments found for '{args.ticker}' (class_code=TQBR). Try --all to see all results.")
        else:
            print(f"{'TICKER':<12} {'FIGI':<16} {'CLASS':<8} {'TYPE':<12} NAME")
            print("-" * 72)
            for r in results:
                print(f"{r['ticker']:<12} {r['figi']:<16} {r['class_code']:<8} {r['instrument_type']:<12} {r['name']}")
    else:
        parser.print_help()

"""Fetch OHLCV candles from T-Bank API with local CSV cache.

Usage:
    from datetime import datetime, timezone
    from candles import get_candles

    df = get_candles("BBG004730N88", "15min",
                     datetime(2024, 1, 1, tzinfo=timezone.utc),
                     datetime(2024, 6, 1, tzinfo=timezone.utc))

Requires TBANK_TOKEN env var (loaded automatically from bot/.env by config.py).
"""
import os
import time

import pandas as pd
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.exceptions import RequestError

from config import ASSETS, CACHE_DIR, PRIMARY_ASSET, PRIMARY_TF, TIMEFRAMES

_INTERVALS = {
    "1min":  CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5min":  CandleInterval.CANDLE_INTERVAL_5_MIN,
    "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "30min": CandleInterval.CANDLE_INTERVAL_30_MIN,
    "1h":    CandleInterval.CANDLE_INTERVAL_HOUR,
    "1day":  CandleInterval.CANDLE_INTERVAL_DAY,
}

# Retry delays (seconds) for transient errors, keyed by gRPC status string.
_RETRY_DELAYS = {
    "RESOURCE_EXHAUSTED": [10, 30, 60, 120],
    "UNAVAILABLE":        [2, 5, 15, 30],
}


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _q(q) -> float:
    return q.units + q.nano / 1e9


def _fetch(figi: str, timeframe: str, from_ts: pd.Timestamp, to_ts: pd.Timestamp) -> pd.DataFrame:
    retry_counts = {status: 0 for status in _RETRY_DELAYS}
    while True:
        try:
            with Client(os.environ["TBANK_TOKEN"]) as client:
                raw = list(client.get_all_candles(
                    instrument_id=figi,
                    from_=from_ts.to_pydatetime(),
                    to=to_ts.to_pydatetime(),
                    interval=_INTERVALS[timeframe],
                ))
            return pd.DataFrame([{
                "timestamp": _utc(c.time),
                "open":  _q(c.open),
                "high":  _q(c.high),
                "low":   _q(c.low),
                "close": _q(c.close),
                "volume": c.volume,
            } for c in raw])
        except RequestError as e:
            err = str(e)
            for status, delays in _RETRY_DELAYS.items():
                if status in err:
                    n = retry_counts[status]
                    if n < len(delays):
                        retry_counts[status] += 1
                        print(f"  {status} — retrying in {delays[n]}s (attempt {n + 1}/{len(delays)})...")
                        time.sleep(delays[n])
                        break
                    raise RuntimeError(
                        f"{status} fetching {figi} {timeframe}: gave up after {len(delays)} retries"
                    ) from e
            else:
                raise RuntimeError(
                    f"Unexpected API error fetching {figi} {timeframe}: {err}"
                ) from e


def get_candles(figi: str, timeframe: str, from_dt, to_dt) -> pd.DataFrame:
    """Return OHLCV candles for [from_dt, to_dt], fetching only uncached ranges."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{figi}_{timeframe}.csv"
    from_ts, to_ts = _utc(from_dt), _utc(to_dt)

    if path.exists():
        cached = pd.read_csv(path, parse_dates=["timestamp"])
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
    else:
        cached = pd.DataFrame()

    to_fetch = []
    if cached.empty:
        to_fetch.append((from_ts, to_ts))
    else:
        if from_ts < cached["timestamp"].min():
            to_fetch.append((from_ts, cached["timestamp"].min()))
        if to_ts > cached["timestamp"].max():
            to_fetch.append((cached["timestamp"].max(), to_ts))

    if to_fetch:
        fetched = pd.concat([_fetch(figi, timeframe, s, e) for s, e in to_fetch], ignore_index=True)
        cached = (pd.concat([cached, fetched], ignore_index=True)
                  .drop_duplicates("timestamp")
                  .sort_values("timestamp")
                  .reset_index(drop=True))
        cached.to_csv(path, index=False)

    mask = (cached["timestamp"] >= from_ts) & (cached["timestamp"] <= to_ts)
    return cached[mask].reset_index(drop=True)


def load_all_candles(from_dt, to_dt, primary: pd.DataFrame | None = None) -> dict:
    """Load candles for every configured (asset, timeframe) pair.

    Returns {(asset, timeframe): DataFrame}. If `primary` is given, it is used
    as-is for (PRIMARY_ASSET, PRIMARY_TF) instead of being loaded again.
    """
    all_candles = {}
    for asset, figi in ASSETS.items():
        for tf in TIMEFRAMES:
            if primary is not None and asset == PRIMARY_ASSET and tf == PRIMARY_TF:
                all_candles[(asset, tf)] = primary
            else:
                all_candles[(asset, tf)] = get_candles(figi, tf, from_dt, to_dt)
            print(f"  {asset} {tf}: {len(all_candles[(asset, tf)])} bars")
    return all_candles

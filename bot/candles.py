"""Fetch OHLCV candles from T-Bank API with local CSV cache.

Usage:
    from datetime import datetime, timezone
    from candles import get_candles

    df = get_candles("BBG004730N88", "15min",
                     datetime(2024, 1, 1, tzinfo=timezone.utc),
                     datetime(2024, 6, 1, tzinfo=timezone.utc))

Requires TBANK_TOKEN env var (loaded automatically from bot/.env if present).
"""
import os
from pathlib import Path

import pandas as pd
from tinkoff.invest import Client, CandleInterval

CACHE_DIR = Path(__file__).parent / "cache"

# Load bot/.env once at import time so all scripts get the token automatically
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

_INTERVALS = {
    "1min":  CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5min":  CandleInterval.CANDLE_INTERVAL_5_MIN,
    "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "30min": CandleInterval.CANDLE_INTERVAL_30_MIN,
    "1h":    CandleInterval.CANDLE_INTERVAL_HOUR,
    "1day":  CandleInterval.CANDLE_INTERVAL_DAY,
}


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _q(q) -> float:
    return q.units + q.nano / 1e9


def _fetch(figi: str, timeframe: str, from_ts: pd.Timestamp, to_ts: pd.Timestamp) -> pd.DataFrame:
    with Client(os.environ["TBANK_TOKEN"]) as client:
        raw = list(client.get_all_candles(
            instrument_id=figi,
            from_=from_ts.to_pydatetime(),
            to=to_ts.to_pydatetime(),
            interval=_INTERVALS[timeframe],
        ))
    rows = [{
        "timestamp": _utc(c.time),
        "open":  _q(c.open),
        "high":  _q(c.high),
        "low":   _q(c.low),
        "close": _q(c.close),
        "volume": c.volume,
    } for c in raw]
    return pd.DataFrame(rows)


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

"""Candle normalization and resampling (SPEC 4.1).

The canonical timestamp is the *open* of the bar in UTC. The broker layer is
responsible for converting T-Bank's convention to this one at ingestion; the
helpers here assume inputs already carry canonical bar-open UTC timestamps.
"""
from __future__ import annotations

import pandas as pd

from src.core.types import Candle, Instrument

_OHLCV = ("open", "high", "low", "close", "volume")


def candles_to_df(candles: list[Candle], instrument: Instrument) -> pd.DataFrame:
    """Build a sorted observation DataFrame from normalized candle dicts."""
    if not candles:
        cols = ["exchange", "instrument", "timestamp", *_OHLCV]
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["exchange"] = instrument.exchange
    df["instrument"] = instrument.symbol
    df = df[["exchange", "instrument", "timestamp", *_OHLCV]]
    return df.sort_values("timestamp").reset_index(drop=True)


def resample_to_15min(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min candles to 15-min, bar-open labeled.

    Uses ``closed="left", label="left"`` so a 15-min bar covers the half-open
    interval ``[open, open+15min)`` and is timestamped by its open, matching the
    canonical convention. Empty buckets are dropped."""
    d = df_1m.set_index("timestamp").sort_index()
    agg = pd.DataFrame(
        {
            "open": d["open"].resample("15min", closed="left", label="left").first(),
            "high": d["high"].resample("15min", closed="left", label="left").max(),
            "low": d["low"].resample("15min", closed="left", label="left").min(),
            "close": d["close"].resample("15min", closed="left", label="left").last(),
            "volume": d["volume"].resample("15min", closed="left", label="left").sum(),
        }
    )
    agg = agg.dropna(subset=["open"]).reset_index()
    for col in ("exchange", "instrument"):
        if col in df_1m.columns:
            agg[col] = df_1m[col].iloc[0]
    return agg

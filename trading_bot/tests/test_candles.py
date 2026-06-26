"""Candle resampling boundary tests (SPEC 4.1, 14)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.candles import candles_to_df, resample_to_15min
from tests.helpers import INSTRUMENT


def test_resample_closed_left_label_left():
    # 30 one-minute bars from 07:00; expect two 15-min bars labeled 07:00 and 07:15.
    ts = pd.date_range("2024-06-03 07:00", periods=30, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": np.arange(30, dtype=float),
        "high": np.arange(30, dtype=float) + 1,
        "low": np.arange(30, dtype=float) - 1,
        "close": np.arange(30, dtype=float),
        "volume": np.ones(30),
        "exchange": INSTRUMENT.exchange,
        "instrument": INSTRUMENT.symbol,
    })
    out = resample_to_15min(df)
    assert list(out["timestamp"]) == [pd.Timestamp("2024-06-03 07:00", tz="UTC"),
                                      pd.Timestamp("2024-06-03 07:15", tz="UTC")]
    # First bar covers minutes 0..14 (open=0, close=14, high=15, low=-1, vol=15).
    first = out.iloc[0]
    assert first["open"] == 0.0
    assert first["close"] == 14.0
    assert first["high"] == 15.0
    assert first["low"] == -1.0
    assert first["volume"] == 15.0
    # Second bar covers 15..29 (open=15, close=29).
    second = out.iloc[1]
    assert second["open"] == 15.0
    assert second["close"] == 29.0


def test_candles_to_df_sorts_and_tags_identity():
    candles = [
        {"timestamp": pd.Timestamp("2024-06-03 07:15", tz="UTC"), "open": 2, "high": 3, "low": 1, "close": 2, "volume": 9},
        {"timestamp": pd.Timestamp("2024-06-03 07:00", tz="UTC"), "open": 1, "high": 2, "low": 0, "close": 1, "volume": 8},
    ]
    df = candles_to_df(candles, INSTRUMENT)
    assert list(df["timestamp"]) == sorted(df["timestamp"])
    assert (df["exchange"] == "MOEX").all()
    assert (df["instrument"] == "TEST").all()


def test_candles_to_df_empty():
    df = candles_to_df([], INSTRUMENT)
    assert df.empty
    assert "close" in df.columns

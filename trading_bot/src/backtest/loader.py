"""Historical candle download + parquet cache (SPEC 10.1).

Candles are fetched via the broker layer (which applies the canonical bar-open
UTC convention on ingestion) and cached under a hive-partitioned tree so slices
can be read without scanning everything:

    data/candles/exchange=MOEX/instrument=SBER/interval=15min/year=2024/month=06/candles.parquet
    data/candles/.../interval=1min/year=2024/month=06/candles.parquet
    data/candles/.../interval=1day/year=2024/candles.parquet

The 1-day partition is coarser (year only) because daily series are tiny.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.core.candles import candles_to_df
from src.core.types import Instrument

_CACHE_ROOT = Path("data/candles")
_OHLCV = ("open", "high", "low", "close", "volume")


def _partition_dir(root: Path, instrument: Instrument, interval: str, ts: pd.Timestamp) -> Path:
    base = root / f"exchange={instrument.exchange}" / f"instrument={instrument.symbol}" / f"interval={interval}"
    if interval == "1day":
        return base / f"year={ts.year:04d}"
    return base / f"year={ts.year:04d}" / f"month={ts.month:02d}"


def _write_cache(df: pd.DataFrame, instrument: Instrument, interval: str, root: Path) -> None:
    if df.empty:
        return
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df = df.assign(timestamp=ts)
    group_key = ts.dt.year if interval == "1day" else ts.dt.strftime("%Y-%m")
    for _, part in df.groupby(group_key):
        first = pd.Timestamp(part["timestamp"].iloc[0])
        out_dir = _partition_dir(root, instrument, interval, first)
        out_dir.mkdir(parents=True, exist_ok=True)
        part.to_parquet(out_dir / "candles.parquet", index=False)


def read_cache(
    instrument: Instrument,
    interval: str,
    start: datetime,
    end: datetime,
    root: Path = _CACHE_ROOT,
) -> pd.DataFrame:
    """Read any cached candles for the window. Returns an empty frame if none."""
    base = root / f"exchange={instrument.exchange}" / f"instrument={instrument.symbol}" / f"interval={interval}"
    if not base.exists():
        return candles_to_df([], instrument)
    files = sorted(base.rglob("candles.parquet"))
    if not files:
        return candles_to_df([], instrument)
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    mask = (ts >= pd.Timestamp(start, tz="UTC")) & (ts < pd.Timestamp(end, tz="UTC"))
    return df.loc[mask].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def load_candles(
    client,
    instrument: Instrument,
    interval: str,
    start: datetime,
    end: datetime,
    root: Path = _CACHE_ROOT,
    refetch: bool = False,
) -> pd.DataFrame:
    """Return candles for ``[start, end)``, using the cache and filling gaps from
    the broker. Newly fetched candles are written back to the cache."""
    from src.broker.market_data import get_candles  # local import: network boundary

    cached = pd.DataFrame() if refetch else read_cache(instrument, interval, start, end, root)
    if not cached.empty and not refetch:
        return cached

    fetched = get_candles(client, instrument, start, end, interval)
    df = candles_to_df(fetched, instrument)
    _write_cache(df, instrument, interval, root)
    return df

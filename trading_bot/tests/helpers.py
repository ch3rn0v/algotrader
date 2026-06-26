"""Shared helpers for building synthetic candle fixtures in tests."""
from __future__ import annotations

import pandas as pd

from src.core.types import Instrument

INSTRUMENT = Instrument(exchange="MOEX", symbol="TEST", figi="FIGITEST0001", currency="RUB", lot_size=10)


def session_open(day: str = "2024-06-03") -> pd.Timestamp:
    """A Monday 07:00 UTC bar open (start of the MOEX main session)."""
    return pd.Timestamp(f"{day} 07:00:00", tz="UTC")


def make_15m_df(closes, *, start=None, instrument=INSTRUMENT, ohlc_spread=0.5, volume=1000.0):
    """Build a 15-min candle DataFrame from a list of closes, contiguous in time
    starting inside the main session. open/high/low are derived simply."""
    start = start or session_open()
    ts = pd.date_range(start=start, periods=len(closes), freq="15min", tz="UTC")
    rows = []
    for t, c in zip(ts, closes):
        rows.append(
            {
                "exchange": instrument.exchange,
                "instrument": instrument.symbol,
                "timestamp": t,
                "open": float(c),
                "high": float(c) + ohlc_spread,
                "low": float(c) - ohlc_spread,
                "close": float(c),
                "volume": float(volume),
            }
        )
    return pd.DataFrame(rows)


def make_1m_for_15m(df15, *, per_bar=15, volume=200.0, instrument=INSTRUMENT):
    """Build 1-min sub-candles tiling each 15-min bar with flat OHLC at the
    parent close (enough to drive the execution model)."""
    rows = []
    for _, bar in df15.iterrows():
        sub_ts = pd.date_range(start=bar["timestamp"], periods=per_bar, freq="1min", tz="UTC")
        for t in sub_ts:
            c = bar["close"]
            rows.append(
                {
                    "exchange": instrument.exchange,
                    "instrument": instrument.symbol,
                    "timestamp": t,
                    "open": float(c),
                    "high": float(c) + 0.2,
                    "low": float(c) - 0.2,
                    "close": float(c),
                    "volume": float(volume),
                }
            )
    return pd.DataFrame(rows)


def make_1day_df(n=120, *, start="2024-01-01", base=100.0, step=0.1, instrument=INSTRUMENT):
    """A simple rising daily series to seed sigma_daily."""
    ts = pd.date_range(start=pd.Timestamp(start, tz="UTC"), periods=n, freq="1D", tz="UTC")
    rows = []
    for i, t in enumerate(ts):
        c = base + i * step
        rows.append(
            {
                "exchange": instrument.exchange,
                "instrument": instrument.symbol,
                "timestamp": t,
                "open": c,
                "high": c + 0.5,
                "low": c - 0.5,
                "close": c,
                "volume": 1.0,
            }
        )
    return pd.DataFrame(rows)


def base_config():
    """A minimal validated-shape config dict (no pydantic needed in tests)."""
    return {
        "instrument": {
            "exchange": "MOEX", "symbol": "TEST", "figi": "FIGITEST0001",
            "currency": "RUB", "lot_size": 10, "timeframe": "15min",
        },
        "strategy": {
            "fast_ewm_span": 3, "slow_ewm_span": 6, "bb_period": 4,
            "bb_std_mult": 2.0, "entry_cooldown_bars": 2,
        },
        "risk": {"max_position_lots": 5},
        "backtest": {
            "commission_bps": 5, "min_commission_per_order": 0.0, "participation_rate": 0.10,
            "execution_window_minutes": 5, "min_spread_bps": 2, "spread_to_range_ratio": 0.10,
            "impact_coeff": 0.1, "sigma_daily_window_days": 20, "initial_cash": 1_000_000,
        },
        "splits": {"warmup_days": 30, "train_months": 18, "validation_months": 6, "test_months": 12},
        "live": {"reconcile_every_n_bars": 1, "startup_warmup_bars": None,
                 "order_timeout_seconds": 30, "signal_log_close_history_bars": None},
        "metrics": {"sharpe_aggregation": "daily", "risk_free_annual": 0.0, "return_method": "simple"},
    }

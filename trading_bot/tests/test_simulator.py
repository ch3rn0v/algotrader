"""Execution-model tests (SPEC 10.3, 14)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.backtest.simulator import compute_sigma_daily, simulate_fill
from tests.helpers import make_1day_df

CFG = {
    "execution_window_minutes": 5, "participation_rate": 0.10, "min_spread_bps": 2,
    "spread_to_range_ratio": 0.10, "impact_coeff": 0.1, "commission_bps": 5,
    "min_commission_per_order": 0.0,
}


def _subs(n=15, close=100.0, vol=200.0, rng=0.4):
    ts = pd.date_range("2024-06-03 07:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [close] * n,
        "high": [close + rng / 2] * n,
        "low": [close - rng / 2] * n,
        "close": [close] * n,
        "volume": [vol] * n,
    })


def test_buy_pays_up_sell_receives_less():
    subs = _subs()
    buy_qty, buy_price, _ = simulate_fill(2.0, subs, 0.02, CFG, lot_size=10)
    sell_qty, sell_price, _ = simulate_fill(-2.0, subs, 0.02, CFG, lot_size=10)
    vwap = 100.0  # flat OHLC typical price
    assert buy_qty == 2.0 and sell_qty == -2.0
    assert buy_price > vwap, "buy should pay above VWAP"
    assert sell_price < vwap, "sell should receive below VWAP"


def test_partial_fill_clips_to_available_and_drops_remainder():
    # K=5 sub-candles * 200 vol = 1000; participation 0.10 -> available 100 lots.
    subs = _subs(vol=200.0)
    filled, price, fees = simulate_fill(250.0, subs, 0.02, CFG, lot_size=10)
    assert math.isclose(filled, 100.0), f"expected clip to 100, got {filled}"
    assert price > 0 and fees > 0


def test_zero_volume_yields_zero_fill():
    subs = _subs(vol=0.0)
    filled, price, fees = simulate_fill(5.0, subs, 0.02, CFG, lot_size=10)
    assert filled == 0.0 and fees == 0.0 and np.isnan(price)


def test_min_commission_floor_applied():
    cfg = {**CFG, "min_commission_per_order": 50.0, "commission_bps": 0.0001}
    subs = _subs()
    _, _, fees = simulate_fill(1.0, subs, 0.02, cfg, lot_size=10)
    assert math.isclose(fees, 50.0), f"floor should dominate, got {fees}"


def test_sigma_daily_causal_no_same_day_leakage():
    from src.backtest.engine import _attach_sigma_daily

    daily = make_1day_df(n=60)  # daily closes on 2024-01-01 .. 2024-02-29
    closes = daily["close"].astype(float)
    rets = closes.pct_change()
    rolling = rets.rolling(20).std().to_numpy()

    # Build 15-min bars, one per daily date at mid-session, and attach sigma.
    bar_dates = pd.to_datetime(daily["timestamp"], utc=True).dt.normalize() + pd.Timedelta(hours=12)
    df15 = pd.DataFrame({"timestamp": bar_dates.to_numpy()})
    attached = _attach_sigma_daily(df15, daily, window_days=20)

    # The sigma on the bar dated D must equal the rolling std ending at D-1
    # (strictly prior), never the same-day value -> no leakage.
    for i in range(2, len(daily)):
        if not np.isnan(rolling[i - 1]):
            assert np.isclose(attached[i], rolling[i - 1]), f"sigma leakage at row {i}"
        # same-day value differs from prior-day once returns vary; ensure we did
        # not pick the same-day rolling value when they differ.
        if not np.isnan(rolling[i]) and not np.isnan(rolling[i - 1]) and not np.isclose(rolling[i], rolling[i - 1]):
            assert not np.isclose(attached[i], rolling[i]), f"same-day sigma used at row {i}"

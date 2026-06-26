"""Backtest engine tests (SPEC 10.2, 14).

Covers the two no-lookahead invariants (mutation form and step-jump form), the
forced partial-fill path, and the canonical self-consistency invariants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from tests.helpers import INSTRUMENT, base_config, make_15m_df, make_1m_for_15m, make_1day_df


def _flat_daily():
    # Enough daily history so sigma_daily is non-NaN over the test window.
    return make_1day_df(n=80, start="2024-05-01")


def _assert_self_consistent(df, lot):
    eq = df["equity"].to_numpy()
    cash = df["cash"].to_numpy()
    pos = df["position"].to_numpy()
    close = df["close"].to_numpy()
    trade = df["trade_qty_from_prev"].to_numpy()
    price = df["fill_price_from_prev"].to_numpy()
    fees = df["fees_from_prev"].to_numpy()
    for t in range(len(df)):
        assert abs(eq[t] - (cash[t] + pos[t] * close[t])) < 1e-6, f"equity mismatch at {t}"
        if t == 0:
            continue
        assert abs(pos[t] - (pos[t - 1] + trade[t])) < 1e-6, f"position evolution at {t}"
        traded = trade[t] * (price[t] if not np.isnan(price[t]) else 0.0) * lot
        assert abs(cash[t] - (cash[t - 1] - traded - fees[t])) < 1e-6, f"cash evolution at {t}"


def _entry_series():
    # Rising trend (regime up) then a sharp dip to the lower band -> long entry.
    closes = [100, 101, 102, 103, 104, 105, 95, 95]
    df15 = make_15m_df(closes)
    return df15


def test_engine_self_consistency():
    df15 = _entry_series()
    c1m = make_1m_for_15m(df15, volume=500.0)
    out = run_backtest(df15, c1m, _flat_daily(), INSTRUMENT, base_config())
    _assert_self_consistent(out, INSTRUMENT.lot_size)
    # Equity column present, no NaNs in accounting columns.
    assert not out["equity"].isna().any()
    assert not out["cash"].isna().any()


def test_engine_takes_long_entry_and_fills_next_bar():
    df15 = _entry_series()
    c1m = make_1m_for_15m(df15, volume=500.0)
    out = run_backtest(df15, c1m, _flat_daily(), INSTRUMENT, base_config())
    # Signal turns +1 at the dip bar (index 6); the fill lands on the NEXT bar (7).
    assert out["signal"].iloc[6] == 1, f"expected long signal at dip, got {out['signal'].iloc[6]}"
    assert out["trade_qty_from_prev"].iloc[6] == 0.0, "no fill on the signal bar itself"
    assert out["trade_qty_from_prev"].iloc[7] > 0.0, "fill should occur on the bar after the signal"
    assert out["triggering_signal"].iloc[7] == 1


def test_engine_partial_fill_clips_to_available():
    df15 = _entry_series()
    # Tiny 1-min volume forces a partial fill: K=5 * vol 1.0 = 5 -> available 0.5 lots.
    c1m = make_1m_for_15m(df15, volume=1.0)
    out = run_backtest(df15, c1m, _flat_daily(), INSTRUMENT, base_config())
    filled = out["trade_qty_from_prev"].iloc[7]
    assert np.isclose(filled, 0.5), f"expected partial clip to 0.5 lots, got {filled}"
    _assert_self_consistent(out, INSTRUMENT.lot_size)


def test_no_lookahead_mutation_form():
    """Mutating candles from t+1 onward must not change row t's canonical output."""
    df15 = _entry_series()
    c1m = make_1m_for_15m(df15, volume=500.0)
    cfg = base_config()
    base = run_backtest(df15, c1m, _flat_daily(), INSTRUMENT, cfg)

    t = 5
    mutated = df15.copy()
    mutated.loc[mutated.index > t, "close"] = mutated.loc[mutated.index > t, "close"] + 50.0
    mutated.loc[mutated.index > t, "high"] = mutated.loc[mutated.index > t, "close"] + 1.0
    mutated.loc[mutated.index > t, "low"] = mutated.loc[mutated.index > t, "close"] - 1.0
    c1m_mut = make_1m_for_15m(mutated, volume=500.0)
    out = run_backtest(mutated, c1m_mut, _flat_daily(), INSTRUMENT, cfg)

    cols = ["open", "high", "low", "close", "signal", "target_position",
            "trade_qty_from_prev", "fill_price_from_prev", "fees_from_prev",
            "position", "cash", "equity"]
    for c in cols:
        a = base[c].iloc[: t + 1].to_numpy()
        b = out[c].iloc[: t + 1].to_numpy()
        assert np.allclose(np.nan_to_num(a), np.nan_to_num(b)), f"row<=t leaked via column {c}"


def test_no_lookahead_step_jump_execution_lag():
    """A step jump at index t leaves every earlier row identical to an all-flat
    baseline (no backward leakage), and the jump can trigger a fill no earlier
    than bar t+1 (execution lags the signal by one bar). The implemented strategy
    treats close[t] as a legitimate same-bar band trigger per SPEC 6.2, so the
    earliest a signal may change is row t and the earliest fill is row t+1."""
    flat = [100.0] * 12
    base = run_backtest(make_15m_df(flat), make_1m_for_15m(make_15m_df(flat), volume=500.0),
                        _flat_daily(), INSTRUMENT, base_config())

    t = 7
    jumped = list(flat)
    for i in range(t, len(jumped)):
        jumped[i] = 90.0  # downward step starting at t
    dj = make_15m_df(jumped)
    out = run_backtest(dj, make_1m_for_15m(dj, volume=500.0), _flat_daily(), INSTRUMENT, base_config())

    # No backward leakage: rows strictly before t identical to the flat baseline.
    for c in ("signal", "trade_qty_from_prev", "position", "cash"):
        assert np.allclose(base[c].iloc[:t].to_numpy(), out[c].iloc[:t].to_numpy()), f"backward leak in {c}"
    # Execution lag: no trade on or before the signal bar t.
    assert np.all(out["trade_qty_from_prev"].iloc[: t + 1].to_numpy() == 0.0), "fill occurred no later than signal bar"

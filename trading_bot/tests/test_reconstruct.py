"""Reconstruction round-trip test (SPEC 11.3, 14).

Synthetic events -> event_log JSONL -> reconstruct must equal a directly-built
canonical DataFrame, with a position mark on every bar (including no-trade bars),
and the self-consistency assertions must pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.reconstruct import reconstruct
from src.live.event_log import make_logger
from tests.helpers import INSTRUMENT


def _emit_log(tmp_dir):
    log, path = make_logger("rt_test", INSTRUMENT, log_dir=tmp_dir)
    lot = INSTRUMENT.lot_size
    cash = 1_000_000.0
    bars = [
        # (bar_ts, close, signal, target, fill_qty, fill_price, fee)
        ("2024-06-03T07:00:00Z", 100.0, 0, 0.0, None, None, None),
        ("2024-06-03T07:15:00Z", 100.0, 1, 5.0, 5.0, 100.0, 25.0),
        ("2024-06-03T07:30:00Z", 101.0, 1, 5.0, None, None, None),
        ("2024-06-03T07:45:00Z", 99.0, 0, 0.0, -5.0, 99.0, 24.75),
    ]
    position = 0.0
    expected_rows = []
    for bar_ts, close, sig, target, fq, fp, fee in bars:
        log("candle", {"bar_ts": bar_ts, "o": close, "h": close + 0.5, "l": close - 0.5,
                       "c": close, "v": 1000.0, "warmup_invalid": False})
        log("signal", {"bar_ts": bar_ts, "value": sig, "target_position": target,
                       "intended_position_before": position, "gate": "open", "gate_after": "open",
                       "regime": "up", "band": "inside", "fast_ewm": None, "slow_ewm": None,
                       "bb_up": None, "bb_lo": None, "bb_mid": None, "recent_closes": [close]})
        trade_qty, fill_price, fees = 0.0, np.nan, 0.0
        if fq is not None:
            log("order", {"bar_ts": bar_ts, "client_order_id": f"oid{bar_ts}", "retry_attempt": 0,
                          "side": "buy" if fq > 0 else "sell", "qty": abs(fq), "reference_price": fp})
            log("order_status", {"client_order_id": f"oid{bar_ts}", "status": "filled", "error": None})
            log("fill", {"client_order_id": f"oid{bar_ts}", "bar_ts": bar_ts, "qty": fq, "price": fp, "fee": fee})
            position += fq
            cash -= fq * fp * lot + fee
            trade_qty, fill_price, fees = fq, fp, fee
        equity = cash + position * close
        log("position", {"bar_ts": bar_ts, "intended_position": position, "broker_position": position,
                         "cash": cash, "equity": equity, "mark_price": close})
        expected_rows.append({
            "timestamp": pd.Timestamp(bar_ts, tz="UTC"), "open": close, "high": close + 0.5,
            "low": close - 0.5, "close": close, "volume": 1000.0, "signal": sig,
            "target_position": target, "trade_qty_from_prev": trade_qty,
            "fill_price_from_prev": fill_price, "fees_from_prev": fees,
            "position": position, "cash": cash, "equity": equity,
        })
    return path, pd.DataFrame(expected_rows)


def test_reconstruct_round_trip(tmp_path_factory=None):
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    path, expected = _emit_log(tmp)

    df = reconstruct(path, INSTRUMENT, assert_consistency=True)
    assert len(df) == len(expected) == 4

    # Position mark on every bar, including the two no-trade bars.
    assert (df["trade_qty_from_prev"].to_numpy() == np.array([0.0, 5.0, 0.0, -5.0])).all()

    for col in ("close", "signal", "target_position", "position", "cash", "equity",
                "trade_qty_from_prev", "fees_from_prev"):
        a = expected[col].to_numpy(dtype=float)
        b = df[col].to_numpy(dtype=float)
        assert np.allclose(a, b), f"round-trip mismatch in {col}: {a} vs {b}"

    # triggering_signal is signal.shift(1).
    assert df["triggering_signal"].tolist() == [0, 0, 1, 1]


def test_reconstruct_truncates_at_halt():
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    log, path = make_logger("halt_test", INSTRUMENT, log_dir=tmp)
    log("candle", {"bar_ts": "2024-06-03T07:00:00Z", "o": 100, "h": 100.5, "l": 99.5, "c": 100, "v": 1000.0, "warmup_invalid": False})
    log("signal", {"bar_ts": "2024-06-03T07:00:00Z", "value": 0, "target_position": 0.0,
                   "intended_position_before": 0.0, "gate": "open", "gate_after": "open",
                   "regime": "up", "band": "inside", "fast_ewm": None, "slow_ewm": None,
                   "bb_up": None, "bb_lo": None, "bb_mid": None, "recent_closes": [100]})
    log("position", {"bar_ts": "2024-06-03T07:00:00Z", "intended_position": 0.0, "broker_position": 0.0,
                     "cash": 1_000_000.0, "equity": 1_000_000.0, "mark_price": 100})
    log("discrepancy", {"bar_ts": "2024-06-03T07:15:00Z", "intended_position": 0.0,
                        "broker_position": 3.0, "context": "test"})
    log("halt", {"reason": "discrepancy", "intended_position": 0.0})
    # A late position event after the halt must be dropped.
    log("position", {"bar_ts": "2024-06-03T07:30:00Z", "intended_position": 0.0, "broker_position": 0.0,
                     "cash": 1_000_000.0, "equity": 1_000_000.0, "mark_price": 100})

    df = reconstruct(path, INSTRUMENT, assert_consistency=True)
    # Halt bar is floored from the halt event's wall-clock ts; rows at/after it drop.
    assert len(df) >= 1
    assert df["timestamp"].max() <= pd.Timestamp("2024-06-03T07:15:00Z", tz="UTC")

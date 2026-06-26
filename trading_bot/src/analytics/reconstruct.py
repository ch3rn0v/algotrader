"""Reconstruct the canonical DataFrame from a live JSONL run log (SPEC 11.3).

Single-run scope: cross-run reconciliation (halts, ``flat`` runs, resumes across
``run_id``s) is out of scope for v1. Each run is reconstructed independently;
if a ``halt`` event appears, rows are truncated at the halt bar and the
self-consistency assertions are evaluated only on the in-run rows.

On a discrepancy/halt, the canonical ``position`` uses the bot's ``intended_position``
(what the strategy acted on); ``broker_position`` is preserved as an auxiliary
column for debugging.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.types import CANONICAL_COLUMNS, Instrument

_EPS = 1e-6


def _read_events(log_path) -> list[dict]:
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _bar_of(ts_iso: str) -> pd.Timestamp:
    return _utc(ts_iso).floor("15min")


def _fill_bar(ev: dict) -> pd.Timestamp:
    """Bar a fill belongs to. Prefer the explicit ``bar_ts`` the live runner
    stamps (so the trade lands on the same bar as its position mark, keeping the
    log internally consistent); fall back to flooring the wall-clock ``ts``."""
    bar_ts = ev.get("bar_ts")
    return _utc(bar_ts) if bar_ts else _bar_of(ev["ts"])


def reconstruct(log_path, instrument: Instrument, *, assert_consistency: bool = True) -> pd.DataFrame:
    """Fold a run log into the canonical DataFrame, asserting self-consistency."""
    events = _read_events(log_path)
    if not events:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    # The halt event carries no ``bar_ts``; the bar it occurred on is the most
    # recent bar-bearing event at or before it (typically the preceding
    # ``discrepancy``). That bar is incomplete, so rows at/after it are dropped.
    halt_bar = None
    current_bar = None
    for ev in events:
        if ev.get("bar_ts"):
            current_bar = _utc(ev["bar_ts"])
        if ev["type"] == "halt":
            halt_bar = current_bar if current_bar is not None else _bar_of(ev["ts"])
            break

    rows: dict[pd.Timestamp, dict] = defaultdict(dict)
    fills_by_order: dict[str, list[dict]] = defaultdict(list)
    fills_by_bar: dict[pd.Timestamp, list[dict]] = defaultdict(list)

    for ev in events:
        kind = ev["type"]
        if kind == "candle":
            bar = _utc(ev["bar_ts"])
            rows[bar].update(
                open=ev["o"], high=ev["h"], low=ev["l"], close=ev["c"],
                volume=ev["v"], warmup_invalid=bool(ev.get("warmup_invalid", False)),
            )
        elif kind == "signal":
            bar = _utc(ev["bar_ts"])
            rows[bar].update(signal=int(ev["value"]), target_position=float(ev["target_position"]))
        elif kind == "position":
            bar = _utc(ev["bar_ts"])
            rows[bar].update(
                position=float(ev["intended_position"]),
                broker_position=ev.get("broker_position"),
                cash=float(ev["cash"]),
                equity=float(ev["equity"]),
            )
        elif kind == "fill":
            bar = _fill_bar(ev)
            fills_by_order[ev["client_order_id"]].append(ev)
            fills_by_bar[bar].append(ev)

    for bar, fills in fills_by_bar.items():
        qty = sum(f["qty"] for f in fills)
        abs_qty = sum(abs(f["qty"]) for f in fills)
        price = sum(abs(f["qty"]) * f["price"] for f in fills) / abs_qty if abs_qty else float("nan")
        rows[bar].update(
            trade_qty_from_prev=float(qty),
            fill_price_from_prev=float(price),
            fees_from_prev=float(sum(f["fee"] for f in fills)),
        )

    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "timestamp"
    df = df.reset_index()
    if halt_bar is not None:
        df = df.loc[df["timestamp"] < halt_bar].reset_index(drop=True)

    df["exchange"] = instrument.exchange
    df["instrument"] = instrument.symbol
    for col, default in (
        ("signal", 0), ("target_position", 0.0), ("trade_qty_from_prev", 0.0),
        ("fill_price_from_prev", np.nan), ("fees_from_prev", 0.0),
        ("warmup_invalid", False), ("broker_position", np.nan),
    ):
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    df["signal"] = df["signal"].astype(int)
    df["triggering_signal"] = df["signal"].shift(1).fillna(0).astype(int)

    if assert_consistency:
        _assert_self_consistent(df, fills_by_order, instrument.lot_size)

    extra = ["warmup_invalid", "broker_position"]
    return df[[*CANONICAL_COLUMNS, *extra]].reset_index(drop=True)


def _assert_self_consistent(df: pd.DataFrame, fills_by_order: dict, lot_size: int) -> None:
    """Cheap invariants that catch live-bot bugs (SPEC 11.3)."""
    equity = df["equity"].to_numpy()
    cash = df["cash"].to_numpy()
    position = df["position"].to_numpy()
    close = df["close"].to_numpy()
    trade = df["trade_qty_from_prev"].to_numpy()
    fill_price = df["fill_price_from_prev"].to_numpy()
    fees = df["fees_from_prev"].to_numpy()

    for t in range(len(df)):
        assert abs(equity[t] - (cash[t] + position[t] * close[t])) < _EPS, f"equity mismatch at row {t}"
        if t == 0:
            continue
        assert abs(position[t] - (position[t - 1] + trade[t])) < _EPS, f"position evolution broken at row {t}"
        traded = trade[t] * (fill_price[t] if not np.isnan(fill_price[t]) else 0.0) * lot_size
        expected_cash = cash[t - 1] - traded - fees[t]
        assert abs(cash[t] - expected_cash) < _EPS, f"cash evolution broken at row {t}"

    fee_by_bar = df.set_index("timestamp")["fees_from_prev"]
    for order_id, fills in fills_by_order.items():
        bar = _fill_bar(fills[0])
        if bar not in fee_by_bar.index:
            continue  # fill landed after a halt; out of in-run scope
        order_fee = sum(f["fee"] for f in fills)
        assert abs(order_fee - float(fee_by_bar.loc[bar])) < _EPS, f"fee sum mismatch for {order_id}"

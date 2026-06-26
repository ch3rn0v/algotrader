"""Live vs backtest comparison (SPEC 12). Side effects: disk only.

The comparison backtest replays the **logged** candles, not freshly downloaded
ones: exchanges revise historical data, and re-downloading would contaminate the
drift signal with data drift. 1-min candles (absent from the v1 log) are pulled
from the cache for the same window; this is the smallest available imperfection
and is documented in SPEC 12 step 3.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.analytics.reconstruct import reconstruct
from src.backtest.engine import run_backtest
from src.backtest.loader import read_cache
from src.core.candles import candles_to_df
from src.core.types import Instrument

_REPORTS = Path("reports") / "compare"


def _logged_candles(log_path, instrument: Instrument) -> pd.DataFrame:
    with open(log_path) as f:
        events = [json.loads(line) for line in f if line.strip()]
    candles = [
        {"timestamp": e["bar_ts"], "open": e["o"], "high": e["h"], "low": e["l"], "close": e["c"], "volume": e["v"]}
        for e in events
        if e["type"] == "candle"
    ]
    return candles_to_df(candles, instrument)


def compare(log_path, instrument: Instrument, config: dict) -> dict[str, Path]:
    """Replay logged candles through the backtest, align with the reconstructed
    live run, and write overlay + drift PNGs."""
    run_id = Path(log_path).stem
    candles_15m = _logged_candles(log_path, instrument)
    start = pd.Timestamp(candles_15m["timestamp"].iloc[0]).to_pydatetime()
    end = (pd.Timestamp(candles_15m["timestamp"].iloc[-1]) + pd.Timedelta(minutes=15)).to_pydatetime()

    candles_1m = read_cache(instrument, "1min", start, end)
    candles_1day = read_cache(instrument, "1day", start - pd.Timedelta(days=120), end)

    bt = run_backtest(candles_15m, candles_1m, candles_1day, instrument, config)
    live = reconstruct(log_path, instrument)

    keys = ["exchange", "instrument", "timestamp"]
    merged = bt.merge(live, on=keys, suffixes=("_bt", "_live"))
    if "warmup_invalid_bt" in merged.columns:
        merged = merged.loc[~merged["warmup_invalid_bt"].astype(bool)].reset_index(drop=True)

    merged["position_diff"] = merged["position_live"] - merged["position_bt"]
    merged["equity_diff"] = merged["equity_live"] - merged["equity_bt"]
    bt_trade_bar = merged["trade_qty_from_prev_bt"] != 0
    live_trade_bar = merged["trade_qty_from_prev_live"] != 0
    merged["disagree"] = (merged["signal_bt"] != merged["signal_live"]) | (bt_trade_bar != live_trade_bar)

    out_dir = _REPORTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(merged["timestamp"], utc=True)
    paths: dict[str, Path] = {}

    for name, col in (("pnl", "equity"), ("position", "position")):
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(ts, merged[f"{col}_bt"], label="backtest", lw=1.2)
        ax.plot(ts, merged[f"{col}_live"], label="live", lw=1.0, alpha=0.8)
        ax.set_title(f"{col} overlay — {run_id}")
        ax.legend()
        fig.autofmt_xdate()
        paths[f"{name}_overlay"] = out_dir / f"{name}_overlay.png"
        fig.savefig(paths[f"{name}_overlay"], dpi=110, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.step(ts, merged["trade_qty_from_prev_bt"].ne(0).cumsum(), where="post", label="backtest", lw=1.2)
    ax.step(ts, merged["trade_qty_from_prev_live"].ne(0).cumsum(), where="post", label="live", lw=1.0, alpha=0.8)
    ax.set_title(f"cumulative trades overlay — {run_id}")
    ax.legend()
    fig.autofmt_xdate()
    paths["trades_overlay"] = out_dir / "trades_overlay.png"
    fig.savefig(paths["trades_overlay"], dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ts, merged["equity_diff"], color="#d62728", lw=1.0)
    marks = merged.loc[merged["disagree"]]
    ax.scatter(pd.to_datetime(marks["timestamp"], utc=True), marks["equity_diff"], color="#000", s=12, zorder=3, label="signal/fill disagreement")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_title(f"drift (live - backtest equity) — {run_id}")
    ax.legend()
    fig.autofmt_xdate()
    paths["drift"] = out_dir / "drift.png"
    fig.savefig(paths["drift"], dpi=110, bbox_inches="tight")
    plt.close(fig)
    return paths

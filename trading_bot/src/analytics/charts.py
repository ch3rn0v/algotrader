"""Charts and the metrics.json writer (SPEC 11.2). Side effects: disk only.

Three PNGs per run under ``reports/<run_id>/`` plus a ``metrics.json``. PNGs
open everywhere and diff visually; no HTML, no interactivity.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_REPORTS = Path("reports")


def _eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "warmup_invalid" in df.columns:
        df = df.loc[~df["warmup_invalid"].astype(bool)]
    return df.sort_values("timestamp").reset_index(drop=True)


def render_charts(df: pd.DataFrame, run_id: str, root: Path = _REPORTS) -> dict[str, Path]:
    """Write pnl.png, position.png and trades.png; return their paths."""
    rows = _eval_rows(df)
    out_dir = root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(rows["timestamp"], utc=True)
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ts, rows["equity"], color="#1f77b4", lw=1.2)
    running_max = rows["equity"].cummax()
    ax.fill_between(ts, rows["equity"], running_max, where=rows["equity"] < running_max, color="#d62728", alpha=0.15)
    ax.set_title(f"Equity curve — {run_id}")
    ax.set_ylabel("equity")
    fig.autofmt_xdate()
    paths["pnl"] = out_dir / "pnl.png"
    fig.savefig(paths["pnl"], dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.step(ts, rows["position"], where="post", color="#2ca02c", lw=1.2)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_title(f"Net position (lots) — {run_id}")
    ax.set_ylabel("lots")
    fig.autofmt_xdate()
    paths["position"] = out_dir / "position.png"
    fig.savefig(paths["position"], dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3))
    cum_trades = (rows["trade_qty_from_prev"] != 0).cumsum()
    ax.plot(ts, cum_trades, color="#9467bd", lw=1.2)
    ax.set_title(f"Cumulative trade count — {run_id}")
    ax.set_ylabel("trades")
    fig.autofmt_xdate()
    paths["trades"] = out_dir / "trades.png"
    fig.savefig(paths["trades"], dpi=110, bbox_inches="tight")
    plt.close(fig)
    return paths


def write_metrics(metrics: dict, run_id: str, root: Path = _REPORTS) -> Path:
    """Write the flat metrics.json alongside the charts."""
    out_dir = root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return path

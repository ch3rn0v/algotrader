"""Strategy diagnostics chart for a specific time window.

Shows price + EWM trend with entry bands and quiet-filter shading, trade
signals, model predictions, equity curve, position, and per-bar trading volume.

Usage:
    python3 bot/diagnostics.py                           # yesterday
    python3 bot/diagnostics.py --from 2025-03-01         # single day
    python3 bot/diagnostics.py --from 2025-03-01 --to 2025-03-03
    python3 bot/diagnostics.py --trend-alpha 0.05 --entry-pct 0.4
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_mean_rev_ewm import run_backtest
from candles import get_candles
from config import BOT_DIR
from model import build_predictions

# Extra history loaded before the display window for trend/volume warmup.
_WARMUP_DAYS = 5


def plot_diagnostics(
    candles: pd.DataFrame,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    predictions: np.ndarray | None,
    trend_alpha: float,
    entry_pct: float,
    max_slope_pct: float,
    slope_window: int,
    vol_alpha: float,
    vol_window: int,
    vol_ratio_max: float,
    min_quiet_bars: int,
    display_from: datetime,
    display_to: datetime,
    path: Path = Path("diagnostics.png"),
) -> None:
    # Compute trend/filters on full candle set (warmup included), then slice
    # to the display window. Mirrors the indicator maths in run_backtest.
    trend = candles["close"].ewm(alpha=trend_alpha).mean()
    upper = trend * (1 + entry_pct / 100)
    lower = trend * (1 - entry_pct / 100)

    slope_pct = (trend / trend.shift(slope_window) - 1.0).abs() * 100.0
    flat_trend = slope_pct < max_slope_pct
    vol_fast = candles["volume"].ewm(alpha=vol_alpha).mean()
    vol_base = candles["volume"].rolling(vol_window).mean()
    low_vol = vol_fast < vol_ratio_max * vol_base
    quiet = (flat_trend & low_vol).to_numpy()
    q = quiet.astype(np.int64)
    cs = np.cumsum(q)
    quiet_run = cs - np.maximum.accumulate(np.where(q == 0, cs, 0))
    eligible = pd.Series(quiet_run >= min_quiet_bars, index=candles.index)

    ts_all = pd.to_datetime(candles["timestamp"], utc=True)
    disp_from_ts = pd.Timestamp(display_from)
    disp_to_ts = pd.Timestamp(display_to)
    mask = (ts_all >= disp_from_ts) & (ts_all < disp_to_ts)

    c_ts = ts_all[mask]
    c = candles.loc[mask]

    eq = equity.copy()
    eq.loc[:, "timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
    eq_mask = (eq["timestamp"] >= disp_from_ts) & (eq["timestamp"] < disp_to_ts)
    eq = eq.loc[eq_mask]

    tr = trades.copy()
    if not tr.empty:
        tr.loc[:, "timestamp"] = pd.to_datetime(tr["timestamp"], utc=True)
        tr = tr[(tr["timestamp"] >= disp_from_ts) & (tr["timestamp"] < disp_to_ts)]

    bar_w = (c_ts.iloc[1] - c_ts.iloc[0]) * 0.8 if len(c_ts) >= 2 else pd.Timedelta(minutes=4)

    has_pred = predictions is not None
    n_panels = 5 if has_pred else 4
    height_ratios = [2.5, 1, 1.5, 1, 1] if has_pred else [2.5, 1.5, 1, 1]

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, 3 * n_panels),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    if has_pred:
        ax_price, ax_pred, ax_eq, ax_pos, ax_vol = axes
    else:
        ax_price, ax_eq, ax_pos, ax_vol = axes

    # --- Panel: Price + trend/entry bands + quiet shading + volume bars ---
    ax_vol_twin = ax_price.twinx()
    ax_vol_twin.bar(c_ts, c["volume"].values, color="gray", alpha=0.2, zorder=1, width=bar_w)
    ax_vol_twin.yaxis.set_visible(False)

    ax_price.plot(c_ts, c["close"].values, color="steelblue", lw=1.0, zorder=2, label="close")
    ax_price.plot(c_ts, trend[mask].values, color="orange", lw=0.8, ls="--", zorder=3, label="trend (ewm)")
    ax_price.plot(c_ts, upper[mask].values, color="tomato", lw=0.8, ls="--", zorder=3, label="short entry")
    ax_price.plot(c_ts, lower[mask].values, color="seagreen", lw=0.8, ls="--", zorder=3, label="long entry")
    ax_price.fill_between(c_ts, lower[mask].values, upper[mask].values, alpha=0.06, color="steelblue")

    # Shade bars where the quiet filter allows entries.
    lo, hi = c["low"].min(), c["high"].max()
    ax_price.fill_between(c_ts, lo, hi, where=eligible[mask].values, alpha=0.08, color="gold", label="quiet (entries allowed)")

    for _, row in tr.iterrows():
        color = "green" if row["qty"] > 0 else "red"
        ax_price.axvline(row["timestamp"], color=color, alpha=0.8, lw=1.2, zorder=5)

    ax_price.set_ylabel("Price (RUB)")
    ax_price.set_title(f"Price + EWM trend  (alpha={trend_alpha}, entry={entry_pct}%)", loc="left")
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.margins(0)
    ax_price.grid(True, lw=0.4)
    ax_vol_twin.margins(0)

    # --- Panel: Model predictions ---
    if has_pred:
        pred_d = predictions[mask.values]
        ax_pred.plot(c_ts, pred_d, color="purple", lw=0.8)
        ax_pred.axhline(1.0, color="gray", lw=0.7, ls="--")
        ax_pred.set_ylabel("Prediction")
        ax_pred.set_title("Model predictions", loc="left")
        ax_pred.margins(0)
        ax_pred.grid(True, lw=0.4)

    # --- Panel: Equity ---
    if not eq.empty:
        ax_eq.plot(eq["timestamp"], eq["equity"], color="steelblue", lw=1.0)
    ax_eq.set_ylabel("Equity (RUB)")
    ax_eq.set_title("PnL", loc="left")
    ax_eq.margins(0)
    ax_eq.grid(True, lw=0.4)

    # --- Panel: Position ---
    if not eq.empty:
        ax_pos.step(eq["timestamp"], eq["position"], where="post", color="steelblue")
    ax_pos.axhline(0, color="gray", lw=0.7)
    ax_pos.set_ylabel("Position (lots)")
    ax_pos.set_title("Position", loc="left")
    ax_pos.margins(0)
    ax_pos.grid(True, lw=0.4)

    # --- Panel: Per-bar trading volume ---
    if not tr.empty:
        buys = tr[tr["qty"] > 0]
        sells = tr[tr["qty"] < 0]
        if not buys.empty:
            ax_vol.bar(buys["timestamp"], buys["qty"].abs(), color="green", alpha=0.7, width=bar_w, label="buy")
        if not sells.empty:
            ax_vol.bar(sells["timestamp"], sells["qty"].abs(), color="red", alpha=0.7, width=bar_w, label="sell")
        ax_vol.legend(loc="upper left", fontsize=8)
    ax_vol.set_ylabel("Lots traded")
    ax_vol.set_title("Trading volume", loc="left")
    ax_vol.margins(0)
    ax_vol.grid(True, lw=0.4)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Diagnostics chart saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy diagnostics chart")
    parser.add_argument("--figi", default="BBG0047315Y7", help="Instrument FIGI (default: SBERP)")
    parser.add_argument("--timeframe", default="1min", help="Candle timeframe (default: 1min)")
    parser.add_argument("--from", dest="date_from", default=None, metavar="DATE", help="Display start YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--to", dest="date_to", default=None, metavar="DATE", help="Display end YYYY-MM-DD (default: date_from + 1 day)")
    parser.add_argument("--trend-alpha",           type=float, default=0.02)
    parser.add_argument("--entry-pct",             type=float, default=0.3)
    parser.add_argument("--exit-pct",              type=float, default=0.05)
    parser.add_argument("--max-slope-pct",         type=float, default=0.05)
    parser.add_argument("--slope-window",          type=int,   default=15)
    parser.add_argument("--vol-alpha",             type=float, default=0.05)
    parser.add_argument("--vol-window",            type=int,   default=240)
    parser.add_argument("--vol-ratio-max",         type=float, default=0.9)
    parser.add_argument("--min-quiet-bars",        type=int,   default=10)
    parser.add_argument("--time-stop-bars",        type=int,   default=60)
    parser.add_argument("--session-start-utc",     type=int,   default=9)
    parser.add_argument("--session-end-utc",       type=int,   default=16)
    parser.add_argument("--pred-long-threshold",   type=float, default=1.0)
    parser.add_argument("--pred-short-threshold",  type=float, default=1.0)
    parser.add_argument("--out", default="outputs/diagnostics", metavar="DIR")
    args = parser.parse_args()

    today = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    display_from = (
        datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.date_from else today - timedelta(days=1)
    )
    display_to = (
        datetime.strptime(args.date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.date_to else display_from + timedelta(days=1)
    )
    load_from = display_from - timedelta(days=_WARMUP_DAYS)

    print(f"Display window: {display_from.date()} → {display_to.date()}")
    print(f"Loading candles from {load_from.date()} for trend/volume warmup...")
    candles = get_candles(args.figi, args.timeframe, load_from, display_to)
    print(f"Candles loaded: {len(candles)}")

    print("Loading model predictions...")
    predictions, _ = build_predictions(candles, args.figi, args.timeframe, load_from, display_to)

    result = run_backtest(
        candles,
        trend_alpha=args.trend_alpha,
        entry_pct=args.entry_pct,
        exit_pct=args.exit_pct,
        max_slope_pct=args.max_slope_pct,
        slope_window=args.slope_window,
        vol_alpha=args.vol_alpha,
        vol_window=args.vol_window,
        vol_ratio_max=args.vol_ratio_max,
        min_quiet_bars=args.min_quiet_bars,
        time_stop_bars=args.time_stop_bars,
        session_start_utc=args.session_start_utc,
        session_end_utc=args.session_end_utc,
        pred_long_threshold=args.pred_long_threshold,
        pred_short_threshold=args.pred_short_threshold,
        position_size=1,
        headless=False,
        predictions=predictions,
    )

    out_dir = BOT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = f"{display_from.date()}_{display_to.date()}"
    out_path = out_dir / f"diagnostics_{date_tag}.png"

    plot_diagnostics(
        candles=candles,
        equity=result["equity"],
        trades=result["trades"],
        predictions=predictions,
        trend_alpha=args.trend_alpha,
        entry_pct=args.entry_pct,
        max_slope_pct=args.max_slope_pct,
        slope_window=args.slope_window,
        vol_alpha=args.vol_alpha,
        vol_window=args.vol_window,
        vol_ratio_max=args.vol_ratio_max,
        min_quiet_bars=args.min_quiet_bars,
        display_from=display_from,
        display_to=display_to,
        path=out_path,
    )

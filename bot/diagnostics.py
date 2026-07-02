"""Strategy diagnostics chart for a specific time window.

Shows price + Bollinger Bands with trade signals, model predictions,
equity curve, position, and per-bar trading volume.

Usage:
    python3 bot/diagnostics.py                           # yesterday
    python3 bot/diagnostics.py --from 2025-03-01         # single day
    python3 bot/diagnostics.py --from 2025-03-01 --to 2025-03-03
    python3 bot/diagnostics.py --bb-period 25 --bb-std 1.5
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_mean_rev_bb import run_backtest
from candles import get_candles
from config import BOT_DIR
from model import build_predictions

# Extra history loaded before the display window for BB/width warmup.
_WARMUP_DAYS = 5


def plot_diagnostics(
    candles: pd.DataFrame,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    predictions: np.ndarray | None,
    bb_alpha: float,
    bb_std: float,
    width_alpha: float,
    display_from: datetime,
    display_to: datetime,
    path: Path = Path("diagnostics.png"),
) -> None:
    # Compute BB on full candle set (warmup included), then slice to display window.
    mid = candles["close"].ewm(alpha=bb_alpha).mean()
    std_bb = candles["close"].ewm(alpha=bb_alpha).std()
    upper = mid + bb_std * std_bb
    lower = mid - bb_std * std_bb

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

    # --- Panel: Price + BB + volume bars ---
    ax_vol_twin = ax_price.twinx()
    ax_vol_twin.bar(c_ts, c["volume"].values, color="gray", alpha=0.2, zorder=1, width=bar_w)
    ax_vol_twin.yaxis.set_visible(False)

    ax_price.plot(c_ts, c["close"].values, color="steelblue", lw=1.0, zorder=2, label="close")
    ax_price.plot(c_ts, mid[mask].values, color="orange", lw=0.8, ls="--", zorder=3, label="mid")
    ax_price.plot(c_ts, upper[mask].values, color="tomato", lw=0.8, ls="--", zorder=3, label="upper")
    ax_price.plot(c_ts, lower[mask].values, color="seagreen", lw=0.8, ls="--", zorder=3, label="lower")
    ax_price.fill_between(c_ts, lower[mask].values, upper[mask].values, alpha=0.06, color="steelblue")

    for _, row in tr.iterrows():
        color = "green" if row["qty"] > 0 else "red"
        ax_price.axvline(row["timestamp"], color=color, alpha=0.8, lw=1.2, zorder=5)

    ax_price.set_ylabel("Price (RUB)")
    ax_price.set_title(f"Price + Bollinger Bands  (alpha={bb_alpha}, std={bb_std})", loc="left")
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
    parser.add_argument("--timeframe", default="5min", help="Candle timeframe (default: 5min)")
    parser.add_argument("--from", dest="date_from", default=None, metavar="DATE", help="Display start YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--to", dest="date_to", default=None, metavar="DATE", help="Display end YYYY-MM-DD (default: date_from + 1 day)")
    parser.add_argument("--bb-alpha",            type=float, default=0.1)
    parser.add_argument("--bb-std",               type=float, default=2.0)
    parser.add_argument("--time-stop-bars",        type=int,   default=12)
    parser.add_argument("--width-alpha",           type=float, default=0.1)
    parser.add_argument("--session-start-utc",     type=int,   default=9)
    parser.add_argument("--session-end-utc",       type=int,   default=12)
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
    print(f"Loading candles from {load_from.date()} for BB warmup...")
    candles = get_candles(args.figi, args.timeframe, load_from, display_to)
    print(f"Candles loaded: {len(candles)}")

    print("Loading model predictions...")
    predictions, _ = build_predictions(candles, args.figi, args.timeframe, load_from, display_to)

    result = run_backtest(
        candles,
        bb_alpha=args.bb_alpha,
        bb_std=args.bb_std,
        time_stop_bars=args.time_stop_bars,
        width_alpha=args.width_alpha,
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
        bb_alpha=args.bb_alpha,
        bb_std=args.bb_std,
        width_alpha=args.width_alpha,
        display_from=display_from,
        display_to=display_to,
        path=out_path,
    )

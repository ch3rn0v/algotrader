"""Minimal z-score mean reversion for SBERP 1min, gated by the 5min model.

Two tunable parameters, everything else is fixed by design:
  alpha   - one EWM alpha for both the trend (ewm mean of close) and the
            deviation scale (ewm std of close).
  z_entry - entry threshold in EWM standard deviations.

Logic:
  - z = (close - ewm_mean) / ewm_std: how far price has strayed from the
    trend, in units of recent volatility. Being volatility-scaled, one
    threshold works across calm and busy regimes with no extra filters.
  - Enter against the deviation when |z| >= z_entry AND the model agrees:
    pred > 1 for longs, pred < 1 for shorts. The 5min model's prediction is
    mapped onto every 1min bar of its 5min block (merge_asof backward), so a
    decision only ever sees a prediction computed from already-closed bars.
  - Exit when z crosses 0 (price is back at the trend), at session end, and
    always at the last bar of the day.
  - Trade only the main MOEX session, 10:00-18:45 MSK = 07:00-15:45 UTC.
  - Decision uses the previous completed bar; fill at the current bar's open.

Usage:
    python3 backtest_mean_rev_z.py                 # config dates, model-gated
    python3 backtest_mean_rev_z.py --alpha 0.05 --z-entry 2.5
"""

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from candles import get_candles
from config import FROM, OUTPUT_DIR, PRIMARY_ASSET, PRIMARY_FIGI, TO
from model import build_predictions

# Main MOEX session in UTC minutes-of-day (10:00-18:45 MSK, MSK = UTC+3).
_SESSION_START_MIN = 7 * 60
_SESSION_END_MIN = 15 * 60 + 45


def run_backtest(
    candles: pd.DataFrame,
    alpha: float = 0.02,
    z_entry: float = 2.0,
    predictions: np.ndarray | None = None,
    position_size: int = 1,
    initial_cash: float = 100_000.0,
    fee_rate: float = 0.0005,  # market-taker fee per side (T-Bank Trader tariff)
    headless: bool = False,
) -> dict:
    df = candles.reset_index(drop=True)
    n = len(df)

    trend = df["close"].ewm(alpha=alpha).mean()
    sigma = df["close"].ewm(alpha=alpha).std()
    z = ((df["close"] - trend) / sigma).to_numpy()

    ts_dt = pd.to_datetime(df["timestamp"])
    minute = (ts_dt.dt.hour * 60 + ts_dt.dt.minute).to_numpy()
    session_a = (minute >= _SESSION_START_MIN) & (minute < _SESSION_END_MIN)

    date = ts_dt.dt.date.to_numpy()
    is_last_bar = np.empty(n, dtype=bool)
    is_last_bar[:-1] = date[:-1] != date[1:]
    is_last_bar[-1] = True

    # Last in-session bar: session is True now and False on the next bar (or last bar overall).
    is_session_end = np.empty(n, dtype=bool)
    is_session_end[:-1] = session_a[:-1] & ~session_a[1:]
    is_session_end[-1] = False

    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()
    timestamps = df["timestamp"].to_numpy()

    # --- Simulate ---
    position = 0
    cash = float(initial_cash)
    desired = 0
    peak_exposure = 0.0
    total_fees = 0.0
    trade_rows = []
    equity = np.empty(n)
    pos_hist = np.empty(n, dtype=np.int64)

    for i in range(n):
        # Decide desired position using the previous bar's completed data.
        # (i == 0 has no previous bar; desired stays 0.)
        if i > 0:
            p = i - 1
            if np.isnan(z[p]) or is_last_bar[p] or is_session_end[p]:
                desired = 0
            elif position != 0:
                back_at_trend = (position > 0 and z[p] >= 0) or (position < 0 and z[p] <= 0)
                desired = 0 if back_at_trend else position
            elif session_a[p]:
                if predictions is not None:
                    pred = float(predictions[i])
                    want_long = z[p] <= -z_entry and pred > 1.0
                    want_short = z[p] >= z_entry and pred < 1.0
                else:
                    want_long = z[p] <= -z_entry
                    want_short = z[p] >= z_entry
                desired = position_size if want_long else -position_size if want_short else 0
            else:
                desired = 0

        # Execute at this bar's open.
        delta = desired - position
        if delta != 0:
            notional = abs(delta) * opens[i]
            fee = notional * fee_rate
            total_fees += fee
            cash -= delta * opens[i] + fee
            position += delta
            peak_exposure = max(peak_exposure, abs(position) * opens[i])
            trade_rows.append({"timestamp": timestamps[i], "qty": delta, "price": opens[i], "fee": fee})

        pos_hist[i] = position
        equity[i] = cash + position * closes[i]

    trades = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(columns=["timestamp", "qty", "price", "fee"])

    if not headless:
        return {
            "equity": pd.DataFrame({"timestamp": timestamps, "equity": equity, "position": pos_hist}),
            "trades": trades,
            "total_fees": total_fees,
            "peak_exposure": peak_exposure,
        }

    # --- Stats (headless) ---
    pnl = equity[-1] - initial_cash
    t0 = pd.to_datetime(timestamps[0], utc=True)
    t1 = pd.to_datetime(timestamps[-1], utc=True)
    n_days = max((t1 - t0).days, 1)

    n_entries = int(np.sum((pos_hist[1:] != 0) & (pos_hist[:-1] == 0)))
    total_bars_held = int(np.sum(pos_hist != 0))
    avg_bars_held = total_bars_held / n_entries if n_entries > 0 else 0.0
    pct_bars_in_pos = total_bars_held / n * 100 if n > 0 else 0.0
    turnover = float((trades["qty"].abs() * trades["price"]).sum())

    # Daily equity: seed with t=0 so the first day's return isn't dropped.
    daily_eq = np.concatenate(([initial_cash], equity[is_last_bar]))

    if peak_exposure > 0 and len(daily_eq) >= 2:
        daily_ret = np.diff(daily_eq) / peak_exposure
        mean_r = float(daily_ret.mean())
        std_r = float(daily_ret.std())
        sharpe = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0.0
        neg = daily_ret[daily_ret < 0]
        s_std = float(neg.std()) if len(neg) > 1 else 0.0
        sortino = mean_r / s_std * np.sqrt(252) if s_std > 0 else 0.0
        base = 1 + pnl / peak_exposure
        cagr = base ** (365 / n_days) - 1 if base > 0 else -1.0
        running_max = np.maximum.accumulate(np.maximum(equity, initial_cash))
        max_dd = float(min((equity - running_max).min(), 0.0)) / peak_exposure
    else:
        sharpe = sortino = cagr = max_dd = 0.0

    return {
        "pnl": round(pnl, 2),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd": round(max_dd, 4),
        "cagr": round(cagr, 4),
        "n_trades": len(trades),
        "avg_bars_held": round(avg_bars_held, 2),
        "pct_bars_in_pos": round(pct_bars_in_pos, 2),
        "turnover": round(turnover, 2),
        "total_fees": round(total_fees, 2),
        "peak_exposure": round(peak_exposure, 2),
    }


def predictions_for_1min(candles_1m: pd.DataFrame, from_dt, to_dt) -> tuple[np.ndarray | None, dict | None]:
    """Map the 5min model's predictions onto 1min bars.

    Builds predictions on the 5min series, then assigns each 1min bar the
    prediction of the 5min block it falls into (merge_asof backward). That
    prediction was computed from bars completed before the block started, so
    using it anywhere inside the block adds no lookahead.
    """
    df5 = get_candles(PRIMARY_FIGI, "5min", from_dt, to_dt)
    preds5, meta = build_predictions(df5, PRIMARY_FIGI, "5min", from_dt, to_dt, fill=1.0)
    if preds5 is None:
        return None, None
    pred5_df = pd.DataFrame({"timestamp": df5["timestamp"], "pred": preds5})
    merged = pd.merge_asof(candles_1m[["timestamp"]], pred5_df, on="timestamp", direction="backward")
    return merged["pred"].fillna(1.0).to_numpy(), meta


def main():
    parser = argparse.ArgumentParser(description="Z-score mean reversion backtest (1min, model-gated)")
    parser.add_argument("--alpha",   type=float, default=0.02)
    parser.add_argument("--z-entry", type=float, default=2.0)
    parser.add_argument("--from", dest="date_from", default=None, metavar="DATE", help="Start date YYYY-MM-DD (default: config FROM)")
    parser.add_argument("--to",   dest="date_to",   default=None, metavar="DATE", help="End date YYYY-MM-DD (default: config TO)")
    args = parser.parse_args()

    from_dt = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.date_from else FROM
    to_dt = datetime.strptime(args.date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.date_to else TO

    candles = get_candles(PRIMARY_FIGI, "1min", from_dt, to_dt)
    print(f"1min candles: {len(candles)}")

    predictions, meta = predictions_for_1min(candles, from_dt, to_dt)

    # Evaluate on the model's test period only (bars after the train cutoff).
    if predictions is not None:
        train_end_ts = pd.Timestamp(meta["train_end_ts"])
        test_mask = (candles["timestamp"] > train_end_ts).to_numpy()
        print(f"Test period: {test_mask.sum()} / {len(candles)} bars (after {train_end_ts})")
        candles = candles.loc[test_mask].reset_index(drop=True)
        predictions = predictions[test_mask]

    stats = run_backtest(candles, alpha=args.alpha, z_entry=args.z_entry,
                         predictions=predictions, headless=True)
    print(f"\nalpha={args.alpha}  z_entry={args.z_entry}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    result = run_backtest(candles, alpha=args.alpha, z_entry=args.z_entry, predictions=predictions)
    from charts import plot_results  # matplotlib import is slow; keep it off the headless path

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_results(
        candles, result["equity"], result["trades"], result["peak_exposure"],
        symbol=f"{PRIMARY_ASSET} (z-rev test)", timeframe="1min",
        path=out_dir / f"result_z_{ts}.png",
    )


if __name__ == "__main__":
    main()

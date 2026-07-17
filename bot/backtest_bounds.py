"""Volume-weighted-band mean reversion for SBERP 1min, tilted by the model.

Instead of a symmetric z-score channel around an EWM mean, this uses the
volume-weighted EWM high as the upper band and the volume-weighted EWM low as
the lower band. These envelopes sit near the actual high/low path of price, so
the entry range is wider than +/-2 sigma -> fewer, more extreme entries.

  vwh = ewm(high * vol, alpha) / ewm(vol, alpha)   (upper band)
  vwl = ewm(low  * vol, alpha) / ewm(vol, alpha)   (lower band)
  mid = (vwh + vwl) / 2                             (exit level)

Logic:
  - Enter long when close falls to/below the lower band (vwl); enter short
    when close rises to/above the upper band (vwh). Only when flat.
  - Exit when close returns to the mid price (and at session end / EOD).
  - Model tilt: (pred - 1) * close is the expected move in price units;
    shifting BOTH entry bands up by pred_gain * that amount makes longs easier
    and shorts harder when the model is bullish (and vice versa). The exit mid
    is left untilted. pred_gain = 0 disables it.
  - Bands can be built from the 1min execution series (--bounds-tf 1min) or a
    higher timeframe (e.g. 5min) mapped onto 1min bars (merge_asof backward,
    completed bars only) so a decision never sees an unfinished band.
  - Decision uses the previous completed bar; fill at the current bar's open.
  - Main MOEX session only, 10:00-18:45 MSK = 07:00-15:45 UTC.

Usage:
    python3 backtest_bounds.py --bounds-tf 1min --alpha 0.05
    python3 backtest_bounds.py --bounds-tf 5min --alpha 0.1 --pred-gain 1.0
"""

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backtest_stats import compute_stats
from candles import get_candles
from config import FROM, OUTPUT_DIR, PRIMARY_ASSET, PRIMARY_FIGI, TO
from features import TF_DURATIONS
from model import build_predictions, load_latest_model

# Main MOEX session in UTC minutes-of-day (10:00-18:45 MSK, MSK = UTC+3).
_SESSION_START_MIN = 7 * 60
_SESSION_END_MIN = 15 * 60 + 45


def vw_bands(candles: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Volume-weighted EWM high/low bands for one candle series."""
    h, l, v = candles["high"], candles["low"], candles["volume"]
    wv = v.ewm(alpha=alpha, adjust=False).mean()
    wv = wv.where(wv > 0)
    vwh = (h * v).ewm(alpha=alpha, adjust=False).mean() / wv
    vwl = (l * v).ewm(alpha=alpha, adjust=False).mean() / wv
    return vwh.to_numpy(), vwl.to_numpy()


def bands_for_1min(candles_1m: pd.DataFrame, bounds_tf: str, alpha: float,
                   from_dt, to_dt) -> tuple[np.ndarray, np.ndarray]:
    """Return (vwh, vwl) aligned to the 1min bars.

    For a higher bounds_tf, bands are computed on that series and mapped onto
    1min bars by bar END (merge_asof backward), so only completed higher-TF
    bars are ever visible.
    """
    if bounds_tf == "1min":
        return vw_bands(candles_1m, alpha)

    df_tf = get_candles(PRIMARY_FIGI, bounds_tf, from_dt, to_dt).sort_values("timestamp").reset_index(drop=True)
    vwh, vwl = vw_bands(df_tf, alpha)
    band_df = pd.DataFrame({
        "timestamp": df_tf["timestamp"] + TF_DURATIONS[bounds_tf],  # bar end
        "vwh": vwh, "vwl": vwl,
    })
    merged = pd.merge_asof(candles_1m[["timestamp"]], band_df, on="timestamp", direction="backward")
    return merged["vwh"].to_numpy(), merged["vwl"].to_numpy()


def run_backtest(
    candles: pd.DataFrame,
    vwh: np.ndarray,
    vwl: np.ndarray,
    max_hold_bars: int | None = None,
    pred_gain: float = 1.0,
    predictions: np.ndarray | None = None,
    position_size: int = 1,
    initial_cash: float = 100_000.0,
    fee_rate: float = 0.0005,
    headless: bool = False,
) -> dict:
    df = candles.reset_index(drop=True)
    n = len(df)

    mid = (vwh + vwl) / 2.0

    ts_dt = pd.to_datetime(df["timestamp"])
    minute = (ts_dt.dt.hour * 60 + ts_dt.dt.minute).to_numpy()
    session_a = (minute >= _SESSION_START_MIN) & (minute < _SESSION_END_MIN)

    date = ts_dt.dt.date.to_numpy()
    is_last_bar = np.empty(n, dtype=bool)
    is_last_bar[:-1] = date[:-1] != date[1:]
    is_last_bar[-1] = True

    is_session_end = np.empty(n, dtype=bool)
    is_session_end[:-1] = session_a[:-1] & ~session_a[1:]
    is_session_end[-1] = False

    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()
    timestamps = df["timestamp"].to_numpy()

    position = 0
    cash = float(initial_cash)
    desired = 0
    entry_bar = 0
    peak_exposure = 0.0
    total_fees = 0.0
    trade_rows = []
    equity = np.empty(n)
    pos_hist = np.empty(n, dtype=np.int64)

    for i in range(n):
        if i > 0:
            p = i - 1
            if np.isnan(mid[p]) or is_last_bar[p] or is_session_end[p]:
                desired = 0
            elif position != 0:
                # Exit when price has returned to the channel mid.
                back_at_mid = (position > 0 and closes[p] >= mid[p]) or \
                              (position < 0 and closes[p] <= mid[p])
                time_stop = max_hold_bars is not None and i - entry_bar >= max_hold_bars
                desired = 0 if back_at_mid or time_stop else position
            elif session_a[p]:
                # Tilt entry bands up by the expected move (price units) so a
                # bullish forecast makes longs easier and shorts harder.
                shift = 0.0
                if predictions is not None:
                    shift = pred_gain * (float(predictions[i]) - 1.0) * closes[p]
                want_long = closes[p] <= vwl[p] + shift
                want_short = closes[p] >= vwh[p] + shift
                desired = position_size if want_long else -position_size if want_short else 0
            else:
                desired = 0

        delta = desired - position
        if delta != 0:
            notional = abs(delta) * opens[i]
            fee = notional * fee_rate
            total_fees += fee
            cash -= delta * opens[i] + fee
            if position == 0:
                entry_bar = i
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

    return compute_stats(equity, pos_hist, trades, timestamps, is_last_bar,
                         peak_exposure, total_fees, initial_cash)


def predictions_for_1min(candles_1m: pd.DataFrame, from_dt, to_dt):
    """Map the latest model's predictions onto 1min bars (see the z backtest)."""
    loaded = load_latest_model()
    if loaded is None:
        print("No model found. Running without predictions.")
        return None, None
    model_tf = loaded[1].get("primary_tf", "5min")
    df_m = get_candles(PRIMARY_FIGI, model_tf, from_dt, to_dt)
    preds_m, meta = build_predictions(df_m, PRIMARY_FIGI, model_tf, from_dt, to_dt, fill=1.0)
    if preds_m is None:
        return None, None
    pred_df = pd.DataFrame({"timestamp": df_m["timestamp"], "pred": preds_m})
    merged = pd.merge_asof(candles_1m[["timestamp"]], pred_df, on="timestamp", direction="backward")
    return merged["pred"].fillna(1.0).to_numpy(), meta


def main():
    parser = argparse.ArgumentParser(description="Volume-weighted-band mean reversion backtest (1min)")
    parser.add_argument("--bounds-tf", default="1min", help="timeframe to build bands from (default 1min)")
    parser.add_argument("--alpha", type=float, default=0.05, help="EWM alpha for the bands (default 0.05)")
    parser.add_argument("--max-hold", type=int, default=0, help="time stop in bars (0 = off)")
    parser.add_argument("--pred-gain", type=float, default=1.0, help="model tilt scale (0 = ignore model)")
    parser.add_argument("--from", dest="date_from", default=None, metavar="DATE", help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", default=None, metavar="DATE", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    from_dt = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.date_from else FROM
    to_dt = datetime.strptime(args.date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.date_to else TO

    candles = get_candles(PRIMARY_FIGI, "1min", from_dt, to_dt)
    print(f"1min candles: {len(candles)}")

    vwh, vwl = bands_for_1min(candles, args.bounds_tf, args.alpha, from_dt, to_dt)
    predictions, meta = predictions_for_1min(candles, from_dt, to_dt)

    if predictions is not None:
        train_end_ts = pd.Timestamp(meta["train_end_ts"])
        test_mask = (candles["timestamp"] > train_end_ts).to_numpy()
        print(f"Test period: {test_mask.sum()} / {len(candles)} bars (after {train_end_ts})")
        candles = candles.loc[test_mask].reset_index(drop=True)
        predictions = predictions[test_mask]
        vwh, vwl = vwh[test_mask], vwl[test_mask]

    max_hold = args.max_hold if args.max_hold > 0 else None
    stats = run_backtest(candles, vwh, vwl, max_hold_bars=max_hold,
                         pred_gain=args.pred_gain, predictions=predictions, headless=True)
    print(f"\nbounds_tf={args.bounds_tf}  alpha={args.alpha}  max_hold={args.max_hold}  pred_gain={args.pred_gain}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

"""Backtest execution model (SPEC 10.3). Pure.

v1 trades at most a few lots on liquid MOEX names, a tiny fraction of any 1-min
bar's volume. Market impact in this regime is dominated by the bid-ask spread,
so the model is: VWAP of the first K 1-min sub-candles, plus a half-spread floor,
plus a small square-root impact term scaled by *daily* volatility.

The point is not to predict fills exactly but to prevent the backtest from
producing unrealistically tight results by filling at a single auction print
with zero cost. Documented limitations live in SPEC 10.3.
"""
from __future__ import annotations

import math

import pandas as pd


def simulate_fill(
    qty: float,
    sub_candles: pd.DataFrame,
    sigma_daily: float,
    config: dict,
    lot_size: int,
) -> tuple[float, float, float]:
    """Return ``(filled_qty, fill_price, fees)`` for a signed ``qty`` (lots).

    ``sub_candles`` are the 1-min candles covering the bar during which the fill
    occurs; only the first ``execution_window_minutes`` (K) are worked.
    ``sigma_daily`` is the ex-ante daily-return volatility valid at this bar.

    No liquidity in the window (zero volume or no sub-candles) yields a zero
    fill. Oversized orders are partially filled and the remainder is dropped
    (not carried forward), per SPEC 10.3 step 4."""
    k = int(config["execution_window_minutes"])
    window = sub_candles.iloc[:k]
    total_vol = float(window["volume"].sum()) if len(window) else 0.0
    if total_vol <= 0.0:
        return 0.0, float("nan"), 0.0

    typical = (window["high"] + window["low"] + window["close"]) / 3.0
    vwap = float((typical * window["volume"]).sum() / total_vol)

    available = float(config["participation_rate"]) * total_vol
    filled = qty
    if abs(qty) > available:
        filled = math.copysign(available, qty)

    # Conservative (wide) half-spread from the max per-candle 1-min range.
    range_bps = float(((window["high"] - window["low"]) / window["close"] * 10_000).max())
    spread_bps = max(float(config["min_spread_bps"]), range_bps * float(config["spread_to_range_ratio"]))
    half_spread_bps = spread_bps / 2.0

    # Square-root impact scaled by daily volatility (small at v1 sizes).
    participation = abs(filled) / available if available > 0 else 0.0
    sigma_bps = (sigma_daily if sigma_daily == sigma_daily else 0.0) * 10_000  # NaN -> 0
    impact_bps = float(config["impact_coeff"]) * sigma_bps * math.sqrt(participation)

    slippage_bps = half_spread_bps + impact_bps
    fill_price = vwap * (1.0 + math.copysign(1.0, qty) * slippage_bps / 10_000)

    notional = abs(filled) * fill_price * lot_size
    fees = max(float(config["min_commission_per_order"]), float(config["commission_bps"]) * notional / 10_000)
    return float(filled), float(fill_price), float(fees)


def compute_sigma_daily(candles_1day: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Return a ``(date, sigma_daily)`` frame: rolling std of daily close-to-close
    returns. ``sigma_daily`` for a given date uses returns up to that date's close
    and is therefore made causal for intraday bars by the strictly-backward merge
    in the engine (SPEC 10.3)."""
    d = candles_1day.sort_values("timestamp").copy()
    d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.normalize()
    daily = d.groupby("date", as_index=False)["close"].last()
    daily["ret"] = daily["close"].pct_change()
    daily["sigma_daily"] = daily["ret"].rolling(window_days).std()
    return daily[["date", "sigma_daily"]]

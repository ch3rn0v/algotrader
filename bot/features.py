"""Feature engineering for the LightGBM model.

Merges OHLCV-derived features from all (asset, timeframe) pairs onto the
primary 5m index without lookahead (see build_features), then adds
cross-series features (trend agreement, market-wide volume anomalies).
"""
import numpy as np
import pandas as pd

from config import PRIMARY_ASSET, PRIMARY_TF

TF_DURATIONS = {
    "1min": pd.Timedelta(minutes=1),
    "5min": pd.Timedelta(minutes=5),
    "15min": pd.Timedelta(minutes=15),
    "30min": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
}


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    # 100*gain/(gain+loss) == classic RSI; flat window (0/0) is neutral 50.
    rsi = 100 * gain / (gain + loss)
    return rsi.where((gain + loss) > 0, 50.0).where(gain.notna())


def _raw_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute per-bar features for one (asset, timeframe) candle DataFrame."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    out = pd.DataFrame({"timestamp": df["timestamp"]})

    def put(name: str, values) -> None:
        out.loc[:, f"{prefix}_{name}"] = values

    # --- returns / range ---
    put("ret1", c.pct_change(1))
    put("ret3", c.pct_change(3))
    put("ret8", c.pct_change(8))
    put("ret20", c.pct_change(20))
    put("spread", (h - l) / c)

    # --- candle anatomy ---
    # Zero-range (doji) bars are common on thin series; use neutral values
    # instead of NaN so they don't knock out whole rows at dropna.
    bar_range = h - l
    has_range = bar_range > 0
    put("body", (c - o) / c)
    put("upper_wick", (h - np.maximum(o, c)) / c)
    put("lower_wick", (np.minimum(o, c) - l) / c)
    put("body_frac", ((c - o).abs() / bar_range).where(has_range, 0.0))
    put("close_pos", ((c - l) / bar_range).where(has_range, 0.5))
    put("dir", np.sign(c - o))
    put("gap", o / c.shift(1) - 1)

    # --- trend / breakout ---
    ema5 = c.ewm(span=5, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    put("trend", ema5 / ema20 - 1)
    hi20 = h.rolling(20, min_periods=20).max()
    lo20 = l.rolling(20, min_periods=20).min()
    put("brk_hi20", c / hi20 - 1)
    put("brk_lo20", c / lo20 - 1)
    put("new_hi20", (c > hi20.shift(1)).astype(float))
    put("new_lo20", (c < lo20.shift(1)).astype(float))
    put("rsi14", _rsi(c) / 100 - 0.5)
    put("volat20", c.pct_change(1).rolling(20, min_periods=20).std())

    # --- volume ---
    vol_ma = v.rolling(20, min_periods=1).mean().replace(0, np.nan)
    put("vol_norm", v / vol_ma)
    vol_std = v.rolling(20, min_periods=20).std()
    vol_z = (v - vol_ma) / vol_std
    put("vol_z", vol_z.where(vol_std > 0, 0.0).where(vol_std.notna()))
    vol_ma5 = v.rolling(5, min_periods=1).mean()
    put("vol_trend", vol_ma5 / vol_ma - 1)

    return out


def _cross_features(result: pd.DataFrame, prefixes: list[str], primary_prefix: str) -> pd.DataFrame:
    """Row-wise features across series. Safe because every input column is
    already lookahead-free after the merge."""
    assets = sorted({p.rsplit("_", 1)[0] for p in prefixes})
    tfs = sorted({p.rsplit("_", 1)[1] for p in prefixes})

    # Trend/direction agreement of each series with the primary series.
    prim_trend = np.sign(result[f"{primary_prefix}_trend"])
    for p in prefixes:
        if p == primary_prefix:
            continue
        result.loc[:, f"x_{p}_trend_match"] = prim_trend * np.sign(result[f"{p}_trend"])

    # Market-wide unusual volume: mean vol_z across assets at each timeframe.
    for tf in tfs:
        cols = [f"{a}_{tf}_vol_z" for a in assets if f"{a}_{tf}_vol_z" in result.columns]
        if len(cols) > 1:
            result.loc[:, f"x_volz_mkt_{tf}"] = result[cols].mean(axis=1)

    # Per-asset unusual volume agreement across timeframes.
    for a in assets:
        cols = [f"{a}_{tf}_vol_z" for tf in tfs if f"{a}_{tf}_vol_z" in result.columns]
        if len(cols) > 1:
            result.loc[:, f"x_volz_{a}_tfs"] = result[cols].mean(axis=1)

    return result


def build_features(
    all_candles: dict,
    primary_asset: str = PRIMARY_ASSET,
    primary_tf: str = PRIMARY_TF,
) -> pd.DataFrame:
    """Merge features from all (asset, timeframe) pairs onto the primary index.

    `primary_asset`/`primary_tf` default to the config values; pass a trained
    model's own primary when building features for it.

    Primary series: feature values are shifted by one bar so that at
    decision time (start of bar t) only bar t-1 data is visible. Target is close[t]/close[t-1].

    All other series: merged on bar END timestamp (start + duration). This ensures
    only fully completed bars are visible at each decision point. A bar ending
    exactly at a 5m bar's start counts as complete (merge uses <=).

    The result has exactly one primary `timestamp` column. Each non-primary series
    also contributes a `{prefix}_ts` column recording which source bar was matched.
    """
    base = all_candles[(primary_asset, primary_tf)].sort_values("timestamp").reset_index(drop=True)
    result = base[["timestamp"]].copy()
    prefixes = []

    for (asset, tf), candles in all_candles.items():
        prefix = f"{asset}_{tf.replace('min', 'm')}"
        prefixes.append(prefix)
        df = candles.sort_values("timestamp").reset_index(drop=True)
        feats = _raw_features(df, prefix)

        if asset == primary_asset and tf == primary_tf:
            # Target is close[t]/close[t-1], so features must only use bar t-1 data.
            # Shift feature values forward by one bar; timestamp stays as the merge key.
            feat_cols = [c for c in feats.columns if c != "timestamp"]
            feats_prev = feats[["timestamp"]].copy()
            feats_prev.loc[:, feat_cols] = feats[feat_cols].shift(1).values
            result = pd.merge_asof(result, feats_prev, on="timestamp", direction="backward")
        else:
            # Replace timestamp with bar end so merge_asof only matches completed bars.
            # {prefix}_ts retains the original bar start for traceability.
            feats_merge = feats.copy()
            feats_merge.loc[:, f"{prefix}_ts"] = feats["timestamp"]
            feats_merge.loc[:, "timestamp"] = feats["timestamp"] + TF_DURATIONS[tf]
            result = pd.merge_asof(result, feats_merge, on="timestamp", direction="backward")

    primary_prefix = f"{primary_asset}_{primary_tf.replace('min', 'm')}"
    result = _cross_features(result, prefixes, primary_prefix)

    return result

"""Feature engineering for the LightGBM model.

Merges OHLCV-derived features from all (asset, timeframe) pairs onto the
primary 5m index without lookahead (see build_features).
"""
import numpy as np
import pandas as pd

from config import PRIMARY_ASSET, PRIMARY_TF

TF_DURATIONS = {
    "5min": pd.Timedelta(minutes=5),
    "15min": pd.Timedelta(minutes=15),
    "30min": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
}


def _raw_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute per-bar features for one (asset, timeframe) candle DataFrame."""
    out = pd.DataFrame({"timestamp": df["timestamp"]})
    out.loc[:, f"{prefix}_ret1"] = df["close"].pct_change(1)
    out.loc[:, f"{prefix}_ret3"] = df["close"].pct_change(3)
    out.loc[:, f"{prefix}_spread"] = (df["high"] - df["low"]) / df["close"]
    vol_ma = df["volume"].rolling(20, min_periods=1).mean().replace(0, np.nan)
    out.loc[:, f"{prefix}_vol_norm"] = df["volume"] / vol_ma
    return out


def build_features(all_candles: dict) -> pd.DataFrame:
    """Merge features from all (asset, timeframe) pairs onto the primary 5m index.

    Primary series (SBERP 5m): feature values are shifted by one bar so that at
    decision time (start of bar t) only bar t-1 data is visible. Target is close[t]/close[t-1].

    All other series: merged on bar END timestamp (start + duration). This ensures
    only fully completed bars are visible at each decision point. A bar ending
    exactly at a 5m bar's start counts as complete (merge uses <=).

    The result has exactly one primary `timestamp` column. Each non-primary series
    also contributes a `{prefix}_ts` column recording which source bar was matched.
    """
    base = all_candles[(PRIMARY_ASSET, PRIMARY_TF)].sort_values("timestamp").reset_index(drop=True)
    result = base[["timestamp"]].copy()

    for (asset, tf), candles in all_candles.items():
        prefix = f"{asset}_{tf.replace('min', 'm')}"
        df = candles.sort_values("timestamp").reset_index(drop=True)
        feats = _raw_features(df, prefix)

        if asset == PRIMARY_ASSET and tf == PRIMARY_TF:
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

    return result

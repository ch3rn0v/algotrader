"""Train a LightGBM model to predict the current 5-min bar's return for SBERP.

Features: OHLCV-derived features for SBERP, TMOS, PLZL at 5m, 15m, 30m, 1h timeframes.
Target:   close[t] / close[t-1] — return of the current 5-min bar.
Metric:   Pearson correlation (np.corrcoef) between predicted and actual return.

Usage (from repo root, with venv activated):
    python3 bot/train_lgbm.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd

from candles import get_candles

# ---------------------------------------------------------------------------
# Config — verify FIGIs at tbank.ru/invest before use
# ---------------------------------------------------------------------------

ASSETS = {
    "SBERP": "BBG0047315Y7",
    "TMOS": "TCSM61901X76",  # T's proxy for Moscow Exchange index
    "PLZL": "BBG000R607Y3",  # Polyus Gold
}
PRIMARY_ASSET = "SBERP"
PRIMARY_TF = "5min"
TIMEFRAMES = ["5min", "15min", "30min", "1h"]

FROM = datetime(2025, 1, 1, tzinfo=timezone.utc)
TO = datetime(2026, 1, 1, tzinfo=timezone.utc)

TRAIN_RATIO = 0.3  # first 30% for training, last 70% for test

MODEL_DIR = Path(__file__).parent / "outputs" / "models"

max_depth = 4
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 200,
    "max_depth": max_depth,
    "learning_rate": 0.05,
    "num_leaves": 2**max_depth,
    "random_state": 0,
    "verbosity": -1,
}

TF_DURATIONS = {
    "5min": pd.Timedelta(minutes=5),
    "15min": pd.Timedelta(minutes=15),
    "30min": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # 1. Determine required (asset, timeframe) pairs
    required = [(asset, tf) for asset in ASSETS for tf in TIMEFRAMES]
    print(f"Required candle series: {len(required)}")
    for asset, tf in required:
        print(f"  {asset} {tf}  (FIGI: {ASSETS[asset]})")

    # 2. Download / load candles (cache is handled by get_candles)
    print("\nLoading candles...")
    all_candles = {}
    for asset, tf in required:
        df = get_candles(ASSETS[asset], tf, FROM, TO)
        all_candles[(asset, tf)] = df
        print(f"  {asset} {tf}: {len(df)} bars")

    # 3. Build feature matrix
    print("\nBuilding features...")
    features = build_features(all_candles)
    primary = all_candles[(PRIMARY_ASSET, PRIMARY_TF)].sort_values("timestamp").reset_index(drop=True)
    features.loc[:, "target"] = (primary["close"] / primary["close"].shift(1)).values

    n_before = len(features)

    # Drop columns that are entirely NaN — these come from an asset/timeframe with no
    # data in the requested date range (e.g. wrong FIGI). Warn so the user can fix it.
    all_null_cols = [c for c in features.columns if features[c].isna().all()]
    if all_null_cols:
        print(f"WARNING: Dropping {len(all_null_cols)} entirely-null columns (asset has no data):")
        print(f"  {all_null_cols}")
        features = features.drop(columns=all_null_cols)

    features = features.dropna().reset_index(drop=True)
    print(f"Rows after dropna: {len(features)} / {n_before} (dropped {n_before - len(features)})")

    if len(features) == 0:
        partial_null = {c: int(features.isnull().sum()[c]) for c in features.columns}
        raise RuntimeError(
            f"Feature matrix is empty after dropna. "
            f"Per-column null counts: {partial_null}. "
            f"Check that the primary asset has data in the requested date range."
        )

    # Exclude primary timestamp, per-series source timestamps, and target from features.
    ts_cols = {"timestamp"} | {c for c in features.columns if c.endswith("_ts")}
    feature_cols = [c for c in features.columns if c not in ts_cols and c != "target"]
    print(f"Feature matrix: {len(features)} rows × {len(feature_cols)} features")

    X = features[feature_cols]
    y = features["target"].to_numpy()

    # 4. Temporal train/test split — no shuffling
    split = int(len(features) * TRAIN_RATIO)
    if split == 0:
        raise RuntimeError(f"TRAIN_RATIO={TRAIN_RATIO} yields 0 training rows from {len(features)} total. " f"Increase TRAIN_RATIO or extend the date range.")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"\nTrain: {len(X_train)} rows  ({features['timestamp'].iloc[0]} → {features['timestamp'].iloc[split - 1]})")
    print(f"Test:  {len(X_test)} rows  ({features['timestamp'].iloc[split]} → {features['timestamp'].iloc[-1]})")

    # 5. Train
    print(f"\nTraining LightGBM ({LGBM_PARAMS['n_estimators']} trees, max_depth={LGBM_PARAMS['max_depth']})...")
    t0 = time.time()
    model = lgbm.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # 6. Evaluate
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    corr_train = float(np.corrcoef(y_train, pred_train)[0, 1])
    corr_test = float(np.corrcoef(y_test, pred_test)[0, 1])
    print(f"\nTraining time:  {train_time:.1f}s")
    print(f"Corr (train):   {corr_train:.4f}")
    print(f"Corr (test):    {corr_test:.4f}")

    # 7. Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"lgbm_{ts}.txt"
    meta_path = MODEL_DIR / f"lgbm_{ts}_meta.json"

    model.booster_.save_model(str(model_path))

    # Verify: reload saved model and compare predictions against originals.
    loaded = lgbm.Booster(model_file=str(model_path))
    pred_test_loaded = loaded.predict(X_test)
    n_diff = int(np.sum(pred_test != pred_test_loaded))
    if n_diff > 0:
        corr_test_loaded = float(np.corrcoef(y_test, pred_test_loaded)[0, 1])
        mean_pct_diff = float(np.mean(np.abs((pred_test_loaded - pred_test) / np.abs(pred_test))) * 100)
        raise RuntimeError(
            f"Model save/load mismatch: {n_diff}/{len(pred_test)} predictions differ, "
            f"corr_test original={corr_test:.6f} loaded={corr_test_loaded:.6f}, "
            f"mean abs % diff={mean_pct_diff:.4f}%"
        )
    print("Model save/load: verified OK")

    meta = {
        "timestamp": ts,
        "primary_asset": PRIMARY_ASSET,
        "primary_tf": PRIMARY_TF,
        "assets": ASSETS,
        "timeframes": TIMEFRAMES,
        "from": FROM.isoformat(),
        "to": TO.isoformat(),
        "train_ratio": TRAIN_RATIO,
        "train_end_ts": str(features["timestamp"].iloc[split - 1]),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_cols": feature_cols,
        "lgbm_params": LGBM_PARAMS,
        "corr_train": round(corr_train, 6),
        "corr_test": round(corr_test, 6),
        "train_time_s": round(train_time, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved model:    {model_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()

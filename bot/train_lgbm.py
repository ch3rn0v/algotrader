"""Train a LightGBM model to predict the current 5-min bar's return for SBERP.

Features: OHLCV-derived features for SBERP, TMOS, PLZL at 5m, 15m, 30m, 1h timeframes.
Target:   close[t] / close[t-1] — return of the current 5-min bar.
Metric:   Pearson correlation (np.corrcoef) between predicted and actual return.

Usage (from repo root, with venv activated):
    python3 bot/train_lgbm.py
"""

import json
import time
from datetime import datetime

import lightgbm as lgbm
import numpy as np

from candles import load_all_candles
from config import ASSETS, FROM, MODEL_DIR, PRIMARY_ASSET, PRIMARY_TF, TIMEFRAMES, TO
from features import build_features

TRAIN_RATIO = 0.3  # first 30% for training, last 70% for test

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


def main():
    # 1. Download / load candles (cache is handled by get_candles)
    print(f"Loading {len(ASSETS) * len(TIMEFRAMES)} candle series...")
    all_candles = load_all_candles(FROM, TO)

    # 2. Build feature matrix
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
        raise RuntimeError(
            "Feature matrix is empty after dropna. "
            "Check that the primary asset has data in the requested date range."
        )

    # Exclude primary timestamp, per-series source timestamps, and target from features.
    ts_cols = {"timestamp"} | {c for c in features.columns if c.endswith("_ts")}
    feature_cols = [c for c in features.columns if c not in ts_cols and c != "target"]
    print(f"Feature matrix: {len(features)} rows × {len(feature_cols)} features")

    X = features[feature_cols]
    y = features["target"].to_numpy()

    # 3. Temporal train/test split — no shuffling
    split = int(len(features) * TRAIN_RATIO)
    if split == 0:
        raise RuntimeError(
            f"TRAIN_RATIO={TRAIN_RATIO} yields 0 training rows from {len(features)} total. "
            f"Increase TRAIN_RATIO or extend the date range."
        )
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"\nTrain: {len(X_train)} rows  ({features['timestamp'].iloc[0]} → {features['timestamp'].iloc[split - 1]})")
    print(f"Test:  {len(X_test)} rows  ({features['timestamp'].iloc[split]} → {features['timestamp'].iloc[-1]})")

    # 4. Train
    print(f"\nTraining LightGBM ({LGBM_PARAMS['n_estimators']} trees, max_depth={LGBM_PARAMS['max_depth']})...")
    t0 = time.time()
    model = lgbm.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # 5. Evaluate
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    corr_train = float(np.corrcoef(y_train, pred_train)[0, 1])
    corr_test = float(np.corrcoef(y_test, pred_test)[0, 1])
    print(f"\nTraining time:  {train_time:.1f}s")
    print(f"Corr (train):   {corr_train:.4f}")
    print(f"Corr (test):    {corr_test:.4f}")

    # 6. Save model
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

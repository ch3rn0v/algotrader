"""Train a LightGBM model to predict the current primary-bar return for SBERP.

Pipeline:
  1. base features for all (asset, timeframe) pairs (features.py)
  2. base-feature prefilter: drop features with |corr(train target)| below
     --min-base-corr before any pairwise generation
  3. iterative pairwise feature generation, fit on train only (feature_generator.py)
  4. greedy correlation-based feature selection on train only (feature_selector.py)
  5. optuna hyperparameter tuning on a train/validation split (tuner.py)
  6. final fit on the full train set, evaluation on the held-out test set

Target: close[t] / close[t-1] — return of the current primary bar (--tf).
Metric: Pearson correlation between predicted and actual return.

Usage (from repo root, with venv activated):
    python3 bot/train_lgbm.py                     # full pipeline, config defaults
    python3 bot/train_lgbm.py --tf 15min --timeframes 5min,15min,30min,1h
    python3 bot/train_lgbm.py --gen-iterations 0  # skip feature generation
    python3 bot/train_lgbm.py --trials 0          # skip tuning, use default params
"""

import argparse
import json
import time
from datetime import datetime, timezone

import lightgbm as lgbm
import numpy as np

from candles import load_all_candles
from config import ASSETS, FROM, MODEL_DIR, PRIMARY_ASSET, PRIMARY_TF, TIMEFRAMES, TO
from feature_generator import apply_recipes, generate_features
from feature_selector import select_features
from features import build_features
from tuner import tune

TRAIN_RATIO = 0.3  # first 30% for training, last 70% for test

max_depth = 4
DEFAULT_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 200,
    "max_depth": max_depth,
    "learning_rate": 0.05,
    "num_leaves": 2**max_depth,
    "random_state": 0,
    "verbosity": -1,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train the SBERP return model")
    p.add_argument("--tf", default=PRIMARY_TF, help=f"primary timeframe to train on (default {PRIMARY_TF})")
    p.add_argument("--timeframes", default=",".join(TIMEFRAMES),
                   help="comma-separated timeframes to build features from (default: config TIMEFRAMES)")
    p.add_argument("--from", dest="date_from", default=None, metavar="DATE", help="start date YYYY-MM-DD (default: config FROM)")
    p.add_argument("--to", dest="date_to", default=None, metavar="DATE", help="end date YYYY-MM-DD (default: config TO)")
    p.add_argument("--min-base-corr", type=float, default=0.05,
                   help="drop base features with |corr(train target)| below this before generation (0 = keep all, default 0.05)")
    p.add_argument("--gen-iterations", type=int, default=2, help="feature generator iterations (0 = skip, default 2)")
    p.add_argument("--min-target-corr", type=float, default=0.02, help="generator: discard candidates below this |corr| with target (default 0.02)")
    p.add_argument("--gen-max-new", type=int, default=200, help="generator: max new features kept per iteration (default 200)")
    p.add_argument("--max-features", type=int, default=300, help="selector: max selected features (default 300)")
    p.add_argument("--max-inter-corr", type=float, default=0.9, help="selector: max |corr| between selected features (default 0.9)")
    p.add_argument("--trials", type=int, default=30, help="optuna trials (0 = skip tuning, default 30)")
    return p.parse_args()


def main():
    args = parse_args()
    tf = args.tf
    tfs = args.timeframes.split(",")
    from_dt = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.date_from else FROM
    to_dt = datetime.strptime(args.date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.date_to else TO

    # 1. Download / load candles (cache is handled by get_candles)
    print(f"Loading {len(ASSETS) * len(tfs)} candle series...")
    all_candles = load_all_candles(from_dt, to_dt, timeframes=tfs)

    # 2. Build feature matrix
    print("\nBuilding features...")
    features = build_features(all_candles, primary_tf=tf)
    primary = all_candles[(PRIMARY_ASSET, tf)].sort_values("timestamp").reset_index(drop=True)
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
    base_cols = [c for c in features.columns if c not in ts_cols and c != "target"]
    print(f"Feature matrix: {len(features)} rows × {len(base_cols)} base features")

    y = features["target"].to_numpy()

    # 3. Temporal train/test split — no shuffling
    split = int(len(features) * TRAIN_RATIO)
    if split == 0:
        raise RuntimeError(
            f"TRAIN_RATIO={TRAIN_RATIO} yields 0 training rows from {len(features)} total. "
            f"Increase TRAIN_RATIO or extend the date range."
        )
    y_train, y_test = y[:split], y[split:]
    print(f"\nTrain: {split} rows  ({features['timestamp'].iloc[0]} → {features['timestamp'].iloc[split - 1]})")
    print(f"Test:  {len(features) - split} rows  ({features['timestamp'].iloc[split]} → {features['timestamp'].iloc[-1]})")

    # 4. Base-feature prefilter — train rows only. Drops unpromising features
    # before the (quadratic) pairwise generator sees them.
    n_base_total = len(base_cols)
    if args.min_base_corr > 0:
        mat = features[base_cols].iloc[:split].to_numpy(dtype=float)
        mat = mat - mat.mean(axis=0)
        yc = y_train - y_train.mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.abs(mat.T @ yc) / (mat.std(axis=0) * yc.std() * split)
        corr[~np.isfinite(corr)] = 0.0  # zero-variance columns
        base_cols = [c for c, v in zip(base_cols, corr) if v >= args.min_base_corr]
        print(f"\nBase-feature filter: kept {len(base_cols)} / {n_base_total} "
              f"with |corr| >= {args.min_base_corr}")
        if not base_cols:
            raise RuntimeError(
                f"No base feature passes |corr| >= {args.min_base_corr}; "
                f"lower --min-base-corr."
            )

    # 5. Feature generation — fit on train rows only, then applied to all rows.
    recipes = []
    if args.gen_iterations > 0:
        print("\nGenerating features (train data only)...")
        t0 = time.time()
        recipes = generate_features(
            features.iloc[:split],
            y_train,
            base_cols,
            n_iterations=args.gen_iterations,
            min_target_corr=args.min_target_corr,
            max_new_per_iter=args.gen_max_new,
        )
        features = apply_recipes(features, recipes)
        print(f"Feature generation: {len(recipes)} new features in {time.time() - t0:.1f}s")

    all_cols = base_cols + [r["name"] for r in recipes]

    # 6. Feature selection — train rows only.
    print("\nSelecting features (train data only)...")
    feature_cols = select_features(
        features.iloc[:split],
        y_train,
        all_cols,
        max_features=args.max_features,
        max_inter_corr=args.max_inter_corr,
    )

    X = features[feature_cols]
    X_train, X_test = X.iloc[:split], X.iloc[split:]

    # 7. Hyperparameter tuning on a temporal train/validation split.
    if args.trials > 0:
        print("\nTuning hyperparameters...")
        lgbm_params = tune(X_train, y_train, n_trials=args.trials)
    else:
        lgbm_params = dict(DEFAULT_LGBM_PARAMS)

    # 8. Final fit on the full train set.
    print(f"\nTraining LightGBM ({lgbm_params['n_estimators']} trees, max_depth={lgbm_params['max_depth']})...")
    t0 = time.time()
    model = lgbm.LGBMRegressor(**lgbm_params)
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # 9. Evaluate
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    corr_train = float(np.corrcoef(y_train, pred_train)[0, 1])
    corr_test = float(np.corrcoef(y_test, pred_test)[0, 1])
    print(f"\nTraining time:  {train_time:.1f}s")
    print(f"Corr (train):   {corr_train:.4f}")
    print(f"Corr (test):    {corr_test:.4f}")

    # 10. Save model
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
        "primary_tf": tf,
        "assets": ASSETS,
        "timeframes": tfs,
        "from": from_dt.isoformat(),
        "to": to_dt.isoformat(),
        "train_ratio": TRAIN_RATIO,
        "train_end_ts": str(features["timestamp"].iloc[split - 1]),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_base_features": n_base_total,
        "min_base_corr": args.min_base_corr,
        "n_base_kept": len(base_cols),
        "gen_iterations": args.gen_iterations,
        "gen_recipes": recipes,
        "feature_cols": feature_cols,
        "tuning_trials": args.trials,
        "lgbm_params": lgbm_params,
        "corr_train": round(corr_train, 6),
        "corr_test": round(corr_test, 6),
        "train_time_s": round(train_time, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved model:    {model_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()

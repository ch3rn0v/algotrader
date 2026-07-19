"""Load the latest trained LightGBM model and build predictions for a candle series."""
import json

import lightgbm as lgbm
import numpy as np
import pandas as pd

from candles import load_all_candles
from config import MODEL_DIR, PRIMARY_ASSET, PRIMARY_FIGI, PRIMARY_TF
from feature_generator import apply_recipes
from features import build_features


def load_latest_model() -> tuple[lgbm.Booster, dict] | None:
    """Return (booster, meta) for the newest model in MODEL_DIR, or None."""
    meta_files = sorted(MODEL_DIR.glob("lgbm_*_meta.json"))
    if not meta_files:
        return None
    meta_path = meta_files[-1]
    model_path = MODEL_DIR / meta_path.name.replace("_meta.json", ".txt")
    meta = json.loads(meta_path.read_text())
    booster = lgbm.Booster(model_file=str(model_path))
    print(f"Using model: {model_path.name}")
    return booster, meta


def build_predictions(
    candles: pd.DataFrame,
    figi: str,
    timeframe: str,
    from_dt,
    to_dt,
    fill: float = np.nan,
    required: bool = False,
) -> tuple[np.ndarray | None, dict | None]:
    """Return (predictions, meta) with one prediction per row of `candles`.

    Loads the latest saved model, fetches the other (asset, timeframe) series
    over [from_dt, to_dt], builds features, and aligns predictions to `candles`
    by timestamp. Bars without features get `fill` (nan and 1.0 are both
    neutral for the backtest: neither passes an entry threshold).

    Returns (None, None) when no model exists, the instrument/timeframe has no
    model, or feature columns are missing — or raises if `required` is True.
    """
    def fail(msg: str, exc: type[Exception]) -> tuple[None, None]:
        if required:
            raise exc(msg)
        print(f"{msg} Running without predictions.")
        return None, None

    loaded = load_latest_model()
    if loaded is None:
        return fail(f"No model found in {MODEL_DIR}.", FileNotFoundError)
    booster, meta = loaded
    model_asset = meta.get("primary_asset", PRIMARY_ASSET)
    model_tf = meta.get("primary_tf", PRIMARY_TF)
    model_figi = meta.get("assets", {}).get(model_asset, PRIMARY_FIGI)
    if figi != model_figi or timeframe != model_tf:
        return fail("No model for this instrument/timeframe.", RuntimeError)

    # Load exactly the (asset, timeframe) universe the model was trained on,
    # which may differ from the current config.
    all_candles = load_all_candles(
        from_dt, to_dt, primary=candles,
        assets=meta.get("assets"), timeframes=meta.get("timeframes"),
        primary_key=(model_asset, model_tf),
    )
    features = build_features(all_candles, primary_asset=model_asset, primary_tf=model_tf,
                              extended_assets=meta.get("extended_assets"))
    features = features.drop(columns=[c for c in features.columns if features[c].isna().all()])
    features = apply_recipes(features, meta.get("gen_recipes", []))

    missing = [c for c in meta["feature_cols"] if c not in features.columns]
    if missing:
        return fail(f"Feature columns missing from current data: {missing}.", RuntimeError)

    preds = booster.predict(features[meta["feature_cols"]])
    pred_map = dict(zip(features["timestamp"].values, preds))
    predictions = np.fromiter(
        (pred_map.get(ts, fill) for ts in candles["timestamp"].values),
        dtype=float,
        count=len(candles),
    )
    print(f"Predictions: {len(predictions)} bars, mean={np.nanmean(predictions):.5f}")
    return predictions, meta

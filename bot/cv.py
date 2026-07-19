"""Time-series (forward-chaining) cross-validation for the return model.

Folds never let a validation block precede its training rows in time, so the
score is an honest estimate of out-of-sample skill within the train range.
Metric is Pearson correlation between predicted and actual returns, matching
the rest of the pipeline.
"""
import lightgbm as lgbm
import numpy as np
import pandas as pd


def timed_folds(n: int, k: int):
    """Yield k (train_idx, val_idx) pairs with an expanding train window.

    The series is cut into k+1 equal blocks; fold i trains on blocks [0..i]
    and validates on block i+1 (the last fold's val block runs to the end, so
    every row after the first block is validated exactly once)."""
    if k < 2:
        raise ValueError(f"need at least 2 folds, got {k}")
    block = n // (k + 1)
    if block == 0:
        raise ValueError(f"{n} rows too few for {k} folds")
    for i in range(1, k + 1):
        tr_end = block * i
        val_end = n if i == k else block * (i + 1)
        yield np.arange(0, tr_end), np.arange(tr_end, val_end)


def cv_corr(X: pd.DataFrame, y: np.ndarray, params: dict, k: int = 5) -> np.ndarray:
    """Per-fold Pearson corr of predicted vs actual returns over k timed folds."""
    corrs = []
    for tr, val in timed_folds(len(X), k):
        model = lgbm.LGBMRegressor(**params)
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[val])
        c = np.corrcoef(y[val], pred)[0, 1]
        corrs.append(c if np.isfinite(c) else -1.0)
    return np.array(corrs)

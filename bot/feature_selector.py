"""Greedy recursive feature selector.

On each step pick the remaining feature with the highest |corr| with the
target, skipping any whose |corr| with an already-selected feature is
max_inter_corr or higher. Stops when every feature has been considered or
max_features are selected. Uses TRAIN data only.
"""
import numpy as np
import pandas as pd


def _standardize(mat: np.ndarray) -> np.ndarray:
    """Column-standardize with NaNs replaced by the column mean (so they
    contribute nothing to correlations)."""
    mean = np.nanmean(mat, axis=0)
    mat = np.where(np.isfinite(mat), mat, mean)
    std = mat.std(axis=0)
    std[std == 0] = np.nan
    return (mat - mean) / std


def select_features(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    feature_cols: list[str],
    max_features: int = 300,
    max_inter_corr: float = 0.9,
) -> list[str]:
    """Return the selected feature names, in selection order."""
    y = np.asarray(y_train, dtype=float)
    mat = _standardize(train_df[feature_cols].to_numpy(dtype=float))
    n = mat.shape[0]

    y_std = (y - y.mean()) / y.std()
    target_corr = np.abs(mat.T @ y_std) / n
    target_corr[~np.isfinite(target_corr)] = 0.0  # zero-variance columns

    order = np.argsort(-target_corr)
    selected_idx: list[int] = []
    for idx in order:
        if len(selected_idx) >= max_features:
            break
        if target_corr[idx] == 0.0:
            break  # everything after is constant/degenerate
        if selected_idx:
            inter = np.abs(mat[:, selected_idx].T @ mat[:, idx]) / n
            if np.nanmax(inter) >= max_inter_corr:
                continue
        selected_idx.append(int(idx))

    selected = [feature_cols[i] for i in selected_idx]
    print(f"[select] {len(selected)} / {len(feature_cols)} features "
          f"(max_features={max_features}, max_inter_corr={max_inter_corr})")
    return selected

"""Optuna hyperparameter tuning for the LightGBM return model.

The train set is split temporally into sub-train / validation; the objective
is the Pearson correlation between predicted and actual returns on the
validation part. A duplicate guard prunes any trial whose sampled params
exactly match an already-evaluated trial, so no parameter set is ever
trained twice within a study.
"""
import lightgbm as lgbm
import numpy as np
import optuna
import pandas as pd

VALID_RATIO = 0.2  # last 20% of train used for validation during tuning

FIXED_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "random_state": 0,
    "verbosity": -1,
    "subsample_freq": 1,  # so tuned `subsample` actually takes effect
}


def _suggest_params(trial: optuna.Trial) -> dict:
    max_depth = trial.suggest_int("max_depth", 3, 8)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 8, min(256, 2**max_depth)),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, step=10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0, step=0.05),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _is_duplicate(trial: optuna.Trial) -> bool:
    """True when an earlier trial already ran with exactly these params."""
    for t in trial.study.get_trials(deepcopy=False):
        if t.number < trial.number and t.params == trial.params and t.state in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.RUNNING,
        ):
            return True
    return False


def tune(X_train: pd.DataFrame, y_train: np.ndarray, n_trials: int = 30, seed: int = 0) -> dict:
    """Return the best LightGBM params (fixed + tuned) found in n_trials."""
    split = int(len(X_train) * (1 - VALID_RATIO))
    X_sub, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_sub, y_val = y_train[:split], y_train[split:]
    print(f"[tune] sub-train {len(X_sub)} rows, valid {len(X_val)} rows, {n_trials} trials")

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        if _is_duplicate(trial):
            raise optuna.TrialPruned("duplicate parameter set")
        model = lgbm.LGBMRegressor(**FIXED_PARAMS, **params)
        model.fit(X_sub, y_sub)
        pred = model.predict(X_val)
        corr = np.corrcoef(y_val, pred)[0, 1]
        return corr if np.isfinite(corr) else -1.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    print(f"[tune] best valid corr: {study.best_value:.4f} "
          f"({n_pruned} duplicate trials pruned)")
    print(f"[tune] best params: {study.best_params}")
    return {**FIXED_PARAMS, **study.best_trial.params}

"""Model-guided backtest runner.

Workflow:
  1. Load best params from the latest optimizer results CSV.
  2. Load the latest LightGBM model and generate predictions (no retraining).
  3. Isolate the test period (bars after train cutoff) and run the backtest.
  4. Report results and save chart.

Usage:
    python3 bot/runner.py                          # best params from optimizer
    python3 bot/runner.py --bb-alpha 0.05          # override specific params
    python3 bot/runner.py --bb-std 2.0 --session-end-utc 14
"""

import argparse
from datetime import datetime, timezone

import pandas as pd

from backtest_mean_rev_bb import run_backtest
from candles import get_candles
from charts import plot_results
from config import FROM, OUTPUT_DIR, PRIMARY_ASSET, PRIMARY_FIGI, PRIMARY_TF, TO
from model import build_predictions
from optimizer import RESULT_COLS

OUT_DIR = OUTPUT_DIR / "backtest"


def parse_args() -> dict:
    """Optional backtest param overrides; any provided value beats the optimizer's best."""
    parser = argparse.ArgumentParser(description="Run backtest with best optimizer params")
    parser.add_argument("--bb-alpha",             type=float)
    parser.add_argument("--bb-std",               type=float)
    parser.add_argument("--time-stop-bars",        type=int)
    parser.add_argument("--width-alpha",           type=float)
    parser.add_argument("--session-start-utc",     type=int)
    parser.add_argument("--session-end-utc",       type=int)
    parser.add_argument("--position-size",         type=int)
    parser.add_argument("--pred-long-threshold",   type=float)
    parser.add_argument("--pred-short-threshold",  type=float)
    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}


def main():
    # 1. Best params from the latest optimizer run, plus CLI overrides
    csv_path = OUTPUT_DIR / "optimizer" / f"results_{PRIMARY_FIGI}_{PRIMARY_TF}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No optimizer results at {csv_path}. Run optimizer.py first.")

    opt_df = pd.read_csv(csv_path)  # already sorted by sortino descending
    best = opt_df.iloc[0]
    bt_params = {k: best[k] for k in opt_df.columns if k not in RESULT_COLS}

    overrides = parse_args()
    bt_params.update(overrides)

    source = "optimizer" if not overrides else f"optimizer + CLI overrides: {list(overrides.keys())}"
    print(f"Params ({source}, sortino={best['sortino']:.4f}):")
    for k, v in bt_params.items():
        marker = " *" if k in overrides else ""
        print(f"  {k}: {v}{marker}")

    # 2. Candles and model predictions
    print("\nLoading candles and building predictions...")
    primary = get_candles(PRIMARY_FIGI, PRIMARY_TF, FROM, TO)
    predictions, meta = build_predictions(primary, PRIMARY_FIGI, PRIMARY_TF, FROM, TO)

    # 3. Backtest on the test period only (or the full range if there's no model)
    if predictions is not None:
        train_end_ts = pd.Timestamp(meta["train_end_ts"])
        print(f"Train cutoff: {train_end_ts}")
        test_mask = primary["timestamp"] > train_end_ts
        bt_candles = primary[test_mask].reset_index(drop=True)
        bt_predictions = predictions[test_mask.to_numpy()]
    else:
        bt_candles = primary
        bt_predictions = None

    if len(bt_candles) == 0:
        raise RuntimeError("Test period is empty. Check train_end_ts in model metadata.")

    print(f"\nTest period: {len(bt_candles)} bars")
    print(f"  from: {bt_candles['timestamp'].iloc[0]}")
    print(f"  to:   {bt_candles['timestamp'].iloc[-1]}")

    result = run_backtest(bt_candles, predictions=bt_predictions, **bt_params)

    equity, trades = result["equity"], result["trades"]
    peak_exposure = result["peak_exposure"]

    pnl = equity["equity"].iloc[-1] - equity["equity"].iloc[0]
    print(f"\nTrades:        {len(trades)}")
    print(f"Total P&L:     {pnl:+,.2f}")
    print(f"Peak exposure: {peak_exposure:,.2f}")
    if peak_exposure > 0:
        print(f"Return:        {pnl / peak_exposure * 100:+.2f}%")

    # 4. Save chart
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_results(
        bt_candles, equity, trades, peak_exposure,
        symbol=f"{PRIMARY_ASSET} (test)", timeframe=PRIMARY_TF,
        path=OUT_DIR / f"result_{ts}.png",
    )


if __name__ == "__main__":
    main()

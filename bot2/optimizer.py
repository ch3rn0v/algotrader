"""Grid-search optimiser for BB mean reversion parameters.

Usage (standalone):
    python optimizer.py

Or from code:
    from optimizer import optimize
    results_df = optimize(candles, param_grid={...}, fixed_params={...})
"""

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd

from backtest_mean_rev_bb import run_backtest

RESULT_COLS = ["pnl", "sharpe", "sortino", "max_dd", "cagr", "n_trades", "avg_bars_held", "turnover", "peak_exposure"]


def _run_one(args: tuple) -> dict:
    candles, params = args
    result = run_backtest(candles, **params, headless=True)
    return {**params, **result}


def optimize(
    candles: pd.DataFrame,
    param_grid: dict,
    fixed_params: Optional[dict] = None,
    n_jobs: Optional[int] = None,
    out_dir: Path = Path("optimizer_output"),
) -> pd.DataFrame:
    """
    Cartesian-product grid search, parallelised across CPU cores.

    Returns a styled DataFrame (background gradient per result column).
    Also writes optimizer_output/results.csv and results.html.

    Parameters
    ----------
    candles     : candle DataFrame as returned by get_candles()
    param_grid  : dict of {param_name: [values…]} to sweep
    fixed_params: params passed verbatim to every backtest call
    n_jobs      : worker processes (default: all CPUs)
    out_dir     : directory for CSV/HTML output
    """
    fixed = fixed_params or {}
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    all_params = [{**dict(zip(keys, c)), **fixed} for c in combos]

    n_workers = n_jobs or os.cpu_count()
    n_total = len(all_params)
    print(f"Grid: {n_total} combinations | {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_run_one, (candles, p)) for p in all_params]
        rows = []
        for i, fut in enumerate(futs, 1):
            rows.append(fut.result())
            print(f"\r  {i}/{n_total}", end="", flush=True)
    print()

    df = pd.DataFrame(rows)

    # Reorder: param cols first, then result cols
    param_cols = [c for c in df.columns if c not in RESULT_COLS]
    result_cols = [c for c in RESULT_COLS if c in df.columns]
    df = df[param_cols + result_cols]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)

    fmt = {
        "pnl": "{:+,.0f}",
        "sharpe": "{:.3f}",
        "sortino": "{:.3f}",
        "max_dd": "{:.3f}",
        "cagr": "{:.3f}",
        "n_trades": "{:.0f}",
        "avg_bars_held": "{:.1f}",
        "turnover": "{:,.0f}",
        "peak_exposure": "{:,.0f}",
    }
    styled = df.style.format({k: v for k, v in fmt.items() if k in df.columns}, na_rep="—").background_gradient(subset=result_cols, cmap="RdYlGn", axis=0)
    styled.to_html(out_dir / "results.html")
    print(f"Saved: {out_dir}/results.{{csv,html}}")

    return styled


if __name__ == "__main__":
    import os
    from datetime import datetime, timezone
    from candles import get_candles

    _env = Path(__file__).parent.parent / "trading_bot" / ".env"
    if _env.exists():
        for line in _env.read_text().splitlines():
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    candles = get_candles(
        "BBG0047315Y7",
        "5min",
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    print(f"Candles: {len(candles)}")

    styled = optimize(
        candles,
        param_grid={
            "bb_period": [10, 15, 20, 25, 30],
            "bb_std": [1.5, 2.0, 2.5, 3.0],
            "time_stop_bars": [6, 12, 18, 24],
            "session_end_utc": [12, 14, 16],
        },
        fixed_params={
            "session_start_utc": 9,
            "width_lookback": 50,
            "position_size": 1,
        },
        out_dir=Path(__file__).parent / "optimizer_output",
    )

    top = styled.data.sort_values("sharpe", ascending=False).head(10)
    print("\nTop 10 by Sharpe:")
    print(top.to_string(index=False))

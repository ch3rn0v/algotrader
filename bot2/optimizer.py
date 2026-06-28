"""Grid-search optimiser for BB mean reversion parameters.

Usage:
    source trading_bot/.venv/bin/activate

    # defaults: SBERP, 5min, 2025, all CPUs
    python3 bot2/optimizer.py

    # custom dates and instrument
    python3 bot2/optimizer.py --figi BBG004730N88 --timeframe 15min --from 2024-01-01 --to 2025-01-01

    # fewer workers, sort by PnL, show top 20
    python3 bot2/optimizer.py --jobs 4 --sort-by pnl --top 20

    # custom output dir
    python3 bot2/optimizer.py --out my_results

Or from code:
    from optimizer import optimize
    results_df = optimize(candles, param_grid={...}, fixed_params={...})
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_mean_rev_bb import run_backtest
from candles import get_candles

RESULT_COLS = ["pnl", "sharpe", "sortino", "max_dd", "cagr", "n_trades", "avg_bars_held", "turnover", "peak_exposure"]


def _run_one(args: tuple) -> dict:
    candles, params = args
    result = run_backtest(candles, **params, headless=True)
    return {**params, **result}


_FMT = {
    "pnl": lambda v: f"{v:+,.0f}",
    "sharpe": lambda v: f"{v:.3f}",
    "sortino": lambda v: f"{v:.3f}",
    "max_dd": lambda v: f"{v:.3f}",
    "cagr": lambda v: f"{v:.3f}",
    "n_trades": lambda v: f"{v:.0f}",
    "avg_bars_held": lambda v: f"{v:.1f}",
    "turnover": lambda v: f"{v:,.0f}",
    "peak_exposure": lambda v: f"{v:,.0f}",
}


def _render_table(df: pd.DataFrame, result_cols: list, path: Path) -> None:
    """Save a color-coded table as a PNG (RdYlGn per result column)."""
    cmap = plt.cm.RdYlGn
    norms = {col: (df[col].min(), df[col].max()) for col in result_cols if col in df.columns}

    def _color(col, val):
        if col not in norms or not np.isfinite(val):
            return (1.0, 1.0, 1.0, 1.0)
        mn, mx = norms[col]
        t = (val - mn) / (mx - mn) if mx > mn else 0.5
        return cmap(t)

    cell_text, cell_colors = [], []
    for _, row in df.iterrows():
        txt, clr = [], []
        for col in df.columns:
            val = row[col]
            fmt_fn = _FMT.get(col)
            txt.append(fmt_fn(val) if fmt_fn and pd.notna(val) else ("—" if pd.isna(val) else str(val)))
            clr.append(_color(col, val) if col in result_cols else (1.0, 1.0, 1.0, 1.0))
        cell_text.append(txt)
        cell_colors.append(clr)

    n_rows, n_cols = len(df), len(df.columns)
    fig, ax = plt.subplots(figsize=(max(12, n_cols * 1.4), max(4, n_rows * 0.28 + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        colLabels=list(df.columns),
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(range(n_cols))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def optimize(
    candles: pd.DataFrame,
    param_grid: dict,
    fixed_params: Optional[dict] = None,
    n_jobs: Optional[int] = None,
    out_dir: Path = Path("optimizer_output"),
    label: str = "",
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Cartesian-product grid search, parallelised across CPU cores.

    Returns a plain DataFrame sorted by sortino descending.
    Also writes results_{label}.csv and results_{label}.png to out_dir.
    The PNG shows the top_n rows with RdYlGn coloring per result column.

    Parameters
    ----------
    candles     : candle DataFrame as returned by get_candles()
    param_grid  : dict of {param_name: [values…]} to sweep
    fixed_params: params passed verbatim to every backtest call
    n_jobs      : worker processes (default: all CPUs)
    out_dir     : directory for output files
    label       : ticker+timeframe tag used in output filenames
    top_n       : rows shown in the PNG (default: 30)
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

    df = df.sort_values("sortino", ascending=False).reset_index(drop=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"results_{label}" if label else "results"
    df.to_csv(out_dir / f"{stem}.csv", index=False)

    display_df = df.head(top_n)
    _render_table(display_df, result_cols, out_dir / f"{stem}.png")
    print(f"Saved: {out_dir}/{stem}.{{csv,png}}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BB mean reversion grid optimiser")
    parser.add_argument("--figi", default="BBG0047315Y7", help="Instrument FIGI (default: SBERP)")
    parser.add_argument("--timeframe", default="5min", help="Candle timeframe (default: 5min)")
    parser.add_argument("--from", dest="date_from", default="2025-01-01", metavar="DATE", help="Start date YYYY-MM-DD (default: 2025-01-01)")
    parser.add_argument("--to", dest="date_to", default="2026-01-01", metavar="DATE", help="End date YYYY-MM-DD (default: 2026-01-01)")
    parser.add_argument("--jobs", type=int, default=None, help="Parallel workers (default: all CPUs)")
    parser.add_argument("--sort-by", default="sortino", help="Result column to sort top-N by (default: sortino)")
    parser.add_argument("--top", type=int, default=20, help="Rows to print (default: 20)")
    parser.add_argument("--out", default="optimizer_output", metavar="DIR", help="Output directory (default: optimizer_output)")
    args = parser.parse_args()

    _env = Path(__file__).parent.parent / "trading_bot" / ".env"
    if _env.exists():
        for line in _env.read_text().splitlines():
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    from_dt = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = datetime.strptime(args.date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    candles = get_candles(args.figi, args.timeframe, from_dt, to_dt)
    print(f"Candles: {len(candles)}")

    results = optimize(
        candles,
        label=f"{args.figi}_{args.timeframe}",
        param_grid={
            "bb_period": [23, 24, 25, 26, 27],
            "bb_std": [0.5, 1, 1.5, 2.0],
            "time_stop_bars": [16, 20, 24, 30],
            "session_end_utc": [14, 15, 16, 17],
            "width_lookback": [25, 50, 75],
        },
        fixed_params={
            "session_start_utc": 9,
            "position_size": 1,
        },
        n_jobs=args.jobs,
        out_dir=Path(__file__).parent / args.out,
        top_n=args.top,
    )

    sort_col = args.sort_by if args.sort_by in results.columns else "sortino"
    top = results.sort_values(sort_col, ascending=False).head(args.top)
    print(f"\nTop {args.top} by {sort_col}:")
    print(top.to_string(index=False))

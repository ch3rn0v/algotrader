"""Grid-search optimiser for BB mean reversion parameters.

Usage:
    # defaults: SBERP, 5min, 2025, all CPUs
    python3 bot/optimizer.py

    # custom dates and instrument
    python3 bot/optimizer.py --figi BBG004730N88 --timeframe 15min --from 2024-01-01 --to 2025-01-01

    # fewer workers
    python3 bot/optimizer.py --jobs 4

    # custom output dir
    python3 bot/optimizer.py --out outputs/my_results

Or from code:
    from optimizer import optimize
    results_df = optimize(candles, param_grid={...}, fixed_params={...})
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Optional

import lightgbm as lgbm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_mean_rev_bb import run_backtest
from candles import get_candles
from train_lgbm import (
    ASSETS as LGBM_ASSETS,
    MODEL_DIR,
    PRIMARY_ASSET,
    PRIMARY_TF,
    TIMEFRAMES as LGBM_TIMEFRAMES,
    build_features,
)

RESULT_COLS = ["pnl", "sharpe", "sortino", "max_dd", "cagr", "n_trades", "avg_bars_held", "turnover", "total_fees", "peak_exposure"]


def _run_one(args: tuple) -> dict:
    candles, params, predictions = args
    result = run_backtest(candles, **params, headless=True, predictions=predictions)
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
    "total_fees": lambda v: f"{v:,.0f}",
    "peak_exposure": lambda v: f"{v:,.0f}",
}


def _write_html(
    df: pd.DataFrame,
    color_cols: list,
    path: Path,
    default_sort_col: str = "sortino",
) -> None:
    """Write a self-contained sortable HTML table with inline RdYlGn cell backgrounds.

    Colors are normalised per column. Clicking a header sorts descending on first
    click, ascending on second click. Default sort column shows ▼ initially.
    """
    cmap = plt.cm.RdYlGn
    norms = {}
    for col in color_cols:
        if col not in df.columns:
            continue
        try:
            norms[col] = (float(df[col].min()), float(df[col].max()))
        except (TypeError, ValueError):
            pass

    def _bg(col, val):
        if col not in norms:
            return ""
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(v):
            return ""
        mn, mx = norms[col]
        t = (v - mn) / (mx - mn) if mx > mn else 0.5
        r, g, b, _ = cmap(t)
        return f"background-color:rgb({int(r*255)},{int(g*255)},{int(b*255)})"

    cols = list(df.columns)
    sort_idx = cols.index(default_sort_col) if default_sort_col in cols else -1

    th_css = "padding:4px 8px;border:1px solid #ccc;background:#f0f0f0;" "white-space:nowrap;cursor:pointer;user-select:none;position:sticky;top:0"
    td_css = "padding:4px 8px;border:1px solid #ccc;white-space:nowrap"

    header_cells = []
    for i, col in enumerate(cols):
        indicator = " ▼" if i == sort_idx else ""
        dir_attr = ' data-dir="desc"' if i == sort_idx else ""
        header_cells.append(f'<th onclick="sortTable(this)" data-col="{col}"{dir_attr} style="{th_css}">' f"{col}{indicator}</th>")

    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            fmt_fn = _FMT.get(col)
            txt = fmt_fn(val) if fmt_fn and pd.notna(val) else ("—" if pd.isna(val) else str(val))
            bg = _bg(col, val)
            style = f"{td_css};{bg}" if bg else td_css
            cells.append(f'<td style="{style}">{txt}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    js = """\
function sortTable(th) {
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const col   = th.cellIndex;
    const asc   = th.dataset.dir === 'desc';          // 'desc' → next click is asc
    th.dataset.dir = asc ? 'asc' : 'desc';
    table.querySelectorAll('th').forEach(h => { if (h !== th) delete h.dataset.dir; });
    const rows = Array.from(tbody.rows);
    rows.sort((a, b) => {
        const av = a.cells[col].textContent.trim().replace(/[,+]/g, '');
        const bv = b.cells[col].textContent.trim().replace(/[,+]/g, '');
        const an = parseFloat(av), bn = parseFloat(bv);
        const cmp = isNaN(an) || isNaN(bn) ? av.localeCompare(bv) : an - bn;
        return asc ? cmp : -cmp;
    });
    rows.forEach(r => tbody.appendChild(r));
    table.querySelectorAll('th').forEach(h => { h.textContent = h.dataset.col; });
    th.textContent = th.dataset.col + (asc ? ' ▲' : ' ▼');
}"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body  {{ font-family: monospace; font-size: 12px; margin: 8px; }}
table {{ border-collapse: collapse; }}
th:hover {{ filter: brightness(0.92); }}
</style>
</head>
<body>
<table>
  <thead><tr>{"".join(header_cells)}</tr></thead>
  <tbody>{"".join(body_rows)}</tbody>
</table>
<script>{js}</script>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")


def optimize(
    candles: pd.DataFrame,
    param_grid: dict,
    fixed_params: Optional[dict] = None,
    n_jobs: Optional[int] = None,
    out_dir: Path = Path("outputs/optimizer"),
    label: str = "",
    predictions: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Cartesian-product grid search, parallelised across CPU cores.

    Returns a plain DataFrame sorted by sortino descending.
    Also writes results_{label}.csv and results_{label}.html to out_dir.

    Parameters
    ----------
    candles     : candle DataFrame as returned by get_candles()
    param_grid  : dict of {param_name: [values…]} to sweep
    fixed_params: params passed verbatim to every backtest call
    n_jobs      : worker processes (default: all CPUs)
    out_dir     : directory for output files
    label       : ticker+timeframe tag used in output filenames
    """
    fixed = fixed_params or {}
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    all_params = [{**dict(zip(keys, c)), **fixed} for c in combos]

    n_workers = n_jobs or os.cpu_count()
    n_total = len(all_params)
    print(f"Grid: {n_total} combinations | {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_run_one, (candles, p, predictions)) for p in all_params]
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

    sweep_cols = [c for c in param_grid.keys() if c in df.columns]
    color_cols = sweep_cols + result_cols

    _write_html(df, color_cols, out_dir / f"{stem}.html")
    print(f"Saved: {out_dir}/{stem}.{{csv,html}}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BB mean reversion grid optimiser")
    parser.add_argument("--figi", default="BBG0047315Y7", help="Instrument FIGI (default: SBERP)")
    parser.add_argument("--timeframe", default="5min", help="Candle timeframe (default: 5min)")
    parser.add_argument("--from", dest="date_from", default="2025-01-01", metavar="DATE", help="Start date YYYY-MM-DD (default: 2025-01-01)")
    parser.add_argument("--to", dest="date_to", default="2026-01-01", metavar="DATE", help="End date YYYY-MM-DD (default: 2026-01-01)")
    parser.add_argument("--jobs", type=int, default=None, help="Parallel workers (default: all CPUs)")
    parser.add_argument("--out", default="outputs/optimizer", metavar="DIR", help="Output directory (default: outputs/optimizer)")
    args = parser.parse_args()

    from_dt = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = datetime.strptime(args.date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    candles = get_candles(args.figi, args.timeframe, from_dt, to_dt)
    print(f"Candles: {len(candles)}")

    # Load model predictions if a model exists for this instrument/timeframe.
    predictions = None
    meta_files = sorted(MODEL_DIR.glob("lgbm_*_meta.json"))
    if meta_files and args.figi == LGBM_ASSETS.get(PRIMARY_ASSET) and args.timeframe == PRIMARY_TF:
        meta_path = meta_files[-1]
        model_path = MODEL_DIR / meta_path.name.replace("_meta.json", ".txt")
        meta = json.loads(meta_path.read_text())
        feature_cols = meta["feature_cols"]

        lgbm_model = lgbm.Booster(model_file=str(model_path))
        print(f"Using model: {model_path.name}")

        print("Loading candles for feature engineering...")
        all_candles = {(PRIMARY_ASSET, PRIMARY_TF): candles}
        for asset in LGBM_ASSETS:
            for tf in LGBM_TIMEFRAMES:
                if not (asset == PRIMARY_ASSET and tf == PRIMARY_TF):
                    all_candles[(asset, tf)] = get_candles(LGBM_ASSETS[asset], tf, from_dt, to_dt)

        features = build_features(all_candles)
        null_cols = [c for c in features.columns if features[c].isna().all()]
        if null_cols:
            features = features.drop(columns=null_cols)

        missing = [c for c in feature_cols if c not in features.columns]
        if missing:
            print(f"WARNING: {len(missing)} feature columns missing — running without predictions.")
        else:
            preds_all = lgbm_model.predict(features[feature_cols])
            pred_map = dict(zip(features["timestamp"].values, preds_all))
            predictions = np.fromiter(
                (pred_map.get(ts, 1.0) for ts in candles["timestamp"].values),
                dtype=float,
                count=len(candles),
            )
            print(f"Predictions: {len(predictions)} bars, mean={predictions.mean():.5f}")
    else:
        print("No model found (or non-primary instrument) — running without predictions.")

    results = optimize(
        candles,
        label=f"{args.figi}_{args.timeframe}",
        param_grid={
            "bb_period": [20, 23, 24, 25, 26, 27],
            "bb_std": [1.5, 2.0, 2.5],
            "time_stop_bars": [16, 20, 24, 30],
            "session_end_utc": [12, 14, 15, 16, 17],
            "width_lookback": [50],
            "pred_long_threshold": [1.0, 1.01, 1.1],
            "pred_short_threshold": [1.0, 0.99, 0.9],
        },
        fixed_params={
            "session_start_utc": 9,
            "position_size": 1,
        },
        n_jobs=args.jobs,
        out_dir=Path(__file__).parent / args.out,
        predictions=predictions,
    )

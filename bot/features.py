"""Feature engineering for the LightGBM model.

Merges OHLCV-derived features from all (asset, timeframe) pairs onto the
primary index without lookahead (see build_features), then adds
cross-series features (trend agreement, market-wide volume anomalies).

The primary series additionally gets an extended set (MACD spreads, intra-bar
ratios, VWAP-like EWMs, N-bar diffs, lags, rolling slopes, EWM volume
ratios); all EWMs there are alpha-parameterized over EWM_ALPHAS.
"""
import itertools

import numpy as np
import pandas as pd

from config import PRIMARY_ASSET, PRIMARY_TF

# Extended-feature knobs (primary series only; see _extended_features).
EWM_ALPHAS = (0.01, 0.05, 0.1, 0.5)
SHIFT_LAGS = (1, 2, 4, 8, 16)
DIFF_LAGS = (1, 2, 4)
SLOPE_WINDOW = 10

TF_DURATIONS = {
    "1min": pd.Timedelta(minutes=1),
    "5min": pd.Timedelta(minutes=5),
    "15min": pd.Timedelta(minutes=15),
    "30min": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
}


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    # 100*gain/(gain+loss) == classic RSI; flat window (0/0) is neutral 50.
    rsi = 100 * gain / (gain + loss)
    return rsi.where((gain + loss) > 0, 50.0).where(gain.notna())


def _alpha_tag(alpha: float) -> str:
    # 0.01 -> a01, 0.05 -> a05, 0.1 -> a1, 0.5 -> a5
    return f"a{str(alpha).replace('0.', '')}"


def _rolling_slope(s: pd.Series, window: int = SLOPE_WINDOW) -> pd.Series:
    """OLS slope of the last `window` values against 0..window-1 (per-bar units).
    Windows containing NaN yield NaN."""
    y = s.to_numpy(dtype=float)
    out = np.full(len(y), np.nan)
    if len(y) >= window:
        x = np.arange(window, dtype=float)
        x -= x.mean()
        sw = np.lib.stride_tricks.sliding_window_view(y, window)
        out[window - 1:] = sw @ x / (x @ x)
    return pd.Series(out, index=s.index)


# Base features by group, used to extend shifts/slopes over the whole set.
_PRICE_BASE = [
    "ret1", "ret3", "ret8", "ret20", "spread", "body", "upper_wick",
    "lower_wick", "body_frac", "close_pos", "dir", "gap", "trend",
    "brk_hi20", "brk_lo20", "new_hi20", "new_lo20", "rsi14", "volat20",
]
_VOL_BASE = ["vol_norm", "vol_z", "vol_trend"]


def _raw_features(df: pd.DataFrame, prefix: str, extended: bool = False) -> pd.DataFrame:
    """Compute per-bar features for one (asset, timeframe) candle DataFrame."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    out = pd.DataFrame({"timestamp": df["timestamp"]})

    def put(name: str, values) -> None:
        out.loc[:, f"{prefix}_{name}"] = values

    # --- returns / range ---
    put("ret1", c.pct_change(1))
    put("ret3", c.pct_change(3))
    put("ret8", c.pct_change(8))
    put("ret20", c.pct_change(20))
    put("spread", (h - l) / c)

    # --- candle anatomy ---
    # Zero-range (doji) bars are common on thin series; use neutral values
    # instead of NaN so they don't knock out whole rows at dropna.
    bar_range = h - l
    has_range = bar_range > 0
    put("body", (c - o) / c)
    put("upper_wick", (h - np.maximum(o, c)) / c)
    put("lower_wick", (np.minimum(o, c) - l) / c)
    put("body_frac", ((c - o).abs() / bar_range).where(has_range, 0.0))
    put("close_pos", ((c - l) / bar_range).where(has_range, 0.5))
    put("dir", np.sign(c - o))
    put("gap", o / c.shift(1) - 1)

    # --- trend / breakout ---
    ema5 = c.ewm(span=5, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    put("trend", ema5 / ema20 - 1)
    hi20 = h.rolling(20, min_periods=20).max()
    lo20 = l.rolling(20, min_periods=20).min()
    put("brk_hi20", c / hi20 - 1)
    put("brk_lo20", c / lo20 - 1)
    put("new_hi20", (c > hi20.shift(1)).astype(float))
    put("new_lo20", (c < lo20.shift(1)).astype(float))
    put("rsi14", _rsi(c) / 100 - 0.5)
    put("volat20", c.pct_change(1).rolling(20, min_periods=20).std())

    # --- volume ---
    vol_ma = v.rolling(20, min_periods=1).mean().replace(0, np.nan)
    put("vol_norm", v / vol_ma)
    vol_std = v.rolling(20, min_periods=20).std()
    vol_z = (v - vol_ma) / vol_std
    put("vol_z", vol_z.where(vol_std > 0, 0.0).where(vol_std.notna()))
    vol_ma5 = v.rolling(5, min_periods=1).mean()
    put("vol_trend", vol_ma5 / vol_ma - 1)

    if not extended:
        return out

    # --- extended set (primary series only: the pairwise generator is
    # quadratic in feature count, so this stays off for context series) ---
    ext: dict[str, pd.Series] = {}
    price_new: list[str] = []
    vol_new: list[str] = []

    def putx(name: str, values: pd.Series, group: list[str]) -> None:
        ext[name] = values
        group.append(name)

    # MACD: normalized EMA spread for every fast>slow alpha pair.
    emas = {a: c.ewm(alpha=a, adjust=False).mean() for a in EWM_ALPHAS}
    for fast, slow in itertools.combinations(sorted(EWM_ALPHAS, reverse=True), 2):
        putx(f"macd_{_alpha_tag(fast)}_{_alpha_tag(slow)}", (emas[fast] - emas[slow]) / c, price_new)

    # Intra-bar price ratios.
    putx("hl", h / l - 1, price_new)
    putx("co", c / o - 1, price_new)
    putx("ch", c / h - 1, price_new)
    putx("cl", c / l - 1, price_new)

    # Volume-weighted EWM close/high/low (VWAP-like), normalized by the
    # latest close, plus the volume-weighted high-low spread.
    for a in EWM_ALPHAS:
        t = _alpha_tag(a)
        wnorm = v.ewm(alpha=a, adjust=False).mean().where(lambda s: s > 0)
        wc = (c * v).ewm(alpha=a, adjust=False).mean() / wnorm
        wh = (h * v).ewm(alpha=a, adjust=False).mean() / wnorm
        wl = (l * v).ewm(alpha=a, adjust=False).mean() / wnorm
        putx(f"vwap_{t}", wc / c - 1, price_new)
        putx(f"vwh_{t}", wh / c - 1, price_new)
        putx(f"vwl_{t}", wl / c - 1, price_new)
        putx(f"vwhl_{t}", (wh - wl) / c, price_new)

    # EWM path efficiency: smoothed signed drift over smoothed absolute
    # per-bar movement. |1| = straight-line move, ~0 = choppy churn. This is
    # the EWM analog of net-change-over-path-length (inverted vs the raw
    # path/net ratio, which blows up when the net move is ~0). The v-weighted
    # variant emphasizes movement that happened on real volume.
    dc = c.diff()
    for a in EWM_ALPHAS:
        t = _alpha_tag(a)
        path = dc.abs().ewm(alpha=a, adjust=False).mean()
        putx(f"eff_{t}", dc.ewm(alpha=a, adjust=False).mean() / path.where(path > 0), price_new)
        pathv = (dc.abs() * v).ewm(alpha=a, adjust=False).mean()
        putx(f"effv_{t}", (dc * v).ewm(alpha=a, adjust=False).mean() / pathv.where(pathv > 0), price_new)

    # N-bar close differences (close - close[t-N]), raw and EWM-smoothed,
    # close-normalized. Raw cd1 is skipped: ret1 already covers it.
    for n in DIFF_LAGS:
        d = c.diff(n)
        if n > 1:
            putx(f"cd{n}", d / c, price_new)
        for a in EWM_ALPHAS:
            putx(f"cd{n}_ewm_{_alpha_tag(a)}", d.ewm(alpha=a, adjust=False).mean() / c, price_new)

    # Volume levels and N-th order differences. Zero-volume bars would blow up
    # the ratios, so the normalizer is the latest positive volume.
    v_ref = v.where(v > 0).ffill()
    for a in EWM_ALPHAS:
        t = _alpha_tag(a)
        vm = v.ewm(alpha=a, adjust=False).mean()
        # pandas only implements ewm sum with adjust=True (sum of (1-a)^i * x).
        vs = v.ewm(alpha=a, adjust=True).sum()
        putx(f"vm_{t}", vm / v_ref, vol_new)
        putx(f"v_over_vm_{t}", v / vm.where(vm > 0), vol_new)
        putx(f"vsum_{t}", vs / v_ref, vol_new)
        putx(f"v_over_vsum_{t}", v / vs.where(vs > 0), vol_new)
    for n in DIFF_LAGS:
        d = v.diff(n)
        putx(f"vd{n}", d / v_ref, vol_new)
        for a in EWM_ALPHAS:
            putx(f"vd{n}_ewm_{_alpha_tag(a)}", d.ewm(alpha=a, adjust=False).mean() / v_ref, vol_new)

    def series_of(name: str) -> pd.Series:
        return ext[name] if name in ext else out[f"{prefix}_{name}"]

    price_all = _PRICE_BASE + price_new
    vol_all = _VOL_BASE + vol_new

    # Lags of every price feature; slopes of every price and volume feature.
    # Slopes/lags of base (unshifted) series only: slope∘shift == shift∘slope.
    for name in price_all:
        s = series_of(name)
        for lag in SHIFT_LAGS:
            ext[f"{name}_lag{lag}"] = s.shift(lag)
    for name in price_all + vol_all:
        ext[f"{name}_slope{SLOPE_WINDOW}"] = _rolling_slope(series_of(name))

    return pd.concat(
        [out, pd.DataFrame({f"{prefix}_{k}": s for k, s in ext.items()}, index=out.index)],
        axis=1,
    )


def _cross_features(result: pd.DataFrame, prefixes: list[str], primary_prefix: str) -> pd.DataFrame:
    """Row-wise features across series. Safe because every input column is
    already lookahead-free after the merge."""
    assets = sorted({p.rsplit("_", 1)[0] for p in prefixes})
    tfs = sorted({p.rsplit("_", 1)[1] for p in prefixes})
    cross: dict[str, pd.Series] = {}

    # Trend/direction agreement of each series with the primary series.
    prim_trend = np.sign(result[f"{primary_prefix}_trend"])
    for p in prefixes:
        if p == primary_prefix:
            continue
        cross[f"x_{p}_trend_match"] = prim_trend * np.sign(result[f"{p}_trend"])

    # Market-wide unusual volume: mean vol_z across assets at each timeframe.
    for tf in tfs:
        cols = [f"{a}_{tf}_vol_z" for a in assets if f"{a}_{tf}_vol_z" in result.columns]
        if len(cols) > 1:
            cross[f"x_volz_mkt_{tf}"] = result[cols].mean(axis=1)

    # Per-asset unusual volume agreement across timeframes.
    for a in assets:
        cols = [f"{a}_{tf}_vol_z" for tf in tfs if f"{a}_{tf}_vol_z" in result.columns]
        if len(cols) > 1:
            cross[f"x_volz_{a}_tfs"] = result[cols].mean(axis=1)

    return pd.concat([result, pd.DataFrame(cross, index=result.index)], axis=1)


def build_features(
    all_candles: dict,
    primary_asset: str = PRIMARY_ASSET,
    primary_tf: str = PRIMARY_TF,
) -> pd.DataFrame:
    """Merge features from all (asset, timeframe) pairs onto the primary index.

    `primary_asset`/`primary_tf` default to the config values; pass a trained
    model's own primary when building features for it.

    Primary series: feature values are shifted by one bar so that at
    decision time (start of bar t) only bar t-1 data is visible. Target is close[t]/close[t-1].

    All other series: merged on bar END timestamp (start + duration). This ensures
    only fully completed bars are visible at each decision point. A bar ending
    exactly at a 5m bar's start counts as complete (merge uses <=).

    The result has exactly one primary `timestamp` column. Each non-primary series
    also contributes a `{prefix}_ts` column recording which source bar was matched.
    """
    base = all_candles[(primary_asset, primary_tf)].sort_values("timestamp").reset_index(drop=True)
    result = base[["timestamp"]].copy()
    prefixes = []

    for (asset, tf), candles in all_candles.items():
        prefix = f"{asset}_{tf.replace('min', 'm')}"
        prefixes.append(prefix)
        df = candles.sort_values("timestamp").reset_index(drop=True)
        is_primary = asset == primary_asset and tf == primary_tf
        feats = _raw_features(df, prefix, extended=is_primary)

        if asset == primary_asset and tf == primary_tf:
            # Target is close[t]/close[t-1], so features must only use bar t-1 data.
            # Shift feature values forward by one bar; timestamp stays as the merge key.
            feat_cols = [c for c in feats.columns if c != "timestamp"]
            feats_prev = feats[["timestamp"]].copy()
            feats_prev.loc[:, feat_cols] = feats[feat_cols].shift(1).values
            result = pd.merge_asof(result, feats_prev, on="timestamp", direction="backward")
        else:
            # Replace timestamp with bar end so merge_asof only matches completed bars.
            # {prefix}_ts retains the original bar start for traceability.
            feats_merge = feats.copy()
            feats_merge.loc[:, f"{prefix}_ts"] = feats["timestamp"]
            feats_merge.loc[:, "timestamp"] = feats["timestamp"] + TF_DURATIONS[tf]
            result = pd.merge_asof(result, feats_merge, on="timestamp", direction="backward")

    primary_prefix = f"{primary_asset}_{primary_tf.replace('min', 'm')}"
    result = _cross_features(result, prefixes, primary_prefix)

    return result

"""Pure indicators over a candle DataFrame (SPEC 6.2).

All indicators feeding signal generation are lagged by one bar so the signal at
row *t* depends only on data observable strictly before bar *t* opens. The
just-observed ``close`` is exposed separately (``close_for_band``) as the trigger
that pierces a pre-existing band.

``adjust=False`` is mandatory: it makes EWMA a true online recursion
(``y_t = a*x_t + (1-a)*y_{t-1}``), which is what live and backtest must agree on.
``adjust=True`` (pandas default) renormalizes weights over the whole prior window
and produces different warmup-region values, breaking live/backtest parity.
"""
from __future__ import annotations

import pandas as pd


def compute_indicators(candles_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Return ``candles_df`` with lagged indicator columns appended.

    Adds (all lagged one bar): ``fast_ewm``, ``slow_ewm``, ``bb_mid``,
    ``bb_std``, ``bb_up``, ``bb_lo``; plus the non-lagged ``close_for_band``
    used only by the band classifier in SPEC 6.3.
    """
    df = candles_df.copy()
    close = df["close"]

    fast = close.ewm(span=params["fast_ewm_span"], adjust=False).mean().shift(1)
    slow = close.ewm(span=params["slow_ewm_span"], adjust=False).mean().shift(1)
    bb_mid = close.rolling(params["bb_period"]).mean().shift(1)
    bb_std = close.rolling(params["bb_period"]).std().shift(1)
    mult = params["bb_std_mult"]

    df["fast_ewm"] = fast
    df["slow_ewm"] = slow
    df["bb_mid"] = bb_mid
    df["bb_std"] = bb_std
    df["bb_up"] = bb_mid + mult * bb_std
    df["bb_lo"] = bb_mid - mult * bb_std
    df["close_for_band"] = close
    return df

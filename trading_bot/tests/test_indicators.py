"""Indicator tests: one-bar lag and EWMA online-recursion parity (SPEC 6.2, 14)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.indicators import compute_indicators
from tests.helpers import make_15m_df

PARAMS = {"fast_ewm_span": 3, "slow_ewm_span": 6, "bb_period": 4, "bb_std_mult": 2.0}


def _series_df(closes):
    return make_15m_df(closes)


def test_indicator_lag_equals_unlagged_truncated_at_t_minus_1():
    closes = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    df = compute_indicators(_series_df(closes), PARAMS)
    close = pd.Series([float(c) for c in closes])

    unlagged_fast = close.ewm(span=3, adjust=False).mean()
    unlagged_slow = close.ewm(span=6, adjust=False).mean()
    unlagged_mid = close.rolling(4).mean()
    unlagged_std = close.rolling(4).std()

    for t in range(1, len(closes)):
        assert np.isclose(df["fast_ewm"].iloc[t], unlagged_fast.iloc[t - 1])
        assert np.isclose(df["slow_ewm"].iloc[t], unlagged_slow.iloc[t - 1])
        if not np.isnan(unlagged_mid.iloc[t - 1]):
            assert np.isclose(df["bb_mid"].iloc[t], unlagged_mid.iloc[t - 1])
        if not np.isnan(unlagged_std.iloc[t - 1]):
            assert np.isclose(df["bb_up"].iloc[t], unlagged_mid.iloc[t - 1] + 2.0 * unlagged_std.iloc[t - 1])
            assert np.isclose(df["bb_lo"].iloc[t], unlagged_mid.iloc[t - 1] - 2.0 * unlagged_std.iloc[t - 1])


def test_first_row_indicators_are_nan():
    df = compute_indicators(_series_df([100, 101, 102, 103, 104]), PARAMS)
    assert np.isnan(df["fast_ewm"].iloc[0])
    assert np.isnan(df["slow_ewm"].iloc[0])
    assert np.isnan(df["bb_mid"].iloc[0])


def test_ewma_recursion_prefix_invariance():
    """adjust=False makes EWMA a true online recursion: the value at index t is
    identical whether computed on the full series or on a longer series sliced to
    t. (The bands DO depend on a fixed rolling window, so this targets the EWMAs,
    which are the parity-critical recursion.)"""
    long_closes = [100 + i * 0.3 for i in range(40)]
    df_long = compute_indicators(_series_df(long_closes), PARAMS)
    # Recompute the unlagged fast EWMA on a prefix and a longer series; compare at t.
    s_full = pd.Series(long_closes).ewm(span=3, adjust=False).mean()
    for t in (10, 20, 30):
        s_prefix = pd.Series(long_closes[: t + 1]).ewm(span=3, adjust=False).mean()
        assert np.isclose(s_full.iloc[t], s_prefix.iloc[t])
        # And the lagged column at t+1 equals the unlagged value at t.
        assert np.isclose(df_long["fast_ewm"].iloc[t + 1], s_full.iloc[t])

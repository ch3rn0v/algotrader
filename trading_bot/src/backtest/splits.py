"""Train / validation / test split resolution (SPEC 10.4).

Splits are anchored to a run date and expressed as durations. Laid out
contiguously in time, oldest first: ``warmup`` precedes ``train``; ``train``,
``validation`` and ``test`` are adjacent and ordered, with ``test`` ending on
the run date. The warmup prefix seeds indicator state only and is excluded from
every metric and from parameter selection.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

_SPLIT_NAMES = ("warmup", "train", "validation", "test")


def resolve_splits(splits_config: dict, today) -> dict:
    """Resolve the relative ``splits`` block into concrete ``(start, end)``
    date ranges for warmup, train, validation and test."""
    today_ts = pd.Timestamp(today).normalize()
    test_end = today_ts
    test_start = test_end - pd.DateOffset(months=splits_config["test_months"])
    val_end = test_start
    val_start = val_end - pd.DateOffset(months=splits_config["validation_months"])
    train_end = val_start
    train_start = train_end - pd.DateOffset(months=splits_config["train_months"])
    warmup_end = train_start
    warmup_start = warmup_end - pd.Timedelta(days=splits_config["warmup_days"])
    return {
        "warmup": (warmup_start.date(), warmup_end.date()),
        "train": (train_start.date(), train_end.date()),
        "validation": (val_start.date(), val_end.date()),
        "test": (test_start.date(), test_end.date()),
    }


def get_split(resolved: dict, name: str) -> tuple[date, date]:
    """Return the ``(start_date, end_date)`` tuple for a split by name."""
    if name not in _SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}; expected one of {_SPLIT_NAMES}")
    return resolved[name]


def warmup_window_for(resolved: dict, run_split: str, warmup_days: int) -> tuple[date, date]:
    """Return the warmup prefix ``(start, end)`` that precedes ``run_split``."""
    start, _ = get_split(resolved, run_split)
    start_ts = pd.Timestamp(start)
    return (start_ts - pd.Timedelta(days=warmup_days)).date(), start


def validate_metric_windows_disjoint(resolved: dict) -> None:
    """Assert the evaluation windows of train/validation/test do not overlap.
    Adjacent-and-ordered is fine; overlapping is not (SPEC 10.4)."""
    train_s, train_e = resolved["train"]
    val_s, val_e = resolved["validation"]
    test_s, test_e = resolved["test"]
    assert train_e <= val_s, f"train window overlaps validation: {train_e} > {val_s}"
    assert val_e <= test_s, f"validation window overlaps test: {val_e} > {test_s}"
    assert train_s < train_e < val_e < test_e, "metric windows are not ordered in time"


def validate_warmup_discipline(resolved: dict, run_split: str, warmup_days: int) -> None:
    """Assert a run's warmup prefix is not metric data of that same run.

    It does *not* prohibit a warmup prefix from falling inside a different
    split's metric window (e.g. validation warming up from train's tail): that
    is the deliberate, accepted design in SPEC 10.4. Only in-run reuse is
    forbidden."""
    eval_start, eval_end = get_split(resolved, run_split)
    warm_start, warm_end = warmup_window_for(resolved, run_split, warmup_days)
    assert warm_end <= eval_start, "warmup prefix overlaps its own run's metric window"
    assert warm_start < warm_end, "warmup window is empty or inverted"
    assert eval_start < eval_end, "run metric window is empty or inverted"

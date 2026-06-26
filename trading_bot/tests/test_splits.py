"""Train/validation/test split tests (SPEC 10.4, 14)."""
from __future__ import annotations

from datetime import date

from src.backtest.splits import (
    get_split,
    resolve_splits,
    validate_metric_windows_disjoint,
    validate_warmup_discipline,
    warmup_window_for,
)

SPLITS = {"warmup_days": 30, "train_months": 18, "validation_months": 6, "test_months": 12}


def test_splits_contiguous_and_ordered():
    resolved = resolve_splits(SPLITS, date(2026, 6, 1))
    w_s, w_e = resolved["warmup"]
    tr_s, tr_e = resolved["train"]
    va_s, va_e = resolved["validation"]
    te_s, te_e = resolved["test"]
    # Contiguity: each window's end is the next window's start.
    assert w_e == tr_s
    assert tr_e == va_s
    assert va_e == te_s
    # Ordered in time and test ends on the run date.
    assert w_s < tr_s < va_s < te_s < te_e
    assert te_e == date(2026, 6, 1)


def test_metric_windows_disjoint_passes():
    resolved = resolve_splits(SPLITS, date(2026, 6, 1))
    validate_metric_windows_disjoint(resolved)  # must not raise


def test_warmup_discipline_in_run():
    resolved = resolve_splits(SPLITS, date(2026, 6, 1))
    for split in ("train", "validation", "test"):
        validate_warmup_discipline(resolved, split, SPLITS["warmup_days"])  # must not raise


def test_warmup_prefix_precedes_eval_window():
    resolved = resolve_splits(SPLITS, date(2026, 6, 1))
    eval_start, _ = get_split(resolved, "validation")
    warm_start, warm_end = warmup_window_for(resolved, "validation", SPLITS["warmup_days"])
    assert warm_end == eval_start
    assert warm_start < warm_end

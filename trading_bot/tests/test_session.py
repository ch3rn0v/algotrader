"""Session predicate tests (SPEC 7.2, 14)."""
from __future__ import annotations

from datetime import datetime, timezone

from src.core.session import is_main_session


def _utc(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def test_session_open_close_boundaries():
    # 2024-06-03 is a Monday.
    assert is_main_session(_utc(2024, 6, 3, 7, 0)) is True       # 10:00 MSK open
    assert is_main_session(_utc(2024, 6, 3, 6, 59)) is False     # one minute before open
    assert is_main_session(_utc(2024, 6, 3, 15, 25)) is True     # last 15-min bar open
    assert is_main_session(_utc(2024, 6, 3, 15, 40)) is False    # right-exclusive close
    assert is_main_session(_utc(2024, 6, 3, 15, 39)) is True


def test_session_weekend_excluded():
    # 2024-06-08 Saturday, 2024-06-09 Sunday.
    assert is_main_session(_utc(2024, 6, 8, 10, 0)) is False
    assert is_main_session(_utc(2024, 6, 9, 10, 0)) is False


def test_session_weekday_grid():
    # Mon-Fri all open at 12:00 UTC (mid-session); Sat/Sun closed.
    for day, expected in ((3, True), (4, True), (5, True), (6, True), (7, True), (8, False), (9, False)):
        assert is_main_session(_utc(2024, 6, day, 12, 0)) is expected

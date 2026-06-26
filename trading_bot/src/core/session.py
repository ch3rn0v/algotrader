"""MOEX main equity session predicate, evaluated in UTC (SPEC 7.2).

Main session: 10:00-18:40 MSK (MSK = UTC+3) = 07:00-15:40 UTC, Mon-Fri.
Half-day schedules and exchange holidays are out of scope for v1; the operator
is responsible for stopping the bot before MOEX half-days and holidays.
"""
from __future__ import annotations

from datetime import datetime, time

# 10:00 MSK and 18:40 MSK expressed in UTC.
_SESSION_START_UTC = time(7, 0)
_SESSION_END_UTC = time(15, 40)


def is_main_session(ts_utc: datetime) -> bool:
    """True iff ``ts_utc`` (bar-open, UTC) is inside the MOEX main session.

    The session is half-open on the right: a bar whose open is exactly 15:40 UTC
    (18:40 MSK) would extend past the close, so the end is exclusive. The last
    in-session 15-min bar therefore opens at 15:25 UTC.
    """
    if ts_utc.weekday() >= 5:  # Saturday == 5, Sunday == 6
        return False
    t = ts_utc.time()  # naive time-of-day; internal timestamps are UTC by contract
    return _SESSION_START_UTC <= t < _SESSION_END_UTC

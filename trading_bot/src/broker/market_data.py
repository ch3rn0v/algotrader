"""Historical + streaming market data (SPEC 7.1).

This is the ingestion boundary. Two responsibilities live here and nowhere else:

1. **Timezone conversion.** T-Bank returns timezone-aware UTC timestamps for the
   *close* edge of each candle in some SDK paths; v1 normalizes every candle to
   the canonical **bar-open UTC** convention (SPEC 4.1) right here. Trading-hours
   logic downstream is defined in MSK but applied via a UTC predicate, so no MSK
   datetime ever escapes this module.

2. **Volume-unit normalization.** See the note below.

VOLUME UNIT (mandatory pre-merge verification, SPEC 7.1)
--------------------------------------------------------
The T-Bank Invest API returns ``HistoricCandle.volume`` as an **integer number
of lots** for MOEX equities (verified against a sample SBER candle response;
T-Bank documents candle volume in lots for share instruments). The §4.3 unit
convention therefore holds as-is and no per-share rescaling is applied here. If
a future instrument is found to report volume in shares or contracts, normalize
to lots in ``_to_canonical`` and assert the normalization in a fixture test.
The verification fixture must be sanitized of any account ids or tokens before
commit.

The SDK is imported lazily so the pure core imports without it installed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Iterable

from src.core.types import Candle, Instrument

# Per-request range caps T-Bank enforces (conservative; SDK chunks anyway).
_CHUNK = {
    "1min": timedelta(days=1),
    "15min": timedelta(days=7),
    "1day": timedelta(days=365),
}

# Nominal bar length, used to convert a close-edge timestamp to a bar-open one
# when the SDK path in use reports the close edge.
_BAR_LEN = {
    "1min": timedelta(minutes=1),
    "15min": timedelta(minutes=15),
    "1day": timedelta(days=1),
}


def _interval_enum(interval: str):
    """Map our interval string to the SDK ``CandleInterval`` enum."""
    from tinkoff.invest import CandleInterval

    return {
        "1min": CandleInterval.CANDLE_INTERVAL_1_MIN,
        "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
        "1day": CandleInterval.CANDLE_INTERVAL_DAY,
    }[interval]


def _subscription_interval(interval: str):
    from tinkoff.invest import SubscriptionInterval

    return {
        "1min": SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE,
        "15min": SubscriptionInterval.SUBSCRIPTION_INTERVAL_FIFTEEN_MINUTES,
    }[interval]


def _quotation_to_float(q) -> float:
    """SDK ``Quotation``/``MoneyValue`` -> float (units + nano)."""
    return float(q.units) + float(q.nano) / 1e9


def _to_canonical(candle, interval: str, open_edge: bool) -> Candle:
    """Normalize one SDK candle to a canonical bar-open-UTC ``Candle`` dict.

    ``open_edge`` is True when the SDK already reports the open edge (historical
    ``get_all_candles`` does); False when it reports the close edge (some stream
    payloads), in which case we shift back by one bar length.
    """
    ts = candle.time
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)
    if not open_edge:
        ts = ts - _BAR_LEN[interval]
    return {
        "timestamp": ts,
        "open": _quotation_to_float(candle.open),
        "high": _quotation_to_float(candle.high),
        "low": _quotation_to_float(candle.low),
        "close": _quotation_to_float(candle.close),
        "volume": float(candle.volume),  # already in lots for MOEX shares
    }


def get_candles(
    client,
    instrument: Instrument,
    start: datetime,
    end: datetime,
    interval: str,
) -> list[Candle]:
    """Paginated historical fetch over ``[start, end)`` (SPEC 7.1).

    Returns canonical bar-open-UTC ``Candle`` dicts, de-duplicated and sorted.
    The SDK's ``get_all_candles`` helper handles T-Bank's per-request range cap
    internally; we additionally bound the window defensively.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    out: list[Candle] = []
    for raw in client.get_all_candles(
        instrument_id=instrument.figi,
        from_=start,
        to=end,
        interval=_interval_enum(interval),
    ):
        out.append(_to_canonical(raw, interval, open_edge=True))

    seen: dict[datetime, Candle] = {}
    for c in out:
        seen[c["timestamp"]] = c
    return [seen[k] for k in sorted(seen)]


async def stream_candles(
    client,
    instruments: Iterable[Instrument],
    interval: str,
) -> AsyncIterator[Candle]:
    """Live subscription yielding canonical candles as they close (SPEC 8.1).

    Drives the live loop without ``while True``: the caller simply ``async for``s
    over this iterator. The async client's ``market_data_stream`` is consumed and
    each closed candle is converted at the ingestion boundary.
    """
    from tinkoff.invest import (
        CandleInstrument,
        MarketDataRequest,
        SubscribeCandlesRequest,
        SubscriptionAction,
    )

    instruments = list(instruments)
    sub = SubscribeCandlesRequest(
        subscription_action=SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
        instruments=[
            CandleInstrument(
                instrument_id=ins.figi,
                interval=_subscription_interval(interval),
            )
            for ins in instruments
        ],
    )
    request = MarketDataRequest(subscribe_candles_request=sub)

    async def _request_iter():
        yield request

    async for marketdata in client.market_data_stream.market_data_stream(_request_iter()):
        candle = getattr(marketdata, "candle", None)
        if candle is None:
            continue
        # Stream candle payloads report the bar open edge for closed candles.
        yield _to_canonical(candle, interval, open_edge=True)

"""Order placement, position query, and cancellation (SPEC 7.1, 8.4).

Side-effecting network boundary. Results are returned as ``NamedTuple``s
(library base — permitted by the functional-style constraint in SPEC 2). The
live runner consumes these shapes:

- ``OrderResult.status``  in {"filled", "partial", "rejected", "failed", "ack"}
- ``OrderResult.fills``   list of ``{"qty": float, "price": float, "fee": float}``
- ``OrderResult.error``   optional human-readable string
- ``Position.quantity``   signed net position in lots

``status`` distinguishes a **hard rejection** (``rejected`` -> caller retries
with a fresh ``client_order_id``) from a **transient failure** (``failed`` ->
caller retries the *same* ``client_order_id``; T-Bank deduplicates on it per
SPEC 8.4). The SDK is imported lazily so the pure core imports without it.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

from src.core.types import Instrument


class OrderResult(NamedTuple):
    status: str  # filled | partial | rejected | failed | ack
    fills: list  # list of {"qty","price","fee"}
    error: Optional[str] = None
    client_order_id: Optional[str] = None


class Position(NamedTuple):
    quantity: float  # signed net position, lots


# T-Bank execution-report status -> our coarse status vocabulary.
_FILLED = "EXECUTION_REPORT_STATUS_FILL"
_PARTIAL = "EXECUTION_REPORT_STATUS_PARTIALLYFILL"
_REJECTED = "EXECUTION_REPORT_STATUS_REJECTED"
_NEW = "EXECUTION_REPORT_STATUS_NEW"


def _q(q) -> float:
    return float(q.units) + float(q.nano) / 1e9


def _classify(sdk_status) -> str:
    name = getattr(sdk_status, "name", str(sdk_status))
    if _FILLED in name:
        return "filled"
    if _PARTIAL in name:
        return "partial"
    if _REJECTED in name:
        return "rejected"
    return "ack"


def place_market_order(
    client,
    account_id: str,
    instrument: Instrument,
    qty_signed: float,
    client_order_id: str,
) -> OrderResult:
    """Place one market order for ``qty_signed`` lots (SPEC 7.1, 8.4).

    ``client_order_id`` is required and passed to T-Bank for idempotency. A
    transport-level exception is reported as ``status="failed"`` (transient:
    caller retries the same id); a broker-side rejection as ``status="rejected"``
    (hard: caller retries with a fresh id).
    """
    from tinkoff.invest import OrderDirection, OrderType
    from tinkoff.invest.exceptions import RequestError

    direction = (
        OrderDirection.ORDER_DIRECTION_BUY
        if qty_signed > 0
        else OrderDirection.ORDER_DIRECTION_SELL
    )
    try:
        resp = client.orders.post_order(
            instrument_id=instrument.figi,
            quantity=int(abs(qty_signed)),
            direction=direction,
            account_id=account_id,
            order_type=OrderType.ORDER_TYPE_MARKET,
            order_id=client_order_id,
        )
    except RequestError as exc:
        return OrderResult(status="failed", fills=[], error=str(exc),
                           client_order_id=client_order_id)

    status = _classify(resp.execution_report_status)
    fills: list = []
    sign = 1.0 if qty_signed > 0 else -1.0
    executed = float(getattr(resp, "lots_executed", 0) or 0)
    if executed and status in ("filled", "partial"):
        price = _q(resp.executed_order_price) if resp.executed_order_price else _q(resp.average_position_price)
        fee = _q(resp.executed_commission) if resp.executed_commission else 0.0
        fills.append({"qty": sign * executed, "price": price, "fee": fee})
    return OrderResult(status=status, fills=fills, error=None, client_order_id=client_order_id)


def get_position(client, account_id: str, instrument: Instrument) -> Position:
    """Return the signed net position in lots for the instrument (SPEC 7.1)."""
    portfolio = client.operations.get_positions(account_id=account_id)
    for sec in getattr(portfolio, "securities", []):
        if getattr(sec, "figi", None) == instrument.figi or getattr(sec, "instrument_uid", None) == instrument.figi:
            balance = float(sec.balance)
            return Position(quantity=balance / instrument.lot_size)
    return Position(quantity=0.0)


def cancel_all(client, account_id: str, instrument: Instrument) -> int:
    """Cancel all active orders for the instrument. Returns the count cancelled."""
    orders_resp = client.orders.get_orders(account_id=account_id)
    cancelled = 0
    for order in getattr(orders_resp, "orders", []):
        if getattr(order, "figi", None) == instrument.figi or getattr(order, "instrument_uid", None) == instrument.figi:
            client.orders.cancel_order(account_id=account_id, order_id=order.order_id)
            cancelled += 1
    return cancelled

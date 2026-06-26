"""Live runner (SPEC 8). Stream-driven; no ``while True``.

The loop consumes ``broker.market_data.stream_candles`` as an async iterator and
acts on each closed candle following the per-bar sequence in SPEC 8.3. The bot
owns its intended position (SPEC 8.2); the broker is polled every bar as a sanity
check and never trusted to drive decisions. On a real discrepancy the bot halts
and stays halted until manually restarted.
"""
from __future__ import annotations

import asyncio
import signal as signal_module
import uuid
from collections import deque

import numpy as np
import pandas as pd

from src.core.indicators import compute_indicators
from src.core.session import is_main_session
from src.core.strategy import classify_band, classify_regime, step
from src.core.types import Instrument
from src.live import state as botstate
from src.live.event_log import make_logger
from src.live.executor import MAX_RETRIES, client_order_id, order_delta
from src.broker import market_data, orders


def _buffer_to_df(buffer: deque, instrument: Instrument) -> pd.DataFrame:
    df = pd.DataFrame(list(buffer))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["exchange"] = instrument.exchange
    df["instrument"] = instrument.symbol
    return df


def _place_with_retry(client, account_id, instrument, signal_ts, target_position, delta, log):
    """Place one market order for ``delta`` lots with the retry policy of SPEC 8.4.

    Returns the terminal ``OrderResult`` or ``None`` if all attempts failed. Uses
    a bounded ``for`` loop over tries (never ``while``). Per SPEC 8.4 a transient
    failure retries the *same* ``retry_attempt`` (identical id -> T-Bank
    deduplicates); a hard rejection increments ``retry_attempt`` (fresh id)."""
    retry_attempt = 0
    coid = client_order_id(instrument, signal_ts, target_position, retry_attempt)
    for _try in range(MAX_RETRIES + 1):
        coid = client_order_id(instrument, signal_ts, target_position, retry_attempt)
        log("order", {"bar_ts": signal_ts, "client_order_id": coid, "retry_attempt": retry_attempt,
                      "side": "buy" if delta > 0 else "sell", "qty": abs(delta), "reference_price": None})
        result = orders.place_market_order(client, account_id, instrument, delta, coid)
        log("order_status", {"client_order_id": coid, "status": result.status,
                             "error": getattr(result, "error", None)})
        if result.status in ("filled", "partial"):
            for fill in getattr(result, "fills", []):
                log("fill", {"client_order_id": coid, "bar_ts": signal_ts, "qty": fill["qty"],
                             "price": fill["price"], "fee": fill["fee"]})
            return result
        if result.status == "rejected":
            retry_attempt += 1  # hard rejection: fresh id on the next try
        # transient ("failed"/other): keep the same retry_attempt -> same id.
    log("order_status", {"client_order_id": coid, "status": "failed", "error": "retries exhausted"})
    return None


def process_bar(candle: dict, *, bot, strat_params, buffer, client, account_id, instrument, config, log):
    """Handle one closed candle (SPEC 8.3). Returns (bot, halted: bool)."""
    t = pd.Timestamp(candle["timestamp"])
    bar_ts = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

    # 1. Poll broker position; reconcile against intended (in-flight = 0 in v1 sync path).
    broker_pos = orders.get_position(client, account_id, instrument).quantity
    if botstate.is_discrepant(bot, broker_pos, 0.0):
        log("discrepancy", {"bar_ts": bar_ts, "intended_position": bot["intended_position"],
                            "broker_position": broker_pos, "context": "per-bar reconciliation"})
        orders.cancel_all(client, account_id, instrument)
        log("halt", {"reason": "discrepancy", "intended_position": bot["intended_position"]})
        return bot, True

    in_session = is_main_session(bar_ts.to_pydatetime())
    buffer.append(candle)
    warmup_invalid = len(buffer) <= max(strat_params["slow_ewm_span"], strat_params["bb_period"])
    log("candle", {"bar_ts": bar_ts, "o": candle["open"], "h": candle["high"], "l": candle["low"],
                   "c": candle["close"], "v": candle["volume"], "warmup_invalid": bool(warmup_invalid)})

    # 2. Out of session: mark only, no signal, no order (position carries).
    if not in_session:
        log("position", {"bar_ts": bar_ts, "intended_position": bot["intended_position"],
                         "broker_position": broker_pos, "cash": bot["cash"],
                         "equity": botstate.mark_equity(bot, candle["close"]), "mark_price": candle["close"]})
        return bot, False

    # 3 + 4. Indicators and the latest bar's signal.
    df = compute_indicators(_buffer_to_df(buffer, instrument), strat_params)
    last = df.iloc[-1]
    strat_state = botstate.to_strategy_state(bot)
    sig, target, regime, band, new_strat = step(
        strat_state, close=last["close"], fast_ewm=last["fast_ewm"], slow_ewm=last["slow_ewm"],
        bb_lo=last["bb_lo"], bb_up=last["bb_up"], bb_mid=last["bb_mid"], in_session=True, params=strat_params,
    )
    log("signal", {"bar_ts": bar_ts, "value": sig, "target_position": target,
                   "intended_position_before": bot["intended_position"], "gate": strat_state["gate"],
                   "gate_after": new_strat["gate"], "regime": regime, "band": band,
                   "fast_ewm": _f(last["fast_ewm"]), "slow_ewm": _f(last["slow_ewm"]),
                   "bb_up": _f(last["bb_up"]), "bb_lo": _f(last["bb_lo"]), "bb_mid": _f(last["bb_mid"]),
                   "recent_closes": [float(c) for c in df["close"].tail(config["live"]["_close_history"]).tolist()]})

    # 6 + 7. Diff against intended; place one delta order if needed.
    pending_order_id = None
    delta = order_delta(target, bot["intended_position"])
    if delta != 0.0:
        result = _place_with_retry(client, account_id, instrument, bar_ts, target, delta, log)
        if result is None:
            log("halt", {"reason": "retry_exhausted", "intended_position": bot["intended_position"]})
            return bot, True
        for fill in getattr(result, "fills", []):
            bot = botstate.apply_fill(bot, fill["qty"], fill["price"], fill["fee"], instrument.lot_size)

    # 8. Update intended + gate; emit the per-bar position mark.
    bot = botstate.merge_strategy_state(bot, sig, new_strat, pending_order_id)
    log("position", {"bar_ts": bar_ts, "intended_position": bot["intended_position"],
                     "broker_position": orders.get_position(client, account_id, instrument).quantity,
                     "cash": bot["cash"], "equity": botstate.mark_equity(bot, candle["close"]),
                     "mark_price": candle["close"]})
    return bot, False


def _f(x) -> "float | None":
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


async def run(client, account_id, instrument: Instrument, config: dict) -> int:
    """Drive the bot until the stream ends or a SIGINT/SIGTERM/halt occurs.
    Returns a CLI exit code (0 clean, 2 halted on discrepancy/retry)."""
    run_id = uuid.uuid4().hex[:12]
    log, _ = make_logger(run_id, instrument)
    strat = config["strategy"]
    risk = config["risk"]
    strat_params = {**strat, "max_position_lots": risk["max_position_lots"]}
    warmup_bars = max(strat["slow_ewm_span"], strat["bb_period"])
    config["live"]["_close_history"] = (config["live"].get("signal_log_close_history_bars") or (warmup_bars + 5))

    buffer: deque = deque(maxlen=warmup_bars * 2 + 16)
    bot = botstate.initial_state(config["backtest"]["initial_cash"])

    broker_pos = orders.get_position(client, account_id, instrument).quantity
    if botstate.is_discrepant(bot, broker_pos, 0.0):
        log("discrepancy", {"bar_ts": None, "intended_position": bot["intended_position"],
                            "broker_position": broker_pos, "context": "startup reconciliation"})
        log("halt", {"reason": "discrepancy", "intended_position": bot["intended_position"]})
        return 2

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig_name in (signal_module.SIGINT, signal_module.SIGTERM):
        loop.add_signal_handler(sig_name, stop.set)

    exit_code = 0
    stream = market_data.stream_candles(client, [instrument], strat["timeframe"])
    async for candle in stream:
        if stop.is_set():
            break
        bot, halted = process_bar(candle, bot=bot, strat_params=strat_params, buffer=buffer,
                                  client=client, account_id=account_id, instrument=instrument,
                                  config=config, log=log)
        if halted:
            exit_code = 2
            break

    # Shutdown (SPEC 8.7): bounded wait handled by the broker layer; positions held.
    log("shutdown", {"intended_position": bot["intended_position"],
                     "reason": "halt" if exit_code == 2 else "signal"})
    return exit_code

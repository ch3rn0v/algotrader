"""Command-line entry points (SPEC 13).

    python -m src.cli backtest --config config/config.yaml [--split S | --from D --to D]
    python -m src.cli run      --config config/config.yaml
    python -m src.cli flat     --config config/config.yaml [--confirm]
    python -m src.cli report   --log data/logs/<run_id>.jsonl
    python -m src.cli compare  --log data/logs/<run_id>.jsonl

Exit-code policy (SPEC 13):
    0 success | 1 recoverable failure | 2 unrecoverable (halt/flat-failed/bad config) | 3 usage error
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

EXIT_OK = 0
EXIT_RECOVERABLE = 1
EXIT_UNRECOVERABLE = 2
EXIT_USAGE = 3


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #
def _cmd_backtest(args) -> int:
    from src.config import instrument_from_config, load_config
    from src.backtest import loader, splits
    from src.backtest.engine import run_backtest
    from src.analytics.metrics import compute_metrics
    from src.analytics.charts import render_charts, write_metrics

    if bool(args.split) == bool(args.from_ or args.to):
        print("error: provide exactly one of --split or (--from and --to)", file=sys.stderr)
        return EXIT_USAGE
    if (args.from_ and not args.to) or (args.to and not args.from_):
        print("error: --from and --to must be given together", file=sys.stderr)
        return EXIT_USAGE

    try:
        config = load_config(args.config)
    except Exception as exc:  # pydantic validation, missing file, bad yaml
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_UNRECOVERABLE

    instrument = instrument_from_config(config)
    warmup_days = config["splits"]["warmup_days"]

    if args.split:
        resolved = splits.resolve_splits(config["splits"], date.today())
        splits.validate_metric_windows_disjoint(resolved)
        splits.validate_warmup_discipline(resolved, args.split, warmup_days)
        eval_start, eval_end = splits.get_split(resolved, args.split)
        warm_start, _ = splits.warmup_window_for(resolved, args.split, warmup_days)
        load_start, load_end, split_name = warm_start, eval_end, args.split
        eval_start_ts = pd.Timestamp(eval_start, tz="UTC")
    else:
        eval_start = datetime.fromisoformat(args.from_).date()
        eval_end = datetime.fromisoformat(args.to).date()
        load_start = (pd.Timestamp(eval_start) - pd.Timedelta(days=warmup_days)).date()
        load_end, split_name = eval_end, "custom"
        eval_start_ts = pd.Timestamp(eval_start, tz="UTC")

    start_dt = datetime.combine(load_start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(load_end, datetime.min.time(), tzinfo=timezone.utc)

    try:
        raw = _maybe_client()
        ctx = raw if raw is not None else contextlib.nullcontext()
        with ctx as client:
            c15 = loader.load_candles(client, instrument, "15min", start_dt, end_dt)
            c1m = loader.load_candles(client, instrument, "1min", start_dt, end_dt)
            c1d = loader.load_candles(client, instrument, "1day", start_dt, end_dt)
    except Exception as exc:
        print(f"data load error: {exc}", file=sys.stderr)
        return EXIT_RECOVERABLE

    if c15.empty:
        print("data load error: no 15-min candles for the requested window", file=sys.stderr)
        return EXIT_RECOVERABLE

    canonical = run_backtest(c15, c1m, c1d, instrument, config)
    # Metrics + charts only over the evaluation window (warmup prefix excluded).
    eval_df = canonical[pd.to_datetime(canonical["timestamp"], utc=True) >= eval_start_ts].reset_index(drop=True)

    run_id = f"bt_{split_name}_{uuid.uuid4().hex[:8]}"
    metrics = compute_metrics(eval_df, instrument, config, run_id=run_id, split=split_name)
    paths = render_charts(eval_df, run_id)
    metrics_path = write_metrics(metrics, run_id)
    print(f"run_id={run_id}")
    print(f"metrics: {metrics_path}")
    for name, p in paths.items():
        print(f"{name}: {p}")
    print(f"final_equity={metrics['final_equity']:.2f} sharpe={metrics['sharpe']:.3f} "
          f"trades={metrics['trade_count']} max_dd={metrics['max_drawdown']:.4f}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# run (live)
# --------------------------------------------------------------------------- #
def _cmd_run(args) -> int:
    from src.config import instrument_from_config, load_config
    from src.broker import client as broker_client
    from src.live.runner import run as live_run

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_UNRECOVERABLE

    instrument = instrument_from_config(config)
    token = broker_client.read_token()
    account_id = broker_client.read_account_id()

    async def _main() -> int:
        async with broker_client.make_async_client(token) as client:
            return await live_run(client, account_id, instrument, config)

    try:
        return asyncio.run(_main())
    except Exception as exc:
        print(f"fatal error: {exc}", file=sys.stderr)
        return EXIT_UNRECOVERABLE


# --------------------------------------------------------------------------- #
# flat (exit to zero)
# --------------------------------------------------------------------------- #
def _cmd_flat(args) -> int:
    from src.config import instrument_from_config, load_config
    from src.broker import client as broker_client, orders
    from src.live.event_log import make_logger
    from src.live.executor import MAX_RETRIES, flat_client_order_id

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_UNRECOVERABLE

    if not args.confirm:
        reply = input("This will place market orders to flatten the position. Proceed? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("aborted")
            return EXIT_USAGE

    instrument = instrument_from_config(config)
    token = broker_client.read_token()
    account_id = broker_client.read_account_id()
    timeout = config["live"]["order_timeout_seconds"]

    flat_run_id = f"flat_{uuid.uuid4().hex[:10]}"
    log, _ = make_logger(flat_run_id, instrument)

    try:
        with broker_client.make_client(token) as client:
            return _flatten(client, account_id, instrument, flat_run_id, timeout, log, orders,
                            flat_client_order_id, MAX_RETRIES)
    except Exception as exc:
        log("halt", {"reason": "fatal_error", "intended_position": None})
        print(f"flat error: {exc}", file=sys.stderr)
        return EXIT_UNRECOVERABLE


def _flatten(client, account_id, instrument, flat_run_id, timeout, log, orders,
             flat_client_order_id, max_retries) -> int:
    """Bring the broker position to zero with bounded retries (SPEC 13)."""
    for retry_attempt in range(max_retries + 1):
        position = orders.get_position(client, account_id, instrument).quantity
        log("position", {"bar_ts": None, "intended_position": 0.0, "broker_position": position,
                         "cash": None, "equity": None, "mark_price": None})
        if abs(position) < 1e-9:
            log("shutdown", {"intended_position": 0.0, "reason": "flat_complete"})
            print("position is flat")
            return EXIT_OK
        delta = -position
        coid = flat_client_order_id(instrument, flat_run_id, retry_attempt)
        log("order", {"bar_ts": None, "client_order_id": coid, "retry_attempt": retry_attempt,
                      "side": "buy" if delta > 0 else "sell", "qty": abs(delta), "reference_price": None})
        result = orders.place_market_order(client, account_id, instrument, delta, coid)
        log("order_status", {"client_order_id": coid, "status": result.status,
                             "error": getattr(result, "error", None)})
        for fill in getattr(result, "fills", []):
            log("fill", {"client_order_id": coid, "qty": fill["qty"], "price": fill["price"], "fee": fill["fee"]})
        _bounded_sleep(timeout)

    final = orders.get_position(client, account_id, instrument).quantity
    if abs(final) < 1e-9:
        log("shutdown", {"intended_position": 0.0, "reason": "flat_complete"})
        print("position is flat")
        return EXIT_OK
    log("halt", {"reason": "retry_exhausted", "intended_position": final})
    print(f"flat failed: position still {final} lots after {max_retries + 1} attempts", file=sys.stderr)
    return EXIT_UNRECOVERABLE


def _bounded_sleep(seconds: float) -> None:
    """Bounded wait for a fill ack (single sleep, never a polling while-loop)."""
    time.sleep(min(float(seconds), 30.0))


# --------------------------------------------------------------------------- #
# report (reconstruct -> metrics + charts)
# --------------------------------------------------------------------------- #
def _cmd_report(args) -> int:
    from src.config import instrument_from_config, load_config
    from src.analytics.reconstruct import reconstruct
    from src.analytics.metrics import compute_metrics
    from src.analytics.charts import render_charts, write_metrics

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"error: log not found: {log_path}", file=sys.stderr)
        return EXIT_USAGE
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_UNRECOVERABLE

    instrument = instrument_from_config(config)
    try:
        df = reconstruct(log_path, instrument, assert_consistency=True)
    except AssertionError as exc:
        print(f"reconstruction self-consistency failed: {exc}", file=sys.stderr)
        return EXIT_UNRECOVERABLE
    except Exception as exc:
        print(f"reconstruction error: {exc}", file=sys.stderr)
        return EXIT_RECOVERABLE

    run_id = log_path.stem
    metrics = compute_metrics(df, instrument, config, run_id=run_id, split="custom")
    paths = render_charts(df, run_id)
    metrics_path = write_metrics(metrics, run_id)
    print(f"run_id={run_id}")
    print(f"metrics: {metrics_path}")
    for name, p in paths.items():
        print(f"{name}: {p}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# compare (live vs backtest)
# --------------------------------------------------------------------------- #
def _cmd_compare(args) -> int:
    from src.config import instrument_from_config, load_config
    from src.analytics.compare import compare

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"error: log not found: {log_path}", file=sys.stderr)
        return EXIT_USAGE
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_UNRECOVERABLE

    instrument = instrument_from_config(config)
    try:
        paths = compare(log_path, instrument, config)
    except Exception as exc:
        print(f"compare error: {exc}", file=sys.stderr)
        return EXIT_RECOVERABLE

    for name, p in paths.items():
        print(f"{name}: {p}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# helpers + arg parsing
# --------------------------------------------------------------------------- #
def _maybe_client():
    """Return a broker client if credentials are present, else None.

    The loader uses the parquet cache first and only touches the network on a
    cache miss, so backtests over cached windows run without credentials.
    """
    try:
        from src.broker import client as broker_client

        return broker_client.make_client(broker_client.read_token())
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading_bot", description="Trend-filtered mean-reversion bot (SPEC).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bt = sub.add_parser("backtest", help="run a backtest over a split or explicit window")
    p_bt.add_argument("--config", default="config/config.yaml")
    p_bt.add_argument("--split", choices=["train", "validation", "test"])
    p_bt.add_argument("--from", dest="from_", metavar="YYYY-MM-DD")
    p_bt.add_argument("--to", dest="to", metavar="YYYY-MM-DD")
    p_bt.set_defaults(func=_cmd_backtest)

    p_run = sub.add_parser("run", help="run the live bot (stream-driven)")
    p_run.add_argument("--config", default="config/config.yaml")
    p_run.set_defaults(func=_cmd_run)

    p_flat = sub.add_parser("flat", help="bring the broker position to zero")
    p_flat.add_argument("--config", default="config/config.yaml")
    p_flat.add_argument("--confirm", action="store_true", help="skip the interactive prompt")
    p_flat.set_defaults(func=_cmd_flat)

    p_rep = sub.add_parser("report", help="reconstruct a live log into metrics + charts")
    p_rep.add_argument("--log", required=True)
    p_rep.add_argument("--config", default="config/config.yaml")
    p_rep.set_defaults(func=_cmd_report)

    p_cmp = sub.add_parser("compare", help="overlay a live log against a backtest of the logged candles")
    p_cmp.add_argument("--log", required=True)
    p_cmp.add_argument("--config", default="config/config.yaml")
    p_cmp.set_defaults(func=_cmd_compare)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

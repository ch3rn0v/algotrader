# Trading Bot

A functional-style Python trading bot for the T-Bank (ex-Tinkoff) Invest API,
paired with a backtester that shares its strategy logic. A live run and a
backtest of the same strategy over the same window produce directly comparable
results.

The v1 strategy is **trend-filtered mean reversion**: a slow/fast EWMA cross
acts as a regime filter, and Bollinger-Band touches trigger reversion entries
taken only in the direction the regime allows. It trades a single MOEX equity
on 15-minute candles during the main equity session (10:00–18:40 MSK).

## Setup

Requires Python 3.11+.

```bash
pip install -e .            # installs pandas, numpy, pyarrow, pydantic, matplotlib, tinkoff-investments
cp .env.example .env        # then fill in TBANK_TOKEN and TBANK_ACCOUNT_ID
```

Edit `config/config.yaml` for the instrument, strategy parameters, risk limits,
backtest execution model, and train/validation/test split durations. Secrets
live in `.env` only, never in the config.

## CLI

All commands are run from the project root as `python -m src.cli <command>`.
Exit codes: `0` success, `1` recoverable failure, `2` operator-attention
required (e.g. live halt on discrepancy), `3` usage error.

```bash
# Backtest a named split (with its warmup prefix) or an explicit window.
python -m src.cli backtest --config config/config.yaml --split train
python -m src.cli backtest --config config/config.yaml --from 2024-01-01 --to 2024-06-30

# Run the live bot (stream-driven; stop with SIGINT/SIGTERM).
python -m src.cli run --config config/config.yaml

# Exit-to-zero safety command (used after a halt or at end of day).
python -m src.cli flat --config config/config.yaml --confirm

# Regenerate charts + metrics.json from a live run log.
python -m src.cli report --log data/logs/<run_id>.jsonl

# Overlay a live run against a backtest of the same logged candles.
python -m src.cli compare --log data/logs/<run_id>.jsonl
```

`backtest` takes either `--split` or `--from/--to`, never both. `report` and
`compare` operate on existing JSONL logs and write PNGs under `reports/`.

## Architecture

One **canonical row schema** (one candle for one instrument) is shared by
backtest output and live-log reconstruction, so all downstream analytics
consume a single contract regardless of source.

- `src/core/` — pure, side-effect-free logic: indicators (lagged EWMA +
  Bollinger Bands), the signal state machine, the MOEX session predicate,
  candle normalization, and shared types.
- `src/broker/` — the only module that performs network I/O against T-Bank
  (candle fetch/stream, market orders, position reads).
- `src/live/` — the stream-driven runner, its executor, in-memory bot state,
  and the append-only JSONL event log (the source of truth for live runs).
- `src/backtest/` — the pandas row-loop engine, the volume-aware execution
  simulator, the historical-candle loader/cache, and split helpers.
- `src/analytics/` — metrics, log reconstruction, matplotlib charts, and the
  live-vs-backtest comparison, all consuming the canonical schema.

Side effects are confined to `broker/` (network) and `event_log.py` (disk);
everything else is pure. State flows through `NamedTuple`/`TypedDict`/dicts —
no custom business-logic classes. All internal timestamps are UTC, with the
canonical convention that a row's `timestamp` is the **open** of its bar.

### No-lookahead discipline

Indicators feeding signal generation are lagged one bar, so the signal at row
*t* depends only on data observable before bar *t* opens, plus *t*'s close as
the trigger that pierces a pre-existing band. Execution fields carry a
`_from_prev` suffix: a fill recorded on row *t* was caused by the signal
computed on row *t-1*. Mutation-form and step-jump-form tests enforce this.

## Documented limitations (v1)

These are explicit scope decisions, several of which should be revisited in v2:

- **No stop-loss.** The regime filter and the `flat` command are the only
  protections against adverse moves.
- **Single instrument.** The schema is multi-instrument-ready, but no engine
  or executor path handles more than one.
- **Fixed session calendar.** The session predicate uses weekday + time logic
  only; MOEX half-days and exchange holidays are not modeled. The operator
  must stop the bot on those days.
- **No remote kill-switch.** v1 stops via `SIGINT`/`SIGTERM` only.
- **Spread is modeled, not measured.** With no order-book data, spread is
  inferred from 1-minute ranges; square-root impact uses a placeholder
  coefficient. The execution model is deterministic (no stochastic fills).
- **Single-run reconstruction.** Cross-run reconciliation across halts,
  `flat` runs, and resumes spanning multiple `run_id`s is out of scope.
- **Volume-unit assumption.** The canonical schema assumes candle `volume`
  arrives in lots; the broker layer documents and verifies this per instrument
  and normalizes at ingestion if needed.

## Testing

The pure core ships with a test suite covering indicator lag/EWMA parity, the
36-entry signal state machine, the cooldown/mid-cross gate, the execution
simulator, session boundaries, splits, no-lookahead invariants, and the
event-log round-trip.

```bash
python -m pytest tests/          # if pytest is installed
python tests/run_tests.py        # otherwise, the bundled pytest-free runner
```

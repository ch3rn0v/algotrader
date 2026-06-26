# Trading Bot — Project Specification
 
## 1. Overview
 
A Python trading bot for the T-Bank (ex-Tinkoff) Invest API, paired with a backtester that shares its strategy logic. The system is designed so that a live run and a backtest of the same strategy over the same window produce directly comparable results.
 
### 1.1 Goals
 
- Run a candle-based **trend-filtered mean-reversion** strategy (EWMA regime filter + Bollinger Band reversion entries) live against the T-Bank API.
- Backtest the **exact same** strategy code on historical candles.
- Produce identical chart shapes from both sources: PnL over time, position over time, cumulative trade count.
- Compare live runs against backtests over the same window to surface real-world drift.
### 1.2 Non-Goals (for v1)
 
- Multi-instrument support. v1 trades exactly one instrument. The schema includes `exchange` and `instrument` fields so logs and reports stay multi-instrument-ready, but no engine, executor, or analytics code is required to handle more than one.
- Multi-strategy orchestration.
- Portfolio-level risk management beyond per-instrument position sizing. In particular, **no stop-loss** in v1 — the regime filter and the `flat` operator command are the only protections against adverse moves. This is an explicit risk-management gap that should be revisited in v2.
- Order types beyond market orders.
- Live paper trading via a separate sandbox layer (the backtester fills that role).
- A live candle-stream collector. v1 backtests on candles fetched ad-hoc via the T-Bank historical API. A separate collector that persists the live stream into the parquet cache is out of scope and tracked as future work.
- **Half-day MOEX sessions and exchange holidays.** The session predicate uses fixed weekday + time logic. The operator is responsible for stopping the bot before MOEX half-days and holidays. v2 should add a `trading_day_override` config list of dates the bot refuses to trade.
- **Remote kill-switch.** v1 stops via `SIGINT`/`SIGTERM` only. If the operator lacks shell access (e.g. the bot runs in an unreachable container), the only way to halt it is to stop its host. v2 should add a control-file watch (e.g. `data/control/halt`) the bot polls each bar.
- Cross-run reconciliation in reconstruction (covering halts, `flat` runs, and resumes spanning multiple `run_id`s).
## 2. Hard Constraints
 
These are non-negotiable and shape every design decision below.
 
- **Language**: Python (3.11+).
- **Style**: Strictly functional. No custom classes for business logic. State flows through `NamedTuple` / `TypedDict` / plain dicts. The only acceptable class usage is library-provided base classes (`pydantic.BaseModel` for config validation at load time only — see §5; SDK-provided types).
- **Loops**: No `while True`. The live loop is driven by an async candle stream (§8.1). The backtest engine iterates rows of a pandas DataFrame (`itertuples` for speed) — pandas is the native data structure throughout, so future strategy work can use vectorized operations directly.
- **Side effects**: Isolated to `broker/` (network) and `event_log.py` (disk). Everything else is pure.
- **Timeframes**: 15-minute candles (`CANDLE_INTERVAL_15_MIN`) drive signal generation. 1-minute candles (`CANDLE_INTERVAL_1_MIN`) drive execution-price modeling in the backtest (see §10.3) and serve as the canonical signal-to-fill resolution.
- **Timezones**: All internal timestamps are UTC. Conversion from Moscow time happens at the broker ingestion boundary and only there. Trading-hours filtering (§7.2) is defined in Moscow time but applied via a UTC predicate.
- **Trading session**: v1 trades the **MOEX main equity session only** (10:00–18:40 Moscow time, MSK = UTC+3). Evening session, weekend session, and auctions are excluded from both signal generation and fills.
## 3. Project Structure
 
```
trading_bot/
├── pyproject.toml
├── .env.example                 # T-Bank token, account id
├── README.md
│
├── config/
│   └── config.yaml              # strategy params, instrument, risk, splits
│
├── src/
│   ├── core/                    # shared, side-effect-free logic
│   │   ├── indicators.py        # EWMA, Bollinger Bands (pure)
│   │   ├── strategy.py          # signal generation (pure)
│   │   ├── types.py             # NamedTuples / TypedDicts
│   │   ├── candles.py           # candle normalization, resampling
│   │   └── session.py           # MOEX main-session predicate (UTC)
│   │
│   ├── broker/                  # T-Bank API adapter
│   │   ├── client.py            # tinkoff-investments SDK wrappers
│   │   ├── market_data.py       # get_candles, stream_candles
│   │   └── orders.py            # place_market_order, get_position
│   │
│   ├── live/                    # the running bot
│   │   ├── runner.py            # main loop (stream-driven)
│   │   ├── executor.py          # signal -> order translation
│   │   ├── state.py             # bot's own intended position (in-memory + persisted)
│   │   └── event_log.py         # structured JSONL logger
│   │
│   ├── backtest/
│   │   ├── engine.py            # pandas row-loop replay
│   │   ├── simulator.py         # execution model: VWAP fills, spread, impact, fees
│   │   ├── splits.py            # train/validation/test split helpers
│   │   └── loader.py            # historical candle download/cache
│   │
│   ├── analytics/               # shared between backtest + live
│   │   ├── metrics.py           # pnl, drawdown, daily Sharpe, trade count
│   │   ├── reconstruct.py       # event log -> canonical df
│   │   ├── charts.py            # matplotlib charts (PNG)
│   │   └── compare.py           # live vs backtest overlay
│   │
│   └── cli.py                   # entry points
│
├── data/
│   ├── candles/                 # parquet cache, hive-partitioned (15min, 1min, 1day)
│   └── logs/                    # JSONL event logs from live runs
│
├── reports/                     # generated PNG charts and metrics
└── tests/
```
 
## 4. Core Data Contract
 
The single most important design decision: one canonical row schema is shared by backtest output and live-log reconstruction. Everything downstream (metrics, charts, comparison) consumes only this schema.
 
### 4.1 Canonical Row
 
A row represents one candle for one instrument. The schema is split into two groups to prevent lookahead bias: fields known **at candle close** (observation-time) and fields produced by **executing during this bar** (execution-time, caused by the *previous* bar's signal).
 
**Timestamp convention (canonical, used everywhere internally):** `timestamp` is the **open of the bar**. A row with `timestamp = 12:15` covers the half-open interval `[12:15, 12:30)` for a 15-min bar. The candle is "closed" at `12:30`, which is also the `timestamp` of the next row. The broker layer is responsible for converting whatever convention T-Bank returns into this one at the ingestion boundary.
 
**Causality of execution-time fields.** Execution fields on row *t* describe a fill that happened **during bar *t*** as a result of the signal computed on row *t-1*. To make this unambiguous in the schema, execution-time field names carry the suffix `_from_prev`. This prevents readers from accidentally joining `signal[t]` against `trade_qty_from_prev[t]` as if they were causally related — they are not. The signal that *caused* `trade_qty_from_prev[t]` lives at row *t-1*. A derived helper column `triggering_signal` is also written, equal to `signal.shift(1)`, for convenience in row-local reads.
 
| Field                  | Type             | Group        | Description                                                               |
|------------------------|------------------|--------------|---------------------------------------------------------------------------|
| `exchange`             | `str`            | identity     | e.g. `MOEX`.                                                              |
| `instrument`           | `str`            | identity     | Symbol or figi.                                                           |
| `timestamp`            | `datetime` (UTC) | identity     | Bar **open** time (see convention above).                                 |
| `open`                 | `float`          | observation  | Candle open price.                                                        |
| `high`                 | `float`          | observation  | Candle high.                                                              |
| `low`                  | `float`          | observation  | Candle low.                                                               |
| `close`                | `float`          | observation  | Candle close — used for indicators and signal generation.                 |
| `volume`               | `float`          | observation  | Candle volume in lots (see §4.3 unit convention).                         |
| `signal`               | `int`            | observation  | Target position sign (`-1`, `0`, `+1`) decided at this bar's close.       |
| `target_position`      | `float`          | observation  | Desired position (lots) implied by `signal` and risk config.              |
| `triggering_signal`    | `int`            | observation  | `signal.shift(1)` — the signal from row *t-1* that caused this bar's fill, if any. Convenience derived field. |
| `trade_qty_from_prev`  | `float`          | execution    | Signed qty actually traded **during this bar**, caused by row *t-1*'s signal (see §10.3). |
| `fill_price_from_prev` | `float`          | execution    | Effective execution price for that fill.                                  |
| `fees_from_prev`       | `float`          | execution    | Fees paid on the fill.                                                    |
| `position`             | `float`          | execution    | Net position **after** the fill.                                          |
| `cash`                 | `float`          | execution    | Cash balance **after** the fill.                                          |
| `equity`               | `float`          | execution    | Marked at this bar's `close`: `cash + position * close`.                  |
 
Primary key: `(exchange, instrument, timestamp)`.
 
### 4.2 Lookahead Discipline
 
The system enforces strict no-lookahead at two layers:
 
- **Predictor causality**: indicators feeding signal generation are lagged by one bar (§6.2). The signal at row *t* depends only on prices from bars strictly before *t*, plus *t*'s close as the trigger that pierces a pre-existing band.
- **Execution causality**: execution fields on row *t* (`trade_qty_from_prev`, `fill_price_from_prev`, `fees_from_prev`) reflect a fill that happened **during bar *t***, triggered by the `signal` / `target_position` computed on row *t-1*. The `_from_prev` suffix encodes this causality directly in the schema. The simulator and the live executor must both respect this delay.
`equity` is marked-to-market at the bar's close — that's a legal use of close data, since it's just measurement, not action.
 
### 4.3 Instrument Reference
 
A small `NamedTuple` passed wherever an instrument is referenced:
 
```python
class Instrument(NamedTuple):
    exchange: str
    symbol: str
    figi: str
    currency: str
    lot_size: int      # number of base units (shares) per lot
```
 
Defined once in `core/types.py`. Strategy, broker, and backtest functions accept `Instrument` rather than bare strings.
 
**Unit convention.** Throughout the spec:
- `qty`, `trade_qty_from_prev`, `position`, `target_position`, `max_position_lots` are all expressed in **lots**.
- `volume` (in candles) is in lots as reported by the T-Bank API. If a future instrument reports volume in shares or contracts, the broker layer normalizes to lots at ingestion.
- `fill_price_from_prev`, `close`, etc. are per-share prices. Notional for fees and cash is computed as `qty * fill_price * lot_size`.
- `cash`, `equity`, `fees_from_prev` are in the instrument's `currency`.
## 5. Configuration
 
`config/config.yaml`:
 
```yaml
instrument:
  exchange: MOEX
  symbol: SBER
  figi: BBG004730N88
  currency: RUB
  lot_size: 10
  timeframe: 15min
 
strategy:
  fast_ewm_span: 20            # EWMA span for fast trend line
  slow_ewm_span: 50            # EWMA span for slow trend line
  bb_period: 20
  bb_std_mult: 2.0             # multiplier for the rolling std (formula in §6.2)
  entry_cooldown_bars: 4       # min bars between exit and next entry (§6.3)
 
risk:
  max_position_lots: 5
 
backtest:
  commission_bps: 5
  min_commission_per_order: 0  # floor in instrument currency; set to broker's per-order minimum if any
  participation_rate: 0.10     # max share of execution-window volume the fill may consume
  execution_window_minutes: 5  # K in §10.3
  min_spread_bps: 2            # spread floor for liquid MOEX names
  spread_to_range_ratio: 0.10  # spread ≈ ratio × range over 1-min candles
  impact_coeff: 0.1            # square-root impact scalar (small at v1 sizes)
  sigma_daily_window_days: 60  # rolling window for ex-ante daily volatility
  initial_cash: 1000000
 
splits:
  # Splits are anchored to "today" (the run date) and expressed as durations.
  # train + validation + test must be contiguous; warmup_days extends *before*
  # train and is excluded from any metric and parameter selection.
  warmup_days: 30
  train_months: 18
  validation_months: 6
  test_months: 12
 
live:
  reconcile_every_n_bars: 1    # poll broker position every N closed bars (v1: every bar)
  startup_warmup_bars: null    # null => max(slow_ewm_span, bb_period) * 2
  order_timeout_seconds: 30    # an order beyond this is considered no longer in-flight
  signal_log_close_history_bars: null  # null => max(slow_ewm_span, bb_period) + 5
 
metrics:
  sharpe_aggregation: daily    # aggregate equity to daily before annualizing
  risk_free_annual: 0.16       # parameterized; defaults to a recent CBR-key-rate-like value
  return_method: simple        # simple daily returns; rf subtracted as daily simple rate
```
 
`backtest/splits.py` resolves the relative `splits` block into concrete dates given a run date and exposes `validate_metric_windows_disjoint` and `validate_warmup_discipline` (see §10.4).
 
`pydantic.BaseModel` is used **at the config-load boundary only**: load YAML → validate → convert to a plain dict (`model_dump()`). No function elsewhere in the project takes a pydantic model as a parameter. This keeps everything downstream dict-native and avoids leaking class-based state into business logic.
 
Secrets (`TBANK_TOKEN`, `TBANK_ACCOUNT_ID`) live in `.env`, never in `config.yaml`.
 
## 6. Strategy Layer (Shared)
 
### 6.1 Framing
 
The v1 strategy is **trend-filtered mean reversion**. The premise: prices mean-revert to their local average on intraday horizons, but reverting *against* a strong trend is structurally adverse. So we use a slow EWMA cross as a regime filter and Bollinger-Band touches as entry triggers, taking reversion trades only in the direction the regime allows.
 
This is *not* a breakout strategy. The bands are reversion levels, not breakout triggers.
 
### 6.2 Indicators
 
`core/indicators.py` exposes pure functions over a candle DataFrame. **All indicators used by signal generation are lagged by one bar so that the signal at row *t* depends only on data observable strictly before bar *t* opens.** This is a stricter no-lookahead discipline than the conventional "use today's close to decide today" pattern; see the rationale below.
 
`compute_indicators(candles_df, params) -> indicators_df` adds these lagged columns:
 
- `fast_ewm`: `close.ewm(span=fast_ewm_span, adjust=False).mean().shift(1)`
- `slow_ewm`: `close.ewm(span=slow_ewm_span, adjust=False).mean().shift(1)`
- `bb_mid`: `close.rolling(bb_period).mean().shift(1)`
- `bb_std`: `close.rolling(bb_period).std().shift(1)`
- `bb_up`: `bb_mid + bb_std_mult * bb_std`
- `bb_lo`: `bb_mid - bb_std_mult * bb_std`
It also adds a non-lagged `close` reference (`close_for_band = close`) used only by the band classifier in §6.3 to decide whether `close` has crossed the (lagged) band. The signal therefore reads: "given the band parameters that were known at the start of bar *t*, did *t*'s close pierce them?" The bands are causal; the close that triggers them is known at the moment the bar closes. This is the cleanest causal split: predictors are strictly historical, the trigger is the just-observed close.
 
`adjust=False` is mandatory: it makes EWMA a true online recursion (`y_t = α·x_t + (1−α)·y_{t-1}`), which is what live and backtest must agree on. `adjust=True` (the pandas default) renormalizes weights using the entire prior window on every point, which is fine offline but produces different values in the warmup region than the live recursion does, breaking parity.
 
**Rationale for lagging.** A 15-min bar's close is observable at the close instant. Acting on it before the bar's `timestamp + interval` is technically permitted in the canonical model. But Bollinger bands and EWMAs computed *at* the bar's close fold today's price into their own mean and dispersion — a `close` at the lower band partly *is* the lower band. Lagging the bands by one bar removes this circularity and makes the predictor strictly historical, which is the cleanest causal contract and the one that survives the no-lookahead invariant test in §14 under the strongest possible interpretation.
 
**Known limitation: short-window estimation noise.** With `bb_period = 20` 15-min bars (~5 hours), `bb_std` is estimated from 20 samples and is itself noisy. Many "band touches" in the resulting series will be artifacts of the band-estimation noise rather than genuine reversion opportunities. This is a well-known issue with short-window Bollinger Bands. v1 keeps `bb_period = 20` for simplicity and visibility; v2 should investigate longer windows, EWMA-based volatility, or a z-score-against-longer-mean trigger as part of strategy iteration on the validation split.
 
### 6.3 Signal Generation (complete state machine)
 
`generate_signals` is a deterministic function of `(prev_signal, regime, band, gate)`:
 
- **Regime** is one of `up` (`fast_ewm > slow_ewm`) or `down` (`fast_ewm <= slow_ewm`). There is no `flat` bucket; ties are mapped to `down`.
- **Band** is one of `at_lower` (`close <= bb_lo`), `at_upper` (`close >= bb_up`), or `inside`.
- **Gate** is a re-entry gate carried as part of the strategy's running state. It blocks a fresh entry until two conditions are both met after the most recent exit:
  1. `bars_since_exit >= entry_cooldown_bars` (config; v1 default **4 bars — this is a guess, not a calibrated value**, and should be tuned on the validation split).
  2. `close` has crossed `bb_mid` at least once since the exit (i.e. price has reverted past the rolling mean).
  Gate values: `open` (entry allowed) or `blocked`. The gate flips to `blocked` on every exit and to `open` only when both conditions are met.
The target signal at each bar is the table below. `prev_signal` is the row's prior signal.
 
| prev_signal | regime | band       | gate    | new_signal | rationale                            |
|-------------|--------|------------|---------|------------|--------------------------------------|
|  0          | up     | at_lower   | open    | +1         | enter long: dip in uptrend           |
|  0          | up     | at_lower   | blocked |  0         | cooldown / no bb_mid re-cross yet    |
|  0          | up     | at_upper   | any     |  0         | no entry: don't fade an uptrend up   |
|  0          | up     | inside     | any     |  0         | no signal                            |
|  0          | down   | at_upper   | open    | -1         | enter short: pop in downtrend        |
|  0          | down   | at_upper   | blocked |  0         | cooldown / no bb_mid re-cross yet    |
|  0          | down   | at_lower   | any     |  0         | no entry: don't fade a downtrend dn  |
|  0          | down   | inside     | any     |  0         | no signal                            |
| +1          | up     | at_upper   | any     |  0         | mean-reversion target hit, exit      |
| +1          | up     | at_lower   | any     | +1         | hold long while uptrend persists     |
| +1          | up     | inside     | any     | +1         | hold long while uptrend persists     |
| +1          | down   | any        | any     |  0         | regime flipped, exit long            |
| -1          | down   | at_lower   | any     |  0         | mean-reversion target hit, exit      |
| -1          | down   | at_upper   | any     | -1         | hold short while downtrend persists  |
| -1          | down   | inside     | any     | -1         | hold short while downtrend persists  |
| -1          | up     | any        | any     |  0         | regime flipped, exit short           |
 
Every (prev, regime, band, gate) combination has a defined output. There are no implicit "hold previous" branches.
 
The gate state itself is updated *after* the signal is computed, using these rules:
- On any transition from non-zero to `0` (an exit), set gate to `blocked` and reset `bars_since_exit = 0` and `crossed_mid_since_exit = False`.
- On every subsequent bar while gate is `blocked`: increment `bars_since_exit`. The mid-cross test uses **the same lagged `bb_mid` value as the entry trigger at this bar** — that is, `bb_mid[t]` per §6.2, which is the rolling mean of closes ending strictly before *t*. The test asserts a sign change of `(close - bb_mid[t])` between consecutive bars: if `(close[t-1] - bb_mid[t]) * (close[t] - bb_mid[t]) < 0`, set `crossed_mid_since_exit = True`. Using the same lag for both the entry trigger and the gate test ensures the gate is tested against the same band geometry the strategy would act on.
- If both gate conditions are now satisfied (`bars_since_exit >= entry_cooldown_bars` AND `crossed_mid_since_exit`), flip gate to `open`.
- The gate is part of the `BotState` (live) and the engine's carried state (backtest); the live event log records its value alongside the signal.
### 6.4 Position Sizing
 
```
target_position = signal * risk.max_position_lots
```
 
Trivial and explicit. Volatility scaling, Kelly, or any other sizing is out of scope for v1.
 
## 7. Broker Layer
 
`src/broker/` is the only module allowed to perform network I/O against T-Bank.
 
### 7.1 Responsibilities
 
- `client.py`: authenticated SDK client factory. One function: `make_client(token) -> SDK client`.
- `market_data.py`:
  - `get_candles(client, instrument, start, end, interval) -> list[Candle]` — paginated historical fetch (T-Bank limits range per request; the function handles chunking). Supports both `15min` and `1min` intervals.
  - `stream_candles(client, instruments, interval) -> AsyncIterator[Candle]` — live subscription.
- `orders.py`:
  - `place_market_order(client, account_id, instrument, qty_signed, client_order_id) -> OrderResult` — `client_order_id` is required (see §8.4).
  - `get_position(client, account_id, instrument) -> Position`.
All functions take dependencies as arguments. No module-level state. The broker layer is also responsible for converting T-Bank's timestamp convention to the canonical bar-open-UTC convention (§4.1) at ingestion.
 
**Volume-unit verification (mandatory pre-merge check).** Before merging the broker layer, the developer **must** verify the actual volume unit returned by the T-Bank Invest API for the configured instrument. Inspect a sample candle response and confirm whether `volume` represents lots, shares, or contracts. Document the finding in a comment at the top of `broker/market_data.py` and add a unit test fixture with a real candle response, **sanitized of any account identifiers, tokens, or other request-scoped identifiers** before commit. The §4.3 unit convention assumes `volume` arrives in lots; if the API returns shares for a given instrument, the broker layer normalizes to lots at ingestion and the test asserts the normalization. Skipping this check risks a silent unit error that contaminates every backtest and every live fill estimate.
 
### 7.2 Trading-Hours Filter
 
`core/session.py` exports a single pure predicate:
 
```python
def is_main_session(ts_utc: datetime) -> bool
```
 
It returns `True` iff `ts_utc` falls inside the MOEX main equity session (10:00–18:40 MSK = 07:00–15:40 UTC) on a regular MOEX trading day. Half-day schedules and exchange holidays are out of scope for v1; the predicate uses fixed weekday + time logic. Both the backtest engine and the live runner must apply this filter before generating a signal or placing an order. Bars whose `timestamp` falls outside the session are kept in the DataFrame for indicator continuity but produce `signal = 0` and `trade_qty_from_prev = 0`.
 
## 8. Live Bot
 
### 8.1 Loop Mechanics
 
`live/runner.py` drives the bot. It does **not** use `while True`. The v1 implementation is **stream-driven**: consume `broker.market_data.stream_candles` as an async iterator and act on each closed candle.
 
The stream provides candles as they close, which is the natural unit of work. There is no scheduler component in v1. If T-Bank's stream proves unreliable in production (e.g. silent disconnects), v2 will add a watchdog that detects gaps via wall-clock and reconnects — but it will not reintroduce a periodic scheduler as the primary loop.
 
### 8.2 Bot-Owned Position State
 
The bot maintains its **own** intended position as the source of truth for execution decisions, persisted via the event log. `live/state.py` exposes pure functions over a small `BotState` dict that includes `intended_position`, `last_signal`, `gate` (§6.3), and `pending_order_id`. The broker is consulted, not trusted, for position information:
 
- At startup, fetch `broker.orders.get_position` once and reconcile against the last persisted intended position.
- During normal operation, signal-to-order decisions are made against `intended_position`. The broker is **not** queried *to drive decisions*.
- **On every closed bar** (config `live.reconcile_every_n_bars: 1`), poll `broker.orders.get_position` as a sanity check. The polled value is recorded in the `position` event's `broker_position` field.
This inversion — bot owns position, broker is checked every bar — eliminates the race between `get_position` and `place_market_order` that would otherwise let an in-flight fill double the size, while keeping the maximum undetected drift to one bar (~15 minutes).
 
**Discrepancy policy.** A discrepancy is any case where, after accounting for in-flight orders (defined below), `broker_position` does not match `intended_position`.
 
**In-flight order accounting.** An order is considered *in-flight* from the moment `place_market_order` is called until either:
- (a) terminal `order_status` (`filled`, `cancelled`, `rejected`, `failed`) is observed, or
- (b) `order_timeout_seconds` (config; v1 default 30s) elapses since placement.
While an order is in-flight, the bot tracks its `qty_remaining` (initially the full requested qty; reduced on each `partial`/`fill` event). The expected broker position is `intended_position − qty_remaining` (the broker has not yet seen the not-yet-filled portion). The reconciliation check is therefore:
 
```
expected_broker_position = intended_position − sum(qty_remaining over in-flight orders)
discrepancy iff broker_position != expected_broker_position
```
 
**Per-bar reconciliation deferral.** If the per-bar reconciliation poll fires while an order is in-flight, the check uses `expected_broker_position` as above. If the order remains in-flight for more than two consecutive bars (`30+ minutes`), this is treated as a stuck order and triggers the discrepancy halt path regardless of the comparison.
 
On detection of a real discrepancy:
 
1. Emit a `discrepancy` event with both values and the diagnostic context.
2. Cancel all pending orders for the instrument.
3. **Halt the bot.** No further signals are computed, no further orders are placed.
4. The bot remains halted until an operator inspects the situation and explicitly restarts it. There is no automatic recovery path in v1.
A discrepancy at startup (intended position from last persisted state vs. current broker position) follows the same rule: log, halt, require manual restart.
 
**Operator resolution procedure.** When halted on discrepancy, the operator's recovery procedure is:
 
1. Inspect the JSONL log to determine the cause (filled order not acked, out-of-band trade, broker-side correction).
2. Decide on the desired starting position.
3. Run the **flat** subcommand (§13) to bring the broker position to zero, or manually adjust via the T-Bank UI.
4. Start a fresh run with a new `run_id`. The new run's startup reconciles the (now zero or operator-set) position and proceeds.
The bot does **not** automatically flatten on discrepancy. v1 prefers halting in place over potentially compounding the problem with automated recovery.
 
### 8.3 Per-Bar Sequence
 
On each closed 15-min candle for the instrument:
 
1. Poll `broker.orders.get_position` (per `live.reconcile_every_n_bars`, v1 default 1). If it disagrees with `intended_position` beyond in-flight expectations, follow the discrepancy policy (§8.2): emit `discrepancy`, cancel pending orders, halt.
2. If `not is_main_session(timestamp)`, emit a `position` mark event (see §8.5) and return — no signal, no order.
3. Append candle to in-memory buffer (capped at warmup length + headroom).
4. Call `compute_indicators`, then `generate_signals` using `(prev_signal, regime, band, gate)` from `BotState`. Emit a `signal` event including `gate`.
5. Compute `target_position` from the latest signal and risk config. Emit it as part of the signal event.
6. Diff `target_position` against `intended_position` (bot's own state).
7. If they differ, call `executor.reconcile`, which constructs a deterministic `client_order_id` (§8.4) and places one market order for the delta. Emit `order`, `order_status`, and `fill` events as they happen.
8. Update `intended_position` and `gate` per §6.3. Emit a `position` mark event with `cash`, `equity`, and the freshly polled `broker_position` regardless of whether a trade occurred.
### 8.4 Order Idempotency and Retries
 
Every market order is placed with a deterministic `client_order_id` that combines the **decision identity** with a **retry attempt counter**:
 
```
client_order_id = sha1(instrument || signal_timestamp || target_position || retry_attempt)[:16]
```
 
`retry_attempt` is `0` for the first placement, `1` for the first retry, `2` for the second.
 
- **Network/transient failures** (timeouts, ack lost, transport errors): the bot retries the *same* `retry_attempt`, so `client_order_id` is identical. T-Bank deduplicates against this id; the same order is not placed twice.
- **Hard rejections** (insufficient margin, rate limit, invalid price collar, etc.): the bot increments `retry_attempt` and retries with a *fresh* `client_order_id`. This avoids being silently deduplicated against the rejected attempt.
**Retry policy**: at most **2 retries** (so 3 attempts total: `retry_attempt` ∈ {0, 1, 2}). After 3 failed attempts, the bot logs `order_status` with `status=failed`, emits a `discrepancy` event if the failure leaves the bot's intended position out of sync with reality, and **halts** per §8.2.
 
The id is logged in every `order` event and used as the join key when matching `fill` events back to their originating attempt.
 
**Restart during in-flight.** v1 does not persist `retry_attempt` across process restarts. If the bot crashes or restarts after placing `retry_attempt=N` for a given decision but before its terminal `order_status` arrives, a fresh process will start from `retry_attempt=0` for the next signal. T-Bank may either deduplicate against the prior attempt or accept the new id depending on whether the prior order is still active broker-side. Either way, the per-bar reconciliation poll (§8.2) will detect any resulting position discrepancy and trigger the halt path. This is the intended behavior: v1 prefers a deterministic halt-and-investigate over silent attempts at automated recovery.
 
### 8.5 Mark Events: Required on Every Bar
 
A `position` mark event MUST be emitted on every 15-min bar close, including bars with no trade and bars outside the trading session. This guarantees that the live equity time series has the same row density as the backtest, so reconstruction produces a comparable canonical DataFrame without phantom flat segments. The mark is computed as `cash + intended_position * close`, where `close` is the close of the just-completed bar.
 
### 8.6 Startup and Warmup
 
- Load config and `.env`. Validate via pydantic; convert to plain dict.
- Authenticate.
- Make **two separate market-data calls**:
  1. `get_candles(interval=15min, ...)` covering at least `2 × max(slow_ewm_span, bb_period)` bars before run start. Used to seed indicator state for signal generation.
  2. `get_candles(interval=1day, ...)` covering at least `sigma_daily_window_days` calendar days before run start. Used **only** to seed the `sigma_daily` column for the execution model (§10.3).
- The first `max(slow_ewm_span, bb_period)` 15-min bars after startup are tagged as `warmup_invalid` in the log (and additionally any bar where `sigma_daily` is NaN, see §10.4); downstream parity comparison ignores this prefix.
- Reconcile broker position against last persisted intended position (§8.2). Initialize `gate = open` if no prior persisted state exists.
### 8.7 Shutdown
 
The runner installs a SIGINT/SIGTERM handler that:
 
1. Stops accepting new candles.
2. Waits for any in-flight order placement to either ack or time out (bounded wait, e.g. 5s).
3. Emits a `shutdown` event with the final intended position.
4. Flushes and closes the event log.
The bot does not auto-cancel open positions on shutdown — that's a deliberate decision. v1's positions are held overnight if the bot stops; an operator decides how to flatten them.
 
## 9. Event Log
 
`live/event_log.py` writes one JSON object per line to `data/logs/<run_id>.jsonl`. Append-only; never edited.
 
### 9.1 Common Fields
 
Every event includes:
 
```json
{"ts": "2026-05-01T12:15:00Z", "exchange": "MOEX", "instrument": "SBER", "type": "...", "run_id": "..."}
```
 
`ts` is the wall-clock UTC time the event was written. Events that refer to a specific bar carry an additional `bar_ts` field with the bar's open timestamp (canonical convention from §4.1).
 
### 9.2 Event Types
 
| `type`         | Additional fields                                                                                                            |
|----------------|------------------------------------------------------------------------------------------------------------------------------|
| `candle`       | `bar_ts`, `o`, `h`, `l`, `c`, `v`, `warmup_invalid` (bool)                                                                   |
| `signal`       | `bar_ts`, `value`, `target_position`, `intended_position_before`, `gate`, `gate_after`, `regime`, `band`, `fast_ewm`, `slow_ewm`, `bb_up`, `bb_lo`, `bb_mid`, `recent_closes` (last `signal_log_close_history_bars` closes used) |
| `order`        | `bar_ts`, `client_order_id`, `retry_attempt`, `side`, `qty`, `reference_price`                                               |
| `order_status` | `client_order_id`, `status` (`placed` / `ack` / `partial` / `filled` / `rejected` / `cancelled` / `failed`), `error` (optional) |
| `fill`         | `client_order_id`, `qty`, `price`, `fee`                                                                                     |
| `position`     | `bar_ts`, `intended_position`, `broker_position`, `cash`, `equity`, `mark_price`                                             |
| `discrepancy`  | `bar_ts`, `intended_position`, `broker_position`, `context`                                                                  |
| `api_error`    | `endpoint`, `error`, `retry_count`                                                                                           |
| `halt`         | `reason` (`discrepancy` / `retry_exhausted` / `fatal_error`), `intended_position`                                            |
| `shutdown`     | `intended_position`, `reason`                                                                                                |
 
A `position` event is emitted on **every** 15-min bar close (§8.5), regardless of whether a trade occurred, and includes `broker_position` from the per-bar poll (§8.2). The `signal` event captures the inputs to the decision (`recent_closes`, indicator values, `gate`, `regime`, `band`) so that drift in the comparison report can be bisected to either input differences (live saw different prices) or decision differences (same prices, different signal — meaning a bug). `gate` is the value used to compute *this bar's* signal (pre-update); `gate_after` is the value carried into the *next bar* (post-update per §6.3). `regime` and `band` are redundant — they are derivable from the indicator and price fields also logged on the same event — but logged anyway so a debugger can read the categorical decision inputs without recomputing them.
 
**`fill` event vs. canonical `fees_from_prev`.** The singular `fee` field on a `fill` event is intentional: a single logical order can produce multiple `fill` events (e.g. partial fills across the K-minute execution window), each contributing its own `fee`. The canonical row's `fees_from_prev` is the sum of all `fee` values from `fill` events that landed during that bar. Reconstruction performs this aggregation.
 
This log is the **source of truth** for live reporting and comparison. Any field that comparison or post-hoc debugging needs must be logged at the moment it happens.
 
## 10. Backtest Engine
 
### 10.1 Loader
 
`backtest/loader.py` fetches historical candles via `broker.market_data.get_candles` and caches them under:
 
```
data/candles/exchange=MOEX/instrument=SBER/interval=15min/year=2024/month=06/candles.parquet
data/candles/exchange=MOEX/instrument=SBER/interval=1min/year=2024/month=06/candles.parquet
data/candles/exchange=MOEX/instrument=SBER/interval=1day/year=2024/candles.parquet
```
 
15-min candles drive the strategy; 1-min candles drive execution-price modeling (§10.3); 1-day candles seed `sigma_daily` (§10.3). Hive-style partitioning enables slice reads via pyarrow without scanning everything. The 1-day partition is coarser (year-only) because daily series are tiny. The broker layer applies the canonical timestamp convention (§4.1) on ingestion before anything is cached.
 
### 10.2 Engine
 
`backtest/engine.py` works on a pandas DataFrame of 15-min candles indexed by `timestamp`, with the corresponding 1-min candles passed alongside:
 
1. **Vectorized prep**: call `compute_indicators` on the 15-min DataFrame (this is also where `is_main_session` produces an `in_session` mask). Force `signal = 0` outside the session. Compute the per-row `sigma_daily` column (§10.3).
2. **Vectorized signal**: call `generate_signals` to produce the `signal` and `target_position` columns. Because §6.3 defines signals as a function of `(prev_signal, regime, band, gate)`, this step is sequential by nature; implement it as an explicit loop or via a stateful scan, but treat the result as derived from observation-only data — no future leakage.
3. **Sequential execution**: iterate rows in order with `itertuples`. For each row *t-1* whose `target_position` differs from the carried `position`, simulate a fill that occurs **during bar *t*** using the 1-min sub-candles of bar *t* and the `sigma_daily` value at bar *t* (§10.3). The resulting `trade_qty_from_prev`, `fill_price_from_prev`, and `fees_from_prev` are stored on **row *t***. Set `triggering_signal[t] = signal[t-1]` for convenience.
4. Carry `position` and `cash` forward across rows; compute `equity = cash + position * close` on every row.
**Mark-to-market convention on trade bars.** When a trade fills on row *t* (triggered by row *t-1*'s signal), the resulting `equity` at row *t*'s close includes both (a) the realized cost of the fill itself and (b) the unrealized P&L from the post-fill price drift over the remainder of bar *t* — typically up to ~10 minutes of price evolution after the K-minute execution window. This is the correct convention but it makes equity on entry/exit bars a mixture of realized and unrealized P&L, which can be visually confusing on the equity curve. Subsequent bars contain only marked unrealized P&L until the next trade.
 
The output is the canonical DataFrame from Section 4.1.
 
The row loop is not a performance bottleneck for v1 (15-min candles; tens of thousands of rows per year per instrument). Keeping it as an explicit loop also keeps the lookahead boundary visible in the code.
 
### 10.3 Execution Model (Volume-Aware, Small-Participant)
 
The execution model is implemented in `backtest/simulator.py`. Throughout this spec, "execution model" refers to the algorithm; "simulator" refers to its implementation file. They are the same thing.
 
Filling at "next bar's open" is not realistic — a single auction print is not a tradeable price for arbitrary size. v1 uses the model below.
 
**Operating regime assumption.** v1 trades at most `max_position_lots = 5` lots, which on liquid MOEX names is a tiny fraction of any single 1-min bar's volume. Market impact in this regime is dominated by the bid-ask spread, not by walking the book. The model below reflects this: a half-spread cost is the floor, and a small square-root impact term scaled by *daily* volatility is added on top.
 
**Inputs** (per fill):
- `qty`: signed quantity to trade (in lots).
- 1-min candles covering **bar *t*** (the bar during which execution occurs, triggered by the signal computed on row *t-1*): 15 sub-candles for a 15-min parent bar, each with `o, h, l, c, volume`.
- `sigma_daily[t]`: an **ex-ante** daily-return volatility estimate valid at bar *t* (see below). Distinct from the realized volatility of the execution window.
**Algorithm**:
1. Determine the **execution window**: the first K 1-min sub-candles of bar *t*, where K is small (config-default 5 minutes). This represents working the order over the early part of the bar rather than touching only the first print.
2. Compute the **VWAP** of those K sub-candles using `(h+l+c)/3` as the typical price weighted by volume.
3. Compute **available volume** as `participation_rate × sum(volume)` over the K sub-candles. This caps how much you may consume.
4. If `|qty| > available_volume`, the fill is **partial**: clip `qty` to `sign(qty) × available_volume`. The canonical row's `trade_qty_from_prev` reflects the actual filled amount, not the requested one. The remainder is **not** carried over to subsequent bars. (At v1 sizes this branch is never taken in practice; it exists for correctness and is exercised by a synthetic low-volume fixture in §14.)
5. Compute the **half-spread cost**. v1 estimates spread per bar from the K 1-min sub-candles using a conservative (wide) proxy: for each sub-candle *i* compute `range_bps_i = (high_i - low_i) / close_i × 10_000`, then take `range_bps = max_i range_bps_i` over the K sub-candles. The bar's spread estimate is `spread_bps = max(min_spread_bps, range_bps × spread_to_range_ratio)` with `min_spread_bps = 2` (config-default, a floor for liquid MOEX names) and `spread_to_range_ratio = 0.10` (config-default; the spread is typically a small fraction of the 1-min range). The half-spread cost is `spread_bps / 2`. Using the *max* per-candle range (rather than mean or window-wide range) is intentionally conservative and gives a stable, slightly wide estimate — the right direction of error for a v1 backtest.
6. Compute **square-root impact** scaled by daily volatility: `impact_bps = impact_coeff × sigma_daily_bps × sqrt(|qty| / available_volume)`, where `sigma_daily_bps = sigma_daily × 10_000`. With `impact_coeff = 0.1` and `sigma_daily = 0.02` (200 bps/day), a 1% participation produces `impact_bps = 0.1 × 200 × 0.1 = 2 bps` — small but non-zero, consistent with the small-participant regime. At v1 sizes this term is dominated by the half-spread.
7. Total adverse offset: `slippage_bps = half_spread_bps + impact_bps`.
8. Final `fill_price = VWAP × (1 + sign(qty) × slippage_bps / 10_000)`.
9. `fees = max(min_commission_per_order, commission_bps × |qty| × fill_price × lot_size / 10_000)`. The `min_commission_per_order` floor (config; v1 default `0`, in instrument currency) reflects that T-Bank's actual fee schedule may impose a minimum charge per order on small trades. Operators on accounts with a known minimum should set this to match; the default of `0` matches a flat-bps schedule. Using the floor prevents the backtest from underestimating costs at small notional sizes.
**`sigma_daily` source.** `sigma_daily` is **a per-row column** computed before the engine's sequential pass, equal to a **60-day rolling standard deviation of daily close-to-close returns** ending strictly before the row's calendar date. This guarantees that the sigma used for a fill on row *t* uses no information from row *t* or later — sigma is causal at every point in time, not a single global value. Implementation: aggregate the cached daily candles to one return per day, compute `daily_returns.rolling(sigma_daily_window_days).std()`, then forward-fill onto the 15-min DataFrame using `merge_asof` with `direction="backward"` so each 15-min bar picks up the most recent strictly-prior daily sigma. The column is reproducible across runs because it depends only on the daily candle cache.
 
**Documented limitations**:
- Spread is modeled, not measured. v1 has no order book data and infers spread from 1-min ranges. Real spreads can deviate, especially around news.
- Square-root impact is a stylized fact, not a calibrated model. The `impact_coeff` is a placeholder; a v2 calibration pass against historical fills would refit it.
- Stochasticity is absent — the same window always produces the same fill price.
- Queue position, hidden orders, locked/crossed markets, and circuit breakers are out of scope.
These are explicit. The point of the model is not to predict fills exactly but to ensure the backtest cannot produce unrealistically tight results by filling at a single auction print with zero cost.
 
### 10.4 Train / Validation / Test Splits
 
`backtest/splits.py` exposes:
 
- `resolve_splits(splits_config, today) -> dict` — resolves the relative `splits` block (durations anchored to `today`) into concrete date ranges for `train`, `validation`, `test`, and `warmup`. The `warmup` window precedes `train`.
- `get_split(resolved, name) -> (start_date, end_date)` for `name` in `{"warmup", "train", "validation", "test"}`.
- `validate_metric_windows_disjoint(resolved)` — asserts that the metric (evaluation) windows of `train`, `validation`, and `test` do not overlap each other. Adjacent and ordered in time is fine; overlapping is not.
- `validate_warmup_discipline(resolved, run_split)` — for a given run's split (e.g. `validation`), asserts that the run's warmup prefix is not used as metric data *within the same run*. It does **not** prohibit a warmup prefix from falling inside a different split's metric window: that is a deliberate, accepted design (see "Warmup discipline" below).
**Warmup discipline.** Indicators with rolling/EWMA history would otherwise be `warmup_invalid` for the first `max(slow_ewm_span, bb_period)` 15-min bars of any backtest. Separately, `sigma_daily` (computed on **daily** candles) is NaN for the first `sigma_daily_window_days` calendar days of available daily history. A 15-min bar is tagged `warmup_invalid` if **either** the indicator warmup is incomplete **or** `sigma_daily` is NaN at that bar. To avoid contaminating validation/test metrics with warmup-region values, **every backtest run loads a warmup prefix from data preceding its evaluation window and runs indicators across the full prefix+window, then computes metrics only on the evaluation window**. The warmup prefix is loaded as follows:
 
- `train` runs warm up from the global `warmup` window (preceding `train`).
- `validation` warms up from data preceding its start, which falls inside `train`'s evaluation window. This is acceptable: `train` data is not "fresh" to the strategy after parameter selection has occurred, and merely reading `train`'s tail to seed indicator state for a `validation` run does not constitute parameter leakage. Only parameter *selection* on `validation` would.
- `test` warms up from data preceding its start (the tail of `validation`).
The warmup data is used **only** to initialize indicator state. No metric is computed over the warmup prefix; no parameter is selected against it. `validate_warmup_discipline` enforces the in-run rule (warmup data of a run is not metric data of that same run); `validate_metric_windows_disjoint` enforces the cross-run rule (no two splits share any metric data).
 
The CLI accepts `--split train|validation|test` as an alternative to `--from / --to` and applies the corresponding window with the right warmup prefix automatically. **All parameter selection and strategy iteration is done on `train`. `validation` is used to pick between candidate parameter sets. `test` is touched at most once and the result is reported as-is, even if disappointing.** This discipline is enforced socially (by the developer), not technically — but the spec calls it out so violations are obvious.
 
### 10.5 Output
 
Engine emits the canonical DataFrame (Section 4.1), one row per bar.
 
## 11. Analytics
 
### 11.1 Metrics
 
`analytics/metrics.py` computes from the canonical DataFrame:
 
- Total return, CAGR.
- Max drawdown, drawdown duration.
- **Sharpe**: aggregate equity to **daily** (last `equity` value per session day), compute **simple** daily returns (`equity_t / equity_{t-1} - 1`), subtract daily simple risk-free rate (`(1 + risk_free_annual)^(1/252) - 1`), then `mean / std × sqrt(252)`. 15-min returns are autocorrelated and heteroskedastic with strong intraday seasonality; computing Sharpe at that frequency and naively annualizing inflates it. Daily aggregation is the standard quant practice and what this spec mandates.
- **Sortino**: same as Sharpe but divides by the standard deviation of *negative* excess daily returns only.
- **Calmar**: `cagr / abs(max_drawdown)`.
- Trade count, win rate, average holding bars, average trade size (lots), longest losing streak, total notional, total fees, total commission bps.
The `warmup_invalid` prefix of the equity series is excluded from all metric computations.
 
**`metrics.json` schema.** A single flat JSON object with these required keys:
 
```json
{
  "run_id":                       "string",
  "split":                        "train|validation|test|custom",
  "start_date":                   "YYYY-MM-DD",
  "end_date":                     "YYYY-MM-DD",
  "session_days":                 124,
  "initial_cash":                 1000000.0,
  "final_equity":                 1023456.78,
  "total_return":                 0.0235,
  "cagr":                         0.0481,
  "sharpe":                       0.92,
  "sortino":                      1.31,
  "calmar":                       1.17,
  "max_drawdown":                 -0.041,
  "max_drawdown_duration_days":   17,
  "longest_losing_streak_trades": 6,
  "trade_count":                  142,
  "win_rate":                     0.54,
  "avg_holding_bars":             6.3,
  "avg_trade_size_lots":          5.0,
  "total_notional":               83456789.0,
  "total_fees":                   1234.56,
  "total_commission_bps":         14.8
}
```
 
`sortino` uses the same daily-aggregation convention as `sharpe` but divides by downside-only standard deviation (negative excess returns). `calmar` is `cagr / abs(max_drawdown)`. `total_commission_bps` is `total_fees / total_notional × 10_000`.
 
Future metrics may be added; consumers should tolerate extra keys but rely on the listed ones being present.
 
### 11.2 Charts
 
`analytics/charts.py` produces three matplotlib figures, saved as PNG files under `reports/<run_id>/`:
 
- `pnl.png` — equity curve with drawdown shaded.
- `position.png` — step chart of net position.
- `trades.png` — cumulative trade count.
A `metrics.json` summary is written alongside. No HTML, no interactivity — a PNG opens everywhere and is trivial to diff visually.
 
### 11.3 Reconstruction
 
`analytics/reconstruct.py` reads a JSONL run log and folds events into the canonical DataFrame — same columns, same dtypes as the backtest output. Specifically:
 
- `candle` events populate `open`/`high`/`low`/`close`/`volume`.
- `signal` events populate `signal`/`target_position` and the indicator inputs (kept as auxiliary columns for debugging). `triggering_signal` is derived as `signal.shift(1)`.
- `fill` events populate `trade_qty_from_prev`/`fill_price_from_prev`/`fees_from_prev` on the bar during which the fill occurred.
- `position` events populate `position`/`cash`/`equity`. Because they're emitted on every bar (§8.5), the resulting DataFrame has the same row density as the backtest.
**Single-run scope.** Reconstruction operates on exactly one run's JSONL log. Cross-run reconciliation — covering halts, operator-driven `flat` runs, restarts, and resumed positions across runs — is **explicitly out of scope for v1**. Each `run_id` is reconstructed independently. If a `halt` event appears in the log, the reconstructed DataFrame is truncated at the halt bar (no rows are produced after it) and assertions below are evaluated only on the in-run rows.
 
**Self-consistency assertions.** For every reconstructed row up to (and not including) any `halt` event, the function asserts:
- `abs(equity - (cash + position * close)) < eps` (equity matches cash + marked position).
- `position[t] = position[t-1] + trade_qty_from_prev[t]` (position evolves only through trades within this run).
- `cash[t] = cash[t-1] - trade_qty_from_prev[t] * fill_price_from_prev[t] * lot_size - fees_from_prev[t]`.
- The sum of `fee` across all reconstructed `fill` events for a `client_order_id` equals the `fees_from_prev` recorded for that fill in the canonical row.
Violations halt reconstruction with a diagnostic. These checks are cheap and catch live-bot bugs that would otherwise silently produce comparison drift attributed to "execution differences."
 
**Discrepancy policy.** If the live log contains `discrepancy` events (which precede a `halt`), the reconstruction uses **`intended_position`** (the bot's view) as the canonical row's `position` value, since the strategy and equity calculations were made against it. The `broker_position` value is preserved in an auxiliary column for debugging. This is the right choice for "what did the strategy do?" questions; if the question is "what is my actual P&L?", the broker view should be reconciled separately at run boundaries — out of scope here.
 
From that point, `metrics.py` and `charts.py` work on the canonical DataFrame identically. This is the keystone of the design: one schema, two sources, one set of analytics functions.
 
## 12. Comparison
 
`analytics/compare.py`:
 
1. Take a live JSONL log; determine its time window and instrument.
2. Extract the `candle` events from the log into a DataFrame. **The comparison backtest is run against these exact candles, not against freshly downloaded historical candles.** Exchanges and APIs sometimes revise historical data after the fact, so re-downloading produces a slightly different series and contaminates the drift signal with data drift. Using the logged candles ensures the only differences between live and backtest are due to the strategy/execution paths, not the inputs.
3. For execution, the backtest still needs 1-min candles. The live log does not contain 1-min candles, so they are fetched from the cache for the same window. This is a known imperfection — execution-window data may have been revised — but it is the smallest one available and is documented here. v2 will log 1-min candles alongside 15-min ones.
4. Reconstruct the live DataFrame from the JSONL log via `analytics/reconstruct.py`.
5. Align both on `(exchange, instrument, timestamp)`. The `warmup_invalid` prefix is dropped from both sides before comparison.
6. Compute per-row diffs: `position_diff`, `equity_diff`, and `trade_timing_diff` (signed bar offset between when a trade occurred in live vs backtest).
7. Render four matplotlib PNGs under `reports/compare/<run_id>/`:
   - `pnl_overlay.png`, `position_overlay.png`, `trades_overlay.png` — backtest vs live, two lines per chart.
   - `drift.png` — `live - backtest` over time, with a marker for any bar where signals or fills disagreed.
Nonzero drift surfaces real-world effects: latency between candle close and order placement, slippage that the model under- or over-estimates, partial fills, rejected orders, and missed candles. A drift report with annotated discrepancy bars is the primary debugging tool for the live bot.
 
## 13. CLI
 
`src/cli.py` exposes five entry points:
 
```
python -m src.cli backtest --config config/config.yaml [--split train|validation|test | --from YYYY-MM-DD --to YYYY-MM-DD]
python -m src.cli run      --config config/config.yaml
python -m src.cli flat     --config config/config.yaml [--confirm]
python -m src.cli report   --log data/logs/<run_id>.jsonl
python -m src.cli compare  --log data/logs/<run_id>.jsonl
```
 
`backtest` requires either `--split` or an explicit `--from / --to` window — never both. `run` reads instrument and credentials from config and `.env`. `report` and `compare` operate on existing logs and emit PNG files under `reports/`.
 
**`flat` — exit-to-zero procedure.** This is the safety subcommand invoked when the bot halts on discrepancy, when an operator wants to stop trading at end of day, or when manual intervention is needed for any other reason. It runs in a separate process from `run` and uses its own `flat_run_id` (a fresh UUID per invocation) for traceability. It:
 
1. Connects to T-Bank.
2. Reads the current broker position for the configured instrument.
3. Places a market order in the opposite direction sized to bring the position to exactly zero, using a `client_order_id` derived as `sha1("flat" || instrument || flat_run_id || retry_attempt)[:16]`. This derivation is intentionally distinct from the `run`-process derivation (§8.4) — the flat subcommand has no `signal_timestamp` and must not collide with any in-progress `run`.
4. Waits for fill confirmation (bounded; defaults to 30 seconds).
5. Verifies position is zero; if not, retries up to 2 times with incremented `retry_attempt` (and therefore fresh `client_order_id`s); if still non-zero, exits with code 2 and a clear diagnostic.
6. Writes a `flat_run` log under `data/logs/<flat_run_id>.jsonl` so the action is auditable.
The `--confirm` flag is required for non-interactive execution; without it the command prompts. The procedure is independent of any running `run` instance — operators must stop `run` first.
 
**Exit code policy.** Every CLI subcommand follows:
- `0` — success (backtest completed, run shut down cleanly, flat reached zero).
- `1` — recoverable failure (data load error, transient broker error during backtest fetch).
- `2` — unrecoverable failure requiring operator attention (live `run` halted on discrepancy, `flat` could not reach zero, validation of config failed).
- `3` — usage error (missing config, conflicting flags).
This makes the CLI safe to wire into cron jobs, CI pipelines, or systemd units.
 
## 14. Testing
 
- Pure functions (`indicators`, `strategy`, `simulator`, `reconstruct`, `session`) get unit tests with hand-crafted candle fixtures.
- **EWMA recursion test**: verify that `compute_indicators` produces the same value at index *t* whether called on the full series ending at *t* or on a longer series and then sliced to *t*. This catches `adjust=True` regressions.
- **State machine completeness test**: enumerate all (prev_signal, regime, band, gate) tuples and assert that `generate_signals` produces the exact target from §6.3. No "default to previous" fallthroughs.
- **Cooldown / mid-cross gate test**: synthetic series that exits at the upper band, then immediately revisits the lower band without crossing `bb_mid`. Assert no re-entry until both gate conditions are satisfied.
- **No-lookahead invariant test (mutation form)**: given a candle DataFrame, mutate any candle from index *t+1* onward and re-run the engine; the canonical row at index *t* must be byte-identical to the original. Catches forward leakage of close prices into row *t*'s decision.
- **No-lookahead invariant test (step-jump form)**: construct a flat series with a single step jump in `close` at index *t*. Assert that `signal` cannot change at any row before *t+1* and `trade_qty_from_prev` cannot be non-zero before *t+2* (signal at *t+1* triggers fill during bar *t+2*). This catches off-by-one shift bugs that the mutation test cannot — bugs that use only past data but with the wrong offset.
- **Indicator lag test**: assert that `fast_ewm[t]`, `slow_ewm[t]`, `bb_mid[t]`, `bb_up[t]`, `bb_lo[t]` are equal to the unlagged-equivalent computation truncated at index *t-1*. Verifies the `.shift(1)` is in place and not silently dropped during a refactor.
- **Resample boundary test**: feed a 1-min synthetic series with a known closed-on/labeled-on boundary into the engine's resampling code (if any) and assert the resulting 15-min bars match expectations bit-for-bit. Catches `closed='left'` vs `'right'` and `label='left'` vs `'right'` confusions.
- **Golden-output engine test**: a known input series produces a known canonical DataFrame, including a synthetic low-volume bar that exercises the partial-fill code path (note: at v1 sizes the partial-fill branch is otherwise unreachable in real data, so the test is the only place it runs).
- **Round-trip test**: feed synthetic events through `event_log` -> `reconstruct` and verify the result matches a directly produced canonical DataFrame, with `position` mark events on every bar (including no-trade bars). Self-consistency assertions (§11.3) must pass.
- **Session predicate test**: assert `is_main_session` returns the expected boolean for a hand-curated grid of (weekday, time) tuples in UTC.
- **Idempotency test**: placing the same `client_order_id` twice via the fake broker results in only one logical order; on rejection, the retry uses a fresh id (§8.4).
- Broker code is tested with a fake client that returns canned responses; no live network calls in CI.
## 15. Libraries
 
| Purpose            | Library                                |
|--------------------|----------------------------------------|
| T-Bank API         | `tinkoff-investments` (official SDK)   |
| Data               | `pandas`, `numpy`, `pyarrow`           |
| Charts             | `matplotlib`                           |
| Config validation  | `pydantic`                             |
| Testing            | `pytest`                               |
 
## 16. Definition of Done (v1)
 
- A backtest run on the **train** window (with its warmup prefix) produces three PNG charts and a `metrics.json` matching the schema in §11.1 in `reports/<run_id>/`.
- A backtest run on the **validation** window produces the same outputs without strategy code changes.
- A live run during the MOEX main session writes a JSONL log with a `position` mark event on **every** 15-min bar in the session window, regardless of whether a trade occurred.
- The bot maintains its own intended position; the broker is polled every bar; in-flight orders are accounted for via `qty_remaining` and `order_timeout_seconds`; on real discrepancy the bot halts and remains halted until manually restarted.
- The `flat` subcommand brings the broker position to zero using its own `flat_run_id`-derived `client_order_id` and exits cleanly. CLI exit codes follow the policy in §13.
- Order placement uses deterministic `client_order_id`s with retry-attempt tie-breaker; verifiably idempotent for transient errors and verifiably fresh-id-on-retry for hard rejections, both covered by tests.
- All indicators feeding signal generation are lagged by one bar; the indicator-lag test passes.
- The state machine (with cooldown and `bb_mid` re-cross gate using the lagged `bb_mid`) is enumerated and the completeness test passes; the cooldown/mid-cross gate test passes.
- The mutation-form and step-jump-form no-lookahead tests both pass against the renamed `trade_qty_from_prev` schema.
- `report` regenerates identical-shape PNG charts from a live log; reconstruction self-consistency assertions (using renamed execution-time fields, scoped to a single run, truncated at any `halt`) pass.
- `compare` produces overlay PNGs and a drift PNG, replaying the **logged** candles (not freshly downloaded ones) through the backtest.
- `sigma_daily` is computed as a strictly causal per-row column; the no-lookahead test verifies this.
- The volume-unit verification (§7.1) has been performed and documented.
- All trading-hours filtering uses `is_main_session`; no signals or fills outside the main MOEX equity session.
- Every constraint in Section 2 holds: no custom classes outside library bases, no `while True`, side effects confined to `broker/` and `event_log.py`, all internal timestamps UTC, pydantic only at the config-load boundary.

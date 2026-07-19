"""Fetch OHLCV candles from T-Bank API with local CSV cache.

Usage:
    from datetime import datetime, timezone
    from candles import get_candles

    df = get_candles("BBG004730N88", "15min",
                     datetime(2024, 1, 1, tzinfo=timezone.utc),
                     datetime(2024, 6, 1, tzinfo=timezone.utc))

Requires TBANK_TOKEN env var (loaded automatically from bot/.env by config.py).
"""
import os
import time

import pandas as pd
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.exceptions import RequestError

from config import ASSETS, CACHE_DIR, PRIMARY_ASSET, PRIMARY_TF, TIMEFRAMES

_INTERVALS = {
    "1min":  CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5min":  CandleInterval.CANDLE_INTERVAL_5_MIN,
    "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "30min": CandleInterval.CANDLE_INTERVAL_30_MIN,
    "1h":    CandleInterval.CANDLE_INTERVAL_HOUR,
    "1day":  CandleInterval.CANDLE_INTERVAL_DAY,
}

# Retry delays (seconds) for transient errors, keyed by gRPC status string.
_RETRY_DELAYS = {
    "RESOURCE_EXHAUSTED": [10, 30, 60, 120],
    "UNAVAILABLE":        [2, 5, 15, 30],
}

_STREAM_LOG_EVERY = 5000    # log fetch progress every N candles
_CHECKPOINT_EVERY = 20000   # persist partial fetch to disk every N candles


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _q(q) -> float:
    return q.units + q.nano / 1e9


def _merge_candles(base: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Combine cached candles with freshly fetched rows, de-duplicated on
    timestamp and sorted. Safe when either side is empty."""
    add = pd.DataFrame(rows)
    if base.empty:
        combined = add
    elif add.empty:
        combined = base
    else:
        combined = pd.concat([base, add], ignore_index=True)
    return combined.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _save_atomic(df: pd.DataFrame, path) -> None:
    """Write via a temp file + rename so an interrupted write can't corrupt
    the cache (rename is atomic on the same filesystem)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _fetch(figi: str, timeframe: str, from_ts: pd.Timestamp, to_ts: pd.Timestamp,
           on_chunk=None) -> pd.DataFrame:
    """Fetch candles, streaming and accumulating rows.

    get_all_candles pages through the API lazily; a multi-year range is many
    sequential requests, so we stream and log progress instead of blocking
    silently on list(...). Rows accumulate ACROSS retries and a transient
    error resumes from the last candle received, so a long fetch survives
    intermittent API failures instead of restarting each time.

    on_chunk(rows), if given, is called every _CHECKPOINT_EVERY candles so the
    caller can persist partial progress to disk. Only pass it when the fetched
    range extends the cache forward (or the cache is empty); checkpointing a
    backward-fill would leave a hole the min/max gap detection can't see.
    """
    retry_counts = {status: 0 for status in _RETRY_DELAYS}
    rows: list[dict] = []
    cur_from = from_ts
    progress_at_last_error = 0
    print(f"    fetching {figi} {timeframe} {from_ts.date()}..{to_ts.date()} "
          f"from API...", flush=True)
    while True:
        try:
            with Client(os.environ["TBANK_TOKEN"]) as client:
                for c in client.get_all_candles(
                    instrument_id=figi,
                    from_=cur_from.to_pydatetime(),
                    to=to_ts.to_pydatetime(),
                    interval=_INTERVALS[timeframe],
                ):
                    ts = _utc(c.time)
                    # from_ is inclusive, so a resumed stream re-sends the last
                    # candle; skip anything we already have.
                    if rows and ts <= rows[-1]["timestamp"]:
                        continue
                    rows.append({
                        "timestamp": ts,
                        "open":  _q(c.open),
                        "high":  _q(c.high),
                        "low":   _q(c.low),
                        "close": _q(c.close),
                        "volume": c.volume,
                    })
                    if len(rows) % _STREAM_LOG_EVERY == 0:
                        print(f"      ...{len(rows)} candles so far "
                              f"(through {rows[-1]['timestamp'].date()})", flush=True)
                    if on_chunk is not None and len(rows) % _CHECKPOINT_EVERY == 0:
                        on_chunk(rows)
                        print(f"      checkpointed {len(rows)} candles to cache", flush=True)
            print(f"    done: {len(rows)} candles for {figi} {timeframe}", flush=True)
            return pd.DataFrame(rows)
        except RequestError as e:
            err = str(e)
            # Resume from the last candle received; reset the retry budget if we
            # made progress since the previous error (budget is per-stall).
            if rows:
                cur_from = rows[-1]["timestamp"]
                if len(rows) > progress_at_last_error:
                    retry_counts = {status: 0 for status in _RETRY_DELAYS}
                progress_at_last_error = len(rows)
            for status, delays in _RETRY_DELAYS.items():
                if status in err:
                    n = retry_counts[status]
                    if n < len(delays):
                        retry_counts[status] += 1
                        print(f"  {status} — retrying in {delays[n]}s "
                              f"(attempt {n + 1}/{len(delays)}, resuming from "
                              f"{cur_from.date()}, {len(rows)} candles so far)...", flush=True)
                        time.sleep(delays[n])
                        break
                    raise RuntimeError(
                        f"{status} fetching {figi} {timeframe}: gave up after {len(delays)} "
                        f"retries ({len(rows)} candles fetched before giving up)"
                    ) from e
            else:
                raise RuntimeError(
                    f"Unexpected API error fetching {figi} {timeframe}: {err}"
                ) from e


def get_candles(figi: str, timeframe: str, from_dt, to_dt) -> pd.DataFrame:
    """Return OHLCV candles for [from_dt, to_dt], fetching only uncached ranges."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{figi}_{timeframe}.csv"
    from_ts, to_ts = _utc(from_dt), _utc(to_dt)

    if path.exists():
        cached = pd.read_csv(path, parse_dates=["timestamp"])
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
    else:
        cached = pd.DataFrame()

    # Each gap carries a `checkpointable` flag: True only when the fetch
    # extends the cache forward (or the cache is empty), so partial saves stay
    # contiguous. A backward-fill can't be checkpointed mid-stream — a partial
    # save would leave a hole between it and the old cache start that the
    # min/max gap detection here would never notice.
    to_fetch = []  # (start, end, checkpointable)
    if cached.empty:
        to_fetch.append((from_ts, to_ts, True))
    else:
        if from_ts < cached["timestamp"].min():
            to_fetch.append((from_ts, cached["timestamp"].min(), False))
        if to_ts > cached["timestamp"].max():
            to_fetch.append((cached["timestamp"].max(), to_ts, True))

    if to_fetch:
        if cached.empty:
            print(f"  {figi} {timeframe}: no local cache", flush=True)
        else:
            print(f"  {figi} {timeframe}: cache covers "
                  f"{cached['timestamp'].min().date()}..{cached['timestamp'].max().date()}", flush=True)
        for s, e, _ in to_fetch:
            print(f"    gap to fetch: {s.date()}..{e.date()}", flush=True)

        # Fetch and persist gap-by-gap so an interruption keeps what arrived.
        for s, e, checkpointable in to_fetch:
            on_chunk = None
            if checkpointable:
                base = cached  # snapshot before this gap; captured per iteration
                on_chunk = lambda rows, base=base: _save_atomic(_merge_candles(base, rows), path)
            fetched = _fetch(figi, timeframe, s, e, on_chunk=on_chunk)
            cached = _merge_candles(cached, fetched.to_dict("records"))
            _save_atomic(cached, path)

    mask = (cached["timestamp"] >= from_ts) & (cached["timestamp"] <= to_ts)
    return cached[mask].reset_index(drop=True)


def load_all_candles(
    from_dt,
    to_dt,
    primary: pd.DataFrame | None = None,
    assets: dict | None = None,
    timeframes: list | None = None,
    primary_key: tuple | None = None,
) -> dict:
    """Load candles for every (asset, timeframe) pair.

    Returns {(asset, timeframe): DataFrame}. If `primary` is given, it is used
    as-is for `primary_key` instead of being loaded again. `assets`,
    `timeframes` and `primary_key` default to the config values; pass a
    model's own lists when building features for an already-trained model.
    """
    assets = assets if assets is not None else ASSETS
    timeframes = timeframes if timeframes is not None else TIMEFRAMES
    primary_asset, primary_tf = primary_key if primary_key is not None else (PRIMARY_ASSET, PRIMARY_TF)
    all_candles = {}
    for asset, figi in assets.items():
        for tf in timeframes:
            if primary is not None and asset == primary_asset and tf == primary_tf:
                all_candles[(asset, tf)] = primary
            else:
                all_candles[(asset, tf)] = get_candles(figi, tf, from_dt, to_dt)
            print(f"  {asset} {tf}: {len(all_candles[(asset, tf)])} bars")
    return all_candles

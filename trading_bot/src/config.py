"""Configuration loading and validation (SPEC 5).

``pydantic.BaseModel`` is used **here only**, at the config-load boundary: load
YAML -> validate -> ``model_dump()`` to a plain dict. No function elsewhere in
the project accepts a pydantic model. This keeps the whole codebase dict-native
and avoids leaking class-based state into business logic (SPEC 2, 5).

Secrets (``TBANK_TOKEN``, ``TBANK_ACCOUNT_ID``) are never read from here; they
live in ``.env`` and are read by ``broker/client.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from src.core.types import Instrument


class _Instrument(BaseModel):
    exchange: str
    symbol: str
    figi: str
    currency: str
    lot_size: int = Field(gt=0)
    timeframe: str = "15min"


class _Strategy(BaseModel):
    fast_ewm_span: int = Field(gt=0)
    slow_ewm_span: int = Field(gt=0)
    bb_period: int = Field(gt=0)
    bb_std_mult: float = Field(gt=0)
    entry_cooldown_bars: int = Field(ge=0)

    @model_validator(mode="after")
    def _fast_below_slow(self):
        if self.fast_ewm_span >= self.slow_ewm_span:
            raise ValueError("fast_ewm_span must be < slow_ewm_span")
        return self


class _Risk(BaseModel):
    max_position_lots: int = Field(gt=0)


class _Backtest(BaseModel):
    commission_bps: float = Field(ge=0)
    min_commission_per_order: float = Field(ge=0, default=0.0)
    participation_rate: float = Field(gt=0, le=1)
    execution_window_minutes: int = Field(gt=0)
    min_spread_bps: float = Field(ge=0)
    spread_to_range_ratio: float = Field(ge=0)
    impact_coeff: float = Field(ge=0)
    sigma_daily_window_days: int = Field(gt=0)
    initial_cash: float = Field(gt=0)


class _Splits(BaseModel):
    warmup_days: int = Field(gt=0)
    train_months: int = Field(gt=0)
    validation_months: int = Field(gt=0)
    test_months: int = Field(gt=0)


class _Live(BaseModel):
    reconcile_every_n_bars: int = Field(gt=0, default=1)
    startup_warmup_bars: Optional[int] = None
    order_timeout_seconds: float = Field(gt=0, default=30.0)
    signal_log_close_history_bars: Optional[int] = None


class _Metrics(BaseModel):
    sharpe_aggregation: str = "daily"
    risk_free_annual: float = 0.0
    return_method: str = "simple"


class _Config(BaseModel):
    instrument: _Instrument
    strategy: _Strategy
    risk: _Risk
    backtest: _Backtest
    splits: _Splits
    live: _Live = _Live()
    metrics: _Metrics = _Metrics()


def load_config(path) -> dict:
    """Load + validate the YAML config, returning a plain dict (SPEC 5).

    Raises ``ValueError`` (via pydantic) on schema violations; the CLI maps that
    to exit code 2 (config validation failed).
    """
    raw = yaml.safe_load(Path(path).read_text())
    return _Config(**raw).model_dump()


def instrument_from_config(config: dict) -> Instrument:
    """Build the ``Instrument`` NamedTuple from a validated config dict."""
    ins = config["instrument"]
    return Instrument(
        exchange=ins["exchange"],
        symbol=ins["symbol"],
        figi=ins["figi"],
        currency=ins["currency"],
        lot_size=ins["lot_size"],
    )

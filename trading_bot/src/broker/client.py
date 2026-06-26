"""Authenticated T-Bank Invest SDK client factory (SPEC 7.1).

This is one of only two side-effecting areas in the project (the other being
``event_log.py``). The ``tinkoff.invest`` SDK is imported lazily inside the
functions so the rest of the project imports cleanly in environments where the
SDK is not installed (e.g. running the pure-core tests).

No module-level state: every function takes its dependencies as arguments.
"""
from __future__ import annotations

import os
from typing import Optional


def read_token(env_var: str = "TBANK_TOKEN") -> str:
    """Read the API token from the environment (never from ``config.yaml``)."""
    token = os.environ.get(env_var)
    if not token:
        raise RuntimeError(
            f"{env_var} is not set. Copy .env.example to .env and fill it in, "
            f"then `export $(grep -v '^#' .env | xargs)` or use a dotenv loader."
        )
    return token


def read_account_id(env_var: str = "TBANK_ACCOUNT_ID") -> str:
    """Read the trading account id from the environment."""
    account_id = os.environ.get(env_var)
    if not account_id:
        raise RuntimeError(f"{env_var} is not set (see .env.example).")
    return account_id


def make_client(token: Optional[str] = None):
    """Return an open ``tinkoff.invest`` services client.

    The caller is responsible for closing it (the SDK's ``Client`` is a context
    manager; ``make_open_client`` below is the preferred entry point for the
    async runner). Kept as a thin factory so the network dependency is isolated.
    """
    from tinkoff.invest import Client  # lazy: network boundary

    token = token or read_token()
    # ``Client`` is a context manager; return the entered services object so the
    # broker functions can call it directly. Callers using the sync API should
    # prefer the ``with Client(token) as client:`` form in their own scope.
    return Client(token)


def make_async_client(token: Optional[str] = None):
    """Return an ``AsyncClient`` context manager for the live streaming runner."""
    from tinkoff.invest import AsyncClient  # lazy: network boundary

    token = token or read_token()
    return AsyncClient(token)

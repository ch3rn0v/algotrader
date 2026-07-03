"""Shared configuration: assets, date range, paths, and .env loading.

Importing this module loads bot/.env (if present) into os.environ, so every
script gets TBANK_TOKEN automatically.
"""
import os
from datetime import datetime, timezone
from pathlib import Path

BOT_DIR = Path(__file__).parent
CACHE_DIR = BOT_DIR / "cache"
OUTPUT_DIR = BOT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

# Verify FIGIs at tbank.ru/invest before use (see find_figi.py).
ASSETS = {
    "SBERP": "BBG0047315Y7",
    "TMOS": "TCSM61901X76",  # T's proxy for Moscow Exchange index
    "PLZL": "BBG000R607Y3",
    "SIBN": "BBG004S684M6",
    "PHOR": "BBG004S689R0",
}
PRIMARY_ASSET = "SBERP"
PRIMARY_FIGI = ASSETS[PRIMARY_ASSET]
PRIMARY_TF = "5min"
TIMEFRAMES = ["5min", "15min", "30min", "1h"]

FROM = datetime(2025, 1, 1, tzinfo=timezone.utc)
TO = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _load_env() -> None:
    env_path = BOT_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_env()

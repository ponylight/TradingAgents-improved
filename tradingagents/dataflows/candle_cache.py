"""Simple file-based cache for OHLCV candles. Avoids re-fetching from ccxt every run.

NOTE: ccxt_crypto.py has its own in-memory cache keyed by (symbol, exchange, timeframe, start_date, end_date).
This file-based cache uses (symbol, timeframe, since_ms, limit) as keys. The two caches don't share state,
so overlapping date ranges may cause redundant fetches. This is non-breaking but could be unified in the future.
"""

import json
import os
import time
import logging
from pathlib import Path

log = logging.getLogger("candle_cache")

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Cache TTL by timeframe
TTL_SECONDS = {
    "1h": 3600,
    "4h": 4 * 3600,
    "1d": 24 * 3600,
}


def _cache_key(symbol: str, timeframe: str, since: int, limit: int) -> str:
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    return f"{safe_symbol}-{timeframe}-{since}-{limit}"


def get_cached(symbol: str, timeframe: str, since: int, limit: int):
    """Return cached candles if fresh enough, else None."""
    key = _cache_key(symbol, timeframe, since, limit)
    path = CACHE_DIR / f"{key}.json"
    
    if not path.exists():
        return None
    
    try:
        with open(path) as f:
            data = json.load(f)
        
        ttl = TTL_SECONDS.get(timeframe, 4 * 3600)
        age = time.time() - data.get("saved_at", 0)
        
        if age > ttl:
            log.debug(f"Cache expired: {key} ({age:.0f}s > {ttl}s)")
            return None
        
        log.info(f"💾 Cache hit: {key} ({len(data['candles'])} candles, {age:.0f}s old)")
        return data["candles"]
    except Exception:
        return None


def save_cache(symbol: str, timeframe: str, since: int, limit: int, candles: list):
    """Save candles to cache."""
    key = _cache_key(symbol, timeframe, since, limit)
    path = CACHE_DIR / f"{key}.json"
    
    try:
        with open(path, "w") as f:
            json.dump({"candles": candles, "saved_at": time.time()}, f)
        log.debug(f"Cache saved: {key} ({len(candles)} candles)")
    except Exception as e:
        log.warning(f"Cache save failed: {e}")

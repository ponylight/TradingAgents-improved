"""
Multi-Exchange Client Manager — singleton that manages ccxt clients
for Bybit (execution), Binance, and Coinbase (confirmation venues).

Binance and Coinbase use public APIs only (no auth required for market data).
Auth credentials are accepted via env vars if available.
"""

import os
import logging
import ccxt

log = logging.getLogger("exchange_manager")

_instances: dict = {}


def _build_config(exchange_id: str) -> dict:
    """Build ccxt config from env vars. Works without auth for public endpoints."""
    config = {"enableRateLimit": True}

    env_prefix = exchange_id.upper()
    api_key = os.environ.get(f"{env_prefix}_API_KEY")
    secret = os.environ.get(f"{env_prefix}_SECRET")

    if api_key:
        config["apiKey"] = api_key
    if secret:
        config["secret"] = secret

    return config


def get_exchange(exchange_id: str) -> ccxt.Exchange:
    """Get or create a cached ccxt exchange instance.

    Supported: bybit, binance, coinbase.
    Instances are cached — safe to call repeatedly.
    """
    if exchange_id in _instances:
        return _instances[exchange_id]

    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unsupported exchange: {exchange_id}")

    config = _build_config(exchange_id)
    exchange_class = getattr(ccxt, exchange_id)

    try:
        exchange = exchange_class(config)
        _instances[exchange_id] = exchange
        log.info(f"Initialized {exchange_id} client (auth={'yes' if config.get('apiKey') else 'no'})")
        return exchange
    except Exception as e:
        log.warning(f"Failed to initialize {exchange_id}: {e}")
        raise


def get_all_exchanges(include: list[str] = None) -> dict[str, ccxt.Exchange]:
    """Get exchange instances for all requested venues.

    Returns only successfully initialized exchanges.
    Failures are logged and skipped (graceful degradation).
    """
    ids = include or ["bybit", "binance", "coinbase"]
    result = {}
    for eid in ids:
        try:
            result[eid] = get_exchange(eid)
        except Exception as e:
            log.warning(f"Skipping {eid}: {e}")
    return result

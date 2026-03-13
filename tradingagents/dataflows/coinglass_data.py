"""
Coinglass liquidation and derivatives data.

Supports optional COINGLASS_API_KEY for authenticated access.
Falls back to free Binance futures API when Coinglass is unavailable.
"""

import logging
import os

import requests
from datetime import datetime

log = logging.getLogger("coinglass")

COINGLASS_API_KEY = os.environ.get("COINGLASS_API_KEY", "")


def _coinglass_headers() -> dict:
    """Build request headers, including API key if available."""
    headers = {}
    if COINGLASS_API_KEY:
        headers["coinglassSecret"] = COINGLASS_API_KEY
    return headers


def _binance_long_short_fallback() -> str:
    """Fallback: get long/short ratio from Binance futures (free, no key)."""
    from tradingagents.dataflows.ccxt_crypto import get_binance_long_short_ratio
    return get_binance_long_short_ratio()


def get_liquidation_data() -> str:
    """Fetch BTC liquidation data from Coinglass, falling back to Binance derivatives positioning."""
    try:
        resp = requests.get(
            "https://open-api.coinglass.com/public/v2/liquidation_history?symbol=BTC&time_type=all",
            headers=_coinglass_headers(),
            timeout=10,
        )

        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                lines = ["# BTC Liquidation Data (Coinglass)\n"]
                for entry in data["data"][:20]:
                    ts = datetime.fromtimestamp(entry.get("t", 0) / 1000).strftime("%Y-%m-%d %H:%M")
                    long_liq = entry.get("longVolUsd", 0)
                    short_liq = entry.get("shortVolUsd", 0)
                    lines.append(f"{ts} | Long liq: ${long_liq:,.0f} | Short liq: ${short_liq:,.0f}")
                return "\n".join(lines)

        # Coinglass failed — log reason and fall through
        if not COINGLASS_API_KEY:
            log.info("No COINGLASS_API_KEY set. Set env var for Coinglass data. Falling back to Binance.")
        else:
            log.info(f"Coinglass returned {resp.status_code}. Falling back to Binance.")

    except Exception as e:
        log.info(f"Coinglass error: {e}. Falling back to Binance.")

    # Fallback: Binance derivatives positioning (funding + OI + L/S ratio)
    from tradingagents.dataflows.ccxt_crypto import get_crypto_liquidations_summary
    return get_crypto_liquidations_summary("BTC/USDT:USDT")


def get_long_short_ratio() -> str:
    """Fetch BTC long/short ratio from Coinglass, falling back to Binance futures."""
    try:
        resp = requests.get(
            "https://open-api.coinglass.com/public/v2/long_short?symbol=BTC&time_type=4",
            headers=_coinglass_headers(),
            timeout=10,
        )

        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                lines = ["# BTC Long/Short Ratio (Coinglass)\n"]
                for entry in data["data"][:10]:
                    exchange = entry.get("exchangeName", "Unknown")
                    long_rate = entry.get("longRate", 0)
                    short_rate = entry.get("shortRate", 0)
                    ratio = entry.get("longShortRatio", 0)
                    lines.append(f"{exchange}: Long {long_rate:.1%} | Short {short_rate:.1%} | Ratio {ratio:.2f}")
                return "\n".join(lines)

        if not COINGLASS_API_KEY:
            log.info("No COINGLASS_API_KEY. Falling back to Binance long/short ratio.")
        else:
            log.info(f"Coinglass L/S returned {resp.status_code}. Falling back to Binance.")

    except Exception as e:
        log.info(f"Coinglass L/S error: {e}. Falling back to Binance.")

    return _binance_long_short_fallback()

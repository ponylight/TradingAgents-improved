"""
Coinglass liquidation and derivatives data (free tier).
"""

import requests
from datetime import datetime


def get_liquidation_data() -> str:
    """Fetch BTC liquidation data from Coinglass public endpoints."""
    try:
        # Public liquidation endpoint
        resp = requests.get(
            "https://open-api.coinglass.com/public/v2/liquidation_history?symbol=BTC&time_type=all",
            timeout=10
        )
        
        if resp.status_code != 200:
            return f"Coinglass API returned {resp.status_code}. Free tier may require API key from coinglass.com"
        
        data = resp.json()
        if data.get("code") != "0":
            return f"Coinglass: {data.get('msg', 'Unknown error')}. Sign up at coinglass.com for free API key."
        
        lines = ["# BTC Liquidation Data (Coinglass)\n"]
        for entry in data.get("data", [])[:20]:
            ts = datetime.fromtimestamp(entry.get("t", 0) / 1000).strftime("%Y-%m-%d %H:%M")
            long_liq = entry.get("longVolUsd", 0)
            short_liq = entry.get("shortVolUsd", 0)
            lines.append(f"{ts} | Long liq: ${long_liq:,.0f} | Short liq: ${short_liq:,.0f}")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Coinglass fetch error: {e}. May need API key from coinglass.com"


def get_long_short_ratio() -> str:
    """Fetch BTC long/short ratio from Coinglass."""
    try:
        resp = requests.get(
            "https://open-api.coinglass.com/public/v2/long_short?symbol=BTC&time_type=4",
            timeout=10
        )
        
        if resp.status_code != 200:
            return f"Coinglass long/short ratio unavailable (status {resp.status_code})"
        
        data = resp.json()
        if data.get("code") != "0":
            return f"Coinglass long/short: {data.get('msg', 'Error')}"
        
        lines = ["# BTC Long/Short Ratio (Coinglass)\n"]
        for entry in data.get("data", [])[:10]:
            exchange = entry.get("exchangeName", "Unknown")
            long_rate = entry.get("longRate", 0)
            short_rate = entry.get("shortRate", 0)
            ratio = entry.get("longShortRatio", 0)
            lines.append(f"{exchange}: Long {long_rate:.1%} | Short {short_rate:.1%} | Ratio {ratio:.2f}")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Long/short ratio error: {e}"

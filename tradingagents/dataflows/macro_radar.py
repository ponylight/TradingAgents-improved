"""
Macro Signal Radar — 7-signal composite BUY/CASH verdict for BTC.

Inspired by World Monitor's market radar. All data from free APIs:
- Yahoo Finance (BTC, JPY, QQQ, XLP)
- alternative.me (Fear & Greed Index)
- mempool.space (Bitcoin hash rate)

Usage:
    from tradingagents.dataflows.macro_radar import get_macro_radar
    result = get_macro_radar()
    print(result["verdict"])  # "BUY" or "CASH"
    print(result["summary"])  # Human-readable summary
"""

import logging
import math
import requests
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

log = logging.getLogger(__name__)

YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}
YAHOO_TIMEOUT = 12
BULLISH_THRESHOLD = 0.57  # ≥57% of known signals must be bullish
MIN_KNOWN_SIGNALS = 5     # Require at least 5/7 signals for a verdict
CACHE_TTL = 300           # 5 minutes

# Shared session with retry/backoff
_session = requests.Session()
_retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retry))
_session.headers.update(YAHOO_HEADERS)

# Simple TTL cache
_cache = {}


def _cached_get(url: str, ttl: int = CACHE_TTL, **kwargs) -> Optional[requests.Response]:
    """GET with TTL cache and shared session."""
    now = time.time()
    if url in _cache and (now - _cache[url][0]) < ttl:
        return _cache[url][1]
    try:
        r = _session.get(url, timeout=kwargs.pop("timeout", YAHOO_TIMEOUT), **kwargs)
        r.raise_for_status()
        _cache[url] = (now, r)
        return r
    except Exception as e:
        log.warning(f"GET {url[:60]}... failed: {e}")
        return None


def _yahoo_chart(symbol: str, range_: str = "210d", interval: str = "1d") -> Optional[dict]:
    """Fetch OHLCV chart data from Yahoo Finance."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_}"
    r = _cached_get(url)
    if not r:
        return None
    try:
        result = r.json().get("chart", {}).get("result")
        if not result:
            log.warning(f"Yahoo returned no result for {symbol}")
            return None
        return result[0]
    except Exception as e:
        log.warning(f"Yahoo parse failed for {symbol}: {e}")
        return None


def _clean_series(data: dict) -> list[tuple[float, float]]:
    """Extract aligned (close, volume) pairs where both are finite."""
    closes = data["indicators"]["quote"][0].get("close", [])
    volumes = data["indicators"]["quote"][0].get("volume", [])
    pairs = []
    for i in range(min(len(closes), len(volumes))):
        c, v = closes[i], volumes[i]
        if c is not None and v is not None and math.isfinite(c) and math.isfinite(v):
            pairs.append((c, v))
    return pairs


def _clean_closes(data: dict) -> list[float]:
    """Extract non-null, finite close prices from Yahoo chart data."""
    return [c for c in data["indicators"]["quote"][0]["close"]
            if c is not None and math.isfinite(c)]


# === Signal 1: Fear & Greed Index ===

def signal_fear_greed() -> dict:
    """Fear & Greed Index. Bullish when > 50."""
    try:
        r = _cached_get("https://api.alternative.me/fng/?limit=1")
        if not r:
            return {"name": "Fear & Greed", "value": None, "bullish": None, "reason": "API unavailable"}
        d = r.json()["data"][0]
        value = int(d["value"])
        classification = d["value_classification"]
        bullish = value > 50
        return {
            "name": "Fear & Greed",
            "value": value,
            "detail": classification,
            "bullish": bullish,
            "reason": f"F&G={value} ({classification}), {'above' if bullish else 'below'} 50 threshold",
        }
    except Exception as e:
        log.warning(f"Fear & Greed signal failed: {e}")
        return {"name": "Fear & Greed", "value": None, "bullish": None, "reason": f"Error: {e}"}


# === Signal 2: BTC Technical Trend ===

def signal_btc_technical() -> dict:
    """BTC vs SMA50 + 30-day VWAP. Bullish when above both."""
    data = _yahoo_chart("BTC-USD", "210d")
    if not data:
        return {"name": "Technical Trend", "value": None, "bullish": None, "reason": "No BTC data"}

    closes = _clean_closes(data)

    if len(closes) < 50:
        return {"name": "Technical Trend", "value": None, "bullish": None, "reason": "Insufficient data"}

    btc_price = closes[-1]
    sma50 = sum(closes[-50:]) / 50
    sma200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None

    # 30-day VWAP using aligned close/volume pairs
    vwap = None
    pairs = _clean_series(data)
    if len(pairs) >= 20:
        recent_pairs = pairs[-30:]
        total_vol = sum(v for _, v in recent_pairs)
        if total_vol > 0:
            vwap = sum(c * v for c, v in recent_pairs) / total_vol

    above_sma50 = btc_price > sma50
    above_vwap = btc_price > vwap if vwap else None
    bullish = above_sma50 and (above_vwap if above_vwap is not None else True)

    # Mayer Multiple
    mayer = btc_price / sma200 if sma200 else None

    return {
        "name": "Technical Trend",
        "value": btc_price,
        "bullish": bullish,
        "detail": {
            "price": round(btc_price, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2) if sma200 else None,
            "vwap30": round(vwap, 2) if vwap else None,
            "mayer_multiple": round(mayer, 3) if mayer else None,
        },
        "reason": f"BTC ${btc_price:,.0f} {'>' if above_sma50 else '<'} SMA50 ${sma50:,.0f}, "
                  f"{'>' if above_vwap else '<'} VWAP ${vwap:,.0f}" if vwap else f"BTC ${btc_price:,.0f} vs SMA50 ${sma50:,.0f}",
    }


# === Signal 3: JPY Liquidity ===

def signal_jpy_liquidity() -> dict:
    """JPY/USD 30-day rate of change. Bullish when ROC > -2% (no yen squeeze)."""
    data = _yahoo_chart("JPY=X", "35d")
    if not data:
        return {"name": "Liquidity (JPY)", "value": None, "bullish": None, "reason": "No JPY data"}

    closes = _clean_closes(data)
    if len(closes) < 20:
        return {"name": "Liquidity (JPY)", "value": None, "bullish": None, "reason": "Insufficient JPY data"}

    jpy_now = closes[-1]
    jpy_30d = closes[0]
    roc = ((jpy_now - jpy_30d) / jpy_30d) * 100
    bullish = roc > -2

    return {
        "name": "Liquidity (JPY)",
        "value": round(roc, 2),
        "bullish": bullish,
        "detail": {"jpy_rate": round(jpy_now, 2), "roc_30d_pct": round(roc, 2)},
        "reason": f"JPY 30d ROC {roc:+.2f}%, {'no' if bullish else 'YES'} yen squeeze",
    }


# === Signal 4: Macro Regime (QQQ vs XLP) ===

def signal_macro_regime() -> dict:
    """QQQ 20-day ROC vs XLP 20-day ROC. Bullish when QQQ outperforms (risk-on)."""
    rocs = {}
    for sym in ["QQQ", "XLP"]:
        data = _yahoo_chart(sym, "25d")
        if not data:
            return {"name": "Macro Regime", "value": None, "bullish": None, "reason": f"No {sym} data"}
        closes = _clean_closes(data)
        if len(closes) < 15:
            return {"name": "Macro Regime", "value": None, "bullish": None, "reason": f"Insufficient {sym} data"}
        rocs[sym] = ((closes[-1] - closes[0]) / closes[0]) * 100

    risk_on = rocs["QQQ"] > rocs["XLP"]
    return {
        "name": "Macro Regime",
        "value": round(rocs["QQQ"] - rocs["XLP"], 2),
        "bullish": risk_on,
        "detail": {"qqq_roc20": round(rocs["QQQ"], 2), "xlp_roc20": round(rocs["XLP"], 2)},
        "reason": f"QQQ {rocs['QQQ']:+.2f}% vs XLP {rocs['XLP']:+.2f}% — {'risk-on' if risk_on else 'defensive'}",
    }


# === Signal 5: Flow Structure (BTC vs QQQ) ===

def signal_flow_structure() -> dict:
    """BTC 5-day return vs QQQ 5-day return. Bullish when gap < 5% (aligned)."""
    btc_data = _yahoo_chart("BTC-USD", "10d")
    qqq_data = _yahoo_chart("QQQ", "10d")

    if not btc_data or not qqq_data:
        return {"name": "Flow Structure", "value": None, "bullish": None, "reason": "Missing data"}

    btc_c = _clean_closes(btc_data)
    qqq_c = _clean_closes(qqq_data)

    if len(btc_c) < 6 or len(qqq_c) < 6:
        return {"name": "Flow Structure", "value": None, "bullish": None, "reason": "Insufficient data"}

    btc_5d = ((btc_c[-1] - btc_c[-6]) / btc_c[-6]) * 100
    qqq_5d = ((qqq_c[-1] - qqq_c[-6]) / qqq_c[-6]) * 100
    gap = abs(btc_5d - qqq_5d)
    aligned = gap < 5

    return {
        "name": "Flow Structure",
        "value": round(gap, 2),
        "bullish": aligned,
        "detail": {"btc_5d_pct": round(btc_5d, 2), "qqq_5d_pct": round(qqq_5d, 2), "gap_pct": round(gap, 2)},
        "reason": f"BTC {btc_5d:+.2f}% vs QQQ {qqq_5d:+.2f}% — gap {gap:.1f}%, {'aligned' if aligned else 'divergent'}",
    }


# === Signal 6: Hash Rate ===

def signal_hash_rate() -> dict:
    """Bitcoin mining hashrate 30-day change. Bullish when growing > 3%."""
    try:
        r = _cached_get("https://mempool.space/api/v1/mining/hashrate/1m", timeout=15)
        if not r:
            return {"name": "Hash Rate", "value": None, "bullish": None, "reason": "API unavailable"}
        hashrates = r.json().get("hashrates", [])

        if len(hashrates) < 2:
            return {"name": "Hash Rate", "value": None, "bullish": None, "reason": "Insufficient hashrate data"}

        current = hashrates[-1]["avgHashrate"]
        month_ago = hashrates[0]["avgHashrate"]
        change_pct = ((current - month_ago) / month_ago) * 100
        bullish = change_pct > 3

        return {
            "name": "Hash Rate",
            "value": round(change_pct, 2),
            "bullish": bullish,
            "detail": {"change_30d_pct": round(change_pct, 2)},
            "reason": f"Hash rate {change_pct:+.2f}% 30d — {'growing' if bullish else 'stagnant/declining'}",
        }
    except Exception as e:
        log.warning(f"Hash rate signal failed: {e}")
        return {"name": "Hash Rate", "value": None, "bullish": None, "reason": f"Error: {e}"}


# === Signal 7: Mining Cost ===

def signal_mining_cost() -> dict:
    """BTC price vs hashrate-implied cost floor. Bullish when price > $60K."""
    # Use the BTC price from technical signal to avoid duplicate API call
    btc_data = _yahoo_chart("BTC-USD", "5d")
    if not btc_data:
        return {"name": "Mining Cost", "value": None, "bullish": None, "reason": "No BTC data"}

    closes = _clean_closes(btc_data)
    if not closes:
        return {"name": "Mining Cost", "value": None, "bullish": None, "reason": "No BTC closes"}

    btc_price = closes[-1]
    # $60K is approximate hashrate-implied production cost (World Monitor methodology)
    cost_floor = 60000
    bullish = btc_price > cost_floor

    return {
        "name": "Mining Cost",
        "value": round(btc_price, 2),
        "bullish": bullish,
        "detail": {"btc_price": round(btc_price, 2), "cost_floor": cost_floor},
        "reason": f"BTC ${btc_price:,.0f} {'>' if bullish else '<'} ${cost_floor:,.0f} production cost",
    }


# === Composite Verdict ===

def get_macro_radar() -> dict:
    """
    Compute 7-signal macro radar with composite BUY/CASH verdict.

    Returns dict with:
        - signals: list of 7 signal dicts
        - bullish_count: number of bullish signals
        - known_count: number of signals with data
        - bullish_pct: percentage bullish
        - verdict: "BUY" or "CASH"
        - summary: human-readable summary for agents
        - timestamp: UTC timestamp
    """
    signals = [
        signal_fear_greed(),
        signal_btc_technical(),
        signal_jpy_liquidity(),
        signal_macro_regime(),
        signal_flow_structure(),
        signal_hash_rate(),
        signal_mining_cost(),
    ]

    known = [s for s in signals if s["bullish"] is not None]
    bullish_count = sum(1 for s in known if s["bullish"])
    known_count = len(known)
    bullish_pct = bullish_count / known_count if known_count > 0 else 0

    # Require minimum signals for a confident verdict
    if known_count < MIN_KNOWN_SIGNALS:
        verdict = "UNKNOWN"
        log.warning(f"Macro radar: only {known_count}/{len(signals)} signals available, verdict UNKNOWN")
    else:
        verdict = "BUY" if bullish_pct >= BULLISH_THRESHOLD else "CASH"

    # Build summary
    bull_signals = [s["name"] for s in known if s["bullish"]]
    bear_signals = [s["name"] for s in known if not s["bullish"]]
    unknown = [s["name"] for s in signals if s["bullish"] is None]

    summary_lines = [
        f"MACRO RADAR: {verdict} ({bullish_count}/{known_count} bullish, {bullish_pct:.0%})",
        "",
        "BULLISH:" if bull_signals else "",
    ]
    for s in known:
        if s["bullish"]:
            summary_lines.append(f"  ✅ {s['name']}: {s['reason']}")

    if bear_signals:
        summary_lines.append("\nBEARISH:")
        for s in known:
            if not s["bullish"]:
                summary_lines.append(f"  ❌ {s['name']}: {s['reason']}")

    if unknown:
        summary_lines.append(f"\nUNKNOWN: {', '.join(unknown)}")

    # Add key metrics inline
    tech = next((s for s in signals if s["name"] == "Technical Trend"), None)
    if tech and tech.get("detail"):
        d = tech["detail"]
        summary_lines.append(f"\nBTC: ${d.get('price', 0):,.0f} | SMA50: ${d.get('sma50', 0):,.0f} | "
                             f"SMA200: ${d.get('sma200', 0):,.0f}" if d.get('sma200') else "")
        if d.get("mayer_multiple"):
            summary_lines.append(f"Mayer Multiple: {d['mayer_multiple']:.3f} "
                                 f"({'overheated' if d['mayer_multiple'] > 2.4 else 'undervalued' if d['mayer_multiple'] < 0.8 else 'normal'})")

    return {
        "signals": signals,
        "bullish_count": bullish_count,
        "known_count": known_count,
        "bullish_pct": round(bullish_pct, 4),
        "verdict": verdict,
        "summary": "\n".join(summary_lines),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# === Stablecoin Peg Health (bonus) ===

def get_stablecoin_health() -> dict:
    """Monitor stablecoin peg health via CoinGecko. Free, no auth."""
    coins = {
        "tether": "USDT",
        "usd-coin": "USDC",
        "dai": "DAI",
    }
    try:
        ids = ",".join(coins.keys())
        r = _cached_get(
            f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true",
        )
        if not r:
            return {"coins": {}, "overall": "UNKNOWN", "summary": "CoinGecko API unavailable"}
        data = r.json()

        results = {}
        any_depeg = False
        any_warning = False
        for cg_id, symbol in coins.items():
            price = data.get(cg_id, {}).get("usd", 1.0)
            deviation = abs(price - 1.0) * 100
            if deviation > 1.0:
                status = "DEPEGGED"
                any_depeg = True
            elif deviation > 0.5:
                status = "SLIGHT_DEPEG"
                any_warning = True
            else:
                status = "ON_PEG"
            results[symbol] = {"price": price, "deviation_pct": round(deviation, 3), "status": status}

        overall = "WARNING" if any_depeg else "CAUTION" if any_warning else "HEALTHY"
        return {
            "coins": results,
            "overall": overall,
            "summary": " | ".join(f"{s}: ${d['price']:.4f} ({d['status']})" for s, d in results.items()),
        }
    except Exception as e:
        log.warning(f"Stablecoin health check failed: {e}")
        return {"coins": {}, "overall": "UNKNOWN", "summary": f"Error: {e}"}


# === Cached wrapper for multi-caller dedup ===

_radar_cache = {"result": None, "ts": 0}
_stable_cache = {"result": None, "ts": 0}


def get_macro_radar_cached(ttl: int = CACHE_TTL) -> dict:
    """Cached macro radar — prevents duplicate API calls from prefetch + tool invocations."""
    now = time.time()
    if _radar_cache["result"] and (now - _radar_cache["ts"]) < ttl:
        return _radar_cache["result"]
    result = get_macro_radar()
    _radar_cache["result"] = result
    _radar_cache["ts"] = now
    return result


def get_stablecoin_health_cached(ttl: int = CACHE_TTL) -> dict:
    """Cached stablecoin health."""
    now = time.time()
    if _stable_cache["result"] and (now - _stable_cache["ts"]) < ttl:
        return _stable_cache["result"]
    result = get_stablecoin_health()
    _stable_cache["result"] = result
    _stable_cache["ts"] = now
    return result


# CLI test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("MACRO SIGNAL RADAR")
    print("=" * 60)
    result = get_macro_radar()
    print(result["summary"])
    print("\n" + "=" * 60)
    print("STABLECOIN HEALTH")
    print("=" * 60)
    health = get_stablecoin_health()
    print(health["summary"])

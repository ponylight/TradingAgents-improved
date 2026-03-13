"""
Bitcoin On-Chain Fundamentals Module

Fetches on-chain data from free public APIs:
- Blockchain.info: hash rate, difficulty, active addresses, tx volume, mempool
- Mempool.space: difficulty adjustment, fee estimates
- CoinGecko: market data, supply metrics, ATH distance

Returns structured data for the Crypto Fundamentals Analyst agent.
No API keys required. All public endpoints.
"""

from __future__ import annotations
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

log = logging.getLogger("onchain")

TIMEOUT = 15
_cache: Dict[str, tuple] = {}  # key -> (timestamp, data)
CACHE_TTL = 300  # 5 minutes


def _get_cached(key: str, fetcher, ttl: int = CACHE_TTL) -> Any:
    """Simple in-memory cache."""
    now = time.time()
    if key in _cache and (now - _cache[key][0]) < ttl:
        return _cache[key][1]
    data = fetcher()
    _cache[key] = (now, data)
    return data


MEMPOOL_MIRRORS = [
    "mempool.space",
    "mempool.emzy.de",
    "mempool.bisq.services",
]


def _safe_get(url: str, params: dict = None, timeout: int = TIMEOUT) -> Optional[dict]:
    """Safe HTTP GET with error handling and mempool.space mirror fallback."""
    # If it's a mempool.space URL, try all mirrors
    if "mempool.space" in url:
        last_err = None
        for mirror in MEMPOOL_MIRRORS:
            mirror_url = url.replace("mempool.space", mirror, 1)
            try:
                r = requests.get(mirror_url, params=params, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                log.info(f"Mempool mirror {mirror} failed: {e}")
                last_err = e
        # For fee endpoints, try blockchain.info as last resort
        if "/fees/" in url:
            try:
                r = requests.get("https://blockchain.info/q/fees", timeout=timeout)
                r.raise_for_status()
                # blockchain.info returns a simple number (sat/byte)
                fee = int(r.text.strip())
                return {"fastestFee": fee, "halfHourFee": fee, "hourFee": max(fee // 2, 1),
                        "economyFee": max(fee // 4, 1), "minimumFee": 1}
            except Exception:
                pass
        # For hashrate endpoints, try blockchain.info/stats
        if "/mining/hashrate/" in url:
            try:
                r = requests.get("https://api.blockchain.info/stats", timeout=timeout)
                r.raise_for_status()
                stats = r.json()
                hr = stats.get("hash_rate", 0)
                # Return minimal structure compatible with get_hashrate_history
                return {"hashrates": [{"avgHashrate": hr, "timestamp": int(time.time())}]}
            except Exception:
                pass
        log.warning(f"All mempool mirrors failed for {url}: {last_err}")
        return None

    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"API error {url}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  Data Sources
# ═══════════════════════════════════════════════════════════════

def get_blockchain_stats() -> Dict[str, Any]:
    """Blockchain.info stats: hash rate, difficulty, tx count, price."""
    def fetch():
        data = _safe_get("https://api.blockchain.info/stats")
        if not data:
            return {}
        return {
            "hash_rate_eh": round(data.get("hash_rate", 0) / 1e9, 1),
            "difficulty_t": round(data.get("difficulty", 0) / 1e12, 2),
            "minutes_between_blocks": round(data.get("minutes_between_blocks", 0), 1),
            "blocks_mined_24h": data.get("n_blocks_mined", 0),
            "btc_sent_24h": round(data.get("total_btc_sent", 0) / 1e8, 0),
            "tx_count_24h": data.get("n_tx", 0),
            "market_price_usd": round(data.get("market_price_usd", 0), 2),
            "miners_revenue_usd": round(data.get("miners_revenue_usd", 0), 0) or round(data.get("n_blocks_mined", 144) * 3.125 * data.get("market_price_usd", 0), 0),  # fallback: blocks × reward × price
            "mempool_size_bytes": data.get("mempool_size_bytes", 0),
            "mempool_count": data.get("mempool_count", 0),
        }
    return _get_cached("blockchain_stats", fetch)


def get_active_addresses(timespan: str = "30days") -> Dict[str, Any]:
    """Unique active addresses over time from blockchain.info."""
    def fetch():
        data = _safe_get(f"https://api.blockchain.info/charts/n-unique-addresses?timespan={timespan}&format=json")
        if not data or "values" not in data:
            return {}
        vals = data["values"]
        if not vals:
            return {}
        latest = vals[-1]["y"]
        avg = sum(v["y"] for v in vals) / len(vals)
        # Trend: compare last 7 days avg to previous 7 days
        if len(vals) >= 14:
            recent_7 = sum(v["y"] for v in vals[-7:]) / 7
            prev_7 = sum(v["y"] for v in vals[-14:-7]) / 7
            trend_pct = ((recent_7 - prev_7) / prev_7) * 100 if prev_7 > 0 else 0
        else:
            trend_pct = 0
        return {
            "latest": int(latest),
            "avg_30d": int(avg),
            "trend_7d_pct": round(trend_pct, 2),
        }
    return _get_cached("active_addresses", fetch)


def get_tx_volume_usd(timespan: str = "30days") -> Dict[str, Any]:
    """Estimated transaction volume in USD."""
    def fetch():
        data = _safe_get(f"https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan={timespan}&format=json")
        if not data or "values" not in data:
            return {}
        vals = data["values"]
        if not vals:
            return {}
        latest = vals[-1]["y"]
        avg = sum(v["y"] for v in vals) / len(vals)
        return {
            "latest_usd": round(latest, 0),
            "avg_30d_usd": round(avg, 0),
            "latest_b": round(latest / 1e9, 2),
            "avg_30d_b": round(avg / 1e9, 2),
        }
    return _get_cached("tx_volume", fetch)


def get_mempool_info() -> Dict[str, Any]:
    """Mempool.space: fee estimates and congestion."""
    def fetch():
        fees = _safe_get("https://mempool.space/api/v1/fees/recommended")
        if not fees:
            return {}
        return {
            "fastest_fee_sat_vb": fees.get("fastestFee", 0),
            "half_hour_fee": fees.get("halfHourFee", 0),
            "hour_fee": fees.get("hourFee", 0),
            "economy_fee": fees.get("economyFee", 0),
            "minimum_fee": fees.get("minimumFee", 0),
        }
    return _get_cached("mempool_fees", fetch)


def get_difficulty_adjustment() -> Dict[str, Any]:
    """Mempool.space: next difficulty adjustment info."""
    def fetch():
        data = _safe_get("https://mempool.space/api/v1/difficulty-adjustment")
        if not data:
            return {}
        return {
            "progress_pct": round(data.get("progressPercent", 0), 1),
            "change_pct": round(data.get("difficultyChange", 0), 2),
            "estimated_change_pct": round(data.get("estimatedChange", data.get("difficultyChange", 0)), 2),
            "remaining_blocks": data.get("remainingBlocks", 0),
            "remaining_time_sec": data.get("remainingTime", 0),
            "previous_retarget_pct": round(data.get("previousRetarget", 0), 2),
        }
    return _get_cached("difficulty_adj", fetch)


def get_hashrate_history() -> Dict[str, Any]:
    """Mempool.space: recent hashrate trend."""
    def fetch():
        data = _safe_get("https://mempool.space/api/v1/mining/hashrate/1m")
        if not data or "hashrates" not in data:
            return {}
        hrs = data["hashrates"]
        if not hrs:
            return {}
        latest = hrs[-1].get("avgHashrate", 0) / 1e18
        # 7d avg vs 30d avg
        if len(hrs) >= 30:
            recent_7 = sum(h["avgHashrate"] for h in hrs[-7:]) / 7 / 1e18
            avg_30 = sum(h["avgHashrate"] for h in hrs) / len(hrs) / 1e18
            trend = ((recent_7 - avg_30) / avg_30) * 100 if avg_30 > 0 else 0
        else:
            recent_7 = latest
            avg_30 = latest
            trend = 0
        return {
            "latest_eh": round(latest, 1),
            "avg_7d_eh": round(recent_7, 1),
            "avg_30d_eh": round(avg_30, 1),
            "trend_pct": round(trend, 2),
        }
    return _get_cached("hashrate_history", fetch)


def get_coingecko_market_data() -> Dict[str, Any]:
    """CoinGecko: market cap, supply, ATH, price changes."""
    def fetch():
        data = _safe_get("https://api.coingecko.com/api/v3/coins/bitcoin",
                        params={"localization": "false", "tickers": "false",
                               "community_data": "false", "developer_data": "false"})
        if not data:
            return {}
        md = data.get("market_data", {})
        return {
            "price_usd": md.get("current_price", {}).get("usd", 0),
            "market_cap_b": round(md.get("market_cap", {}).get("usd", 0) / 1e9, 1),
            "volume_24h_b": round(md.get("total_volume", {}).get("usd", 0) / 1e9, 1),
            "circulating_supply_m": round(md.get("circulating_supply", 0) / 1e6, 3),
            "max_supply_m": 21.0,
            "supply_pct_mined": round((md.get("circulating_supply", 0) / 21e6) * 100, 2),
            "ath_usd": md.get("ath", {}).get("usd", 0),
            "ath_change_pct": round(md.get("ath_change_percentage", {}).get("usd", 0), 1),
            "change_24h_pct": round(md.get("price_change_percentage_24h", 0) or 0, 2),
            "change_7d_pct": round(md.get("price_change_percentage_7d", 0) or 0, 2),
            "change_30d_pct": round(md.get("price_change_percentage_30d", 0) or 0, 2),
            "change_1y_pct": round(md.get("price_change_percentage_1y", 0) or 0, 2),
        }
    return _get_cached("coingecko_market", fetch)


def get_halving_info() -> Dict[str, Any]:
    """Calculate halving cycle position (deterministic)."""
    HALVING_INTERVAL = 210_000
    LAST_HALVING_BLOCK = 840_000  # April 2024
    NEXT_HALVING_BLOCK = 1_050_000
    
    # Get current block height
    stats = get_blockchain_stats()
    # Approximate from blocks_mined
    try:
        data = _safe_get("https://mempool.space/api/blocks/tip/height")
        if isinstance(data, int):
            current_block = data
        else:
            current_block = 880_000  # fallback estimate for early 2026
    except:
        current_block = 880_000
    
    blocks_since_halving = current_block - LAST_HALVING_BLOCK
    blocks_until_next = NEXT_HALVING_BLOCK - current_block
    cycle_progress = (blocks_since_halving / HALVING_INTERVAL) * 100
    # Estimate days until next halving (~10 min per block)
    days_until = (blocks_until_next * 10) / (60 * 24)
    
    return {
        "current_block": current_block,
        "last_halving_block": LAST_HALVING_BLOCK,
        "next_halving_block": NEXT_HALVING_BLOCK,
        "blocks_since_halving": blocks_since_halving,
        "blocks_until_next": blocks_until_next,
        "cycle_progress_pct": round(cycle_progress, 1),
        "est_days_until_halving": round(days_until, 0),
        "current_block_reward": 3.125,
        "next_block_reward": 1.5625,
    }


# ═══════════════════════════════════════════════════════════════
#  Composite: Full Fundamentals Report
# ═══════════════════════════════════════════════════════════════

def get_btc_fundamentals() -> Dict[str, Any]:
    """Fetch all on-chain fundamentals. Returns structured dict."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "network_health": {
            **get_blockchain_stats(),
            "hashrate_trend": get_hashrate_history(),
            "difficulty_adjustment": get_difficulty_adjustment(),
            "mempool_fees": get_mempool_info(),
        },
        "adoption": {
            "active_addresses": get_active_addresses(),
            "tx_volume": get_tx_volume_usd(),
        },
        "valuation": {
            **get_coingecko_market_data(),
        },
        "supply": get_halving_info(),
    }


def format_fundamentals_report(data: Dict[str, Any]) -> str:
    """Format fundamentals data into a readable report for the LLM agent."""
    nh = data.get("network_health", {})
    ad = data.get("adoption", {})
    val = data.get("valuation", {})
    sup = data.get("supply", {})
    ht = nh.get("hashrate_trend", {})
    da = nh.get("difficulty_adjustment", {})
    fees = nh.get("mempool_fees", {})
    aa = ad.get("active_addresses", {})
    tv = ad.get("tx_volume", {})

    # === Weekend low-volume detection (Issue 5) ===
    _tv_latest = tv.get("latest_b", 0)
    _tv_avg = tv.get("avg_30d_b", 1) or 1
    _tv_pct = ((_tv_latest - _tv_avg) / _tv_avg * 100) if _tv_avg > 0 else 0
    _is_weekend = datetime.now(timezone.utc).weekday() >= 5
    _vol_note = ""
    if _tv_pct < -30 and _is_weekend:
        _vol_note = "  ⚠️ Weekend — low volume expected, not necessarily bearish."
    elif _tv_pct < -30:
        _vol_note = f"  ⚠️ Volume {_tv_pct:.0f}% below 30d avg — potentially bearish signal."
    _tx_volume_line = (
        f"- Tx volume (24h): ${_tv_latest:.1f}B (30d avg: ${tv.get('avg_30d_b', 0):.1f}B){_vol_note}"
    )

    # === Hash rate staleness detection (Issue 6) ===
    _generated_at = data.get("generated_at", "")
    _hashrate_stale_note = ""
    try:
        from datetime import timedelta
        _gen_time = datetime.fromisoformat(_generated_at.replace("Z", "+00:00")) if _generated_at else None
        if _gen_time:
            _age_hours = (datetime.now(timezone.utc) - _gen_time).total_seconds() / 3600
            if _age_hours > 24:
                _hashrate_stale_note = f"  ⚠️ STALE: data is {_age_hours:.0f}h old — do not rely on hash rate for decisions."
    except Exception:
        pass

    lines = [
        "# Bitcoin On-Chain Fundamentals Report",
        f"Generated: {data.get('generated_at', 'N/A')}",
        "",
        "## Network Health & Security",
        f"- Hash Rate: {nh.get('hash_rate_eh', 0)} EH/s (7d avg: {ht.get('avg_7d_eh', 0)} EH/s, 30d avg: {ht.get('avg_30d_eh', 0)} EH/s, trend: {ht.get('trend_pct', 0):+.1f}%){_hashrate_stale_note}",
        f"- Difficulty: {nh.get('difficulty_t', 0)}T",
        f"- Next adjustment: {da.get('change_pct', 0):+.2f}% in {da.get('remaining_blocks', 0)} blocks",
        f"- Block time: {nh.get('minutes_between_blocks', 0)} min (target: 10 min)",
        f"- Mempool fees: fastest {fees.get('fastest_fee_sat_vb', 0)} sat/vB, economy {fees.get('economy_fee', 0)} sat/vB",
        f"- Miners revenue (24h): ${nh.get('miners_revenue_usd', 0):,.0f}",
        "",
        "## Adoption & Usage",
        f"- Active addresses (latest): {aa.get('latest', 0):,} (30d avg: {aa.get('avg_30d', 0):,}, 7d trend: {aa.get('trend_7d_pct', 0):+.1f}%)",
        f"- Tx count (24h): {nh.get('tx_count_24h', 0):,}",
        _tx_volume_line,
        f"- BTC transferred (24h): {nh.get('btc_sent_24h', 0):,.0f} BTC",
        "",
        "## Valuation & Market",
        f"- Price: ${val.get('price_usd', 0):,.0f}",
        f"- Market Cap: ${val.get('market_cap_b', 0):.1f}B",
        f"- 24h Volume: ${val.get('volume_24h_b', 0):.1f}B",
        f"- Price changes: 24h {val.get('change_24h_pct', 0):+.1f}%, 7d {val.get('change_7d_pct', 0):+.1f}%, 30d {val.get('change_30d_pct', 0):+.1f}%, 1y {val.get('change_1y_pct', 0):+.1f}%",
        f"- ATH: ${val.get('ath_usd', 0):,.0f} (current is {val.get('ath_change_pct', 0):+.1f}% from ATH)",
        "",
        "## Supply & Halving Cycle",
        f"- Circulating: {sup.get('current_block_reward', 0)} BTC/block, {val.get('circulating_supply_m', 0):.3f}M / 21M ({val.get('supply_pct_mined', 0)}% mined)",
        f"- Halving cycle: {sup.get('cycle_progress_pct', 0):.1f}% complete (block {sup.get('current_block', 0):,})",
        f"- Next halving in ~{sup.get('est_days_until_halving', 0):.0f} days ({sup.get('blocks_until_next', 0):,} blocks)",
        f"- Current reward: {sup.get('current_block_reward', 0)} BTC → next: {sup.get('next_block_reward', 0)} BTC",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    data = get_btc_fundamentals()
    print(format_fundamentals_report(data))

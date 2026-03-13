"""
Cross-venue data aggregation for BTC/USDT.

Provides price alignment, funding rate divergence, OI comparison,
and spot/perp basis across Bybit, Binance, and Coinbase.

All functions use public APIs only — no auth required.
Partial results returned if some exchanges are unreachable.
"""

import time
import logging
from typing import Optional

from .exchange_manager import get_all_exchanges

log = logging.getLogger("cross_venue")

# Cache with TTL
_cache: dict = {}
_CACHE_TTL = 300  # 5 minutes for non-realtime data


def _cached(key: str, ttl: int = _CACHE_TTL):
    """Check cache. Returns (hit, value)."""
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < ttl:
        return True, entry["value"]
    return False, None


def _set_cache(key: str, value):
    _cache[key] = {"value": value, "ts": time.time()}


# ---------------------------------------------------------------------------
# Symbol mapping per exchange
# ---------------------------------------------------------------------------

def _spot_symbol(exchange_id: str) -> str:
    """Map to exchange-specific spot symbol."""
    if exchange_id == "coinbase":
        return "BTC/USD"
    return "BTC/USDT"


def _perp_symbol(exchange_id: str) -> str:
    """Map to exchange-specific perpetual symbol."""
    if exchange_id == "coinbase":
        return None  # Coinbase doesn't have perps on public API
    return "BTC/USDT:USDT"


# ---------------------------------------------------------------------------
# Core data functions
# ---------------------------------------------------------------------------

def get_multi_exchange_ticker(symbol: str = "BTC/USDT") -> dict:
    """Get price/spread from each exchange.

    Returns dict with per-exchange ticker data and a summary.
    """
    cache_key = f"ticker:{symbol}"
    hit, cached = _cached(cache_key, ttl=60)  # 1 min cache for prices
    if hit:
        return cached

    exchanges = get_all_exchanges()
    tickers = {}

    for eid, ex in exchanges.items():
        spot_sym = _spot_symbol(eid)
        try:
            t = ex.fetch_ticker(spot_sym)
            tickers[eid] = {
                "last": t.get("last"),
                "bid": t.get("bid"),
                "ask": t.get("ask"),
                "spread_pct": ((t["ask"] - t["bid"]) / t["bid"] * 100) if t.get("bid") and t.get("ask") else None,
                "volume_24h": t.get("quoteVolume"),
                "timestamp": t.get("timestamp"),
            }
        except Exception as e:
            log.warning(f"Ticker fetch failed for {eid}: {e}")
            tickers[eid] = {"error": str(e)}

    # Compute price divergence
    prices = {k: v["last"] for k, v in tickers.items() if isinstance(v.get("last"), (int, float))}
    summary = {}
    if len(prices) >= 2:
        max_p, min_p = max(prices.values()), min(prices.values())
        summary["max_divergence_pct"] = (max_p - min_p) / min_p * 100
        summary["price_range"] = f"${min_p:,.2f} - ${max_p:,.2f}"
        summary["venues_reporting"] = len(prices)
    else:
        summary["venues_reporting"] = len(prices)

    result = {"exchanges": tickers, "summary": summary, "ts": time.time()}
    _set_cache(cache_key, result)
    return result


def get_cross_venue_funding(symbol: str = "BTC/USDT") -> dict:
    """Get funding rates from Bybit + Binance.

    Coinbase excluded (no perps on public API).
    """
    cache_key = f"funding:{symbol}"
    hit, cached = _cached(cache_key)
    if hit:
        return cached

    exchanges = get_all_exchanges(include=["bybit", "binance"])
    funding = {}

    for eid, ex in exchanges.items():
        perp_sym = _perp_symbol(eid)
        if not perp_sym:
            continue
        try:
            rates = ex.fetch_funding_rate(perp_sym)
            funding[eid] = {
                "rate": rates.get("fundingRate"),
                "next_funding_time": rates.get("fundingDatetime"),
                "timestamp": rates.get("timestamp"),
            }
        except Exception as e:
            log.warning(f"Funding fetch failed for {eid}: {e}")
            funding[eid] = {"error": str(e)}

    # Divergence analysis
    valid_rates = {k: v["rate"] for k, v in funding.items() if isinstance(v.get("rate"), (int, float))}
    divergence = {}
    if len(valid_rates) >= 2:
        rates_list = list(valid_rates.values())
        divergence["spread_bps"] = abs(rates_list[0] - rates_list[1]) * 10000
        divergence["same_direction"] = all(r > 0 for r in rates_list) or all(r < 0 for r in rates_list)
        divergence["interpretation"] = (
            "aligned" if divergence["same_direction"] and divergence["spread_bps"] < 5
            else "minor_divergence" if divergence["spread_bps"] < 10
            else "significant_divergence"
        )

    result = {"exchanges": funding, "divergence": divergence, "ts": time.time()}
    _set_cache(cache_key, result)
    return result


def get_cross_venue_oi(symbol: str = "BTC/USDT") -> dict:
    """Get open interest from Bybit + Binance."""
    cache_key = f"oi:{symbol}"
    hit, cached = _cached(cache_key)
    if hit:
        return cached

    exchanges = get_all_exchanges(include=["bybit", "binance"])
    oi_data = {}

    for eid, ex in exchanges.items():
        perp_sym = _perp_symbol(eid)
        if not perp_sym:
            continue
        try:
            oi = ex.fetch_open_interest(perp_sym)
            oi_data[eid] = {
                "open_interest": oi.get("openInterestAmount"),
                "open_interest_value": oi.get("openInterestValue"),
                "timestamp": oi.get("timestamp"),
            }
        except Exception as e:
            log.warning(f"OI fetch failed for {eid}: {e}")
            oi_data[eid] = {"error": str(e)}

    result = {"exchanges": oi_data, "ts": time.time()}
    _set_cache(cache_key, result)
    return result


def get_spot_perp_basis(symbol: str = "BTC/USDT") -> dict:
    """Calculate spot vs perp basis.

    Spot price from Coinbase (or Binance spot fallback).
    Perp price from Bybit and Binance.
    """
    cache_key = f"basis:{symbol}"
    hit, cached = _cached(cache_key)
    if hit:
        return cached

    exchanges = get_all_exchanges()
    spot_price = None
    perp_prices = {}

    # Get spot price — prefer Coinbase, fallback to Binance spot
    for eid in ["coinbase", "binance", "bybit"]:
        if eid not in exchanges:
            continue
        try:
            t = exchanges[eid].fetch_ticker(_spot_symbol(eid))
            spot_price = t.get("last")
            spot_source = eid
            if spot_price:
                break
        except Exception as e:
            log.warning(f"Spot price fetch failed for {eid}: {e}")

    # Get perp prices
    for eid in ["bybit", "binance"]:
        if eid not in exchanges:
            continue
        perp_sym = _perp_symbol(eid)
        if not perp_sym:
            continue
        try:
            t = exchanges[eid].fetch_ticker(perp_sym)
            perp_prices[eid] = t.get("last")
        except Exception as e:
            log.warning(f"Perp price fetch failed for {eid}: {e}")

    basis = {}
    if spot_price and perp_prices:
        for eid, perp_p in perp_prices.items():
            if perp_p:
                basis_pct = (perp_p - spot_price) / spot_price * 100
                basis[eid] = {
                    "spot": spot_price,
                    "spot_source": spot_source,
                    "perp": perp_p,
                    "basis_pct": round(basis_pct, 4),
                    "interpretation": (
                        "contango" if basis_pct > 0.02
                        else "backwardation" if basis_pct < -0.02
                        else "flat"
                    ),
                }

    # Flag derivatives-led moves
    warning = None
    for eid, b in basis.items():
        if abs(b["basis_pct"]) > 0.5:
            warning = f"Large basis ({b['basis_pct']:+.3f}%) suggests derivatives-led move — likely short-lived"

    result = {"basis": basis, "warning": warning, "ts": time.time()}
    _set_cache(cache_key, result)
    return result


def get_cross_venue_confirmation(symbol: str = "BTC/USDT") -> dict:
    """Master summary: are venues aligned on direction?

    Returns a structured confirmation report suitable for analyst consumption.
    """
    cache_key = f"confirmation:{symbol}"
    hit, cached = _cached(cache_key, ttl=120)  # 2 min cache
    if hit:
        return cached

    # Gather all data
    ticker = get_multi_exchange_ticker(symbol)
    funding = get_cross_venue_funding(symbol)
    oi = get_cross_venue_oi(symbol)
    basis = get_spot_perp_basis(symbol)

    # Build confirmation summary
    venues_up = 0
    venues_down = 0
    venues_total = 0

    # Price direction heuristic: compare last to 24h vwap via bid/ask midpoint
    for eid, t in ticker.get("exchanges", {}).items():
        if isinstance(t.get("last"), (int, float)) and isinstance(t.get("bid"), (int, float)):
            venues_total += 1
            mid = (t["bid"] + t["ask"]) / 2 if t.get("ask") else t["last"]
            if t["last"] >= mid:
                venues_up += 1
            else:
                venues_down += 1

    # Funding direction
    funding_directions = []
    for eid, f in funding.get("exchanges", {}).items():
        if isinstance(f.get("rate"), (int, float)):
            funding_directions.append("long_paying" if f["rate"] > 0 else "short_paying" if f["rate"] < 0 else "neutral")

    # Confirmation verdict
    confirmed_venues = max(venues_up, venues_down)
    direction = "bullish" if venues_up > venues_down else "bearish" if venues_down > venues_up else "neutral"

    confirmation_level = (
        "strong" if confirmed_venues >= 3
        else "moderate" if confirmed_venues == 2
        else "weak"
    )

    # Build text summary
    lines = [
        f"=== Cross-Venue Confirmation Report ===",
        f"Venues reporting: {venues_total}",
        f"Direction: {direction.upper()} ({confirmation_level} confirmation, {confirmed_venues}/{venues_total} venues agree)",
        "",
    ]

    # Price alignment
    price_summary = ticker.get("summary", {})
    if price_summary.get("max_divergence_pct") is not None:
        lines.append(f"Price divergence: {price_summary['max_divergence_pct']:.4f}% across {price_summary.get('price_range', 'N/A')}")
    for eid, t in ticker.get("exchanges", {}).items():
        if isinstance(t.get("last"), (int, float)):
            lines.append(f"  {eid}: ${t['last']:,.2f} (spread: {t.get('spread_pct', 'N/A'):.4f}%)" if isinstance(t.get("spread_pct"), (int, float)) else f"  {eid}: ${t['last']:,.2f}")

    # Funding
    lines.append("")
    lines.append("Funding rates:")
    for eid, f in funding.get("exchanges", {}).items():
        if isinstance(f.get("rate"), (int, float)):
            lines.append(f"  {eid}: {f['rate']*100:.4f}% ({'longs paying' if f['rate'] > 0 else 'shorts paying'})")
        elif f.get("error"):
            lines.append(f"  {eid}: unavailable")
    fdiv = funding.get("divergence", {})
    if fdiv.get("interpretation"):
        lines.append(f"  Funding divergence: {fdiv['interpretation']} ({fdiv.get('spread_bps', 0):.1f} bps)")

    # OI
    lines.append("")
    lines.append("Open interest:")
    for eid, o in oi.get("exchanges", {}).items():
        if isinstance(o.get("open_interest_value"), (int, float)):
            lines.append(f"  {eid}: ${o['open_interest_value']:,.0f}")
        elif isinstance(o.get("open_interest"), (int, float)):
            lines.append(f"  {eid}: {o['open_interest']:,.2f} BTC")
        elif o.get("error"):
            lines.append(f"  {eid}: unavailable")

    # Basis
    lines.append("")
    lines.append("Spot/perp basis:")
    for eid, b in basis.get("basis", {}).items():
        lines.append(f"  {eid} perp vs {b['spot_source']} spot: {b['basis_pct']:+.4f}% ({b['interpretation']})")
    if basis.get("warning"):
        lines.append(f"  WARNING: {basis['warning']}")

    # Final verdict
    lines.append("")
    lines.append(f"CONFIRMATION: {confirmation_level.upper()} {direction.upper()}")
    if confirmation_level == "weak":
        lines.append("  Only 1 of 3 venues confirms direction — REDUCE confidence by 1 level")
    if basis.get("warning"):
        lines.append("  Derivatives-led move detected — treat as SHORT-LIVED until spot confirms")

    summary_text = "\n".join(lines)

    result = {
        "direction": direction,
        "confirmation_level": confirmation_level,
        "confirmed_venues": confirmed_venues,
        "total_venues": venues_total,
        "funding_divergence": fdiv.get("interpretation", "unknown"),
        "basis_warning": basis.get("warning"),
        "summary": summary_text,
        "ts": time.time(),
    }

    _set_cache(cache_key, result)
    return result

"""
CryptoMonitor — Geopolitical Intelligence for Crypto Trading

Produces a Crisis Impact Index (CII) score 0-100 by aggregating:
1. GDELT event tone & volume (conflict, sanctions, military)
2. GDELT article tone for crypto-relevant geopolitical events
3. Known conflict zone proximity to mining/exchange hubs
4. Keyword-scored RSS headlines (BBC, Al Jazeera)

Output: single JSON dict consumed by the fund manager.

Rate limits:
  - GDELT Doc API: 1 req per 5s (we cache aggressively, TTL 30min)
  - GDELT Bulk CSV: no limit (updated every 15min)
  - RSS: no limit

Cost: $0 (all free APIs)
"""

import io
import csv
import math
import time
import zipfile
import logging
import requests
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_LAST_UPDATE = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
GDELT_MIN_INTERVAL = 6  # seconds between GDELT API calls

# Queries that matter for crypto markets
GDELT_QUERIES = {
    "conflict": "(war OR military OR airstrike OR missile OR invasion)",
    "sanctions": "(sanctions OR embargo OR freeze OR seize OR ban)",
    "crypto_regulation": "(cryptocurrency regulation OR bitcoin ban OR crypto crackdown)",
    "energy_disruption": "(oil pipeline OR energy crisis OR power grid OR mining ban)",
    "financial_crisis": "(bank failure OR default OR currency crisis OR capital controls)",
}

# Mining/exchange hub countries — disruptions here directly impact crypto
CRITICAL_REGIONS = {
    "United States", "China", "Russia", "Kazakhstan", "Iran",
    "Canada", "Germany", "Singapore", "Japan", "South Korea",
    "United Arab Emirates", "Hong Kong",
}

# Goldstein scale: negative = conflict, positive = cooperation
# Events with Goldstein < -5 are significant conflicts
GOLDSTEIN_CONFLICT_THRESHOLD = -5.0
GOLDSTEIN_SEVERE_THRESHOLD = -8.0

# RSS feeds for headline scanning
RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
]

# Crypto-impact keywords with weights
IMPACT_KEYWORDS = {
    # Direct crypto impact (weight 3)
    "bitcoin": 3, "cryptocurrency": 3, "crypto": 3, "mining ban": 3,
    "exchange hack": 3, "stablecoin": 3, "defi": 3, "cbdc": 3,
    # Financial system (weight 2)
    "sanctions": 2, "capital controls": 2, "bank run": 2, "default": 2,
    "currency crisis": 2, "inflation": 2, "interest rate": 2,
    "federal reserve": 2, "central bank": 2,
    # Geopolitical (weight 1)
    "war": 1, "military": 1, "nuclear": 1, "invasion": 1,
    "missile": 1, "airstrike": 1, "conflict": 1, "escalation": 1,
    "ceasefire": -1, "peace": -1, "agreement": -1, "de-escalation": -1,
}

# ─── Session with retry ──────────────────────────────────────────────────────

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(
    max_retries=requests.packages.urllib3.util.retry.Retry(
        total=2, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504]
    )
)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)
_session.headers["User-Agent"] = "CryptoMonitor/1.0"

# ─── Cache ────────────────────────────────────────────────────────────────────

_cache = {}
_CACHE_TTL = 1800  # 30 minutes


def _cached_get(url: str, params: dict = None, timeout: int = 15) -> Optional[str]:
    """GET with TTL cache and rate limiting."""
    cache_key = f"{url}:{str(sorted(params.items()) if params else '')}"
    now = time.time()

    if cache_key in _cache:
        text, ts = _cache[cache_key]
        if now - ts < _CACHE_TTL:
            return text

    try:
        resp = _session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        _cache[cache_key] = (resp.text, now)
        return resp.text
    except Exception as e:
        log.warning(f"CryptoMonitor fetch failed {url}: {e}")
        # Return stale cache if available
        if cache_key in _cache:
            return _cache[cache_key][0]
        return None


# ─── GDELT Article Tone Analysis ─────────────────────────────────────────────

_last_gdelt_call = 0.0


def _gdelt_query(query: str, timespan: str = "24h", max_records: int = 10) -> list:
    """Query GDELT Doc API for articles matching a query. Returns list of article dicts."""
    global _last_gdelt_call

    # Rate limit: 1 request per GDELT_MIN_INTERVAL seconds
    now = time.time()
    wait = GDELT_MIN_INTERVAL - (now - _last_gdelt_call)
    if wait > 0:
        time.sleep(wait)

    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(max_records),
        "format": "json",
        "timespan": timespan,
    }

    text = _cached_get(GDELT_DOC_API, params, timeout=20)
    _last_gdelt_call = time.time()

    if not text:
        return []

    try:
        import json
        data = json.loads(text)
        return data.get("articles", [])
    except Exception:
        return []


def get_gdelt_tone_scores() -> dict:
    """
    Query GDELT for each category and compute average tone.
    Returns dict of category -> {count, avg_tone, sources}.
    """
    results = {}
    for category, query in GDELT_QUERIES.items():
        articles = _gdelt_query(query, timespan="24h", max_records=20)
        if not articles:
            results[category] = {"count": 0, "avg_tone": 0.0, "sources": []}
            continue

        # GDELT articles don't include tone in artlist mode
        # Use article count and source diversity as proxies
        sources = set()
        for a in articles:
            domain = a.get("domain", "")
            country = a.get("sourcecountry", "")
            sources.add(f"{domain}({country})")

        results[category] = {
            "count": len(articles),
            "avg_tone": 0.0,  # artlist doesn't include tone
            "sources": list(sources)[:5],
        }

    return results


# ─── RSS Headline Scanning ───────────────────────────────────────────────────

def _parse_rss(xml_text: str) -> list:
    """Parse RSS XML and return list of {title, link, pubDate}."""
    import xml.etree.ElementTree as ET
    items = []
    try:
        root = ET.fromstring(xml_text)
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            date_el = item.find("pubDate")
            if title_el is not None and title_el.text:
                items.append({
                    "title": title_el.text.strip(),
                    "link": link_el.text.strip() if link_el is not None and link_el.text else "",
                    "pubDate": date_el.text.strip() if date_el is not None and date_el.text else "",
                })
    except Exception as e:
        log.debug(f"RSS parse error: {e}")
    return items


def get_headline_scores() -> dict:
    """
    Scan RSS headlines for crypto-impacting keywords.
    Returns {total_score, headline_count, top_headlines, sentiment}.
    """
    all_headlines = []
    for feed_url in RSS_FEEDS:
        text = _cached_get(feed_url, timeout=10)
        if text:
            all_headlines.extend(_parse_rss(text))

    if not all_headlines:
        return {"total_score": 0, "headline_count": 0, "top_headlines": [], "sentiment": "UNKNOWN"}

    scored = []
    for h in all_headlines:
        title_lower = h["title"].lower()
        score = 0
        matched = []
        for keyword, weight in IMPACT_KEYWORDS.items():
            if keyword in title_lower:
                score += weight
                matched.append(keyword)
        if score != 0:
            scored.append({**h, "score": score, "keywords": matched})

    scored.sort(key=lambda x: abs(x["score"]), reverse=True)

    total = sum(s["score"] for s in scored)
    negative = sum(1 for s in scored if s["score"] > 0)  # conflict keywords have positive weight
    positive = sum(1 for s in scored if s["score"] < 0)  # peace keywords have negative weight

    if total > 5:
        sentiment = "RISK_OFF"
    elif total < -2:
        sentiment = "RISK_ON"
    else:
        sentiment = "NEUTRAL"

    return {
        "total_score": total,
        "headline_count": len(scored),
        "top_headlines": [{"title": s["title"], "score": s["score"]} for s in scored[:5]],
        "sentiment": sentiment,
    }


# ─── Mining Region Risk ──────────────────────────────────────────────────────

def get_mining_region_risk() -> dict:
    """
    Check GDELT for disruptions in critical mining/exchange regions.
    Returns {risk_score, affected_regions, events}.
    """
    region_query = " OR ".join(f'sourcecountry:"{r}"' for r in list(CRITICAL_REGIONS)[:5])
    conflict_query = f"({region_query}) (conflict OR sanctions OR ban OR crisis)"

    articles = _gdelt_query(conflict_query, timespan="48h", max_records=15)

    affected = set()
    for a in articles:
        country = a.get("sourcecountry", "")
        if country in CRITICAL_REGIONS:
            affected.add(country)

    # Score: 0-30 based on number of affected critical regions
    risk_score = min(30, len(affected) * 8 + len(articles) * 1)

    return {
        "risk_score": risk_score,
        "affected_regions": list(affected),
        "article_count": len(articles),
    }


# ─── Crisis Impact Index ─────────────────────────────────────────────────────

def compute_cii() -> dict:
    """
    Compute the Crisis Impact Index (0-100).

    Components:
    - GDELT event volume & tone (0-40 points)
    - RSS headline sentiment (0-30 points)
    - Mining region disruption (0-30 points)

    Output:
    {
        "cii_score": int,
        "level": "LOW" | "MODERATE" | "ELEVATED" | "HIGH" | "SEVERE",
        "crypto_impact": "BULLISH" | "NEUTRAL" | "BEARISH" | "VERY_BEARISH",
        "top_events": [...],
        "components": {...},
        "timestamp": "...",
    }
    """
    log.info("CryptoMonitor: computing CII...")

    # Component 1: GDELT tone scores
    gdelt = get_gdelt_tone_scores()
    gdelt_score = 0
    for cat, data in gdelt.items():
        count = data["count"]
        if cat == "conflict":
            gdelt_score += min(15, count * 1.5)
        elif cat == "sanctions":
            gdelt_score += min(10, count * 2)
        elif cat == "crypto_regulation":
            gdelt_score += min(8, count * 2)
        elif cat == "energy_disruption":
            gdelt_score += min(5, count * 1)
        elif cat == "financial_crisis":
            gdelt_score += min(7, count * 1.5)
    gdelt_score = min(40, gdelt_score)

    # Component 2: RSS headline sentiment
    headlines = get_headline_scores()
    headline_score = min(30, max(0, headlines["total_score"] * 3))

    # Component 3: Mining region risk
    mining = get_mining_region_risk()
    mining_score = mining["risk_score"]

    # Composite CII
    cii = int(min(100, gdelt_score + headline_score + mining_score))

    # Classify level
    if cii >= 80:
        level = "SEVERE"
        crypto_impact = "VERY_BEARISH"
    elif cii >= 60:
        level = "HIGH"
        crypto_impact = "BEARISH"
    elif cii >= 40:
        level = "ELEVATED"
        crypto_impact = "BEARISH"
    elif cii >= 20:
        level = "MODERATE"
        crypto_impact = "NEUTRAL"
    else:
        level = "LOW"
        crypto_impact = "BULLISH"

    # Gather top events for context
    top_events = []
    for h in headlines.get("top_headlines", [])[:3]:
        top_events.append(h["title"])

    for cat in ["conflict", "sanctions", "crypto_regulation"]:
        if gdelt.get(cat, {}).get("count", 0) > 5:
            top_events.append(f"High {cat} activity: {gdelt[cat]['count']} articles in 24h")

    if mining.get("affected_regions"):
        top_events.append(f"Disruptions in: {', '.join(mining['affected_regions'])}")

    result = {
        "cii_score": cii,
        "level": level,
        "crypto_impact": crypto_impact,
        "top_events": top_events[:5],
        "components": {
            "gdelt_score": round(gdelt_score, 1),
            "headline_score": round(headline_score, 1),
            "mining_region_score": round(mining_score, 1),
        },
        "details": {
            "gdelt_categories": {k: v["count"] for k, v in gdelt.items()},
            "headline_sentiment": headlines["sentiment"],
            "affected_mining_regions": mining.get("affected_regions", []),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    log.info(
        f"CryptoMonitor CII: {cii}/100 ({level}) | "
        f"GDELT={gdelt_score:.0f} Headlines={headline_score:.0f} Mining={mining_score:.0f} | "
        f"Impact={crypto_impact}"
    )

    return result


# ─── Cached entry point ──────────────────────────────────────────────────────

_cii_cache = {"result": None, "ts": 0}
_CII_CACHE_TTL = 1800  # 30 min


def get_crisis_impact_index(ttl: int = None) -> dict:
    """Get CII with caching. Default TTL 30 min."""
    cache_ttl = ttl or _CII_CACHE_TTL
    now = time.time()
    if _cii_cache["result"] and (now - _cii_cache["ts"]) < cache_ttl:
        return _cii_cache["result"]

    result = compute_cii()
    _cii_cache["result"] = result
    _cii_cache["ts"] = now
    return result


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    result = get_crisis_impact_index()
    print(json.dumps(result, indent=2))

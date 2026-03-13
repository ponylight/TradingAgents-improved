"""
Geopolitical News & Intelligence Module

Inspired by situation-monitor (github.com/hipcityreg/situation-monitor).
Fetches news from 30+ RSS feeds + GDELT, scores for market impact.

Categories:
- Finance: CNBC, MarketWatch, Yahoo Finance, BBC Business, FT
- Politics: BBC World, NPR, Guardian, NYT
- Government: White House, Federal Reserve, SEC, DoD
- Intel: CSIS, Brookings, CFR, Bellingcat, Defense One
- Regional: The Diplomat (APAC), Al-Monitor (MENA)

Plus GDELT API for real-time global event detection.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger("geopolitical_news")

TIMEOUT = 5  # Per-feed timeout (reduced from 15s)
OVERALL_TIMEOUT = 30  # Max total time for all feeds
HEADERS = {"User-Agent": "TradingBot/2.0 (crypto news aggregator)"}

# === RSS FEEDS (curated from situation-monitor) ===

FEEDS = {
    "finance": [
        ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories"),
        ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
        ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ],
    "politics": [
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("NPR News", "https://feeds.npr.org/1001/rss.xml"),
        ("Guardian World", "https://www.theguardian.com/world/rss"),
    ],
    "government": [
        ("Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml"),
        ("SEC", "https://www.sec.gov/news/pressreleases.rss"),
    ],
    "intel": [
        ("CSIS", "https://www.csis.org/analysis/feed"),
        ("Brookings", "https://www.brookings.edu/feed/"),
        ("CFR", "https://www.cfr.org/rss.xml"),
        ("Defense One", "https://www.defenseone.com/rss/all/"),
        ("The Diplomat", "https://thediplomat.com/feed/"),
        ("Al-Monitor", "https://www.al-monitor.com/rss"),
    ],
    "crypto": [
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("The Block", "https://www.theblock.co/rss.xml"),
    ],
}

# === ALERT KEYWORDS (from situation-monitor) ===

ALERT_KEYWORDS = {
    "war", "invasion", "military", "nuclear", "sanctions", "missile",
    "attack", "troops", "conflict", "strike", "bomb", "casualties",
    "ceasefire", "treaty", "nato", "coup", "martial law", "emergency",
    "assassination", "terrorist", "hostage", "evacuation",
}

# Keywords that specifically impact crypto/BTC
CRYPTO_IMPACT_KEYWORDS = {
    # High impact (direct)
    "bitcoin": 3, "btc": 3, "cryptocurrency": 3, "crypto": 3,
    "sec": 2, "etf": 3, "spot etf": 3, "regulation": 2,
    "stablecoin": 2, "tether": 2, "usdt": 2, "usdc": 2,
    "binance": 2, "coinbase": 2, "exchange": 1,
    # Medium impact (macro)
    "fed": 2, "federal reserve": 3, "interest rate": 3, "rate cut": 3, "rate hike": 3,
    "inflation": 2, "cpi": 3, "fomc": 3, "powell": 2,
    "treasury": 2, "yield": 1, "dollar": 2, "dxy": 2,
    # Geopolitical risk
    "sanctions": 2, "war": 2, "missile": 1, "iran": 2, "china": 1,
    "taiwan": 2, "russia": 1, "tariff": 2, "trade war": 2,
    # Liquidity / risk
    "bank run": 3, "banking crisis": 3, "liquidity": 2, "default": 2,
    "recession": 2, "layoffs": 1, "unemployment": 1,
}

REGION_KEYWORDS = {
    "MENA": ["iran", "israel", "saudi", "syria", "iraq", "gaza", "lebanon", "yemen", "houthi"],
    "APAC": ["china", "taiwan", "japan", "korea", "south china sea", "philippines"],
    "EUROPE": ["nato", "ukraine", "russia", "germany", "france", "uk"],
}


def _fetch_rss(url: str, limit: int = 15) -> List[Dict]:
    """Fetch and parse an RSS feed."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return []
        
        root = ET.fromstring(r.content)
        items = []
        
        # Handle both RSS 2.0 and Atom feeds
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            description = (item.findtext("description") or "")[:300].strip()
            
            if title:
                items.append({
                    "title": title[:200],
                    "link": link,
                    "pub_date": pub_date,
                    "description": re.sub(r'<[^>]+>', '', description)[:200],
                })
            if len(items) >= limit:
                break
        
        return items
    except Exception as e:
        log.debug(f"RSS fetch failed {url}: {e}")
        return []


_gdelt_cache: Dict[str, Any] = {}  # query -> {"ts": float, "data": list}
GDELT_CACHE_TTL = 600  # 10 minutes


def _fetch_gdelt(query: str = "bitcoin OR crypto OR cryptocurrency", limit: int = 75) -> List[Dict]:
    """Fetch articles from GDELT API with cache and exponential backoff on 429."""
    cache_key = f"{query}:{limit}"
    now = time.time()

    # Check cache first
    if cache_key in _gdelt_cache and (now - _gdelt_cache[cache_key]["ts"]) < GDELT_CACHE_TTL:
        return _gdelt_cache[cache_key]["data"]

    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={query}&mode=artlist&maxrecords={limit}&format=json&sort=datedesc"
    last_err = None

    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 429:
                wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                log.info(f"GDELT rate limited (429), waiting {wait}s (attempt {attempt+1}/3)")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                last_err = f"status {r.status_code}"
                break

            data = r.json()
            articles = []
            for article in data.get("articles", []):
                title = article.get("title", "").strip()
                if title:
                    articles.append({
                        "title": title[:200],
                        "link": article.get("url", ""),
                        "source": article.get("domain", "GDELT"),
                        "pub_date": article.get("seendate", ""),
                    })
            # Update cache
            _gdelt_cache[cache_key] = {"ts": now, "data": articles}
            return articles
        except Exception as e:
            last_err = e
            wait = 2 ** (attempt + 1)
            log.info(f"GDELT error: {e}, retrying in {wait}s (attempt {attempt+1}/3)")
            time.sleep(wait)

    # All retries failed — return stale cache if available
    if cache_key in _gdelt_cache:
        log.warning(f"GDELT failed after retries ({last_err}), returning stale cache")
        return _gdelt_cache[cache_key]["data"]

    log.warning(f"GDELT unavailable: {last_err}")
    return []


def _score_headline(title: str) -> Dict:
    """Score a headline for crypto market impact."""
    lower = title.lower()
    
    # Alert keyword check
    is_alert = any(kw in lower for kw in ALERT_KEYWORDS)
    
    # Crypto impact score
    impact_score = 0
    matched_keywords = []
    for keyword, weight in CRYPTO_IMPACT_KEYWORDS.items():
        if keyword in lower:
            impact_score += weight
            matched_keywords.append(keyword)
    
    # Region detection
    region = None
    for reg, keywords in REGION_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            region = reg
            break
    
    # Sentiment (basic)
    bullish_words = {"rally", "surge", "buy", "bullish", "approval", "adopt", "inflow", "cut rate", "dovish"}
    bearish_words = {"crash", "sell", "bearish", "ban", "reject", "outflow", "hike rate", "hawkish", "crack down"}
    
    bull = sum(1 for w in bullish_words if w in lower)
    bear = sum(1 for w in bearish_words if w in lower)
    sentiment = "bullish" if bull > bear else "bearish" if bear > bull else "neutral"
    
    return {
        "impact_score": min(impact_score, 10),
        "is_alert": is_alert,
        "matched_keywords": matched_keywords[:5],
        "region": region,
        "sentiment": sentiment,
    }


def fetch_all_news(categories: List[str] = None, max_per_feed: int = 10) -> Dict[str, Any]:
    """Fetch news from all configured RSS feeds + GDELT.
    
    Args:
        categories: Which feed categories to fetch. Default: all.
        max_per_feed: Max articles per feed source.
    
    Returns:
        Structured news data with scoring.
    """
    if categories is None:
        categories = list(FEEDS.keys())

    all_articles = []
    source_stats = {}

    # Build list of (name, url, category) for concurrent fetching
    feed_tasks = []
    for category in categories:
        for name, url in FEEDS.get(category, []):
            feed_tasks.append((name, url, category))

    # Fetch all RSS feeds concurrently with overall timeout
    deadline = time.monotonic() + OVERALL_TIMEOUT

    def _fetch_one(task):
        name, url, category = task
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return name, category, []
        articles = _fetch_rss(url, max_per_feed)
        for article in articles:
            article["source"] = name
            article["category"] = category
            article["scoring"] = _score_headline(article["title"])
        return name, category, articles

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in feed_tasks}
        for future in as_completed(futures, timeout=OVERALL_TIMEOUT):
            try:
                name, category, articles = future.result(timeout=1)
                all_articles.extend(articles)
                source_stats[name] = len(articles)
            except Exception as e:
                name = futures[future][0]
                log.debug(f"Feed {name} failed: {e}")
                source_stats[name] = 0

    # GDELT for crypto-specific global coverage
    gdelt_articles = _fetch_gdelt("bitcoin OR crypto OR cryptocurrency", 75)
    for article in gdelt_articles:
        article["category"] = "gdelt"
        article["scoring"] = _score_headline(article["title"])
        all_articles.append(article)
    source_stats["GDELT"] = len(gdelt_articles)
    
    # Sort by impact score
    all_articles.sort(key=lambda a: a.get("scoring", {}).get("impact_score", 0), reverse=True)
    
    # Extract alerts (high-impact geopolitical events)
    alerts = [a for a in all_articles if a.get("scoring", {}).get("is_alert", False)]
    
    # Top crypto-relevant headlines
    crypto_relevant = [a for a in all_articles if a.get("scoring", {}).get("impact_score", 0) >= 2]
    
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_articles": len(all_articles),
        "sources_fetched": len(source_stats),
        "source_stats": source_stats,
        "alerts": [{"title": a["title"], "source": a.get("source", ""), "region": a["scoring"].get("region")} for a in alerts[:10]],
        "crypto_relevant": [{"title": a["title"], "source": a.get("source", ""), "score": a["scoring"]["impact_score"], "sentiment": a["scoring"]["sentiment"]} for a in crypto_relevant[:15]],
        "top_headlines": [{"title": a["title"], "source": a.get("source", ""), "category": a["category"]} for a in all_articles[:20]],
        "category_counts": {cat: sum(1 for a in all_articles if a.get("category") == cat) for cat in categories + ["gdelt"]},
        "alert_count": len(alerts),
        "avg_impact": round(sum(a.get("scoring", {}).get("impact_score", 0) for a in all_articles) / max(len(all_articles), 1), 2),
    }


def format_geopolitical_report(data: Dict[str, Any]) -> str:
    """Format news data into a report for the News Analyst agent."""
    if not data or data.get("total_articles", 0) == 0:
        return "Geopolitical news unavailable."
    
    lines = [
        "# Geopolitical & Market News Intelligence",
        f"Sources: {data['sources_fetched']} feeds | Articles: {data['total_articles']} | Alerts: {data['alert_count']}",
        "",
    ]
    
    if data["alerts"]:
        lines.append("## 🚨 ALERTS (geopolitical events)")
        for a in data["alerts"]:
            region = f" [{a['region']}]" if a.get("region") else ""
            lines.append(f"- {a['title']} ({a['source']}{region})")
        lines.append("")
    
    if data["crypto_relevant"]:
        lines.append("## 📊 Crypto-Relevant Headlines (impact ≥ 2)")
        for a in data["crypto_relevant"]:
            lines.append(f"- [{a['sentiment'].upper()}] (impact {a['score']}) {a['title']} — {a['source']}")
        lines.append("")
    
    lines.append("## 📰 Top Headlines")
    for a in data["top_headlines"][:10]:
        lines.append(f"- [{a['category'].upper()}] {a['title']} — {a['source']}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    data = fetch_all_news()
    print(format_geopolitical_report(data))
    print(f"\n--- Stats ---")
    for source, count in sorted(data["source_stats"].items()):
        print(f"  {source}: {count}")

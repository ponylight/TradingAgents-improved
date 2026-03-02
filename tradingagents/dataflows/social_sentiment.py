"""
Social Sentiment Module for Crypto

Sources:
- Reddit (r/bitcoin, r/cryptocurrency, r/CryptoMarkets, r/BitcoinMarkets)
  Free JSON API, no auth needed.

Provides:
- Top posts with scores and comment counts
- Simple sentiment scoring (keyword-based, fast)
- Discussion volume as a proxy for engagement
- Dominant narratives extraction
"""

from __future__ import annotations
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("social_sentiment")

TIMEOUT = 15
HEADERS = {"User-Agent": "TradingBot/1.0"}

SUBREDDITS = ["bitcoin", "cryptocurrency", "CryptoMarkets", "BitcoinMarkets"]

# Keyword sentiment scoring (fast, no ML dependency)
BULLISH_WORDS = {
    "bull", "bullish", "moon", "mooning", "pump", "rally", "buy", "buying",
    "long", "accumulate", "hodl", "hold", "undervalued", "breakout", "support",
    "golden cross", "higher low", "recovery", "ath", "all time high",
    "institutional", "adoption", "etf", "halving", "scarcity", "demand",
    "green", "bounce", "reversal", "bottom", "oversold", "cheap", "dip",
}

BEARISH_WORDS = {
    "bear", "bearish", "dump", "crash", "sell", "selling", "short",
    "overvalued", "bubble", "top", "resistance", "death cross", "lower high",
    "capitulation", "fear", "panic", "rug", "scam", "regulation", "ban",
    "red", "breakdown", "overbought", "expensive", "exit", "liquidation",
    "recession", "inflation", "hawkish", "taper",
}

FEAR_WORDS = {
    "fear", "panic", "crash", "worried", "scared", "uncertain", "risk",
    "liquidated", "margin call", "rekt", "wrecked", "blood", "massacre",
}

GREED_WORDS = {
    "moon", "lambo", "rich", "millionaire", "100k", "200k", "500k",
    "guaranteed", "easy money", "free money", "cant lose", "to the moon",
}


def _safe_get(url: str, timeout: int = TIMEOUT) -> Optional[dict]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            log.warning("Reddit rate limited, waiting 2s")
            time.sleep(2)
        return None
    except Exception as e:
        log.warning(f"Reddit error {url}: {e}")
        return None


def _score_text(text: str) -> Dict[str, float]:
    """Score a text for bullish/bearish/fear/greed signals."""
    lower = text.lower()
    words = set(re.findall(r'\b\w+\b', lower))
    bigrams = set()
    word_list = re.findall(r'\b\w+\b', lower)
    for i in range(len(word_list) - 1):
        bigrams.add(f"{word_list[i]} {word_list[i+1]}")
    all_tokens = words | bigrams

    bull = len(all_tokens & BULLISH_WORDS)
    bear = len(all_tokens & BEARISH_WORDS)
    fear = len(all_tokens & FEAR_WORDS)
    greed = len(all_tokens & GREED_WORDS)
    total = bull + bear + 1  # avoid div by zero

    return {
        "bullish_score": round(bull / total, 2),
        "bearish_score": round(bear / total, 2),
        "fear_signals": fear,
        "greed_signals": greed,
        "net_sentiment": round((bull - bear) / total, 2),
    }


def get_reddit_posts(subreddit: str, sort: str = "hot", limit: int = 25) -> List[Dict]:
    """Fetch posts from a subreddit."""
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"
    data = _safe_get(url)
    if not data:
        return []

    posts = []
    for child in data.get("data", {}).get("children", []):
        d = child.get("data", {})
        if d.get("stickied"):
            continue
        title = d.get("title", "")
        selftext = d.get("selftext", "")[:500]
        score = d.get("score", 0)
        comments = d.get("num_comments", 0)
        created = d.get("created_utc", 0)
        
        sentiment = _score_text(f"{title} {selftext}")
        
        posts.append({
            "subreddit": subreddit,
            "title": title[:150],
            "score": score,
            "comments": comments,
            "created_utc": created,
            "age_hours": round((time.time() - created) / 3600, 1) if created else 0,
            "sentiment": sentiment,
        })

    return posts


def get_social_sentiment() -> Dict[str, Any]:
    """Fetch and aggregate social sentiment from all crypto subreddits."""
    all_posts = []
    sub_stats = {}

    for sub in SUBREDDITS:
        posts = get_reddit_posts(sub, "hot", 25)
        all_posts.extend(posts)
        
        if posts:
            scores = [p["sentiment"]["net_sentiment"] for p in posts]
            engagement = sum(p["score"] + p["comments"] for p in posts)
            avg_sentiment = sum(scores) / len(scores)
            sub_stats[sub] = {
                "post_count": len(posts),
                "avg_sentiment": round(avg_sentiment, 3),
                "total_engagement": engagement,
                "top_post": posts[0]["title"] if posts else "",
            }
        time.sleep(0.5)  # Rate limit courtesy

    if not all_posts:
        return {"error": "No data from Reddit", "generated_at": datetime.now(timezone.utc).isoformat()}

    # Aggregate
    all_sentiments = [p["sentiment"]["net_sentiment"] for p in all_posts]
    all_bull = sum(p["sentiment"]["bullish_score"] for p in all_posts)
    all_bear = sum(p["sentiment"]["bearish_score"] for p in all_posts)
    all_fear = sum(p["sentiment"]["fear_signals"] for p in all_posts)
    all_greed = sum(p["sentiment"]["greed_signals"] for p in all_posts)
    total_engagement = sum(p["score"] + p["comments"] for p in all_posts)

    avg_sentiment = sum(all_sentiments) / len(all_sentiments)

    # Classify overall mood
    if avg_sentiment > 0.15:
        mood = "BULLISH"
    elif avg_sentiment < -0.15:
        mood = "BEARISH"
    elif all_fear > all_greed * 2:
        mood = "FEARFUL"
    elif all_greed > all_fear * 2:
        mood = "GREEDY"
    else:
        mood = "NEUTRAL"

    # Extract dominant narratives (most common topics in high-engagement posts)
    top_posts = sorted(all_posts, key=lambda p: p["score"] + p["comments"], reverse=True)[:10]
    narratives = [p["title"] for p in top_posts]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall_mood": mood,
        "avg_net_sentiment": round(avg_sentiment, 3),
        "total_posts_analyzed": len(all_posts),
        "total_engagement": total_engagement,
        "bullish_total": round(all_bull, 1),
        "bearish_total": round(all_bear, 1),
        "fear_signals": all_fear,
        "greed_signals": all_greed,
        "subreddit_breakdown": sub_stats,
        "top_narratives": narratives[:5],
    }


def format_social_sentiment_report(data: Dict[str, Any]) -> str:
    """Format social sentiment into a report for the LLM agent."""
    if "error" in data:
        return f"Social sentiment unavailable: {data['error']}"

    lines = [
        "# Crypto Social Sentiment Report",
        f"Generated: {data.get('generated_at', 'N/A')}",
        "",
        f"## Overall Mood: {data['overall_mood']}",
        f"- Net sentiment score: {data['avg_net_sentiment']:+.3f} (-1 = extreme bearish, +1 = extreme bullish)",
        f"- Posts analyzed: {data['total_posts_analyzed']}",
        f"- Total engagement (upvotes + comments): {data['total_engagement']:,}",
        f"- Bullish signals: {data['bullish_total']:.0f} | Bearish signals: {data['bearish_total']:.0f}",
        f"- Fear signals: {data['fear_signals']} | Greed signals: {data['greed_signals']}",
        "",
        "## Subreddit Breakdown",
    ]

    for sub, stats in data.get("subreddit_breakdown", {}).items():
        lines.append(f"- r/{sub}: sentiment {stats['avg_sentiment']:+.3f}, "
                    f"engagement {stats['total_engagement']:,}, "
                    f"posts {stats['post_count']}")

    lines.append("")
    lines.append("## Top Narratives (highest engagement)")
    for i, title in enumerate(data.get("top_narratives", []), 1):
        lines.append(f"{i}. {title}")

    return "\n".join(lines)



def classify_posts_with_llm(posts: List[Dict], llm=None) -> Dict[str, Any]:
    """Use an auxiliary LLM to classify top posts for sentiment.
    
    Per paper Section 5.2: "sentiment scores of posts calculated by auxiliary language models"
    
    This is more accurate than keyword scoring because it handles:
    - Sarcasm ("great, another crash")
    - Crypto memes ("number go up technology")  
    - Context-dependent terms ("short" can mean brief or short-selling)
    - Coded language ("diamond hands", "paper hands", "ngmi")
    
    Args:
        posts: List of post dicts from get_reddit_posts()
        llm: A LangChain-compatible LLM (quick-thinking model)
    
    Returns:
        Dict with LLM-classified sentiment scores and narratives
    """
    if not llm or not posts:
        return {}
    
    # Take top 15 by engagement (more signal, less noise)
    top = sorted(posts, key=lambda p: p["score"] + p["comments"], reverse=True)[:15]
    
    post_text = ""
    for i, p in enumerate(top, 1):
        post_text += f"{i}. [{p['subreddit']}] (↑{p['score']}, 💬{p['comments']}) {p['title']}\n"
    
    prompt = f"""Classify the sentiment of these crypto Reddit posts. Consider sarcasm, memes, and crypto slang.

{post_text}

For each post, classify as: BULLISH, BEARISH, or NEUTRAL.
Then provide an overall summary.

Respond in this exact JSON format (no markdown):
{{"posts": [{{"id": 1, "sentiment": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0}}], "overall": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "dominant_narrative": "one sentence summary of what crypto Reddit is focused on", "contrarian_signal": "none|weak|moderate|strong — is sentiment extreme enough to be contrarian?"}}"""

    try:
        result = llm.invoke(prompt)
        import json
        # Try to parse JSON from response
        text = result.content if hasattr(result, 'content') else str(result)
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        parsed = json.loads(text)
        
        # Calculate scores from classifications
        sentiments = [p["sentiment"] for p in parsed.get("posts", [])]
        bull_count = sentiments.count("BULLISH")
        bear_count = sentiments.count("BEARISH")
        neutral_count = sentiments.count("NEUTRAL")
        total = len(sentiments) or 1
        
        return {
            "llm_overall": parsed.get("overall", "NEUTRAL"),
            "llm_confidence": parsed.get("confidence", 0.5),
            "llm_bull_pct": round(bull_count / total, 2),
            "llm_bear_pct": round(bear_count / total, 2),
            "llm_neutral_pct": round(neutral_count / total, 2),
            "dominant_narrative": parsed.get("dominant_narrative", ""),
            "contrarian_signal": parsed.get("contrarian_signal", "none"),
            "post_count": total,
        }
    except Exception as e:
        log.warning(f"LLM sentiment classification failed: {e}")
        return {}


def get_social_sentiment_enhanced(llm=None) -> Dict[str, Any]:
    """Get social sentiment with optional LLM classification overlay.
    
    Always runs keyword scoring (fast, free).
    If an LLM is provided, also runs auxiliary classification on top posts.
    LLM results are merged as an additional "llm_classification" field.
    """
    base = get_social_sentiment()
    
    if llm and "error" not in base:
        # Collect all posts for LLM classification
        all_posts = []
        for sub in SUBREDDITS:
            posts = get_reddit_posts(sub, "hot", 25)
            all_posts.extend(posts)
            time.sleep(0.3)
        
        if all_posts:
            llm_result = classify_posts_with_llm(all_posts, llm)
            if llm_result:
                base["llm_classification"] = llm_result
                # If LLM disagrees with keyword scoring, note it
                keyword_mood = base.get("overall_mood", "NEUTRAL")
                llm_mood = llm_result.get("llm_overall", "NEUTRAL")
                if keyword_mood != llm_mood:
                    base["sentiment_divergence"] = f"Keywords say {keyword_mood}, LLM says {llm_mood}"
    
    return base


def format_social_sentiment_report_enhanced(data: Dict[str, Any]) -> str:
    """Format social sentiment with LLM overlay into a report."""
    base_report = format_social_sentiment_report(data)
    
    llm = data.get("llm_classification")
    if not llm:
        return base_report
    
    lines = [
        base_report,
        "",
        "## LLM Sentiment Classification (auxiliary model)",
        f"- Overall: {llm.get('llm_overall', 'N/A')} (confidence: {llm.get('llm_confidence', 0):.0%})",
        f"- Bullish: {llm.get('llm_bull_pct', 0):.0%} | Bearish: {llm.get('llm_bear_pct', 0):.0%} | Neutral: {llm.get('llm_neutral_pct', 0):.0%}",
        f"- Dominant narrative: {llm.get('dominant_narrative', 'N/A')}",
        f"- Contrarian signal: {llm.get('contrarian_signal', 'none')}",
    ]
    
    divergence = data.get("sentiment_divergence")
    if divergence:
        lines.append(f"- ⚠️ DIVERGENCE: {divergence}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    data = get_social_sentiment()
    print(format_social_sentiment_report(data))

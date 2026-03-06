"""
Geopolitical Event Detection

Scans news text for high-impact geopolitical events (wars, sanctions, crises)
and returns a severity score that can override technical analysis.

Severity: 0 (none) to -100 (extreme risk-off event)
"""

import re
import logging
from typing import List, Tuple

log = logging.getLogger("geopolitical")

# Keyword tiers with severity weights
TIER1_CRITICAL = [
    # Active warfare / military action
    r"\bwar\b", r"\battack(?:s|ed|ing)?\b", r"\bstrike[sd]?\b", r"\bbomb(?:s|ed|ing)?\b",
    r"\binvasion\b", r"\bmilitary\s+action\b", r"\bescalat(?:e|ion|ing)\b",
    r"\bnuclear\b", r"\bmissile[sd]?\b", r"\bairstrikes?\b",
]

TIER2_HIGH = [
    # Sanctions, crises, regime changes
    r"\bsanctions?\b", r"\bembargo\b", r"\bblockade\b",
    r"\bcrisis\b", r"\bconflict\b", r"\bcoup\b",
    r"\bgeopolitical\b", r"\btrade\s+war\b", r"\bcold\s+war\b",
    r"\bdefault(?:s|ed)?\b", r"\bcollapse[sd]?\b",
]

TIER3_MODERATE = [
    # Tensions, disputes
    r"\btension[sd]?\b", r"\bdispute[sd]?\b", r"\bterroris[tm]\b",
    r"\bprotest[sd]?\b", r"\bunrest\b", r"\binstability\b",
    r"\bthreat(?:s|en|ened)?\b", r"\bretaliat(?:e|ion|ing)\b",
]

# Geopolitical actors that amplify severity when combined with keywords
ACTORS = [
    r"\b(?:US|USA|United\s+States|America)\b",
    r"\bChina\b", r"\bRussia\b", r"\bIran\b", r"\bIsrael\b",
    r"\bNorth\s+Korea\b", r"\bUkraine\b", r"\bTaiwan\b",
    r"\bNATO\b", r"\bEU\b", r"\bOPEC\b",
    r"\bMiddle\s+East\b", r"\bGulf\b", r"\bSouth\s+China\s+Sea\b",
]


def detect_geopolitical_events(texts: List[str]) -> Tuple[float, List[str]]:
    """
    Scan a list of news texts for geopolitical events.

    Args:
        texts: list of news headlines/summaries

    Returns:
        (severity_score, list_of_detected_events)
        severity_score: 0 to -100 (negative = bearish/risk-off)
        detected_events: human-readable descriptions
    """
    total_severity = 0.0
    detected = []

    for text in texts:
        if not text:
            continue
        lower = text.lower()

        # Check each tier
        tier1_hits = [p for p in TIER1_CRITICAL if re.search(p, lower)]
        tier2_hits = [p for p in TIER2_HIGH if re.search(p, lower)]
        tier3_hits = [p for p in TIER3_MODERATE if re.search(p, lower)]

        if not (tier1_hits or tier2_hits or tier3_hits):
            continue

        # Base severity by tier
        if tier1_hits:
            base = -40
        elif tier2_hits:
            base = -20
        else:
            base = -10

        # Amplify if major geopolitical actors are involved
        actor_hits = [p for p in ACTORS if re.search(p, text, re.IGNORECASE)]
        if len(actor_hits) >= 2:
            base *= 1.5  # Two major actors = amplified
        elif len(actor_hits) >= 1:
            base *= 1.2

        total_severity += base

        tier_label = "CRITICAL" if tier1_hits else "HIGH" if tier2_hits else "MODERATE"
        actors_str = ", ".join(re.search(p, text, re.IGNORECASE).group() for p in actor_hits if re.search(p, text, re.IGNORECASE))
        snippet = text[:120].strip()
        detected.append(f"[{tier_label}] {snippet}" + (f" (actors: {actors_str})" if actors_str else ""))

    # Cap at -100
    total_severity = max(-100, total_severity)

    if detected:
        log.warning(f"🌍 Geopolitical events detected: severity={total_severity:.0f}, events={len(detected)}")
        for d in detected[:5]:
            log.warning(f"  → {d}")

    return round(total_severity, 1), detected

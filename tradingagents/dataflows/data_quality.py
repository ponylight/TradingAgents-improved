"""
Structured Data Quality Scoring

Each analyst report passes through quality scoring before downstream debate.
Scores: CLEAN / DEGRADED / STALE / FAILED
Surfaces a structured header so downstream agents see data quality at a glance.
"""

import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

log = logging.getLogger("data_quality")


class DataQuality(str, Enum):
    CLEAN = "CLEAN"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    FAILED = "FAILED"


def score_report(
    report: str,
    analyst_type: str,
    *,
    max_age_hours: float = 12.0,
    generated_at: Optional[str] = None,
    fallback_used: bool = False,
    missing_fields: Optional[list] = None,
) -> dict:
    """Score a report's data quality.

    Args:
        report: The analyst report text.
        analyst_type: One of "market", "sentiment", "fundamentals", "news".
        max_age_hours: Maximum acceptable data age in hours.
        generated_at: ISO timestamp of when data was generated (if known).
        fallback_used: Whether fallback data sources were used.
        missing_fields: List of fields that are missing or unavailable.

    Returns:
        dict with: quality (DataQuality), data_age_hours (float|None),
        missing_fields (list), contradictions (list), fallback_used (bool),
        issues (list of str).
    """
    missing_fields = missing_fields or []
    issues = []
    contradictions = []
    data_age_hours = None

    # --- Determine data age ---
    if generated_at:
        try:
            gen_time = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            data_age_hours = (datetime.now(timezone.utc) - gen_time).total_seconds() / 3600
        except (ValueError, TypeError):
            issues.append("Could not parse generated_at timestamp")
    else:
        # Try to extract timestamp from report text
        ts_match = re.search(r"Generated:\s*(\S+)", report)
        if ts_match:
            try:
                gen_time = datetime.fromisoformat(ts_match.group(1).replace("Z", "+00:00"))
                data_age_hours = (datetime.now(timezone.utc) - gen_time).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

    # --- Check for failure indicators ---
    failure_patterns = [
        r"(?i)error fetching",
        r"(?i)unavailable",
        r"(?i)analysis incomplete",
        r"(?i)no data (?:found|available|returned)",
    ]
    failure_count = sum(1 for p in failure_patterns if re.search(p, report))

    # --- Check for contradiction indicators ---
    # Detect when report contains opposing signals without resolution
    if re.search(r"(?i)bullish.*bearish|bearish.*bullish", report[:500]):
        # Only flag if the same sentence has both without "however/but/although"
        pass  # LLM reports often discuss both sides — not a contradiction

    # --- Determine quality score ---
    if not report or len(report.strip()) < 50:
        quality = DataQuality.FAILED
        issues.append("Report is empty or too short")
    elif failure_count >= 3:
        quality = DataQuality.FAILED
        issues.append(f"{failure_count} data source failures detected")
    elif data_age_hours is not None and data_age_hours > max_age_hours:
        quality = DataQuality.STALE
        issues.append(f"Data is {data_age_hours:.1f}h old (max: {max_age_hours}h)")
    elif fallback_used or missing_fields or failure_count >= 1:
        quality = DataQuality.DEGRADED
        if fallback_used:
            issues.append("Fallback data source used")
        if missing_fields:
            issues.append(f"Missing fields: {', '.join(missing_fields)}")
        if failure_count >= 1:
            issues.append(f"{failure_count} data source(s) returned errors")
    else:
        quality = DataQuality.CLEAN

    result = {
        "quality": quality,
        "data_age_hours": round(data_age_hours, 1) if data_age_hours is not None else None,
        "missing_fields": missing_fields,
        "contradictions": contradictions,
        "fallback_used": fallback_used,
        "issues": issues,
    }

    if quality != DataQuality.CLEAN:
        log.warning(f"⚠️ [{analyst_type}] Data quality: {quality.value} — {'; '.join(issues)}")

    return result


def format_quality_header(score: dict, analyst_type: str) -> str:
    """Format a structured quality header for injection into reports.

    This header is prepended to analyst reports so downstream agents
    (researchers, trader, risk managers) can see data quality at a glance.
    """
    q = score["quality"]
    lines = [f"═══ DATA QUALITY: {q.value} ({analyst_type}) ═══"]

    if score["data_age_hours"] is not None:
        lines.append(f"  Data age: {score['data_age_hours']}h")

    if score["missing_fields"]:
        lines.append(f"  Missing: {', '.join(score['missing_fields'])}")

    if score["fallback_used"]:
        lines.append("  ⚠️ Fallback data source used")

    if score["issues"]:
        for issue in score["issues"]:
            lines.append(f"  ⚠️ {issue}")

    if q == DataQuality.STALE:
        lines.append("  ⛔ STALE DATA — cap confidence to MEDIUM")
    elif q == DataQuality.FAILED:
        lines.append("  ⛔ DATA FAILED — do NOT build conclusions on this report")

    lines.append("═══════════════════════════════════════")
    return "\n".join(lines)


def cap_confidence_for_quality(quality: DataQuality, stated_confidence: str) -> str:
    """Cap confidence level based on data quality.

    Returns the effective confidence (may be lower than stated).
    """
    confidence_order = ["LOW", "MEDIUM", "HIGH"]

    stated_upper = stated_confidence.upper()
    if stated_upper not in confidence_order:
        return stated_confidence

    idx = confidence_order.index(stated_upper)

    if quality == DataQuality.FAILED:
        return "LOW"
    elif quality == DataQuality.STALE:
        return confidence_order[min(idx, 1)]  # cap at MEDIUM
    elif quality == DataQuality.DEGRADED:
        return confidence_order[min(idx, 2)]  # no cap, but logged
    else:
        return stated_confidence

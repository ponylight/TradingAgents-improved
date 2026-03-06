"""
Tiered Analyst Scheduling

Not all analysts need to run every cycle. Data decay rates differ:
  - Market (technical): Every 4H — price action changes every candle
  - Sentiment (funding/OI): Every 8H — funding resets every 8H
  - News: Every 12H — news doesn't change every 4H
  - Fundamentals (on-chain): Every 24H — network metrics barely move intraday

Between full runs, cached reports from the last run are used.
Cuts LLM costs ~40% and speeds up the 4H cycle.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("analyst_schedule")

# Default refresh intervals (hours)
DEFAULT_SCHEDULE = {
    "market": 4,        # Every cycle
    "sentiment": 8,     # Every 2nd cycle
    "news": 12,         # Every 3rd cycle
    "fundamentals": 24, # Once per day
}

CACHE_FILE = Path(__file__).parent.parent.parent / "logs" / "analyst_cache.json"


class AnalystScheduler:
    """Manages which analysts need to run based on staleness."""

    def __init__(self, schedule: Optional[Dict[str, int]] = None):
        self.schedule = schedule or DEFAULT_SCHEDULE
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict:
        try:
            if CACHE_FILE.exists():
                return json.loads(CACHE_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    def _save_cache(self):
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(self._cache, indent=2))
        except IOError as e:
            log.warning(f"Failed to save analyst cache: {e}")

    def get_analysts_to_run(self) -> Tuple[List[str], List[str]]:
        """
        Determine which analysts need fresh runs vs cached reports.

        Returns:
            (analysts_to_run, analysts_to_cache)
        """
        now = datetime.now(timezone.utc)
        to_run = []
        to_cache = []

        for analyst, interval_hours in self.schedule.items():
            last_run = self._cache.get(analyst, {}).get("last_run")
            cached_report = self._cache.get(analyst, {}).get("report")

            if not last_run or not cached_report:
                # Never run before — must run
                to_run.append(analyst)
                continue

            try:
                last_dt = datetime.fromisoformat(last_run)
                age_hours = (now - last_dt).total_seconds() / 3600

                if age_hours >= interval_hours:
                    to_run.append(analyst)
                    log.info(f"📊 {analyst}: stale ({age_hours:.1f}h >= {interval_hours}h) — will refresh")
                else:
                    to_cache.append(analyst)
                    log.info(f"💾 {analyst}: fresh ({age_hours:.1f}h < {interval_hours}h) — using cache")
            except (ValueError, TypeError):
                to_run.append(analyst)

        # Market analyst ALWAYS runs (it's the core signal)
        if "market" not in to_run:
            to_run.append("market")
            if "market" in to_cache:
                to_cache.remove("market")

        log.info(f"📋 Schedule: run={to_run}, cache={to_cache}")
        return to_run, to_cache

    def get_cached_report(self, analyst: str) -> Optional[str]:
        """Get the cached report for an analyst."""
        entry = self._cache.get(analyst, {})
        report = entry.get("report")
        if report:
            age_hours = 0
            try:
                last_dt = datetime.fromisoformat(entry.get("last_run", ""))
                age_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
            except (ValueError, TypeError):
                pass
            return f"[CACHED REPORT — {age_hours:.1f}h old]\n\n{report}"
        return None

    def update_cache(self, analyst: str, report: str):
        """Store a fresh report in the cache."""
        self._cache[analyst] = {
            "last_run": datetime.now(timezone.utc).isoformat(),
            "report": report,
        }
        self._save_cache()
        log.info(f"💾 Cached {analyst} report ({len(report)} chars)")

    def force_full_run(self):
        """Clear all cache timestamps to force every analyst to run next cycle."""
        for analyst in self._cache:
            self._cache[analyst]["last_run"] = None
        self._save_cache()
        log.info("🔄 Forced full run — all analysts will refresh next cycle")

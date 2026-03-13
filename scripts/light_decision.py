#!/usr/bin/env python3
"""
Light Decision Layer — Fast trading decisions between full agent runs.

Reads:
  - Real-time price from websocket (realtime_price.json)
  - Cached analyst reports from last full run (agent_reports_YYYY-MM-DD.json)
  - Executor state (executor_state.json)
  - CII + macro radar (cached in memory)

Uses a small/fast model (Sonnet) for quick BUY/SELL/HOLD evaluation.
Runs every 15-30 min. Can trigger entries/exits between 4H cron cycles.

Cost: ~$0.01-0.02 per run (Sonnet, small prompt).
"""

import json
import os
import sys
import logging
import time
import warnings
import fcntl
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

SYMBOL = "BTC/USDT:USDT"
OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"


def _get_gateway():
    """Read OpenClaw gateway URL and auth token."""
    try:
        cfg = json.loads(OPENCLAW_CONFIG.read_text())
        gw = cfg.get("gateway", {})
        port = gw.get("port", 18789)
        token = gw.get("auth", {}).get("token", "")
        return f"http://127.0.0.1:{port}/v1", token
    except Exception:
        return "http://127.0.0.1:18789/v1", ""
LOGS = PROJECT_ROOT / "logs"
REALTIME_FILE = LOGS / "realtime_price.json"
STATE_FILE = LOGS / "executor_state.json"
LIGHT_LOG = LOGS / "light_decision.log"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python3"
LOCK_FILE = LOGS / ".light_decision.lock"

# Only act if confidence is HIGH and move is significant
MIN_CONFIDENCE = 8         # 1-10, only act on high conviction
MIN_MOVE_PCT = 1.5         # Minimum % move from last decision price to consider
MAX_LIGHT_TRADES_PER_DAY = 2  # Don't overtrade
MIN_REPORT_FRESHNESS_HOURS = 4
OVERRIDE_COOLDOWN_SECONDS = 20 * 60

LOGS.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
)

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_root_logger.handlers.clear()
_root_logger.addHandler(logging.FileHandler(LIGHT_LOG))
log = logging.getLogger("light_decision")
log.propagate = True


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def get_realtime_price() -> dict:
    """Read latest price from websocket monitor."""
    data = load_json(REALTIME_FILE)
    if not data:
        return {}
    # Check freshness — must be <2 min old
    ts = data.get("timestamp", "")
    if ts:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
            if age > 120:
                log.warning(f"Realtime price stale ({age:.0f}s old)")
                return {}
        except Exception:
            pass
    return data


def get_cached_reports() -> tuple[dict, Path | None]:
    """Load most recent analyst reports and return the file used."""
    today = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d")
    reports_file = LOGS / f"agent_reports_{today}.json"
    if not reports_file.exists():
        yesterday = (datetime.now(ZoneInfo("Australia/Sydney")) - timedelta(days=1)).strftime("%Y-%m-%d")
        reports_file = LOGS / f"agent_reports_{yesterday}.json"
    if not reports_file.exists():
        return {}, None
    return load_json(reports_file), reports_file


def report_file_age_hours(path: Path | None) -> float | None:
    if not path or not path.exists():
        return None
    try:
        return (time.time() - path.stat().st_mtime) / 3600
    except OSError:
        return None


def get_executor_state() -> dict:
    return load_json(STATE_FILE)


def get_macro_summary() -> str:
    """Quick macro check without full radar (use cached if available)."""
    try:
        from tradingagents.dataflows.macro_radar import get_macro_radar_cached
        macro = get_macro_radar_cached(ttl=3600)  # Accept 1h stale
        return macro.get("summary", "Macro radar unavailable")
    except Exception:
        return "Macro radar unavailable"


def get_cii_summary() -> str:
    """Quick CII check."""
    try:
        from tradingagents.dataflows.crypto_monitor import get_crisis_impact_index
        cii = get_crisis_impact_index(ttl=3600)  # Accept 1h stale
        return f"CII: {cii['cii_score']}/100 ({cii['level']}) — {cii['crypto_impact']}"
    except Exception:
        return "CII unavailable"


def run_light_evaluation(price_data: dict, reports: dict, state: dict) -> dict:
    """Run fast LLM evaluation using cached reports + live price."""
    import requests

    price = price_data["price"]
    change_pct = price_data.get("change_pct", 0)
    funding = price_data.get("funding_rate", "?")

    # Build compact prompt
    market_report = reports.get("market", "No market report cached")[:800]
    news_report = reports.get("news", "No news report cached")[:800]
    last_decision = state.get("last_decision", "?")
    last_decision_time = state.get("last_decision_time", "?")
    # Check both active_trade and positions.committee for current position
    active = state.get("active_trade") or {}
    committee = state.get("positions", {}).get("committee") or {}
    
    # Prefer active_trade if it exists, fall back to positions.committee
    if active:
        has_position = True
        direction = active.get("direction", "?")
        entry = active.get("entry", 0)
        amount = active.get("amount", 0)
        trail = active.get("trailing_stop", 0)
    elif committee:
        has_position = True
        # positions.committee uses 'side' not 'direction'
        side = committee.get("side", "?")
        direction = "BUY" if side == "long" else ("SELL" if side == "short" else side)
        entry = committee.get("entry", 0)
        amount = committee.get("amount", 0)
        trail = committee.get("trailing_stop", committee.get("stop_loss", 0))
    else:
        has_position = False
        direction = "?"
        entry = 0
        amount = 0
        trail = 0

    position_str = "FLAT (no position)"
    if has_position:
        if direction == "BUY" and entry > 0:
            pnl_pct = (price - entry) / entry * 100
        elif direction == "SELL" and entry > 0:
            pnl_pct = (entry - price) / entry * 100
        else:
            pnl_pct = 0
        position_str = f"{direction} {amount:.3f} BTC @ ${entry:,.0f} | PnL: {pnl_pct:+.2f}% | SL/Trail: ${trail:,.0f}"

    macro = get_macro_summary()
    # CII disabled — GDELT rate-limited, low decision value
    # cii = get_cii_summary()

    prompt = f"""You are a fast-response crypto trading monitor. Make a QUICK decision based on cached intelligence + live price.

LIVE DATA:
- Price: ${price:,.2f} ({change_pct:+.2f}% from reference)
- Funding rate: {funding}
- Position: {position_str}
- Last full decision: {last_decision} at {last_decision_time}

CACHED INTELLIGENCE (from last full agent run):
Market: {market_report}
News: {news_report}

MACRO: {macro}
GEO: (CII disabled)

RULES:
- You are a SUPPLEMENT to the full agent pipeline, not a replacement
- Only override the last decision if conditions have MATERIALLY changed
- If flat: only enter on HIGH confidence (8+/10) setups with clear catalyst
- If in position: only exit early if stop/invalidation level breached or major adverse event
- Default to HOLD unless something is clearly wrong or clearly right

Output EXACTLY:
---LIGHT_DECISION---
ACTION: [HOLD/BUY/SELL/CLOSE]
CONFIDENCE: [1-10]
REASON: [one line]
---END---"""

    # Use OpenClaw gateway (Anthropic subscription, same as executor)
    try:
        gw_url, gw_token = _get_gateway()
        headers = {"Content-Type": "application/json"}
        if gw_token:
            headers["Authorization"] = f"Bearer {gw_token}"
        resp = requests.post(
            f"{gw_url}/chat/completions",
            headers=headers,
            json={
                "model": "anthropic/claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices or not choices[0].get("message", {}).get("content"):
            log.warning(f"LLM returned empty response: {json.dumps(data)[:200]}")
            return {}
        content = choices[0]["message"]["content"]
        return {"raw": content, "price": price, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        return {}


def parse_decision(raw: str) -> dict:
    """Parse the structured decision output."""
    result = {"action": "HOLD", "confidence": 0, "reason": ""}
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("ACTION:"):
            result["action"] = line.split(":", 1)[1].strip().upper()
        elif line.startswith("CONFIDENCE:"):
            try:
                result["confidence"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("REASON:"):
            result["reason"] = line.split(":", 1)[1].strip()
    return result


def should_act(decision: dict, state: dict, price_data: dict, report_age_hours: float | None = None) -> bool:
    """Determine if we should act on a light decision."""
    action = decision["action"]
    confidence = decision["confidence"]

    if action == "HOLD":
        return False

    # CLOSE requires position validation — don't close based on stale state
    if action == "CLOSE":
        # Verify we actually have a position in state
        active = state.get("active_trade") or state.get("positions", {}).get("committee")
        if not active:
            log.warning("⚠️ CLOSE rejected: no position in state (active_trade or positions.committee)")
            return False
        # Verify the CLOSE reason makes sense for the position direction
        # Use specific position-referencing phrases to avoid false positives
        # on common words like "short-term", "long-term", "no longer", etc.
        import re
        side = active.get("direction") or active.get("side", "")
        reason_lower = decision.get("reason", "").lower()
        # Patterns that indicate the LLM thinks we have a short/long position
        _short_position_re = re.compile(r'\bshort\b(?!\s*-?\s*term)')
        _long_position_re = re.compile(r'\blong\b(?!\s*-?\s*term|\s*-?\s*run)')
        if side in ("long", "BUY") and _short_position_re.search(reason_lower):
            log.warning(f"⚠️ CLOSE rejected: reason references SHORT but position is LONG — stale context")
            return False
        if side in ("short", "SELL") and _long_position_re.search(reason_lower):
            log.warning(f"⚠️ CLOSE rejected: reason references LONG but position is SHORT — stale context")
            return False
        return True

    if report_age_hours is not None and report_age_hours > MIN_REPORT_FRESHNESS_HOURS:
        log.warning(
            f"Light decision {action} rejected: cached reports too stale "
            f"({report_age_hours:.1f}h > {MIN_REPORT_FRESHNESS_HOURS}h)"
        )
        return False

    if confidence < MIN_CONFIDENCE:
        log.info(f"Light decision {action} rejected: confidence {confidence} < {MIN_CONFIDENCE}")
        return False

    # Check minimum price move from last decision
    abs_change = abs(price_data.get("change_pct", 0))
    if abs_change < MIN_MOVE_PCT:
        log.info(f"Light decision {action} rejected: move {abs_change:.2f}% < {MIN_MOVE_PCT}% minimum")
        return False

    # Check daily trade limit
    today = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d")
    light_trades = state.get("light_trades", {}).get(today, 0)
    if light_trades >= MAX_LIGHT_TRADES_PER_DAY:
        log.warning(f"Light trade limit reached ({light_trades}/{MAX_LIGHT_TRADES_PER_DAY} today)")
        return False

    last_override_at = state.get("last_light_override_time")
    if last_override_at:
        try:
            last_dt = datetime.fromisoformat(last_override_at)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            seconds_since = (datetime.now(timezone.utc) - last_dt).total_seconds()
            if seconds_since < OVERRIDE_COOLDOWN_SECONDS:
                log.warning(
                    f"Light decision {action} rejected: override cooldown active "
                    f"({seconds_since:.0f}s < {OVERRIDE_COOLDOWN_SECONDS}s)"
                )
                return False
        except Exception as e:
            log.warning(f"Failed to parse last_light_override_time '{last_override_at}': {e} — cooldown bypassed")

    return True


def check_pattern_signals() -> dict | None:
    """Run pattern scanner and return the highest-confidence actionable signal, or None."""
    try:
        from tradingagents.dataflows.pattern_scanner import scan_all_patterns_structured
        results = scan_all_patterns_structured("BTC/USDT")
        # Filter: detected, confidence >= 7, has a tradeable direction
        actionable = [
            r for r in results
            if r["detected"]
            and r["confidence"] >= 7
            and r.get("direction") in ("BUY", "SELL", "SHORT")
        ]
        if not actionable:
            return None
        # Highest confidence wins
        best = max(actionable, key=lambda r: r["confidence"])
        # Normalize direction: SHORT → SELL for executor
        direction = best["direction"]
        if direction == "SHORT":
            direction = "SELL"
        return {
            "action": direction,
            "pattern_name": best["pattern"],
            "confidence": best["confidence"],
            "reason": best["details"],
            "entry_price": best.get("entry_price"),
            "stop_price": best.get("stop_price"),
        }
    except Exception as e:
        log.warning(f"Pattern scanner failed: {e}")
        return None


def executor_running() -> bool:
    """Best-effort check: don't spawn executor if its lock is already held."""
    executor_lock = LOGS / ".executor.lock"
    executor_lock.parent.mkdir(parents=True, exist_ok=True)
    fd = open(executor_lock, "a+")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except (IOError, OSError):
        return True
    finally:
        fd.close()


def main():
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        log.warning("⏸️ Another light decision instance is running — exiting")
        lock_fd.close()
        return

    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))

    log.info("=" * 40)
    log.info("⚡ Light Decision Layer")
    log.info("=" * 40)

    # Get real-time price
    price_data = get_realtime_price()
    if not price_data:
        # Fallback to REST
        try:
            import ccxt
            ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})
            ticker = ex.fetch_ticker(SYMBOL)
            price_data = {"price": ticker["last"], "change_pct": 0, "funding_rate": "?", "timestamp": datetime.now(timezone.utc).isoformat()}
            log.info(f"Using REST fallback: ${ticker['last']:,.2f}")
        except Exception as e:
            log.error(f"Cannot get price: {e}")
            return

    log.info(f"💰 Price: ${price_data['price']:,.2f} ({price_data.get('change_pct', 0):+.2f}%)")

    # Load cached data
    reports, reports_file = get_cached_reports()
    state = get_executor_state()
    report_age_hours = report_file_age_hours(reports_file)

    if not reports:
        log.warning("No cached reports — skipping evaluation")
        return

    if report_age_hours is not None:
        log.info(f"📦 Cached reports age: {report_age_hours:.1f}h")

    # ─── Pattern scanner fast-path (replaces old Green Lane) ─────────
    pattern_signal = check_pattern_signals()
    if pattern_signal:
        log.info(
            f"🔍 Pattern detected: {pattern_signal['pattern_name']} "
            f"(conf={pattern_signal['confidence']}, dir={pattern_signal['action']})"
        )

        # Position gate: check both active_trade and positions.committee
        active = state.get("active_trade") or {}
        committee = state.get("positions", {}).get("committee") or {}
        current_pos = active or committee
        if current_pos:
            # Determine current direction
            pos_dir = current_pos.get("direction") or current_pos.get("side", "")
            if pos_dir in ("long", "BUY"):
                pos_dir = "BUY"
            elif pos_dir in ("short", "SELL"):
                pos_dir = "SELL"

            if pattern_signal["action"] == pos_dir:
                log.info(
                    f"⏸️  Pattern {pattern_signal['pattern_name']} same direction as "
                    f"existing {pos_dir} position — skipping (no doubling)"
                )
            else:
                # Opposite direction: write reversal override and spawn executor
                log.warning(
                    f"🔄 Pattern {pattern_signal['pattern_name']} OPPOSITE to existing "
                    f"{pos_dir} position — triggering full pipeline reversal"
                )
                reversal_file = LOGS / "pattern_reversal.json"
                tmp_file = LOGS / ".pattern_reversal.tmp"
                tmp_file.write_text(json.dumps({
                    "action": pattern_signal["action"],
                    "pattern_name": pattern_signal["pattern_name"],
                    "confidence": pattern_signal["confidence"],
                    "reason": f"Reversal: {pattern_signal['reason']}",
                    "entry_price": pattern_signal.get("entry_price"),
                    "stop_price": pattern_signal.get("stop_price"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "pattern_scanner",
                }))
                tmp_file.rename(reversal_file)

                if executor_running():
                    log.warning("⏸️ Executor already running — pattern reversal queued")
                else:
                    import subprocess
                    subprocess.Popen(
                        [str(VENV_PYTHON), str(PROJECT_ROOT / "scripts" / "live_executor.py")],
                        cwd=str(PROJECT_ROOT),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
        else:
            # No position: write pattern override and spawn executor
            log.warning(
                f"🚨 PATTERN ENTRY: {pattern_signal['action']} via "
                f"{pattern_signal['pattern_name']} — launching executor"
            )
            override_file = LOGS / "pattern_override.json"
            tmp_file = LOGS / ".pattern_override.tmp"
            tmp_file.write_text(json.dumps({
                "action": pattern_signal["action"],
                "pattern_name": pattern_signal["pattern_name"],
                "confidence": pattern_signal["confidence"],
                "reason": pattern_signal["reason"],
                "entry_price": pattern_signal.get("entry_price"),
                "stop_price": pattern_signal.get("stop_price"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "pattern_scanner",
            }))
            tmp_file.rename(override_file)

            # Update daily trade count and cooldown
            today = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d")
            state_lock_fd = None
            try:
                state_lock_fd = open(STATE_FILE.with_suffix(".lock"), "w")
                fcntl.flock(state_lock_fd, fcntl.LOCK_EX)
                fresh_state = load_json(STATE_FILE)
                fresh_state.setdefault("light_trades", {})[today] = fresh_state.get("light_trades", {}).get(today, 0) + 1
                fresh_state["last_light_override_time"] = datetime.now(timezone.utc).isoformat()
                tmp = STATE_FILE.with_suffix(".tmp")
                tmp.write_text(json.dumps(fresh_state, indent=2))
                tmp.rename(STATE_FILE)
            except Exception as e:
                log.warning(f"Failed to update light trade count: {e}")
            finally:
                if state_lock_fd is not None:
                    try:
                        fcntl.flock(state_lock_fd, fcntl.LOCK_UN)
                        state_lock_fd.close()
                    except Exception:
                        pass

            if executor_running():
                log.warning("⏸️ Executor already running — pattern override queued")
            else:
                import subprocess
                subprocess.Popen(
                    [str(VENV_PYTHON), str(PROJECT_ROOT / "scripts" / "live_executor.py")],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        return  # Pattern path handled — skip LLM evaluation
    # ─── End pattern scanner fast-path ────────────────────────────────

    # Run evaluation
    result = run_light_evaluation(price_data, reports, state)
    if not result:
        return

    decision = parse_decision(result.get("raw", ""))
    log.info(f"⚡ Decision: {decision['action']} (confidence={decision['confidence']}) — {decision['reason']}")

    if should_act(decision, state, price_data, report_age_hours=report_age_hours):
        action = decision["action"]
        if action in ("BUY", "SELL", "CLOSE"):
            label = "LIGHT ENTRY" if action in ("BUY", "SELL") else "LIGHT CLOSE"
            log.warning(f"🚨 {label}: {action} — launching executor with override")
            # Atomic write: write to temp then rename
            light_file = LOGS / "light_override.json"
            tmp_file = LOGS / ".light_override.tmp"
            tmp_file.write_text(json.dumps({
                "action": action,
                "confidence": decision["confidence"],
                "reason": decision["reason"],
                "price_at_decision": price_data["price"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "light_decision",
            }))
            tmp_file.rename(light_file)

            # Increment daily light trade count (merge-patch: re-read, update key, atomic write)
            today = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d")
            state_lock_fd = None
            try:
                state_lock_fd = open(STATE_FILE.with_suffix(".lock"), "w")
                fcntl.flock(state_lock_fd, fcntl.LOCK_EX)
                fresh_state = load_json(STATE_FILE)
                fresh_state.setdefault("light_trades", {})[today] = fresh_state.get("light_trades", {}).get(today, 0) + 1
                fresh_state["last_light_override_time"] = datetime.now(timezone.utc).isoformat()
                tmp = STATE_FILE.with_suffix(".tmp")
                tmp.write_text(json.dumps(fresh_state, indent=2))
                tmp.rename(STATE_FILE)
            except Exception as e:
                log.warning(f"Failed to update light trade count: {e}")
            finally:
                if state_lock_fd is not None:
                    try:
                        fcntl.flock(state_lock_fd, fcntl.LOCK_UN)
                        state_lock_fd.close()
                    except Exception:
                        pass

            if executor_running():
                log.warning("⏸️ Executor already running — light override queued, not spawning another executor")
            else:
                import subprocess
                subprocess.Popen(
                    [str(VENV_PYTHON), str(PROJECT_ROOT / "scripts" / "live_executor.py")],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
    else:
        log.info("⏸️  No action needed")


if __name__ == "__main__":
    main()

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
from datetime import datetime, timezone
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

# Only act if confidence is HIGH and move is significant
MIN_CONFIDENCE = 8         # 1-10, only act on high conviction
MIN_MOVE_PCT = 1.5         # Minimum % move from last decision price to consider
MAX_LIGHT_TRADES_PER_DAY = 2  # Don't overtrade

LOGS.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LIGHT_LOG),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("light_decision")


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


def get_cached_reports() -> dict:
    """Load most recent analyst reports."""
    today = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d")
    reports_file = LOGS / f"agent_reports_{today}.json"
    if not reports_file.exists():
        # Try yesterday
        from datetime import timedelta
        yesterday = (datetime.now(ZoneInfo("Australia/Sydney")) - timedelta(days=1)).strftime("%Y-%m-%d")
        reports_file = LOGS / f"agent_reports_{yesterday}.json"
    return load_json(reports_file)


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
    has_position = bool(state.get("active_trade"))
    active = state.get("active_trade", {})

    position_str = "FLAT (no position)"
    if has_position:
        direction = active.get("direction", "?")
        entry = active.get("entry", 0)
        amount = active.get("amount", 0)
        trail = active.get("trailing_stop", 0)
        pnl_pct = ((price - entry) / entry * 100) if direction == "BUY" and entry > 0 else (((entry - price) / entry * 100) if entry > 0 else 0)
        position_str = f"{direction} {amount:.3f} BTC @ ${entry:,.0f} | PnL: {pnl_pct:+.2f}% | Trail: ${trail:,.0f}"

    macro = get_macro_summary()
    cii = get_cii_summary()

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
GEO: {cii}

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
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
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


def should_act(decision: dict, state: dict, price_data: dict) -> bool:
    """Determine if we should act on a light decision."""
    action = decision["action"]
    confidence = decision["confidence"]

    if action == "HOLD":
        return False

    if confidence < MIN_CONFIDENCE:
        log.info(f"Light decision {action} rejected: confidence {confidence} < {MIN_CONFIDENCE}")
        return False

    # Check minimum price move from last decision
    abs_change = abs(price_data.get("change_pct", 0))
    if abs_change < MIN_MOVE_PCT and action in ("BUY", "SELL"):
        log.info(f"Light decision {action} rejected: move {abs_change:.2f}% < {MIN_MOVE_PCT}% minimum")
        return False

    # Check daily trade limit
    today = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d")
    light_trades = state.get("light_trades", {}).get(today, 0)
    if light_trades >= MAX_LIGHT_TRADES_PER_DAY:
        log.warning(f"Light trade limit reached ({light_trades}/{MAX_LIGHT_TRADES_PER_DAY} today)")
        return False

    return True


def check_green_lane_setup() -> dict | None:
    """Run green lane scanner, return signal dict if triggered."""
    try:
        from tradingagents.graph.green_lane import check_green_lane
        signal = check_green_lane("BTC/USDT", min_quality=7)
        if signal:
            log.info(f"🟢 Green lane: {signal.direction.upper()} q={signal.quality_score} entry=${signal.entry_price:,.2f}")
            return {
                "action": signal.direction.upper(),
                "entry": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "tp1": signal.tp1,
                "tp2": signal.tp2,
                "quality": signal.quality_score,
                "reasoning": signal.reasoning,
            }
    except Exception as e:
        log.debug(f"Green lane scan skipped: {e}")
    return None


def main():
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
    reports = get_cached_reports()
    state = get_executor_state()

    if not reports:
        log.warning("No cached reports — skipping evaluation")
        return

    # Check green lane setup (deterministic, no LLM cost)
    gl_signal = check_green_lane_setup()
    if gl_signal:
        # Green lane found — write override and launch executor directly (no LLM needed)
        log.warning(f"🟢 GREEN LANE TRIGGER: {gl_signal['action']} q={gl_signal['quality']}")
        gl_file = LOGS / "green_lane_override.json"
        tmp = LOGS / ".green_lane_override.tmp"
        tmp.write_text(json.dumps({**gl_signal, "timestamp": datetime.now(timezone.utc).isoformat(), "source": "light_green_lane"}))
        tmp.rename(gl_file)
        import subprocess
        subprocess.Popen(
            [str(VENV_PYTHON), str(PROJECT_ROOT / "scripts" / "live_executor.py")],
            cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return

    # Run evaluation
    result = run_light_evaluation(price_data, reports, state)
    if not result:
        return

    decision = parse_decision(result.get("raw", ""))
    log.info(f"⚡ Decision: {decision['action']} (confidence={decision['confidence']}) — {decision['reason']}")

    if should_act(decision, state, price_data):
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

            # Increment daily light trade count
            today = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d")
            state.setdefault("light_trades", {})[today] = state.get("light_trades", {}).get(today, 0) + 1
            STATE_FILE.write_text(json.dumps(state, indent=2))

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

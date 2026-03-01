#!/usr/bin/env python3
"""
Price Sentinel — Monitors BTC price vs last 4H candle close.
Triggers the full TradingAgents pipeline on big moves.

Runs every 5 min via cron. Zero LLM cost.

Thresholds:
  3-5%  → Alert only (log)
  ≥5%   → Trigger full agent pipeline
"""

import ccxt
import json
import subprocess
import sys
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "logs" / "sentinel_state.json"
EXECUTOR = PROJECT_ROOT / "scripts" / "live_executor.py"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python3"

SYMBOL = "BTC/USDT:USDT"
TRIGGER_THRESHOLD = 0.05  # 5% — wake agents
COOLDOWN_MINUTES = 120    # Don't re-trigger within 2 hours

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "sentinel.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("sentinel")


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_last_4h_close():
    """Get the close price of the last completed 4H candle."""
    exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})
    candles = exchange.fetch_ohlcv(SYMBOL, "4h", limit=2)
    # candles[-1] is current (incomplete), candles[-2] is last closed
    last_closed = candles[-2]
    return {
        "close": last_closed[4],
        "timestamp": last_closed[0],
        "time_str": datetime.fromtimestamp(last_closed[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def get_current_price():
    exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})
    ticker = exchange.fetch_ticker(SYMBOL)
    return ticker["last"]


def main():
    state = load_state()
    now = datetime.now(timezone.utc)

    # Check cooldown
    last_trigger = state.get("last_trigger")
    if last_trigger:
        last_dt = datetime.fromisoformat(last_trigger)
        mins_since = (now - last_dt).total_seconds() / 60
        if mins_since < COOLDOWN_MINUTES:
            log.debug(f"⏳ Cooldown: {COOLDOWN_MINUTES - mins_since:.0f}min remaining")
            return

    # Get prices
    candle = get_last_4h_close()
    current = get_current_price()
    change_pct = (current - candle["close"]) / candle["close"]
    abs_change = abs(change_pct)

    state["last_check"] = now.isoformat()
    state["last_price"] = current
    state["last_4h_close"] = candle["close"]
    state["last_change_pct"] = round(change_pct * 100, 2)

    if abs_change >= TRIGGER_THRESHOLD:
        direction = "📈" if change_pct > 0 else "📉"
        log.warning(f"🚨 {direction} BIG MOVE: {change_pct*100:+.2f}% | ${candle['close']:,.0f} → ${current:,.0f} | TRIGGERING PIPELINE")
        
        state["last_trigger"] = now.isoformat()
        state["trigger_reason"] = f"{change_pct*100:+.2f}% from 4H close"
        save_state(state)

        # Run the full executor
        result = subprocess.run(
            [str(VENV_PYTHON), str(EXECUTOR)],
            capture_output=True, text=True, timeout=600,
            cwd=str(PROJECT_ROOT),
        )
        log.info(f"Pipeline exit code: {result.returncode}")
        if result.returncode != 0:
            log.error(f"Pipeline stderr: {result.stderr[-500:]}")
        return

    else:
        log.debug(f"😴 Quiet: {change_pct*100:+.2f}% | ${current:,.0f}")

    save_state(state)


if __name__ == "__main__":
    main()

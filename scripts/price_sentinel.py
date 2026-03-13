#!/usr/bin/env python3
"""
Price Sentinel — Monitors BTC price vs last 4H candle close.
Triggers the full TradingAgents pipeline on big moves.

Runs every 5 min via cron. Zero LLM cost.

Thresholds:
  3-5%  → Alert only (log)
  >=5%  → Trigger full agent pipeline
"""

import ccxt
import json
import subprocess
import sys
import logging
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "logs" / "sentinel_state.json"
EXECUTOR = PROJECT_ROOT / "scripts" / "live_executor.py"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python3"
SYMBOL = "BTC/USDT:USDT"
TRIGGER_THRESHOLD = 0.03  # 3% — wake agents (was 5%, too slow)
COOLDOWN_MINUTES = 60     # Don't re-trigger within 1 hour (was 2h)

warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
)

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_root_logger.handlers.clear()
_root_logger.addHandler(logging.FileHandler(PROJECT_ROOT / "logs" / "sentinel.log"))
log = logging.getLogger("sentinel")
log.propagate = True

for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.FileHandler):
        handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))

CANDLE_MOVE_THRESHOLD = 0.03   # 3% candle body triggers pipeline
VOLUME_SPIKE_MULTIPLIER = 2.5  # Volume >= 2.5x 6-candle average triggers pipeline




def get_last_two_4h_candles():
    """Get last 8 completed 4H candles for candle-over-candle + volume spike detection."""
    exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})
    candles = exchange.fetch_ohlcv(SYMBOL, "4h", limit=10)
    # candles[-1] = current incomplete, [-2] = last closed
    closed = candles[:-1]  # all completed candles
    prev = closed[-2]
    last = closed[-1]
    # 6-candle average volume excluding the last candle
    avg_vol = sum(c[5] for c in closed[-7:-1]) / 6
    return {
        "prev": {"open": prev[1], "high": prev[2], "low": prev[3], "close": prev[4], "ts": prev[0]},
        "last": {"open": last[1], "high": last[2], "low": last[3], "close": last[4],
                 "volume": last[5], "ts": last[0]},
        "avg_vol_6": avg_vol,
    }


# ─────────────────────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))



# ─────────────────────────────────────────────────────────────────────────────
# Price helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    state = load_state()
    now = datetime.now(timezone.utc)

    # Check cooldown
    last_trigger = state.get("last_trigger")
    if last_trigger:
        last_dt = datetime.fromisoformat(last_trigger)
        mins_since = (now - last_dt).total_seconds() / 60
        if mins_since < COOLDOWN_MINUTES:
            log.debug(f"Cooldown: {COOLDOWN_MINUTES - mins_since:.0f}min remaining")
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

    # Candle-over-candle check: did the last closed 4H candle have a big body vs prior close?
    # This catches fast moves that already closed before we check current price
    candle_trigger = False
    candle_trigger_reason = ""
    try:
        two_candles = get_last_two_4h_candles()
        prev_close = two_candles["prev"]["close"]
        last_close = two_candles["last"]["close"]
        last_high = two_candles["last"]["high"]
        last_low = two_candles["last"]["low"]
        last_ts = two_candles["last"]["ts"]

        # Check if we already triggered on this candle
        last_candle_triggered = state.get("last_candle_trigger_ts", 0)

        if last_ts != last_candle_triggered and prev_close > 0:
            body_pct = abs(last_close - prev_close) / prev_close
            range_pct = (last_high - last_low) / prev_close
            direction_candle = "UP" if last_close > prev_close else "DOWN"

            last_vol = two_candles["last"]["volume"]
            avg_vol = two_candles.get("avg_vol_6", 0)
            vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
            vol_spike = vol_ratio >= VOLUME_SPIKE_MULTIPLIER

            if range_pct >= CANDLE_MOVE_THRESHOLD:
                # Variation (high-low range) >= 3% — covers both big body and big wick
                log.warning(
                    f"BIG CANDLE ({direction_candle}): variation={range_pct*100:.2f}% "
                    f"body={body_pct*100:.2f}% vol={vol_ratio:.1f}x avg | "
                    f"${last_low:,.0f}-${last_high:,.0f} | TRIGGERING"
                )
                candle_trigger = True
                candle_trigger_reason = f"{range_pct*100:.2f}% candle variation, vol={vol_ratio:.1f}x"
                state["last_candle_trigger_ts"] = last_ts
            elif vol_spike:
                # High volume even without big price move — unusual activity, worth checking
                log.warning(
                    f"VOLUME SPIKE: {vol_ratio:.1f}x avg | "
                    f"${last_close:,.0f} body={body_pct*100:.2f}% | TRIGGERING"
                )
                candle_trigger = True
                candle_trigger_reason = f"volume spike {vol_ratio:.1f}x avg (body={body_pct*100:.2f}%)"
                state["last_candle_trigger_ts"] = last_ts
    except Exception as e:
        log.warning(f"Candle-over-candle check failed: {e}")

    triggered = abs_change >= TRIGGER_THRESHOLD or candle_trigger
    if triggered:
        if candle_trigger and abs_change < TRIGGER_THRESHOLD:
            direction = "UP" if candle_trigger_reason.startswith("+") or "UP" in candle_trigger_reason else "DOWN"
            log.warning(f"BIG CANDLE TRIGGER: {candle_trigger_reason} | current=${current:,.0f}")
            trigger_reason = candle_trigger_reason
        else:
            direction = "UP" if change_pct > 0 else "DOWN"
            log.warning(f"BIG MOVE ({direction}): {change_pct*100:+.2f}% | ${candle['close']:,.0f} -> ${current:,.0f} | TRIGGERING PIPELINE")
            trigger_reason = f"{change_pct*100:+.2f}% from 4H close"

    if triggered:
        state["last_trigger"] = now.isoformat()
        state["trigger_reason"] = trigger_reason
        save_state(state)

        # Run the full executor — executor itself holds the fcntl lock and will
        # exit immediately if already running. No sentinel-side pre-check needed
        # (avoids TOCTOU race between lock probe and subprocess start).
        log.info(f"🚨 SENTINEL TRIGGER: {trigger_reason} — launching executor")
        result = subprocess.run(
            [str(VENV_PYTHON), str(EXECUTOR)],
            capture_output=True, text=True, timeout=600,
            cwd=str(PROJECT_ROOT),
        )
        log.info(f"Pipeline exit code: {result.returncode}")
        if result.returncode != 0:
            log.error(f"Pipeline stderr: {result.stderr[-500:]}")
        save_state(state)
        return

    else:
        log.debug(f"Quiet: {change_pct*100:+.2f}% | ${current:,.0f}")

    save_state(state)


if __name__ == "__main__":
    main()

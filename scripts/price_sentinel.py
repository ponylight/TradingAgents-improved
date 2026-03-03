#!/usr/bin/env python3
"""
Price Sentinel — Monitors BTC price vs last 4H candle close.
Triggers the full TradingAgents pipeline on big moves.

Runs every 5 min via cron. Zero LLM cost.

Thresholds:
  3-5%  → Alert only (log)
  >=5%  → Trigger full agent pipeline

Green Lane scanning runs every cycle (~3 sec when quiet).
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
GREEN_LANE_LOG = PROJECT_ROOT / "logs" / "green_lane_signals.json"

SYMBOL = "BTC/USDT:USDT"
GL_SYMBOL = "BTC/USDT"  # Green lane scanner expects spot symbol format
TRIGGER_THRESHOLD = 0.05  # 5% — wake agents
COOLDOWN_MINUTES = 120    # Don't re-trigger within 2 hours

# Green Lane rate limits
GL_COOLDOWN_HOURS = 4     # Minimum hours between green lane triggers
GL_DAILY_MAX = 3          # Maximum triggers per day
GL_MIN_QUALITY = 7        # Minimum quality score to act on

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "sentinel.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("sentinel")

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
# Green Lane imports — optional, sentinel keeps running if unavailable
# ─────────────────────────────────────────────────────────────────────────────
_GREEN_LANE_AVAILABLE = False
_gl_import_err_msg = None

try:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from tradingagents.graph.green_lane import check_green_lane, format_green_lane_alert
    from live_executor import can_open_green_lane
    _GREEN_LANE_AVAILABLE = True
    log.debug("Green lane modules loaded successfully")
except Exception as _e:
    _gl_import_err_msg = str(_e)
    log.warning(f"Green lane unavailable (sentinel still runs): {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _ensure_gl_state(state: dict) -> dict:
    """Ensure green lane rate-limit fields exist in state."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if "green_lane_last_trigger" not in state:
        state["green_lane_last_trigger"] = None
    if "green_lane_triggers_today" not in state:
        state["green_lane_triggers_today"] = 0
    if "green_lane_today_date" not in state:
        state["green_lane_today_date"] = today
    # Reset daily counter if date has changed
    if state["green_lane_today_date"] != today:
        log.info(f"New day ({today}) — resetting green lane daily counter")
        state["green_lane_triggers_today"] = 0
        state["green_lane_today_date"] = today
    return state


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
# Green Lane scan
# ─────────────────────────────────────────────────────────────────────────────

import time as _time

def check_green_lane_signal(state: dict = None) -> dict:
    """
    Run green lane scanner and log signal if triggered and passes all gates.
    # Brief delay to avoid Bybit rate limits after sentinel price check
    import time; time.sleep(2)

    Returns updated state dict (caller must save_state).
    Does NOT execute trades — logs to green_lane_signals.json only.
    Actual order execution will be wired in a future phase.
    """
    if state is None:
        state = load_state()

    state = _ensure_gl_state(state)
    now = datetime.now(timezone.utc)

    if not _GREEN_LANE_AVAILABLE:
        log.debug(f"Green lane scan skipped (module unavailable): {_gl_import_err_msg}")
        return state

    # ── Run scanner ───────────────────────────────────────────────────────────
    log.info(f"Green lane scan: {GL_SYMBOL} (min_quality={GL_MIN_QUALITY})")
    try:
        signal = check_green_lane(GL_SYMBOL, min_quality=GL_MIN_QUALITY)
    except Exception as e:
        log.error(f"Green lane scanner error (sentinel continues): {e}")
        return state

    if signal is None:
        log.info("Green lane: no qualifying setup detected")
        return state

    log.info(
        f"Green lane signal detected! dir={signal.direction} "
        f"quality={signal.quality_score}/10 entry=${signal.entry_price:,.2f}"
    )

    # ── Rate limit: daily cap ─────────────────────────────────────────────────
    triggers_today = state.get("green_lane_triggers_today", 0)
    if triggers_today >= GL_DAILY_MAX:
        log.info(f"Green lane: daily limit reached ({triggers_today}/{GL_DAILY_MAX} today) — skipping")
        return state

    # ── Rate limit: cooldown ──────────────────────────────────────────────────
    last_trigger_str = state.get("green_lane_last_trigger")
    if last_trigger_str:
        last_dt = datetime.fromisoformat(last_trigger_str)
        hours_since = (now - last_dt).total_seconds() / 3600
        if hours_since < GL_COOLDOWN_HOURS:
            log.info(
                f"Green lane: cooldown active ({hours_since:.1f}h since last, "
                f"need {GL_COOLDOWN_HOURS}h) — skipping"
            )
            return state

    # ── Position gate: can_open_green_lane ───────────────────────────────────
    try:
        executor_state_file = PROJECT_ROOT / "logs" / "executor_state.json"
        if executor_state_file.exists():
            exec_state = json.loads(executor_state_file.read_text())
        else:
            exec_state = {}

        equity = exec_state.get("equity", 0.0)
        allowed, reason = can_open_green_lane(exec_state, equity)
        if not allowed:
            log.info(f"Green lane: position gate blocked — {reason}")
            return state
        log.info(f"Green lane: position gate passed — {reason}")
    except Exception as e:
        log.warning(f"Green lane: position gate check failed ({e}) — skipping conservatively")
        return state

    # ── All gates passed — log signal ─────────────────────────────────────────
    log.info(
        f"GREEN LANE SIGNAL LOGGED: {signal.direction.upper()} "
        f"entry=${signal.entry_price:,.2f} sl=${signal.stop_loss:,.2f} "
        f"tp1=${signal.tp1:,.2f} tp2=${signal.tp2:,.2f} q={signal.quality_score}"
    )

    # Update rate limit state
    state["green_lane_last_trigger"] = now.isoformat()
    state["green_lane_triggers_today"] = triggers_today + 1

    # Append to green_lane_signals.json
    try:
        GREEN_LANE_LOG.parent.mkdir(exist_ok=True)
        existing = []
        if GREEN_LANE_LOG.exists():
            try:
                existing = json.loads(GREEN_LANE_LOG.read_text())
                if not isinstance(existing, list):
                    existing = [existing]
            except Exception:
                existing = []

        record = {
            "timestamp": now.isoformat(),
            "symbol": GL_SYMBOL,
            "signal": signal.model_dump(),
            "rate_limit_state": {
                "triggers_today": state["green_lane_triggers_today"],
                "today_date": state["green_lane_today_date"],
            },
            "execution_status": "pending_wire",  # actual order wiring coming later
        }
        existing.append(record)
        GREEN_LANE_LOG.write_text(json.dumps(existing, indent=2, default=str))
        log.info(f"Green lane signal appended to {GREEN_LANE_LOG}")
    except Exception as e:
        log.error(f"Failed to write green_lane_signals.json: {e}")

    # Print formatted alert
    try:
        alert_text = format_green_lane_alert(signal, "PENDING — execution not yet wired")
        log.info(f"\n{alert_text}")
        print(f"\n{alert_text}\n")
    except Exception as e:
        log.warning(f"format_green_lane_alert failed: {e}")

    return state


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
            # Still run green lane scan even during price-trigger cooldown
            state = check_green_lane_signal(state)
            save_state(state)
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

            if body_pct >= CANDLE_MOVE_THRESHOLD:
                log.warning(
                    f"BIG CANDLE ({direction_candle}): body={body_pct*100:.2f}% "
                    f"vol={vol_ratio:.1f}x avg | "
                    f"prev_close=${prev_close:,.0f} → candle_close=${last_close:,.0f} | TRIGGERING"
                )
                candle_trigger = True
                candle_trigger_reason = f"{body_pct*100:+.2f}% candle body, vol={vol_ratio:.1f}x"
                state["last_candle_trigger_ts"] = last_ts
            elif range_pct >= CANDLE_MOVE_THRESHOLD * 2:
                # Wick-inclusive: candle range is 6%+ even if body is smaller
                log.warning(
                    f"BIG CANDLE RANGE ({direction_candle}): range={range_pct*100:.2f}% "
                    f"vol={vol_ratio:.1f}x avg | "
                    f"${last_low:,.0f}-${last_high:,.0f} | TRIGGERING"
                )
                candle_trigger = True
                candle_trigger_reason = f"{range_pct*100:.2f}% candle range, vol={vol_ratio:.1f}x"
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

        # Run the full executor
        result = subprocess.run(
            [str(VENV_PYTHON), str(EXECUTOR)],
            capture_output=True, text=True, timeout=600,
            cwd=str(PROJECT_ROOT),
        )
        log.info(f"Pipeline exit code: {result.returncode}")
        if result.returncode != 0:
            log.error(f"Pipeline stderr: {result.stderr[-500:]}")
        # Run green lane scan after pipeline trigger
        state = check_green_lane_signal(state)
        save_state(state)
        return

    else:
        log.debug(f"Quiet: {change_pct*100:+.2f}% | ${current:,.0f}")

    # Run green lane scan every cycle
    state = check_green_lane_signal(state)
    save_state(state)


if __name__ == "__main__":
    main()

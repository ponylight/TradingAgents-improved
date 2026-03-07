#!/usr/bin/env python3
"""
Real-time price monitor via Bybit websocket.

Replaces polling-based sentinel with sub-second price detection.
Triggers:
  - Sentinel logic on threshold breach (launches executor)
  - Writes real-time state to shared file for light decision layer

Runs as a persistent daemon. Zero API cost (websocket is free).
"""

import asyncio
import json
import time
import logging
import subprocess
import signal
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt.pro as ccxtpro

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "logs" / "ws_state.json"
REALTIME_FILE = PROJECT_ROOT / "logs" / "realtime_price.json"  # Shared with light layer
EXECUTOR = PROJECT_ROOT / "scripts" / "live_executor.py"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python3"

SYMBOL = "BTC/USDT:USDT"

# Trigger thresholds
TRIGGER_PCT = 0.03          # 3% move from reference → launch executor
ALERT_PCT = 0.02            # 2% move → log alert
COOLDOWN_SECONDS = 3600     # 1 hour between executor launches
REFERENCE_RESET_HOURS = 4   # Reset reference price every 4h (align with cron)

(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "ws_monitor.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ws_monitor")

# Graceful shutdown
_running = True


def _signal_handler(sig, frame):
    global _running
    log.info(f"Received signal {sig} — shutting down")
    _running = False


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def write_realtime(data: dict):
    """Write real-time price data for light decision layer (atomic)."""
    REALTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = REALTIME_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(REALTIME_FILE)


def launch_executor(reason: str):
    """Launch executor in background (non-blocking). Executor handles its own locking."""
    log.info(f"🚨 LAUNCHING EXECUTOR: {reason}")
    try:
        subprocess.Popen(
            [str(VENV_PYTHON), str(EXECUTOR)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        log.error(f"Failed to launch executor: {e}")


async def run_monitor():
    global _running

    exchange = ccxtpro.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})

    state = load_state()
    reference_price = state.get("reference_price", 0)
    reference_time = state.get("reference_time", 0)
    last_trigger_time = state.get("last_trigger_time", 0)
    tick_count = 0

    log.info("🔌 Websocket monitor starting...")
    log.info(f"   Trigger: {TRIGGER_PCT*100}% | Alert: {ALERT_PCT*100}% | Cooldown: {COOLDOWN_SECONDS}s")

    try:
        while _running:
            try:
                ticker = await exchange.watch_ticker(SYMBOL)
            except Exception as e:
                log.warning(f"Websocket error: {e} — reconnecting in 5s")
                await asyncio.sleep(5)
                continue

            now = time.time()
            price = ticker["last"]
            bid = ticker.get("bid", price)
            ask = ticker.get("ask", price)
            volume_24h = ticker.get("quoteVolume", 0)
            funding_rate = ticker.get("info", {}).get("fundingRate", "")
            tick_count += 1

            # Set reference on first tick or every REFERENCE_RESET_HOURS
            if reference_price == 0 or (now - reference_time) > REFERENCE_RESET_HOURS * 3600:
                reference_price = price
                reference_time = now
                log.info(f"📌 Reference price set: ${price:,.2f}")

            change_pct = (price - reference_price) / reference_price
            abs_change = abs(change_pct)

            # Write real-time data for light decision layer
            if tick_count % 10 == 0:  # Every ~10 ticks to reduce disk IO
                write_realtime({
                    "price": price,
                    "bid": bid,
                    "ask": ask,
                    "spread": round(ask - bid, 2),
                    "change_pct": round(change_pct * 100, 4),
                    "reference_price": reference_price,
                    "volume_24h": volume_24h,
                    "funding_rate": funding_rate,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "tick_count": tick_count,
                })

            # Alert threshold
            if abs_change >= ALERT_PCT and tick_count % 60 == 0:
                direction = "🟢 UP" if change_pct > 0 else "🔴 DOWN"
                log.info(f"{direction} {change_pct*100:+.2f}% | ${price:,.2f} (ref ${reference_price:,.2f})")

            # Trigger threshold
            if abs_change >= TRIGGER_PCT:
                since_last = now - last_trigger_time
                if since_last >= COOLDOWN_SECONDS:
                    direction = "UP" if change_pct > 0 else "DOWN"
                    reason = f"{change_pct*100:+.2f}% {direction} from ${reference_price:,.0f} → ${price:,.0f}"
                    log.warning(f"🚨 TRIGGER: {reason}")

                    launch_executor(reason)
                    last_trigger_time = now
                    reference_price = price  # Reset reference after trigger
                    reference_time = now

                    state.update({
                        "reference_price": reference_price,
                        "reference_time": reference_time,
                        "last_trigger_time": last_trigger_time,
                        "last_trigger_reason": reason,
                    })
                    save_state(state)
                else:
                    remaining = COOLDOWN_SECONDS - since_last
                    if tick_count % 300 == 0:
                        log.debug(f"Cooldown: {remaining:.0f}s remaining")

            # Periodic state save (every ~5 min)
            if tick_count % 3000 == 0:
                state.update({
                    "reference_price": reference_price,
                    "reference_time": reference_time,
                    "last_trigger_time": last_trigger_time,
                    "last_price": price,
                    "last_update": now,
                })
                save_state(state)
                log.info(f"💓 Alive: ${price:,.2f} ({change_pct*100:+.2f}%) | {tick_count} ticks")

    except asyncio.CancelledError:
        log.info("Monitor cancelled")
    finally:
        await exchange.close()
        log.info("🔌 Websocket closed")


if __name__ == "__main__":
    log.info("=" * 50)
    log.info("Bybit Websocket Price Monitor v1")
    log.info("=" * 50)
    try:
        asyncio.run(run_monitor())
    except KeyboardInterrupt:
        log.info("Interrupted — exiting")

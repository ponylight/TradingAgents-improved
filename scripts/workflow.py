#!/usr/bin/env python3
"""V20 Unified Workflow — Single entry point for all trading operations.

Runs every 30 minutes via one cron job. Decides what to do based on timing:

- EVERY RUN (30min): Position monitor + Risk Guard
- ON 4H CANDLE CLOSE: Full V20 signal detection + execution
- DAILY 8AM AEDT: Morning briefing
- DAILY 9AM AEDT: GM health check

Usage:
    python scripts/workflow.py              # auto-detect what to do
    python scripts/workflow.py --force-scan # force signal scan regardless of timing
    python scripts/workflow.py --dry-run    # skip order placement
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import ccxt
from dotenv import load_dotenv

# ── Setup ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
STATE_FILE = SCRIPT_DIR / ".workflow_state.json"

load_dotenv(PROJECT_DIR / ".env")

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_SECRET = os.getenv("BYBIT_SECRET", "")
SYMBOL = "BTC/USDT:USDT"

# Sydney timezone offset (AEDT = UTC+11)
AEDT_OFFSET = timedelta(hours=11)

# 4H candle close times in UTC: 0, 4, 8, 12, 16, 20
CANDLE_CLOSE_HOURS_UTC = {0, 4, 8, 12, 16, 20}


def get_exchange():
    exchange = ccxt.bybit({
        "apiKey": BYBIT_API_KEY,
        "secret": BYBIT_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "linear"},
    })
    exchange.urls["api"] = {
        "public": "https://api-demo.bybit.com",
        "private": "https://api-demo.bybit.com",
    }
    exchange.has["fetchCurrencies"] = False
    exchange.load_markets()
    return exchange


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "last_scan_candle": None,  # timestamp of last scanned 4H candle
        "last_briefing_date": None,
        "last_gm_date": None,
        "last_run": None,
    }


def save_state(state: dict):
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Timing Helpers ────────────────────────────────────────────────────

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_sydney() -> datetime:
    return now_utc() + AEDT_OFFSET


def is_4h_candle_close() -> bool:
    """Check if we're within 10 minutes after a 4H candle close."""
    utc = now_utc()
    if utc.hour in CANDLE_CLOSE_HOURS_UTC and utc.minute <= 10:
        return True
    return False


def get_current_candle_id() -> str:
    """Return identifier for the current 4H candle (e.g., '2026-02-21T16')."""
    utc = now_utc()
    candle_hour = (utc.hour // 4) * 4
    return f"{utc.strftime('%Y-%m-%d')}T{candle_hour:02d}"


def is_briefing_time() -> bool:
    """8:00-8:30 AM Sydney time."""
    syd = now_sydney()
    return syd.hour == 8 and syd.minute <= 30


def is_gm_time() -> bool:
    """9:00-9:30 AM Sydney time."""
    syd = now_sydney()
    return syd.hour == 9 and syd.minute <= 30


# ── Position Monitor ──────────────────────────────────────────────────

def monitor_positions(exchange) -> list[str]:
    """Check open positions and update trailing stops. Returns status lines."""
    output = []

    try:
        positions = exchange.fetch_positions([SYMBOL])
        open_pos = [p for p in positions if abs(float(p.get("contracts", 0) or 0)) > 0]
    except Exception as e:
        return [f"⚠️ Position check failed: {e}"]

    if not open_pos:
        output.append("📊 No open positions")
        return output

    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        price = ticker["last"]
    except Exception as e:
        return [f"⚠️ Ticker fetch failed: {e}"]

    for pos in open_pos:
        side = pos["side"]
        contracts = float(pos["contracts"])
        entry = float(pos["entryPrice"])
        pnl = float(pos.get("unrealizedPnl", 0))
        leverage = float(pos.get("leverage", 1))

        if side == "long":
            pnl_pct = (price - entry) / entry * 100
            liq_dist = (price - float(pos.get("liquidationPrice", 0))) / price * 100
        else:
            pnl_pct = (entry - price) / entry * 100
            liq_dist = (float(pos.get("liquidationPrice", 0)) - price) / price * 100

        output.append(
            f"📊 {side.upper()} {contracts} BTC @ ${entry:,.0f} | "
            f"PnL: ${pnl:+,.2f} ({pnl_pct:+.1f}%) | "
            f"{leverage:.0f}x | Liq dist: {liq_dist:.1f}%"
        )

        # Warn if close to liquidation
        if liq_dist < 3:
            output.append(f"🚨 WARNING: {liq_dist:.1f}% from liquidation!")

    return output


# ── Signal Scanner (delegates to live_scanner) ────────────────────────

def run_signal_scan(dry_run: bool = False) -> str:
    """Run V20 signal detection. Returns output text."""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "live_scanner.py"),
    ]
    if dry_run:
        cmd.append("--dry-run")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_DIR),
            env={**os.environ, "PYTHONPATH": str(PROJECT_DIR)},
        )
        output = result.stdout + result.stderr
        return output.strip()
    except subprocess.TimeoutExpired:
        return "⚠️ Scanner timed out (120s)"
    except Exception as e:
        return f"⚠️ Scanner failed: {e}"


# ── Risk Guard ────────────────────────────────────────────────────────

def run_risk_guard() -> str:
    """Run risk guard checks. Returns output text."""
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "risk_guard.py")],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_DIR),
            env={**os.environ, "PYTHONPATH": str(PROJECT_DIR)},
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "⚠️ Risk Guard timed out"
    except Exception as e:
        return f"⚠️ Risk Guard failed: {e}"


# ── Morning Briefing ──────────────────────────────────────────────────

def run_briefing(exchange) -> str:
    """Generate morning briefing."""
    lines = []
    syd = now_sydney()
    lines.append(f"☀️ Morning Briefing — {syd.strftime('%a %b %d, %Y')}")
    lines.append("")

    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        price = ticker["last"]
        pct_24h = ticker.get("percentage", 0)
        lines.append(f"BTC: ${price:,.0f} ({pct_24h:+.1f}% 24h)")
    except Exception as e:
        lines.append(f"BTC: fetch failed ({e})")

    try:
        balance = exchange.fetch_balance()
        usdt = float(balance.get("USDT", {}).get("total", 0))
        lines.append(f"Balance: ${usdt:,.2f} USDT")
    except Exception as e:
        lines.append(f"Balance: fetch failed ({e})")

    try:
        positions = exchange.fetch_positions([SYMBOL])
        open_pos = [p for p in positions if abs(float(p.get("contracts", 0) or 0)) > 0]
        if open_pos:
            for p in open_pos:
                side = p["side"]
                contracts = float(p["contracts"])
                entry = float(p["entryPrice"])
                pnl = float(p.get("unrealizedPnl", 0))
                lines.append(f"Position: {side} {contracts} BTC @ ${entry:,.0f}, PnL: ${pnl:+,.2f}")
        else:
            lines.append("No open positions")
    except Exception as e:
        lines.append(f"Positions: fetch failed ({e})")

    return "\n".join(lines)


# ── General Manager ───────────────────────────────────────────────────

def run_gm() -> str:
    """Run GM health check. Returns output text."""
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "general_manager.py")],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_DIR),
            env={**os.environ, "PYTHONPATH": str(PROJECT_DIR)},
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "⚠️ GM health check timed out"
    except Exception as e:
        return f"⚠️ GM health check failed: {e}"


# ── Main Workflow ─────────────────────────────────────────────────────

def main(force_scan: bool = False, dry_run: bool = False):
    state = load_state()
    output_sections = []
    syd = now_sydney()

    print(f"{'='*60}")
    print(f"  V20 Workflow — {syd.strftime('%Y-%m-%d %H:%M AEDT')}")
    print(f"{'='*60}")
    print()

    # Connect to exchange
    try:
        exchange = get_exchange()
    except Exception as e:
        msg = f"🚨 CRITICAL: Cannot connect to Bybit: {e}"
        print(msg)
        output_sections.append(msg)
        save_state(state)
        return "\n\n".join(output_sections)

    # ── ALWAYS: Position Monitor ──────────────────────────────────
    pos_lines = monitor_positions(exchange)
    pos_section = "\n".join(pos_lines)
    print(pos_section)
    print()

    # ── ALWAYS: Risk Guard ────────────────────────────────────────
    rg_output = run_risk_guard()
    print(rg_output)
    print()

    # ── ON 4H CANDLE CLOSE: Signal Scan ──────────────────────────
    candle_id = get_current_candle_id()
    should_scan = force_scan or (is_4h_candle_close() and state.get("last_scan_candle") != candle_id)

    if should_scan:
        print("⚡ 4H candle close detected — running signal scan...")
        print()
        scan_output = run_signal_scan(dry_run=dry_run)
        print(scan_output)
        state["last_scan_candle"] = candle_id
        output_sections.append(scan_output)
        print()
    else:
        print("⏳ Not a 4H candle close — skipping signal scan")
        print()

    # ── DAILY 8AM: Morning Briefing ──────────────────────────────
    today = syd.strftime("%Y-%m-%d")
    if is_briefing_time() and state.get("last_briefing_date") != today:
        briefing = run_briefing(exchange)
        print(briefing)
        output_sections.append(briefing)
        state["last_briefing_date"] = today
        print()

    # ── DAILY 9AM: GM Health Check ───────────────────────────────
    if is_gm_time() and state.get("last_gm_date") != today:
        gm_output = run_gm()
        print(gm_output)
        # Only include in output if issues found
        if "⚠️" in gm_output or "🚨" in gm_output:
            output_sections.append(gm_output)
        state["last_gm_date"] = today
        print()

    # ── Summary ───────────────────────────────────────────────────
    # Build compact output for alerts
    has_alerts = any("🚨" in s or "⚠️" in s for s in [rg_output] + pos_lines)
    has_signal = should_scan and "SIGNAL FIRED" in (output_sections[0] if output_sections else "")

    if has_alerts:
        output_sections.insert(0, rg_output)
    if has_alerts or has_signal or output_sections:
        # There's something worth reporting
        pass
    else:
        # Quiet run — just positions + risk guard, nothing noteworthy
        output_sections = [pos_section]

    save_state(state)

    final = "\n\n".join(output_sections) if output_sections else pos_section
    print()
    print(f"{'='*60}")
    print("  Workflow complete")
    print(f"{'='*60}")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V20 Unified Workflow")
    parser.add_argument("--force-scan", action="store_true", help="Force signal scan")
    parser.add_argument("--dry-run", action="store_true", help="Skip order placement")
    args = parser.parse_args()
    main(force_scan=args.force_scan, dry_run=args.dry_run)

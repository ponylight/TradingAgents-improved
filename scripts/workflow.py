#!/usr/bin/env python3
"""V20 Unified Workflow — Single entry point for all trading operations.

Runs every 15 minutes via one cron job. Decides what to do based on timing:

EVERY RUN (15 min):
  - Position monitor
  - Risk Guard
  - Volatility monitor
  - Update trade statuses

ON 4H CANDLE CLOSE:
  - Signal scan + execution
  - Record trade if signal fired

DAILY 8AM AEDT:
  - Morning briefing
  - Data freshness check

DAILY 9AM AEDT:
  - GM health check
  - Signal drought check
  - Git auto-commit

WEEKLY SUNDAY 8PM AEDT:
  - Weekly performance report
  - Log cleanup

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

# ── Setup ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
STATE_FILE  = SCRIPT_DIR / ".workflow_state.json"

# Ensure scripts/ is importable as a package when run from PROJECT_DIR
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

load_dotenv(PROJECT_DIR / ".env")

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "hr3xN6ozshJJhFbgxs")
BYBIT_SECRET  = os.getenv("BYBIT_SECRET",  "Q5LcRuP4XxcUiasxAZm76FKbdrCcoCFDRQy0")
SYMBOL        = "BTC/USDT:USDT"

# Sydney AEDT offset (UTC+11). DST-safe approximation — adjust to +10 if needed.
AEDT_OFFSET = timedelta(hours=11)

# 4H candle close times in UTC: 0, 4, 8, 12, 16, 20
CANDLE_CLOSE_HOURS_UTC = {0, 4, 8, 12, 16, 20}


# ── Exchange ───────────────────────────────────────────────────────────────────

def get_exchange():
    exchange = ccxt.bybit({
        "apiKey":          BYBIT_API_KEY,
        "secret":          BYBIT_SECRET,
        "enableRateLimit": True,
        "options":         {"defaultType": "linear"},
    })
    exchange.urls["api"] = {
        "public":  "https://api-demo.bybit.com",
        "private": "https://api-demo.bybit.com",
    }
    exchange.has["fetchCurrencies"] = False
    exchange.load_markets()
    return exchange


# ── State ──────────────────────────────────────────────────────────────────────

# Keys owned by the workflow scheduler (not written by submodules)
_WORKFLOW_KEYS = {
    "last_scan_candle",
    "last_briefing_date",
    "last_gm_date",
    "last_weekly_report_date",
    "last_cleanup_date",
    "last_drought_date",
    "last_git_date",
    "last_data_date",
    "last_run",
}


def load_state() -> dict:
    """Load ONLY workflow scheduling keys (ignores submodule keys like ATR)."""
    defaults = {k: None for k in _WORKFLOW_KEYS}
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                raw = json.load(f)
            # Only return keys we own
            for k in _WORKFLOW_KEYS:
                if k in raw:
                    defaults[k] = raw[k]
        except (json.JSONDecodeError, IOError):
            pass
    return defaults


def save_state(state: dict):
    """Merge state with current file contents (preserves keys written by submodules)."""
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Read current file to preserve keys written by submodules (e.g., ATR from volatility monitor)
    existing = {}
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                existing = json.load(f)
        except Exception:
            pass

    # Merge: existing keys updated by state (our keys take precedence)
    merged = {**existing, **state}

    with open(STATE_FILE, "w") as f:
        json.dump(merged, f, indent=2)


# ── Timing Helpers ─────────────────────────────────────────────────────────────

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_sydney() -> datetime:
    """Current time in approximate Sydney AEDT (UTC+11)."""
    return now_utc() + AEDT_OFFSET


def is_4h_candle_close() -> bool:
    """True if within 10 minutes after a 4H candle close."""
    utc = now_utc()
    return utc.hour in CANDLE_CLOSE_HOURS_UTC and utc.minute <= 10


def get_current_candle_id() -> str:
    """Identifier for the current 4H candle, e.g. '2026-02-21T16'."""
    utc         = now_utc()
    candle_hour = (utc.hour // 4) * 4
    return f"{utc.strftime('%Y-%m-%d')}T{candle_hour:02d}"


def is_briefing_time() -> bool:
    """8:00–8:30 AM Sydney."""
    syd = now_sydney()
    return syd.hour == 8 and syd.minute <= 30


def is_gm_time() -> bool:
    """9:00–9:30 AM Sydney."""
    syd = now_sydney()
    return syd.hour == 9 and syd.minute <= 30


def is_sunday_evening() -> bool:
    """Sunday 8:00–8:30 PM Sydney (weekday 6 = Sunday)."""
    syd = now_sydney()
    return syd.weekday() == 6 and syd.hour == 20 and syd.minute <= 30


# ── Position Monitor ───────────────────────────────────────────────────────────

def monitor_positions(exchange) -> list[str]:
    """Check open positions. Returns status lines."""
    output = []

    try:
        positions = exchange.fetch_positions([SYMBOL])
        open_pos  = [p for p in positions if abs(float(p.get("contracts", 0) or 0)) > 0]
    except Exception as e:
        return [f"⚠️ Position check failed: {e}"]

    if not open_pos:
        output.append("📊 No open positions")
        return output

    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        price  = ticker["last"]
    except Exception as e:
        return [f"⚠️ Ticker fetch failed: {e}"]

    for pos in open_pos:
        side      = pos["side"]
        contracts = float(pos["contracts"])
        entry     = float(pos["entryPrice"])
        pnl       = float(pos.get("unrealizedPnl", 0))
        leverage  = float(pos.get("leverage", 1))
        liq_price = float(pos.get("liquidationPrice", 0) or 0)

        if side == "long":
            pnl_pct  = (price - entry) / entry * 100
            liq_dist = (price - liq_price) / price * 100 if liq_price else 999.0
        else:
            pnl_pct  = (entry - price) / entry * 100
            liq_dist = (liq_price - price) / price * 100 if liq_price else 999.0

        output.append(
            f"📊 {side.upper()} {contracts} BTC @ ${entry:,.0f} | "
            f"PnL: ${pnl:+,.2f} ({pnl_pct:+.1f}%) | "
            f"{leverage:.0f}x | Liq dist: {liq_dist:.1f}%"
        )

        if liq_dist < 3:
            output.append(f"🚨 WARNING: {liq_dist:.1f}% from liquidation!")

    return output


# ── Signal Scanner (delegates to live_scanner) ─────────────────────────────────

def run_signal_scan(dry_run: bool = False) -> tuple[str, bool]:
    """Run V20 signal detection.

    Returns (output_text, signal_fired).
    """
    cmd = [sys.executable, str(SCRIPT_DIR / "live_scanner.py")]
    if dry_run:
        cmd.append("--dry-run")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_DIR),
            env={**os.environ, "PYTHONPATH": str(PROJECT_DIR)},
        )
        output = result.stdout + result.stderr
        signal_fired = "V20 SIGNAL FIRED" in output
        return output.strip(), signal_fired
    except subprocess.TimeoutExpired:
        return "⚠️ Scanner timed out (120s)", False
    except Exception as e:
        return f"⚠️ Scanner failed: {e}", False


# ── Risk Guard ─────────────────────────────────────────────────────────────────

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


# ── Volatility Monitor ─────────────────────────────────────────────────────────

def run_volatility_monitor() -> str:
    """Run the ATR-based volatility monitor. Returns status string."""
    try:
        from general_manager import check_volatility
        msg, _ = check_volatility()
        return msg
    except Exception as e:
        return f"⚠️ Volatility monitor failed: {e}"


# ── Update Trade Statuses ──────────────────────────────────────────────────────

def run_update_trades(exchange) -> list[str]:
    """Update open trade statuses from exchange. Returns list of status msgs."""
    try:
        from gm_performance import update_trades
        msgs = update_trades(exchange)
        return msgs if msgs else []
    except Exception as e:
        return [f"⚠️ update_trades failed: {e}"]


# ── Record Trade (if signal fired) ────────────────────────────────────────────

def try_record_trade(scan_output: str):
    """Attempt to parse scan_output and record the trade in trade history.

    Only call this when scan_output contains 'V20 SIGNAL FIRED'.
    The scanner already printed signal details; we parse them best-effort.
    """
    try:
        from gm_performance import record_trade

        # Build a minimal signal_data dict from parsed output
        signal_data = _parse_signal_from_output(scan_output)
        if signal_data is None:
            print("⚠️  Could not parse signal data for trade recording — skipping")
            return

        # We don't have the raw order_data here; pass empty dict
        record_trade(signal_data, order_data={})
        print("📝 Trade recorded in performance tracker")
    except Exception as e:
        print(f"⚠️  Trade recording failed: {e}")


def _parse_signal_from_output(output: str) -> dict | None:
    """Parse signal details from live_scanner stdout.

    Returns a minimal dict compatible with gm_performance.record_trade(),
    or None if parsing fails.
    """
    import re

    def _find(pattern, text, default=None, cast=float):
        m = re.search(pattern, text)
        if m:
            try:
                return cast(m.group(1).replace(",", ""))
            except Exception:
                pass
        return default

    # Verify signal fired
    if "V20 SIGNAL FIRED" not in output:
        return None

    entry      = _find(r"Entry:\s+\$?([\d,.]+)",     output, default=0.0)
    stop_loss  = _find(r"Stop Loss:\s+\$?([\d,.]+)", output, default=0.0)
    tp1        = _find(r"TP1.*?:\s+\$?([\d,.]+)",    output, default=0.0)
    tp2        = _find(r"TP2.*?:\s+\$?([\d,.]+)",    output, default=0.0)
    leverage   = _find(r"Leverage:\s+(\d+)x",        output, default=0, cast=int)
    size_btc   = _find(r"Position:\s+([\d.]+) BTC",  output, default=0.0)

    # Confluence
    conf_match = re.search(r"Confluence:\s+(.+)", output)
    confluence = []
    if conf_match:
        raw = conf_match.group(1).strip()
        confluence = [c.strip() for c in re.split(r"[ +]+", raw) if c.strip()]

    # candle_time
    ct_match = re.search(r"Candle time:\s+(.+)", output)
    candle_time = ct_match.group(1).strip() if ct_match else None

    return {
        "entry_price":   entry,
        "stop_loss":     stop_loss,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "leverage":      leverage,
        "side":          "long",
        "confluence":    confluence,
        "candle_time":   candle_time,
        "size_btc":      size_btc,
    }


# ── Morning Briefing ───────────────────────────────────────────────────────────

def run_briefing(exchange) -> str:
    """Generate morning briefing text."""
    lines = []
    syd   = now_sydney()
    lines.append(f"☀️ Morning Briefing — {syd.strftime('%a %b %d, %Y')}")
    lines.append("")

    try:
        ticker  = exchange.fetch_ticker(SYMBOL)
        price   = ticker["last"]
        pct_24h = ticker.get("percentage", 0)
        lines.append(f"BTC: ${price:,.0f} ({pct_24h:+.1f}% 24h)")
    except Exception as e:
        lines.append(f"BTC: fetch failed ({e})")

    try:
        balance = exchange.fetch_balance()
        usdt    = float(balance.get("USDT", {}).get("total", 0))
        lines.append(f"Balance: ${usdt:,.2f} USDT")
    except Exception as e:
        lines.append(f"Balance: fetch failed ({e})")

    try:
        positions = exchange.fetch_positions([SYMBOL])
        open_pos  = [p for p in positions if abs(float(p.get("contracts", 0) or 0)) > 0]
        if open_pos:
            for p in open_pos:
                side      = p["side"]
                contracts = float(p["contracts"])
                entry     = float(p["entryPrice"])
                pnl       = float(p.get("unrealizedPnl", 0))
                lines.append(f"Position: {side} {contracts} BTC @ ${entry:,.0f}, PnL: ${pnl:+,.2f}")
        else:
            lines.append("No open positions")
    except Exception as e:
        lines.append(f"Positions: fetch failed ({e})")

    # Performance snapshot
    try:
        from gm_performance import get_performance_summary
        s = get_performance_summary()
        lines.append(
            f"Performance: {s['total_trades']} trades | "
            f"WR: {s['win_rate']:.1f}% | "
            f"Total PnL: ${s['total_pnl']:+,.0f}"
        )
    except Exception:
        pass

    return "\n".join(lines)


# ── General Manager ────────────────────────────────────────────────────────────

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


def run_signal_drought_check() -> str:
    """Run signal drought check from GM module."""
    try:
        from general_manager import check_signal_drought
        return check_signal_drought()
    except Exception as e:
        return f"⚠️ Drought check failed: {e}"


def run_data_freshness() -> str:
    """Run data freshness check from GM module."""
    try:
        from general_manager import run_data_freshness as _rdf
        return _rdf()
    except Exception as e:
        return f"⚠️ Data freshness failed: {e}"


def run_git_commit() -> str:
    """Run git auto-commit from GM module."""
    try:
        from general_manager import run_git_commit as _rgc
        return _rgc()
    except Exception as e:
        return f"⚠️ Git commit failed: {e}"


def run_weekly_report() -> str:
    """Generate weekly report from GM module."""
    try:
        from general_manager import generate_weekly_report
        return generate_weekly_report()
    except Exception as e:
        return f"⚠️ Weekly report failed: {e}"


def run_log_cleanup() -> str:
    """Run log cleanup from GM module."""
    try:
        from general_manager import run_log_cleanup as _rlc
        return _rlc()
    except Exception as e:
        return f"⚠️ Log cleanup failed: {e}"


# ── Main Workflow ──────────────────────────────────────────────────────────────

def main(force_scan: bool = False, dry_run: bool = False):
    state  = load_state()
    output_sections = []
    syd    = now_sydney()
    today  = syd.strftime("%Y-%m-%d")

    print(f"{'='*60}")
    print(f"  V20 Workflow — {syd.strftime('%Y-%m-%d %H:%M AEDT')}")
    print(f"{'='*60}")
    print()

    # ── Connect to exchange ────────────────────────────────────────
    try:
        exchange = get_exchange()
    except Exception as e:
        msg = f"🚨 CRITICAL: Cannot connect to Bybit: {e}"
        print(msg)
        output_sections.append(msg)
        save_state(state)
        return "\n\n".join(output_sections)

    # ══════════════════════════════════════════════════════════════
    # EVERY RUN: Position Monitor
    # ══════════════════════════════════════════════════════════════
    pos_lines  = monitor_positions(exchange)
    pos_section = "\n".join(pos_lines)
    print(pos_section)
    print()

    # ══════════════════════════════════════════════════════════════
    # EVERY RUN: Risk Guard
    # ══════════════════════════════════════════════════════════════
    rg_output = run_risk_guard()
    print(rg_output)
    print()

    # ══════════════════════════════════════════════════════════════
    # EVERY RUN: Volatility Monitor
    # ══════════════════════════════════════════════════════════════
    vol_output = run_volatility_monitor()
    print(vol_output)
    print()

    # ══════════════════════════════════════════════════════════════
    # EVERY RUN: Update trade statuses
    # ══════════════════════════════════════════════════════════════
    trade_updates = run_update_trades(exchange)
    if trade_updates:
        for msg in trade_updates:
            print(msg)
            output_sections.append(msg)
        print()

    # ══════════════════════════════════════════════════════════════
    # ON 4H CANDLE CLOSE: Signal Scan + Record Trade
    # ══════════════════════════════════════════════════════════════
    candle_id   = get_current_candle_id()
    should_scan = force_scan or (
        is_4h_candle_close() and state.get("last_scan_candle") != candle_id
    )

    if should_scan:
        print("⚡ 4H candle close detected — running signal scan...")
        print()
        scan_output, signal_fired = run_signal_scan(dry_run=dry_run)
        print(scan_output)
        state["last_scan_candle"] = candle_id

        if signal_fired:
            output_sections.append(scan_output)
            # Record trade in performance tracker
            if not dry_run:
                try_record_trade(scan_output)

        print()
    else:
        print("⏳ Not a 4H candle close — skipping signal scan")
        print()

    # ══════════════════════════════════════════════════════════════
    # DAILY 8AM: Morning Briefing + Data Freshness
    # ══════════════════════════════════════════════════════════════
    if is_briefing_time() and state.get("last_briefing_date") != today:
        briefing = run_briefing(exchange)
        print(briefing)
        output_sections.append(briefing)
        state["last_briefing_date"] = today
        print()

        # Data freshness (piggyback 8AM run)
        if state.get("last_data_date") != today:
            print("📡 Running data freshness check...")
            data_output = run_data_freshness()
            print(data_output)
            state["last_data_date"] = today
            print()

    # ══════════════════════════════════════════════════════════════
    # DAILY 9AM: GM Health Check + Drought Check + Git Commit
    # ══════════════════════════════════════════════════════════════
    if is_gm_time() and state.get("last_gm_date") != today:
        gm_output = run_gm()
        print(gm_output)
        # Only include in output if issues found
        if "⚠️" in gm_output or "🚨" in gm_output:
            output_sections.append(gm_output)
        state["last_gm_date"] = today
        print()

        # Signal drought check
        if state.get("last_drought_date") != today:
            drought_output = run_signal_drought_check()
            print(drought_output)
            if "⚠️" in drought_output or "🚨" in drought_output:
                output_sections.append(drought_output)
            state["last_drought_date"] = today
            print()

        # Git auto-commit
        if state.get("last_git_date") != today:
            git_output = run_git_commit()
            print(git_output)
            if "⚠️" in git_output:
                output_sections.append(git_output)
            state["last_git_date"] = today
            print()

    # ══════════════════════════════════════════════════════════════
    # WEEKLY SUNDAY 8PM: Weekly Report + Log Cleanup
    # ══════════════════════════════════════════════════════════════
    sunday_key = syd.strftime("%Y-%W")   # year + week number

    if is_sunday_evening() and state.get("last_weekly_report_date") != sunday_key:
        print("📊 Sunday 8PM — generating weekly report...")
        weekly_output = run_weekly_report()
        print(weekly_output)
        output_sections.append(weekly_output)
        state["last_weekly_report_date"] = sunday_key
        print()

        # Log cleanup
        if state.get("last_cleanup_date") != sunday_key:
            print("🧹 Running weekly log cleanup...")
            cleanup_output = run_log_cleanup()
            print(cleanup_output)
            state["last_cleanup_date"] = sunday_key
            print()

    # ── Summary ────────────────────────────────────────────────────
    has_alerts = any(
        "🚨" in s or "⚠️" in s
        for s in [rg_output] + pos_lines + [vol_output]
    )

    if has_alerts:
        output_sections.insert(0, rg_output)

    if not output_sections:
        output_sections = [pos_section]

    save_state(state)

    final = "\n\n".join(output_sections)
    print()
    print(f"{'='*60}")
    print("  Workflow complete")
    print(f"{'='*60}")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V20 Unified Workflow")
    parser.add_argument("--force-scan", action="store_true", help="Force signal scan")
    parser.add_argument("--dry-run",    action="store_true", help="Skip order placement")
    args = parser.parse_args()
    main(force_scan=args.force_scan, dry_run=args.dry_run)

#!/usr/bin/env python3
"""General Manager — V20 Daily Health Check

Verifies all systems are operational: scanner logs, Bybit connectivity,
balance sanity, orphaned positions, cron jobs, and disk space.

Usage:
    python scripts/general_manager.py
"""

import os
import re
import json
import shutil
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple

import ccxt
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
LOGS_DIR    = PROJECT_DIR / "logs"

load_dotenv(PROJECT_DIR / ".env")

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_SECRET  = os.getenv("BYBIT_SECRET", "")

# ── Exchange ──────────────────────────────────────────────────────────

SYMBOL = "BTC/USDT:USDT"


def get_exchange() -> ccxt.Exchange:
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


# ── 1. Scanner Health ─────────────────────────────────────────────────

def check_scanner_health() -> Tuple[str, bool]:
    """Count scanner runs in the last 24h from log files."""
    now_utc = datetime.now(timezone.utc)
    cutoff  = now_utc - timedelta(hours=24)

    # Check today's and yesterday's log file
    dates_to_check = [
        now_utc.strftime("%Y-%m-%d"),
        (now_utc - timedelta(days=1)).strftime("%Y-%m-%d"),
    ]

    run_times = []

    for date_str in dates_to_check:
        log_file = LOGS_DIR / f"scanner_{date_str}.log"
        if not log_file.exists():
            continue

        try:
            content = log_file.read_text(errors="replace")
        except IOError:
            continue

        # Each scanner run logs a separator or "V20 Signal Scanner" header
        # Count lines matching run start markers
        for line in content.splitlines():
            # Look for timestamp lines matching run starts
            # Pattern: "2026-02-21 17:05:00  INFO     V20 Signal Scanner"
            # or any line with a recognisable timestamp + known marker
            ts_match = re.match(
                r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+INFO\s+(?:={10}|V20 Signal Scanner)",
                line
            )
            if ts_match:
                try:
                    ts = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
                    ts = ts.replace(tzinfo=timezone.utc)
                    if ts >= cutoff:
                        run_times.append(ts)
                except ValueError:
                    pass

    # Deduplicate (in case multiple marker lines per run)
    run_times = sorted(set(run_times))

    # Expected runs: 6 per 24h (every 4h)
    expected = 6
    actual   = len(run_times)

    if actual >= 4:
        ok = True
        status = f"✅ Scanner: {actual}/{expected} runs in last 24h"
    else:
        ok = False
        status = f"⚠️  Scanner: Only {actual}/{expected} runs in last 24h — MISSING RUNS"

    return status, ok


# ── 2 & 3. Bybit API + Balance ────────────────────────────────────────

def check_bybit_api() -> Tuple[str, bool, Optional[float], Optional[ccxt.Exchange]]:
    """Check API connectivity and return balance."""
    try:
        exchange = get_exchange()
        bal = exchange.fetch_balance()
        usdt = bal.get("USDT", {})
        equity = float(usdt.get("total", usdt.get("free", 0)) or 0)
        ok = True
        status = f"✅ Bybit API: Connected, ${equity:,.2f} USDT"
        return status, ok, equity, exchange
    except Exception as e:
        return f"⚠️  Bybit API: FAILED — {e}", False, None, None


def check_balance(equity: float) -> Tuple[str, bool]:
    if equity is None:
        return "⚠️  Balance: Unknown (API failed)", False
    if equity <= 0:
        return f"⚠️  Balance: ${equity:,.2f} USDT — ZERO OR NEGATIVE!", False
    return f"✅ Balance: ${equity:,.2f} USDT", True


# ── 4. Orphaned Positions ─────────────────────────────────────────────

def check_orphaned_positions(exchange: Optional[ccxt.Exchange]) -> Tuple[str, bool]:
    """Check for positions without a corresponding stop-loss order."""
    if exchange is None:
        return "⚠️  Positions: Cannot check (API failed)", False

    try:
        positions = exchange.fetch_positions([SYMBOL])
        open_pos  = [p for p in positions if abs(float(p.get("contracts", 0) or 0)) > 0]
    except Exception as e:
        return f"⚠️  Positions: Failed to fetch — {e}", False

    if not open_pos:
        return "✅ Positions: 0 open, 0 orphaned orders", True

    try:
        open_orders = exchange.fetch_open_orders(SYMBOL)
    except Exception as e:
        return f"⚠️  Positions: {len(open_pos)} open, but failed to fetch orders — {e}", False

    # Stop-loss orders are typically "stop_market" or "stop" type, reduceOnly=True
    sl_orders = [
        o for o in open_orders
        if o.get("type") in ("stop_market", "stop", "stop_limit")
        and o.get("reduceOnly", False)
    ]

    orphaned = 0
    for pos in open_pos:
        # Check if there's at least one stop-loss order for this position's symbol
        pos_sym = pos.get("symbol", SYMBOL)
        matching_sl = [o for o in sl_orders if o.get("symbol") == pos_sym]
        if not matching_sl:
            orphaned += 1

    if orphaned > 0:
        return (
            f"⚠️  Positions: {len(open_pos)} open, {orphaned} WITHOUT stop-loss — ALERT",
            False,
        )
    else:
        return (
            f"✅ Positions: {len(open_pos)} open, all have stop-loss orders",
            True,
        )


# ── 5. Cron Health ────────────────────────────────────────────────────

CRON_JOBS_REQUIRED = ["v20-scanner", "morning-briefing"]


def check_cron_health() -> Tuple[str, bool]:
    """Run `openclaw cron list` and verify required jobs are healthy."""
    nvm_node_path = os.path.expanduser("~/.nvm/versions/node/v24.13.1/bin")
    env = os.environ.copy()
    env["PATH"] = f"{nvm_node_path}:{env.get('PATH', '')}"

    try:
        result = subprocess.run(
            ["openclaw", "cron", "list"],
            capture_output=True, text=True, timeout=15, env=env
        )
        output = result.stdout + result.stderr
    except FileNotFoundError:
        return "⚠️  Cron: `openclaw` not found in PATH", False
    except subprocess.TimeoutExpired:
        return "⚠️  Cron: `openclaw cron list` timed out", False
    except Exception as e:
        return f"⚠️  Cron: Failed to run command — {e}", False

    if result.returncode != 0 and not output.strip():
        return f"⚠️  Cron: Command failed (exit {result.returncode})", False

    # Parse output looking for required jobs
    lines = output.splitlines()
    found_jobs = {}
    issues = []

    for line in lines:
        for job_name in CRON_JOBS_REQUIRED:
            if job_name in line:
                # Look for signs of "enabled" / "disabled" / "error"
                line_lower = line.lower()
                if "disabled" in line_lower:
                    found_jobs[job_name] = "disabled"
                    issues.append(f"{job_name} is DISABLED")
                elif "error" in line_lower or "fail" in line_lower:
                    found_jobs[job_name] = "error"
                    issues.append(f"{job_name} has errors")
                else:
                    found_jobs[job_name] = "ok"

    missing = [j for j in CRON_JOBS_REQUIRED if j not in found_jobs]
    for j in missing:
        issues.append(f"{j} NOT FOUND in cron list")

    active_count = sum(1 for v in found_jobs.values() if v == "ok")
    total_required = len(CRON_JOBS_REQUIRED)

    if issues:
        issues_str = "; ".join(issues)
        return f"⚠️  Cron: {active_count}/{total_required} jobs healthy — {issues_str}", False
    else:
        return f"✅ Cron: {active_count} jobs active, all healthy", True


# ── 6. Disk Space ─────────────────────────────────────────────────────

def check_disk_space() -> Tuple[str, bool]:
    """Check disk usage. Alert if > 90% full."""
    try:
        total, used, free = shutil.disk_usage("/")
        pct = (used / total) * 100
        used_gb  = used  / (1024 ** 3)
        total_gb = total / (1024 ** 3)

        if pct > 90:
            return (
                f"⚠️  Disk: {pct:.0f}% used ({used_gb:.1f}GB / {total_gb:.1f}GB) — NEARLY FULL!",
                False,
            )
        else:
            return f"✅ Disk: {pct:.0f}% used ({used_gb:.1f}GB / {total_gb:.1f}GB)", True
    except Exception as e:
        return f"⚠️  Disk: Failed to check — {e}", False


# ── Main ──────────────────────────────────────────────────────────────

def main():
    # Timezone-aware timestamp (Sydney / AEDT)
    try:
        from zoneinfo import ZoneInfo
        sydney_tz = ZoneInfo("Australia/Sydney")
        now_local = datetime.now(sydney_tz)
        tz_label  = now_local.strftime("%Z")
    except Exception:
        now_local = datetime.now(timezone.utc)
        tz_label  = "UTC"

    now_str = now_local.strftime(f"%Y-%m-%d %H:%M {tz_label}")
    print(f"GM Health Check — {now_str}")
    print("=" * 55)
    print()

    results   = []
    all_ok    = True

    # 1. Scanner health
    status, ok = check_scanner_health()
    results.append(status)
    if not ok:
        all_ok = False

    # 2 & 3. Bybit API + balance
    api_status, api_ok, equity, exchange = check_bybit_api()
    results.append(api_status)
    if not api_ok:
        all_ok = False
    else:
        bal_status, bal_ok = check_balance(equity)
        if not bal_ok:
            results.append(bal_status)
            all_ok = False

    # 4. Orphaned positions
    pos_status, pos_ok = check_orphaned_positions(exchange)
    results.append(pos_status)
    if not pos_ok:
        all_ok = False

    # 5. Cron health
    cron_status, cron_ok = check_cron_health()
    results.append(cron_status)
    if not cron_ok:
        all_ok = False

    # 6. Disk space
    disk_status, disk_ok = check_disk_space()
    results.append(disk_status)
    if not disk_ok:
        all_ok = False

    # Print all results
    for r in results:
        print(r)

    print()
    if all_ok:
        print("STATUS: ALL SYSTEMS OPERATIONAL")
    else:
        print("STATUS: ISSUES DETECTED — SEE ABOVE")


if __name__ == "__main__":
    main()

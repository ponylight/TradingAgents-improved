#!/usr/bin/env python3
"""General Manager — V20 Ops Management

Runs health checks, performance reporting, data maintenance, and monitoring.

Health checks (daily 9AM):
  1. Scanner log continuity
  2. Bybit API connectivity & balance
  3. Orphaned positions
  4. Cron job health
  5. Disk space

Ops features (called from workflow.py):
  6.  check_volatility()         — every run, ATR-based market regime
  7.  check_signal_drought()     — daily, alert if no signals in 7/14 days
  8.  run_data_freshness()       — daily, append new 4H candles to CSVs
  9.  run_git_commit()           — daily, auto-commit & push changes
  10. generate_weekly_report()   — Sunday 8PM Sydney
  11. run_log_cleanup()          — Sunday, delete old logs, prune history

Usage:
    python scripts/general_manager.py [--volatility] [--drought] [--data]
                                       [--git] [--weekly] [--cleanup]
    (no flags = full health check)
"""

import os
import re
import json
import shutil
import subprocess
import argparse
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional, Tuple, List

SYDNEY_TZ = ZoneInfo("Australia/Sydney")

import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent
LOGS_DIR     = PROJECT_DIR / "logs"
DATA_DIR     = PROJECT_DIR / "backtest" / "data"

HISTORY_FILE  = SCRIPT_DIR / ".trade_history.json"
STATE_FILE    = SCRIPT_DIR / ".workflow_state.json"

load_dotenv(PROJECT_DIR / ".env")

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "hr3xN6ozshJJhFbgxs")
BYBIT_SECRET  = os.getenv("BYBIT_SECRET",  "Q5LcRuP4XxcUiasxAZm76FKbdrCcoCFDRQy0")

SYMBOL = "BTC/USDT:USDT"

# ── Exchange ───────────────────────────────────────────────────────────────────

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


# ════════════════════════════════════════════════════════════════════
#  HEALTH CHECKS (existing functionality)
# ════════════════════════════════════════════════════════════════════

def check_scanner_health() -> Tuple[str, bool]:
    """Count scanner runs in the last 24h from log files."""
    now_syd = datetime.now(SYDNEY_TZ)
    cutoff  = now_syd - timedelta(hours=24)

    dates_to_check = [
        now_syd.strftime("%Y-%m-%d"),
        (now_syd - timedelta(days=1)).strftime("%Y-%m-%d"),
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

        for line in content.splitlines():
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

    run_times = sorted(set(run_times))
    # 4H candles = 6/day, but only warn if we've missed more than 2 consecutive
    # Calculate expected based on hours since first log today
    hours_covered = (now_syd - cutoff).total_seconds() / 3600
    expected = max(1, int(hours_covered / 4))
    actual   = len(run_times)

    if actual >= max(1, expected - 2):
        return f"✅ Scanner: {actual} runs in last 24h (expected ~{expected})", True
    else:
        return f"⚠️  Scanner: Only {actual} runs in last 24h (expected ~{expected}) — MISSING RUNS", False


def check_bybit_api() -> Tuple[str, bool, Optional[float], Optional[ccxt.Exchange]]:
    """Check API connectivity and return balance."""
    try:
        exchange = get_exchange()
        bal      = exchange.fetch_balance()
        usdt     = bal.get("USDT", {})
        equity   = float(usdt.get("total", usdt.get("free", 0)) or 0)
        return f"✅ Bybit API: Connected, ${equity:,.2f} USDT", True, equity, exchange
    except Exception as e:
        return f"⚠️  Bybit API: FAILED — {e}", False, None, None


def check_balance(equity: float) -> Tuple[str, bool]:
    if equity is None:
        return "⚠️  Balance: Unknown (API failed)", False
    if equity <= 0:
        return f"⚠️  Balance: ${equity:,.2f} USDT — ZERO OR NEGATIVE!", False
    return f"✅ Balance: ${equity:,.2f} USDT", True


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

    sl_orders = [
        o for o in open_orders
        if o.get("type") in ("stop_market", "stop", "stop_limit")
        and o.get("reduceOnly", False)
    ]

    orphaned = sum(
        1 for pos in open_pos
        if not [o for o in sl_orders if o.get("symbol") == pos.get("symbol", SYMBOL)]
    )

    if orphaned > 0:
        return f"⚠️  Positions: {len(open_pos)} open, {orphaned} WITHOUT stop-loss — ALERT", False
    return f"✅ Positions: {len(open_pos)} open, all have stop-loss orders", True


CRON_JOBS_REQUIRED = ["v20-workflow"]


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

    lines      = output.splitlines()
    found_jobs = {}
    issues     = []

    for line in lines:
        for job_name in CRON_JOBS_REQUIRED:
            if job_name in line:
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

    active_count   = sum(1 for v in found_jobs.values() if v == "ok")
    total_required = len(CRON_JOBS_REQUIRED)

    if issues:
        return f"⚠️  Cron: {active_count}/{total_required} jobs healthy — {'; '.join(issues)}", False
    return f"✅ Cron: {active_count} jobs active, all healthy", True


def check_disk_space() -> Tuple[str, bool]:
    """Check disk usage. Alert if > 90% full."""
    try:
        total, used, free = shutil.disk_usage("/")
        pct      = (used / total) * 100
        used_gb  = used  / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        if pct > 90:
            return f"⚠️  Disk: {pct:.0f}% used ({used_gb:.1f}GB / {total_gb:.1f}GB) — NEARLY FULL!", False
        return f"✅ Disk: {pct:.0f}% used ({used_gb:.1f}GB / {total_gb:.1f}GB)", True
    except Exception as e:
        return f"⚠️  Disk: Failed to check — {e}", False


# ════════════════════════════════════════════════════════════════════
#  6. VOLATILITY MONITOR
# ════════════════════════════════════════════════════════════════════

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state_key(key: str, value):
    """Update a single key in .workflow_state.json without clobbering."""
    state = _load_state()
    state[key] = value
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _compute_atr(highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray, period: int = 14) -> np.ndarray:
    """True Range → rolling mean ATR."""
    n  = len(highs)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
    atr = np.zeros(n)
    for i in range(period, n):
        atr[i] = np.mean(tr[i - period + 1: i + 1])
    return atr


def check_volatility() -> Tuple[str, Optional[float]]:
    """Fetch last 100 4H candles from Bybit public, compute 14-period ATR,
    compare to 100-candle average ATR.

    Returns (status_message, current_atr).
    """
    try:
        exch = ccxt.bybit({"enableRateLimit": True})
        exch.urls["api"] = {
            "public":  "https://api-demo.bybit.com",
            "private": "https://api-demo.bybit.com",
        }
        exch.has["fetchCurrencies"] = False

        raw  = exch.fetch_ohlcv("BTC/USDT", "4h", limit=100)
        data = np.array([[r[2], r[3], r[4]] for r in raw])   # high, low, close
        highs  = data[:, 0]
        lows   = data[:, 1]
        closes = data[:, 2]

        atr_series = _compute_atr(highs, lows, closes, period=14)

        # Valid ATR values (after warm-up period)
        valid_atr = atr_series[atr_series > 0]
        if len(valid_atr) < 2:
            return "⚠️  Volatility: Not enough ATR data", None

        current_atr = valid_atr[-1]
        avg_atr     = float(np.mean(valid_atr))

        ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

        # Store in state
        _save_state_key("last_atr",       round(float(current_atr), 2))
        _save_state_key("avg_atr_100",    round(avg_atr, 2))
        _save_state_key("atr_ratio",      round(ratio, 3))
        _save_state_key("atr_updated_at", datetime.now(SYDNEY_TZ).isoformat())

        if ratio < 0.5:
            msg = (
                f"⚠️  Low volatility — choppy conditions, V20 signals may underperform "
                f"(ATR ${current_atr:.0f} = {ratio:.0%} of avg ${avg_atr:.0f})"
            )
        elif ratio > 2.0:
            msg = (
                f"⚡ High volatility — increased signal quality expected "
                f"(ATR ${current_atr:.0f} = {ratio:.0%} of avg ${avg_atr:.0f})"
            )
        else:
            msg = (
                f"✅ Volatility: Normal "
                f"(ATR ${current_atr:.0f} = {ratio:.0%} of avg ${avg_atr:.0f})"
            )

        return msg, float(current_atr)

    except Exception as e:
        return f"⚠️  Volatility: Failed to compute ATR — {e}", None


# ════════════════════════════════════════════════════════════════════
#  7. LOG CLEANUP
# ════════════════════════════════════════════════════════════════════

def run_log_cleanup() -> str:
    """Delete log files older than 30 days. Prune trade history > 90 days.

    Returns a summary string.
    """
    lines = ["🧹 Log Cleanup"]

    # ── 1. Delete old log files ────────────────────────────────────
    cutoff_30 = datetime.now(SYDNEY_TZ) - timedelta(days=30)
    deleted_logs = []

    if LOGS_DIR.exists():
        for log_file in LOGS_DIR.glob("*.log"):
            try:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff_30:
                    log_file.unlink()
                    deleted_logs.append(log_file.name)
            except Exception as e:
                lines.append(f"  ⚠️  Could not delete {log_file.name}: {e}")

    if deleted_logs:
        lines.append(f"  Deleted {len(deleted_logs)} log files (>30 days old):")
        for name in deleted_logs[:5]:
            lines.append(f"    - {name}")
        if len(deleted_logs) > 5:
            lines.append(f"    ... and {len(deleted_logs) - 5} more")
    else:
        lines.append("  No old log files to delete")

    # ── 2. Prune trade history older than 90 days ──────────────────
    cutoff_90 = datetime.now(SYDNEY_TZ) - timedelta(days=90)

    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                history = json.load(f)

            original_count = len(history.get("trades", []))
            kept_trades = []
            pruned_count = 0

            for t in history.get("trades", []):
                try:
                    entry_dt = datetime.fromisoformat(t["entry_time"])
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                    if entry_dt >= cutoff_90 or t.get("status") == "open":
                        kept_trades.append(t)
                    else:
                        pruned_count += 1
                except Exception:
                    kept_trades.append(t)  # keep on parse error

            history["trades"] = kept_trades
            with open(HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=2, default=str)

            lines.append(
                f"  Trade history: {original_count} → {len(kept_trades)} entries "
                f"({pruned_count} pruned, summary kept)"
            )
        except Exception as e:
            lines.append(f"  ⚠️  Could not prune trade history: {e}")
    else:
        lines.append("  No trade history file yet")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
#  5. SIGNAL DROUGHT ALERT
# ════════════════════════════════════════════════════════════════════

def check_signal_drought() -> str:
    """Check days since last trade was recorded. Return alert if stale."""
    if not HISTORY_FILE.exists():
        return "ℹ️  Signal drought: No trade history file yet (system new)"

    try:
        with open(HISTORY_FILE) as f:
            history = json.load(f)
        trades = history.get("trades", [])
    except Exception as e:
        return f"⚠️  Signal drought: Could not read history — {e}"

    if not trades:
        return "ℹ️  Signal drought: No trades recorded yet"

    # Find most recent trade entry
    latest_dt = None
    for t in trades:
        try:
            dt = datetime.fromisoformat(t["entry_time"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if latest_dt is None or dt > latest_dt:
                latest_dt = dt
        except Exception:
            pass

    if latest_dt is None:
        return "ℹ️  Signal drought: Could not parse trade timestamps"

    days_since = (datetime.now(SYDNEY_TZ) - latest_dt).days

    if days_since >= 14:
        return (
            f"🚨 SIGNAL DROUGHT ALERT: {days_since} days since last signal "
            f"(last: {latest_dt.strftime('%Y-%m-%d')}). "
            f"Review strategy / market conditions."
        )
    elif days_since >= 7:
        return (
            f"⚠️  Signal drought warning: {days_since} days since last signal "
            f"(last: {latest_dt.strftime('%Y-%m-%d')}). "
            f"Normal bear-market behaviour, but worth monitoring."
        )
    else:
        return (
            f"✅ Signal drought: {days_since} days since last signal "
            f"(last: {latest_dt.strftime('%Y-%m-%d')})"
        )


# ════════════════════════════════════════════════════════════════════
#  3. DATA FRESHNESS
# ════════════════════════════════════════════════════════════════════

def _parse_csv_last_ts(csv_path: Path) -> Optional[datetime]:
    """Return the last timestamp in a OHLCV CSV (first column = timestamp)."""
    try:
        # Read last few lines efficiently
        with open(csv_path, "rb") as f:
            # Seek to near end
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode(errors="replace")

        last_line = None
        for line in tail.splitlines():
            if line.strip() and not line.startswith("timestamp"):
                last_line = line

        if not last_line:
            return None

        ts_str = last_line.split(",")[0].strip()
        # Handle both "2026-02-20 20:00:00" and ISO formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
            try:
                dt = datetime.strptime(ts_str, fmt)
                return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            except ValueError:
                pass
        return None
    except Exception:
        return None


def run_data_freshness() -> str:
    """Fetch latest 4H candles from Binance (public) and append new rows to CSVs."""
    lines = ["📡 Data Freshness Check"]

    # Fetch from Binance (public, no auth)
    try:
        binance = ccxt.binance({"enableRateLimit": True})
        raw     = binance.fetch_ohlcv("BTC/USDT", "4h", limit=100)
    except Exception as e:
        return f"⚠️  Data freshness: Binance fetch failed — {e}"

    # Convert to DataFrame
    new_df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
    new_df["timestamp"] = new_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Process each CSV
    csv_files = [
        DATA_DIR / "btc_usdt_4h.csv",
        DATA_DIR / "btc_usdt_4h_2017.csv",
    ]

    for csv_path in csv_files:
        if not csv_path.exists():
            lines.append(f"  ⚠️  {csv_path.name}: File not found, skipping")
            continue

        try:
            last_ts = _parse_csv_last_ts(csv_path)
            if last_ts is None:
                lines.append(f"  ⚠️  {csv_path.name}: Could not parse last timestamp")
                continue

            # Filter new candles strictly after last_ts
            # new_df timestamps are UTC strings; compare properly
            new_candles = []
            for _, row in new_df.iterrows():
                try:
                    row_ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    if row_ts > last_ts:
                        new_candles.append(row)
                except Exception:
                    pass

            if not new_candles:
                lines.append(f"  ✅ {csv_path.name}: Up to date (last: {last_ts.strftime('%Y-%m-%d %H:%M')})")
                continue

            # Append to CSV
            with open(csv_path, "a") as f:
                for row in new_candles:
                    f.write(
                        f"{row['timestamp']},{row['open']},{row['high']},"
                        f"{row['low']},{row['close']},{row['volume']}\n"
                    )

            lines.append(
                f"  ✅ {csv_path.name}: +{len(new_candles)} new candles "
                f"(up to {new_candles[-1]['timestamp']})"
            )

        except Exception as e:
            lines.append(f"  ⚠️  {csv_path.name}: Error — {e}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
#  4. GIT AUTO-COMMIT
# ════════════════════════════════════════════════════════════════════

def run_git_commit() -> str:
    """git add -A && git commit && git push mine main (only if changes)."""
    nvm_node_path = os.path.expanduser("~/.nvm/versions/node/v24.13.1/bin")
    env = os.environ.copy()
    env["PATH"] = f"{nvm_node_path}:{env.get('PATH', '')}"

    today = datetime.now(SYDNEY_TZ).strftime("%Y-%m-%d")

    try:
        # Check if there are changes to commit
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=15,
            cwd=str(PROJECT_DIR), env=env,
        )
        if not status_result.stdout.strip():
            return "✅ Git: No changes to commit"

        # Stage all
        subprocess.run(
            ["git", "add", "-A"],
            capture_output=True, text=True, timeout=15,
            cwd=str(PROJECT_DIR), env=env,
        )

        # Commit
        commit_msg = f"Daily auto-commit: logs, state, data [{today}]"
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True, text=True, timeout=15,
            cwd=str(PROJECT_DIR), env=env,
        )

        if commit_result.returncode != 0:
            return f"⚠️  Git commit failed: {commit_result.stderr.strip()}"

        # Push
        push_result = subprocess.run(
            ["git", "push", "mine", "main"],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_DIR), env=env,
        )

        if push_result.returncode == 0:
            return f"✅ Git: Committed & pushed [{today}]"
        else:
            return f"⚠️  Git: Committed but push failed — {push_result.stderr.strip()}"

    except subprocess.TimeoutExpired:
        return "⚠️  Git: Command timed out"
    except Exception as e:
        return f"⚠️  Git: Error — {e}"


# ════════════════════════════════════════════════════════════════════
#  2. WEEKLY PERFORMANCE REPORT
# ════════════════════════════════════════════════════════════════════

def generate_weekly_report() -> str:
    """Generate a formatted weekly performance report.

    Week = Monday 00:00 → Sunday 23:59 Sydney time.
    """
    now_syd = datetime.now(SYDNEY_TZ)

    # Week start = last Monday
    days_since_monday = now_syd.weekday()  # 0=Mon, 6=Sun
    week_start_syd    = (now_syd - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    week_end_syd = week_start_syd + timedelta(days=7)

    week_start_utc = week_start_syd.astimezone(timezone.utc)
    week_end_utc   = week_end_syd.astimezone(timezone.utc)

    week_label = (
        f"Week of {week_start_syd.strftime('%b %-d')}–"
        f"{(week_end_syd - timedelta(days=1)).strftime('%-d, %Y')}"
    )

    # Load trade history
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                history = json.load(f)
        except Exception:
            history = {"trades": [], "summary": {}}
    else:
        history = {"trades": [], "summary": {}}

    all_trades = history.get("trades", [])
    summary    = history.get("summary", {})

    # Filter trades in this week
    week_trades = []
    for t in all_trades:
        try:
            entry_dt = datetime.fromisoformat(t["entry_time"])
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            if week_start_utc <= entry_dt < week_end_utc:
                week_trades.append(t)
        except Exception:
            pass

    closed_week = [t for t in week_trades if t.get("status") in ("closed", "liquidated")]
    wins_week   = [t for t in closed_week if (t.get("pnl") or 0) > 0]
    losses_week = [t for t in closed_week if (t.get("pnl") or 0) <= 0]
    week_pnl    = sum(t.get("pnl") or 0 for t in closed_week)

    n_closed = len(closed_week)
    wr_str   = f"{len(wins_week) / n_closed * 100:.1f}%" if n_closed else "N/A"
    pnl_str  = f"+${week_pnl:,.0f}" if week_pnl >= 0 else f"-${abs(week_pnl):,.0f}"

    start_equity   = summary.get("start_equity",   169_751)
    current_equity = summary.get("current_equity", start_equity)
    pnl_pct_str    = f"{week_pnl / start_equity * 100:+.2f}%"

    equity_mult    = current_equity / start_equity if start_equity else 1.0
    equity_mult_str = f"{equity_mult:.2f}x"

    # Best / worst
    best_str  = "n/a"
    worst_str = "n/a"
    if wins_week:
        best_t    = max(wins_week, key=lambda t: t.get("pnl") or 0)
        conf_str  = "+".join((best_t.get("confluence") or [])[:2])
        best_str  = f"+${best_t['pnl']:,.0f} ({best_t['side'].upper()} {best_t['leverage']}x, {conf_str or 'n/a'})"
    if losses_week:
        worst_t   = min(losses_week, key=lambda t: t.get("pnl") or 0)
        conf_str  = "+".join((worst_t.get("confluence") or [])[:2])
        worst_str = f"-${abs(worst_t['pnl']):,.0f} ({worst_t['side'].upper()} {worst_t['leverage']}x, {conf_str or 'n/a'})"

    report_lines = [
        f"📊 V20 Weekly Report — {week_label}",
        f"Trades: {n_closed} ({len(wins_week)}W / {len(losses_week)}L) | Win Rate: {wr_str}",
        f"PnL: {pnl_str} ({pnl_pct_str})",
        f"Best: {best_str}",
        f"Worst: {worst_str}",
        f"Equity: ${current_equity:,.0f} ({equity_mult_str} initial)",
        f"Backtest expects: ~2 trades/week, 51% WR",
    ]

    return "\n".join(report_lines)


# ════════════════════════════════════════════════════════════════════
#  MAIN — Full Health Check
# ════════════════════════════════════════════════════════════════════

def main(args=None):
    now_local = datetime.now(SYDNEY_TZ)
    now_str   = now_local.strftime("%Y-%m-%d %H:%M %Z")

    # Parse CLI flags
    parser = argparse.ArgumentParser(description="V20 General Manager")
    parser.add_argument("--volatility", action="store_true", help="Run volatility monitor only")
    parser.add_argument("--drought",    action="store_true", help="Run signal drought check only")
    parser.add_argument("--data",       action="store_true", help="Run data freshness only")
    parser.add_argument("--git",        action="store_true", help="Run git auto-commit only")
    parser.add_argument("--weekly",     action="store_true", help="Generate weekly report only")
    parser.add_argument("--cleanup",    action="store_true", help="Run log cleanup only")

    if args is None:
        parsed = parser.parse_args()
    else:
        parsed = parser.parse_args(args)

    # Single-operation modes
    if parsed.volatility:
        msg, _ = check_volatility()
        print(msg)
        return

    if parsed.drought:
        print(check_signal_drought())
        return

    if parsed.data:
        print(run_data_freshness())
        return

    if parsed.git:
        print(run_git_commit())
        return

    if parsed.weekly:
        print(generate_weekly_report())
        return

    if parsed.cleanup:
        print(run_log_cleanup())
        return

    # ── Full health check ──────────────────────────────────────────
    print(f"GM Health Check — {now_str}")
    print("=" * 55)
    print()

    results = []
    all_ok  = True

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

    # 6b. Volatility monitor (informational)
    vol_status, _ = check_volatility()
    results.append(vol_status)

    for r in results:
        print(r)

    print()
    if all_ok:
        print("STATUS: ALL SYSTEMS OPERATIONAL")
    else:
        print("STATUS: ISSUES DETECTED — SEE ABOVE")


if __name__ == "__main__":
    main()

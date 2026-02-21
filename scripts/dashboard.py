#!/usr/bin/env python3
"""V20 Trading System Dashboard — Mission Control

A Flask web app serving a live dashboard at http://localhost:5555

Usage:
    source ~/trading-agents-env/bin/activate
    python scripts/dashboard.py
"""

import os
import sys
import json
import re
import subprocess
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

SYDNEY_TZ = ZoneInfo("Australia/Sydney")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
LOGS_DIR    = PROJECT_DIR / "logs"

WORKFLOW_STATE = SCRIPT_DIR / ".workflow_state.json"
RISK_STATE     = SCRIPT_DIR / ".risk_state.json"
TRADE_HISTORY  = SCRIPT_DIR / ".trade_history.json"

# Ensure scripts/ importable
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ── Bybit credentials ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")
except ImportError:
    pass

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "hr3xN6ozshJJhFbgxs")
BYBIT_SECRET  = os.getenv("BYBIT_SECRET",  "Q5LcRuP4XxcUiasxAZm76FKbdrCcoCFDRQy0")
SYMBOL        = "BTC/USDT:USDT"
INITIAL_CAPITAL = 169_751.0

# 4H candle close hours UTC → workflow runs at :05 past these in Sydney
CANDLE_CLOSE_HOURS_UTC = [0, 4, 8, 12, 16, 20]


def _next_4h_scan() -> str:
    """Calculate next 4H scan time in Sydney."""
    from zoneinfo import ZoneInfo
    now = datetime.now(timezone.utc)
    for h in sorted(CANDLE_CLOSE_HOURS_UTC * 2):  # double to wrap next day
        candidate = now.replace(hour=h % 24, minute=5, second=0, microsecond=0)
        if h >= 24:
            candidate += timedelta(days=1)
            candidate = candidate.replace(hour=h % 24)
        if candidate > now:
            syd = candidate.astimezone(SYDNEY_TZ)
            return syd.strftime("%H:%M %Z")
    return "—"


def _to_sydney(ts_str):
    """Convert a UTC timestamp string to Sydney time string for display."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        syd = dt.astimezone(SYDNEY_TZ)
        return syd.strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return ts_str

# ── Flask ──────────────────────────────────────────────────────────────────────
from flask import Flask, jsonify
app = Flask(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def get_exchange():
    try:
        import ccxt
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
    except Exception as e:
        return None


def load_json(path: Path, default=None):
    """Load a JSON file gracefully. Returns default on failure."""
    if default is None:
        default = {}
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return default


def get_live_data():
    """Fetch all live data from Bybit and JSON state files."""
    data = {
        "btc_price":    None,
        "btc_change":   None,
        "equity":       None,
        "positions":    [],
        "open_orders":  [],
        "balance":      None,
        "fetch_error":  None,
    }

    exchange = get_exchange()
    if exchange is None:
        data["fetch_error"] = "Cannot connect to Bybit"
        return data, None

    try:
        ticker          = exchange.fetch_ticker("BTC/USDT")
        data["btc_price"]  = ticker.get("last")
        data["btc_change"] = ticker.get("percentage")
    except Exception as e:
        data["fetch_error"] = f"Ticker: {e}"

    try:
        bal  = exchange.fetch_balance()
        usdt = bal.get("USDT", {})
        data["equity"]  = float(usdt.get("total", 0) or 0)
        data["balance"] = data["equity"]
    except Exception as e:
        data["fetch_error"] = (data.get("fetch_error") or "") + f" | Balance: {e}"

    try:
        positions = exchange.fetch_positions([SYMBOL])
        data["positions"] = [
            p for p in positions if abs(float(p.get("contracts", 0) or 0)) > 0
        ]
    except Exception as e:
        data["fetch_error"] = (data.get("fetch_error") or "") + f" | Positions: {e}"

    try:
        data["open_orders"] = exchange.fetch_open_orders(SYMBOL)
    except Exception:
        pass

    return data, exchange


def get_workflow_state():
    return load_json(WORKFLOW_STATE, {})


def get_risk_state():
    return load_json(RISK_STATE, {
        "peak_equity":      INITIAL_CAPITAL,
        "initial_capital":  INITIAL_CAPITAL,
        "day_start_equity": INITIAL_CAPITAL,
        "day_start_date":   "",
        "no_trade_until":   None,
        "milestones_hit":   [],
        "last_check":       None,
    })


def get_trade_history():
    return load_json(TRADE_HISTORY, {"trades": [], "summary": {}})


def get_cron_status():
    """Run openclaw cron list and return raw output."""
    nvm_path = os.path.expanduser("~/.nvm/versions/node/v24.13.1/bin")
    env = os.environ.copy()
    env["PATH"] = f"{nvm_path}:{env.get('PATH', '')}"
    try:
        result = subprocess.run(
            ["openclaw", "cron", "list"],
            capture_output=True, text=True, timeout=10, env=env
        )
        return (result.stdout + result.stderr).strip()
    except Exception as e:
        return f"Error: {e}"


def get_recent_logs(max_entries=100):
    """Parse log files from logs/ directory. Returns list of dicts."""
    entries = []

    if not LOGS_DIR.exists():
        return entries

    log_files = sorted(LOGS_DIR.glob("*.log"), reverse=True)[:5]  # last 5 files

    for log_file in log_files:
        # Determine agent from filename
        fname = log_file.stem
        if "scanner" in fname:
            agent = "Scanner"
        elif "workflow" in fname:
            agent = "Workflow"
        elif "risk" in fname:
            agent = "Risk Guard"
        elif "briefing" in fname or "brief" in fname:
            agent = "Briefer"
        elif "gm" in fname or "manager" in fname:
            agent = "GM"
        else:
            agent = fname.replace("_", " ").title()

        try:
            content = log_file.read_text(errors="replace")
            for line in content.splitlines():
                # Standard log format: 2026-02-21 10:00:00 INFO message
                m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s+(.+)$", line)
                if m:
                    ts_str, level, msg = m.groups()
                    entries.append({
                        "timestamp": ts_str,
                        "agent":     agent,
                        "level":     level,
                        "message":   msg.strip(),
                    })
                elif line.strip() and not line.startswith("="):
                    # Try to grab any meaningful line
                    entries.append({
                        "timestamp": "",
                        "agent":     agent,
                        "level":     "INFO",
                        "message":   line.strip()[:200],
                    })
        except Exception:
            pass

    # Sort by timestamp desc, filter out blanks
    ts_entries = [e for e in entries if e["timestamp"]]
    no_ts      = [e for e in entries if not e["timestamp"]]

    ts_entries.sort(key=lambda e: e["timestamp"], reverse=True)
    combined = ts_entries + no_ts
    return combined[:max_entries]


def parse_scanner_signals(logs):
    """Extract signal events from log entries."""
    signals = []
    for entry in logs:
        if entry["agent"] == "Scanner":
            msg = entry["message"]
            if "V20 SIGNAL FIRED" in msg or "signal" in msg.lower():
                signals.append(entry)
    return signals


def get_last_briefing():
    """Find last briefing from workflow state or logs."""
    ws = get_workflow_state()
    d = ws.get("last_briefing_date")
    if d:
        return f"{d} 08:00"
    return None


def compute_next_briefing():
    """Next 8AM Sydney."""
    now_syd = datetime.now(SYDNEY_TZ)
    next_8  = now_syd.replace(hour=8, minute=0, second=0, microsecond=0)
    if now_syd.hour >= 8:
        next_8 += timedelta(days=1)
    return next_8.strftime("%Y-%m-%d 08:00 %Z")


# ═══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/data")
def api_data():
    """Return all dashboard data as JSON."""
    live, exchange = get_live_data()
    ws   = get_workflow_state()
    rs   = get_risk_state()
    th   = get_trade_history()
    logs = get_recent_logs(100)

    # ── Equity / PnL ──────────────────────────────────────────────
    equity   = live.get("equity") or INITIAL_CAPITAL
    pnl_abs  = equity - INITIAL_CAPITAL
    pnl_pct  = (pnl_abs / INITIAL_CAPITAL) * 100

    # ── System status ──────────────────────────────────────────────
    issues   = []
    warnings = []

    if live.get("fetch_error"):
        issues.append(f"Bybit API: {live['fetch_error']}")

    # Circuit breaker
    peak = rs.get("peak_equity", INITIAL_CAPITAL)
    if equity < peak * 0.85:
        issues.append("CIRCUIT BREAKER TRIGGERED")
    elif equity < peak * 0.92:
        warnings.append("Equity 8%+ below peak")

    # Daily loss
    if rs.get("no_trade_until"):
        warnings.append("Daily loss limit — trades blocked")

    # ATR
    atr_ratio = ws.get("atr_ratio", 1.0)
    if atr_ratio and atr_ratio < 0.5:
        warnings.append("Low volatility regime")

    if issues:
        system_status = "ISSUES"
    elif warnings:
        system_status = "WARNINGS"
    else:
        system_status = "ALL CLEAR"

    # ── Scanner agent ──────────────────────────────────────────────
    now_syd = datetime.now(SYDNEY_TZ)
    today_str = now_syd.strftime("%Y-%m-%d")

    scanner_logs = [e for e in logs if e["agent"] == "Scanner"]
    signal_logs  = [e for e in scanner_logs if "SIGNAL FIRED" in e.get("message", "") or "V20 SIGNAL" in e.get("message", "")]

    last_scan_candle = ws.get("last_scan_candle")
    last_scan_str    = None
    if last_scan_candle:
        try:
            dt = datetime.strptime(last_scan_candle, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            dt_syd = dt.astimezone(SYDNEY_TZ)
            last_scan_str = dt_syd.strftime("%Y-%m-%d %H:00 %Z")
        except Exception:
            last_scan_str = last_scan_candle

    # Check if scanner ran in last 5h (4H + buffer)
    scanner_active = False
    if last_scan_candle:
        try:
            dt = datetime.strptime(last_scan_candle, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - dt).total_seconds() < 5 * 3600:
                scanner_active = True
        except Exception:
            pass

    last_signal = None
    if signal_logs:
        last_signal = signal_logs[0]["message"][:100]

    # Count signals today / this week
    signals_today = 0
    signals_week  = 0
    week_start    = now_syd - timedelta(days=now_syd.weekday())
    week_start    = week_start.replace(hour=0, minute=0, second=0)

    for e in signal_logs:
        ts = e.get("timestamp", "")
        if ts.startswith(today_str):
            signals_today += 1
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            syd_dt = dt.astimezone(SYDNEY_TZ)
            if syd_dt >= week_start:
                signals_week += 1
        except Exception:
            pass

    # ── Executor (positions + orders) ─────────────────────────────
    btc_price = live.get("btc_price") or 0
    positions_fmt = []
    for pos in live.get("positions", []):
        side      = pos.get("side", "?")
        contracts = float(pos.get("contracts", 0) or 0)
        entry     = float(pos.get("entryPrice", 0) or 0)
        upnl      = float(pos.get("unrealizedPnl", 0) or 0)
        leverage  = float(pos.get("leverage", 1) or 1)
        liq       = float(pos.get("liquidationPrice", 0) or 0)
        notional  = contracts * (btc_price or entry)

        if btc_price and entry:
            pnl_pct_pos = (btc_price - entry) / entry * 100 if side == "long" else (entry - btc_price) / entry * 100
        else:
            pnl_pct_pos = 0

        positions_fmt.append({
            "side":      side.upper(),
            "contracts": contracts,
            "entry":     entry,
            "upnl":      upnl,
            "upnl_pct":  round(pnl_pct_pos, 2),
            "leverage":  leverage,
            "liq_price": liq,
            "notional":  notional,
        })

    orders_fmt = []
    for o in live.get("open_orders", []):
        orders_fmt.append({
            "id":        o.get("id", "?")[:8],
            "type":      o.get("type", "?"),
            "side":      o.get("side", "?"),
            "price":     o.get("price"),
            "amount":    o.get("amount"),
            "reduce":    o.get("reduceOnly", False),
        })

    # ── Risk Guard ─────────────────────────────────────────────────
    day_start  = rs.get("day_start_equity", INITIAL_CAPITAL)
    daily_loss_pct = ((equity - day_start) / day_start * 100) if day_start else 0
    milestones_hit = rs.get("milestones_hit", [])
    milestones_all = ["100x", "500x", "1000x"]
    circuit_ok     = equity >= peak * 0.85
    daily_ok       = rs.get("no_trade_until") is None

    # ── Briefer ───────────────────────────────────────────────────
    last_briefing = get_last_briefing() or ws.get("last_briefing_date")
    next_briefing = compute_next_briefing()

    # ── General Manager ───────────────────────────────────────────
    last_git_date   = ws.get("last_git_date")
    last_data_date  = ws.get("last_data_date")
    atr_current     = ws.get("last_atr")
    avg_atr         = ws.get("avg_atr_100")
    atr_updated_at  = ws.get("atr_updated_at")

    drought_days = None
    all_trades   = th.get("trades", [])
    if all_trades:
        latest_dt = None
        for t in all_trades:
            try:
                dt = datetime.fromisoformat(t["entry_time"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if latest_dt is None or dt > latest_dt:
                    latest_dt = dt
            except Exception:
                pass
        if latest_dt:
            drought_days = (datetime.now(SYDNEY_TZ) - latest_dt).days

    # Cron status
    cron_raw = get_cron_status()
    cron_ok  = "v20-workflow" in cron_raw and "ok" in cron_raw.lower()

    # GM health checks summary
    gm_checks = {
        "Cron jobs":     cron_ok,
        "Bybit API":     live.get("fetch_error") is None,
        "Scanner log":   len(scanner_logs) > 0,
        "Data file":     (PROJECT_DIR / "backtest" / "data" / "btc_usdt_4h.csv").exists(),
    }

    # Data freshness: last candle from workflow state (candle is UTC, display in Sydney)
    last_candle_ts = None
    if last_scan_candle:
        try:
            dt = datetime.strptime(last_scan_candle, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            dt_syd = dt.astimezone(SYDNEY_TZ)
            last_candle_ts = dt_syd.strftime("%Y-%m-%d %H:00 %Z")
        except Exception:
            last_candle_ts = last_scan_candle

    # Git: check last commit
    last_git_commit = None
    try:
        git_result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            capture_output=True, text=True, timeout=5,
            cwd=str(PROJECT_DIR)
        )
        if git_result.returncode == 0:
            last_git_commit = git_result.stdout.strip()[:80]
    except Exception:
        pass

    # ── Performance ───────────────────────────────────────────────
    summary = th.get("summary", {})
    perf = {
        "total_trades": summary.get("total_trades", 0),
        "wins":         summary.get("wins", 0),
        "losses":       summary.get("losses", 0),
        "liquidations": summary.get("liquidations", 0),
        "win_rate":     summary.get("win_rate", 0.0),
        "total_pnl":    summary.get("total_pnl", 0.0),
        "avg_win":      summary.get("avg_win", 0.0),
        "avg_loss":     summary.get("avg_loss", 0.0),
        "best_trade":   summary.get("best_trade", 0.0),
        "worst_trade":  summary.get("worst_trade", 0.0),
        "start_equity":   summary.get("start_equity", INITIAL_CAPITAL),
        "current_equity": summary.get("current_equity", equity),
        "pnl_pct": round((summary.get("current_equity", equity) - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
    }

    # Equity history for chart
    equity_chart = []
    for t in sorted(all_trades, key=lambda x: x.get("entry_time", "")):
        if t.get("exit_time") and t.get("pnl") is not None:
            equity_chart.append({
                "time": t["exit_time"][:10],
                "pnl":  t.get("pnl", 0),
            })

    # ── Activity log ──────────────────────────────────────────────
    activity = []
    for e in logs[:100]:
        activity.append({
            "timestamp": e.get("timestamp", ""),
            "agent":     e.get("agent", "?"),
            "level":     e.get("level", "INFO"),
            "message":   e.get("message", "")[:200],
        })

    return jsonify({
        "generated_at": datetime.now(SYDNEY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "system_status": system_status,
        "issues":   issues,
        "warnings": warnings,

        # Header
        "btc_price":   live.get("btc_price"),
        "btc_change":  live.get("btc_change"),
        "equity":      equity,
        "pnl_abs":     round(pnl_abs, 2),
        "pnl_pct":     round(pnl_pct, 2),

        # Scanner
        "scanner": {
            "status":        "Active" if scanner_active else "Idle",
            "last_run":      last_scan_str,
            "next_run":      _next_4h_scan(),
            "last_signal":   last_signal,
            "signals_today": signals_today,
            "signals_week":  signals_week,
        },

        # Executor
        "executor": {
            "positions":  positions_fmt,
            "orders":     orders_fmt,
            "last_trade": _to_sydney(all_trades[-1]["entry_time"]) if all_trades else None,
        },

        # Risk Guard
        "risk": {
            "equity":          round(equity, 2),
            "peak_equity":     round(peak, 2),
            "day_start":       round(day_start, 2),
            "daily_loss_pct":  round(daily_loss_pct, 2),
            "circuit_ok":      circuit_ok,
            "daily_ok":        daily_ok,
            "no_trade_until":  _to_sydney(rs.get("no_trade_until")),
            "milestones_hit":  milestones_hit,
            "milestones_all":  milestones_all,
            "initial_capital": INITIAL_CAPITAL,
            "equity_multiple": round(equity / INITIAL_CAPITAL, 4),
        },

        # Briefer
        "briefer": {
            "last_briefing": last_briefing,
            "next_briefing": next_briefing,
        },

        # GM
        "gm": {
            "health_checks":   gm_checks,
            "atr_current":     atr_current,
            "avg_atr_100":     avg_atr,
            "atr_ratio":       atr_ratio,
            "atr_updated_at":  _to_sydney(atr_updated_at),
            "drought_days":    drought_days,
            "last_git_commit": last_git_commit,
            "last_git_date":   _to_sydney(last_git_date),
            "last_data_date":  _to_sydney(last_data_date),
            "last_candle_ts":  last_candle_ts,
            "cron_raw":        cron_raw,
        },

        # Performance
        "performance": perf,
        "equity_chart": equity_chart,

        # Activity
        "activity": activity,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V20 Trading System — Mission Control</title>
<style>
  :root {
    --bg:       #1a1a2e;
    --card:     #16213e;
    --card2:    #0f3460;
    --green:    #00ff88;
    --red:      #ff4444;
    --yellow:   #ffaa00;
    --blue:     #4fc3f7;
    --text:     #e0e0e0;
    --text-dim: #888;
    --header:   #ffffff;
    --border:   #2a3a5a;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 14px;
    min-height: 100vh;
  }

  /* ─── Header ─── */
  #header {
    background: linear-gradient(135deg, #0a0a1a 0%, #16213e 100%);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
  }
  #header h1 {
    color: var(--header);
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  #header h1 span { color: var(--green); }

  .header-stats {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    align-items: center;
  }
  .stat-block { text-align: center; }
  .stat-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-value { font-size: 18px; font-weight: 700; color: var(--header); }
  .stat-sub   { font-size: 12px; }

  .status-badge {
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.5px;
  }
  .badge-clear    { background: rgba(0,255,136,0.15); color: var(--green); border: 1px solid var(--green); }
  .badge-warnings { background: rgba(255,170,0,0.15); color: var(--yellow); border: 1px solid var(--yellow); }
  .badge-issues   { background: rgba(255,68,68,0.15);  color: var(--red);    border: 1px solid var(--red); }

  /* ─── Main Layout ─── */
  .container { padding: 20px 24px; max-width: 1800px; margin: 0 auto; }

  /* ─── Agent Cards ─── */
  .agents-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }
  @media (max-width: 1200px) { .agents-grid { grid-template-columns: repeat(3, 1fr); } }
  @media (max-width: 800px)  { .agents-grid { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 500px)  { .agents-grid { grid-template-columns: 1fr; } }

  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    transition: border-color 0.2s;
  }
  .card:hover { border-color: #3a5a8a; }

  .card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }
  .card-icon { font-size: 20px; }
  .card-title { font-size: 15px; font-weight: 700; color: var(--header); }

  .card-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
    gap: 8px;
  }
  .card-key   { color: var(--text-dim); font-size: 12px; white-space: nowrap; }
  .card-val   { color: var(--text); font-size: 12px; text-align: right; font-weight: 500; word-break: break-word; max-width: 60%; }

  .ok    { color: var(--green) !important; }
  .warn  { color: var(--yellow) !important; }
  .bad   { color: var(--red) !important; }
  .dim   { color: var(--text-dim) !important; }
  .blue  { color: var(--blue) !important; }

  .pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 700;
  }
  .pill-green  { background: rgba(0,255,136,0.15); color: var(--green); }
  .pill-red    { background: rgba(255,68,68,0.15);  color: var(--red); }
  .pill-yellow { background: rgba(255,170,0,0.15);  color: var(--yellow); }
  .pill-blue   { background: rgba(79,195,247,0.15); color: var(--blue); }
  .pill-dim    { background: rgba(136,136,136,0.1); color: var(--text-dim); }

  /* Position sub-card */
  .pos-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
  }
  .pos-header { font-weight: 700; font-size: 13px; margin-bottom: 4px; }

  /* Milestone dots */
  .milestone-dots { display: flex; gap: 6px; }
  .m-dot {
    width: 12px; height: 12px; border-radius: 50%;
    background: var(--border);
  }
  .m-dot.hit { background: var(--green); }

  /* ─── Performance Section ─── */
  .section-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--header);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 8px;
  }

  .perf-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }
  .perf-stat {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
  }
  .perf-stat-val   { font-size: 22px; font-weight: 700; color: var(--header); }
  .perf-stat-label { font-size: 11px; color: var(--text-dim); margin-top: 4px; text-transform: uppercase; }

  /* Chart */
  #equityChart {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 24px;
  }
  #equityChart canvas { width: 100% !important; height: 200px !important; }

  /* ─── Activity Log ─── */
  .activity-table-wrapper {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 24px;
  }
  .activity-table-scroll { max-height: 360px; overflow-y: auto; }
  .activity-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  .activity-table th {
    background: var(--card2);
    color: var(--text-dim);
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
    padding: 10px 12px;
    text-align: left;
    position: sticky;
    top: 0;
    z-index: 1;
  }
  .activity-table td {
    padding: 7px 12px;
    border-bottom: 1px solid rgba(42,58,90,0.5);
    vertical-align: top;
  }
  .activity-table tr:last-child td { border-bottom: none; }
  .activity-table tr:hover td { background: rgba(255,255,255,0.02); }
  .agent-badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 8px;
    font-size: 10px;
    font-weight: 700;
    white-space: nowrap;
  }
  .ab-scanner  { background: rgba(0,255,136,0.12); color: #00ff88; }
  .ab-workflow { background: rgba(79,195,247,0.12); color: #4fc3f7; }
  .ab-risk     { background: rgba(255,68,68,0.12);  color: #ff4444; }
  .ab-briefer  { background: rgba(255,170,0,0.12);  color: #ffaa00; }
  .ab-gm       { background: rgba(200,150,255,0.12); color: #c89bff; }
  .ab-default  { background: rgba(136,136,136,0.12); color: #888; }

  .log-error   { color: var(--red); }
  .log-warning { color: var(--yellow); }
  .log-info    { color: var(--text); }

  /* ─── Footer ─── */
  #footer {
    text-align: center;
    color: var(--text-dim);
    font-size: 11px;
    padding: 16px;
    border-top: 1px solid var(--border);
  }
  #refresh-timer { color: var(--green); font-weight: 700; }

  /* ─── Loading overlay ─── */
  #loading {
    position: fixed; inset: 0;
    background: rgba(26,26,46,0.85);
    display: flex; align-items: center; justify-content: center;
    z-index: 999;
    font-size: 18px;
    color: var(--green);
  }
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid var(--border);
    border-top-color: var(--green);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 14px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<div id="loading">
  <div class="spinner"></div> Loading data…
</div>

<!-- Header -->
<div id="header">
  <h1>V20 Trading System — <span>Mission Control</span></h1>
  <div class="header-stats">
    <div class="stat-block">
      <div class="stat-label">BTC Price</div>
      <div class="stat-value" id="btc-price">—</div>
      <div class="stat-sub" id="btc-change">—</div>
    </div>
    <div class="stat-block">
      <div class="stat-label">Account Equity</div>
      <div class="stat-value" id="equity">—</div>
      <div class="stat-sub" id="pnl">—</div>
    </div>
    <div id="status-badge" class="status-badge badge-clear">🟢 ALL CLEAR</div>
  </div>
</div>

<!-- Main -->
<div class="container">

  <!-- Agent Cards -->
  <div style="margin-bottom:12px" class="section-title">🤖 Agent Status</div>
  <div class="agents-grid">

    <!-- Scanner -->
    <div class="card">
      <div class="card-header">
        <span class="card-icon">🔍</span>
        <span class="card-title">Scanner</span>
        <span id="scanner-status-pill" class="pill pill-dim" style="margin-left:auto">Idle</span>
      </div>
      <div class="card-row">
        <span class="card-key">Last run</span>
        <span class="card-val" id="scanner-last-run">—</span>
      </div>
      <div class="card-row">
        <span class="card-key">Next scan</span>
        <span class="card-val" id="scanner-next-run" style="color:var(--blue)">—</span>
      </div>
      <div class="card-row">
        <span class="card-key">Last signal</span>
        <span class="card-val" id="scanner-last-signal">No signal</span>
      </div>
      <div class="card-row">
        <span class="card-key">Today / Week</span>
        <span class="card-val" id="scanner-counts">0 / 0</span>
      </div>
    </div>

    <!-- Executor -->
    <div class="card">
      <div class="card-header">
        <span class="card-icon">💰</span>
        <span class="card-title">Executor</span>
        <span id="executor-badge" class="pill pill-dim" style="margin-left:auto">—</span>
      </div>
      <div id="executor-positions">
        <div class="dim" style="font-size:12px;text-align:center;padding:8px 0">No open positions</div>
      </div>
      <div style="margin-top:8px">
        <div class="card-row">
          <span class="card-key">Open orders</span>
          <span class="card-val" id="executor-orders">0</span>
        </div>
        <div class="card-row">
          <span class="card-key">Last trade</span>
          <span class="card-val" id="executor-last-trade">—</span>
        </div>
      </div>
    </div>

    <!-- Risk Guard -->
    <div class="card">
      <div class="card-header">
        <span class="card-icon">🛡️</span>
        <span class="card-title">Risk Guard</span>
      </div>
      <div class="card-row">
        <span class="card-key">Circuit breaker</span>
        <span class="card-val" id="risk-circuit">—</span>
      </div>
      <div class="card-row">
        <span class="card-key">Daily loss</span>
        <span class="card-val" id="risk-daily">—</span>
      </div>
      <div class="card-row">
        <span class="card-key">Trade block</span>
        <span class="card-val" id="risk-block">None</span>
      </div>
      <div class="card-row">
        <span class="card-key">Milestones</span>
        <span class="card-val">
          <div class="milestone-dots" id="milestones"></div>
        </span>
      </div>
      <div class="card-row">
        <span class="card-key">Equity multiple</span>
        <span class="card-val" id="risk-multiple">—</span>
      </div>
    </div>

    <!-- Briefer -->
    <div class="card">
      <div class="card-header">
        <span class="card-icon">☀️</span>
        <span class="card-title">Briefer</span>
      </div>
      <div class="card-row">
        <span class="card-key">Last briefing</span>
        <span class="card-val" id="briefer-last">—</span>
      </div>
      <div class="card-row">
        <span class="card-key">Next briefing</span>
        <span class="card-val" id="briefer-next">—</span>
      </div>
      <div style="margin-top: 12px; font-size:11px; color: var(--text-dim)">
        Daily at 8:00 AM Sydney (AEDT)
      </div>
    </div>

    <!-- General Manager -->
    <div class="card">
      <div class="card-header">
        <span class="card-icon">👔</span>
        <span class="card-title">General Manager</span>
      </div>
      <div id="gm-checks"></div>
      <div class="card-row" style="margin-top:6px">
        <span class="card-key">Volatility (ATR)</span>
        <span class="card-val" id="gm-atr">—</span>
      </div>
      <div class="card-row">
        <span class="card-key">Signal drought</span>
        <span class="card-val" id="gm-drought">—</span>
      </div>
      <div class="card-row">
        <span class="card-key">Last candle</span>
        <span class="card-val" id="gm-candle">—</span>
      </div>
      <div class="card-row">
        <span class="card-key">Git commit</span>
        <span class="card-val" id="gm-git" style="font-size:10px;max-width:70%">—</span>
      </div>
    </div>

  </div><!-- /agents-grid -->

  <!-- Issues/Warnings banner -->
  <div id="alerts-banner" style="display:none;margin-bottom:18px"></div>

  <!-- Performance -->
  <div class="section-title">📊 Performance</div>
  <div class="perf-grid">
    <div class="perf-stat">
      <div class="perf-stat-val" id="p-trades">0</div>
      <div class="perf-stat-label">Total Trades</div>
    </div>
    <div class="perf-stat">
      <div class="perf-stat-val ok" id="p-wins">0</div>
      <div class="perf-stat-label">Wins</div>
    </div>
    <div class="perf-stat">
      <div class="perf-stat-val bad" id="p-losses">0</div>
      <div class="perf-stat-label">Losses</div>
    </div>
    <div class="perf-stat">
      <div class="perf-stat-val" id="p-wr">0%</div>
      <div class="perf-stat-label">Win Rate</div>
    </div>
    <div class="perf-stat">
      <div class="perf-stat-val" id="p-pnl">$0</div>
      <div class="perf-stat-label">Total PnL</div>
    </div>
    <div class="perf-stat">
      <div class="perf-stat-val" id="p-pnl-pct">0%</div>
      <div class="perf-stat-label">PnL %</div>
    </div>
    <div class="perf-stat">
      <div class="perf-stat-val ok" id="p-best">$0</div>
      <div class="perf-stat-label">Best Trade</div>
    </div>
    <div class="perf-stat">
      <div class="perf-stat-val bad" id="p-worst">$0</div>
      <div class="perf-stat-label">Worst Trade</div>
    </div>
    <div class="perf-stat">
      <div class="perf-stat-val" id="p-equity">$0</div>
      <div class="perf-stat-label">Current Equity</div>
    </div>
  </div>

  <!-- Equity Chart -->
  <div id="equityChart">
    <div class="section-title" style="font-size:14px;margin-bottom:10px">📈 Cumulative PnL</div>
    <canvas id="chartCanvas"></canvas>
    <div id="chart-empty" style="text-align:center;color:var(--text-dim);padding:20px;font-size:13px">
      No trade history yet — chart will appear after first closed trade
    </div>
  </div>

  <!-- Activity Log -->
  <div class="section-title">📋 Activity Log</div>
  <div class="activity-table-wrapper">
    <div class="activity-table-scroll">
      <table class="activity-table">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Agent</th>
            <th>Level</th>
            <th>Message</th>
          </tr>
        </thead>
        <tbody id="activity-log">
          <tr><td colspan="4" style="text-align:center;color:var(--text-dim);padding:20px">Loading…</td></tr>
        </tbody>
      </table>
    </div>
  </div>

</div><!-- /container -->

<!-- Footer -->
<div id="footer">
  V20 Trading System Dashboard &nbsp;|&nbsp;
  Auto-refresh in <span id="refresh-timer">60</span>s &nbsp;|&nbsp;
  <span id="generated-at">—</span>
</div>

<script>
// ─── Utilities ───────────────────────────────────────────────────────────────
const fmt$ = (v, d=0) => v != null ? '$' + Number(v).toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d}) : '—';
const fmtPct = (v, d=2) => v != null ? (v >= 0 ? '+' : '') + Number(v).toFixed(d) + '%' : '—';
const cls = (v, zero='') => v > 0 ? 'ok' : v < 0 ? 'bad' : zero;
const el = id => document.getElementById(id);

function agentClass(agent) {
  if (!agent) return 'ab-default';
  const a = agent.toLowerCase();
  if (a.includes('scanner'))  return 'ab-scanner';
  if (a.includes('workflow')) return 'ab-workflow';
  if (a.includes('risk'))     return 'ab-risk';
  if (a.includes('brief'))    return 'ab-briefer';
  if (a.includes('gm') || a.includes('manager')) return 'ab-gm';
  return 'ab-default';
}

function shortTs(ts) {
  if (!ts) return '—';
  // Parse and convert to Sydney time for display
  try {
    let d = new Date(ts.includes('+') || ts.includes('Z') ? ts : ts + 'Z');
    if (isNaN(d)) return ts.replace('T', ' ').substring(0, 16);
    return d.toLocaleString('en-AU', {timeZone: 'Australia/Sydney', month:'short', day:'numeric', hour:'2-digit', minute:'2-digit', hour12:false});
  } catch(e) {
    return ts.replace('T', ' ').substring(0, 16);
  }
}

// ─── Chart ───────────────────────────────────────────────────────────────────
let chartDrawn = false;
function drawChart(equityData) {
  if (!equityData || equityData.length === 0) {
    el('chart-empty').style.display = 'block';
    el('chartCanvas').style.display = 'none';
    return;
  }
  el('chart-empty').style.display = 'none';
  el('chartCanvas').style.display = 'block';
  chartDrawn = true;

  // Compute cumulative PnL
  let cum = 0;
  const labels = [];
  const values = [];
  equityData.forEach(d => {
    cum += (d.pnl || 0);
    labels.push(d.time);
    values.push(Math.round(cum));
  });

  const canvas = el('chartCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth;
  const H = 200;
  canvas.width = W;
  canvas.height = H;

  const pad = { t: 20, r: 20, b: 30, l: 60 };
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;

  const minV = Math.min(0, ...values);
  const maxV = Math.max(0, ...values);
  const range = maxV - minV || 1;

  function xp(i) { return pad.l + (i / (values.length - 1 || 1)) * cw; }
  function yp(v) { return pad.t + ch - ((v - minV) / range) * ch; }

  ctx.clearRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = 'rgba(42,58,90,0.5)';
  ctx.lineWidth = 1;
  [0, 0.25, 0.5, 0.75, 1].forEach(f => {
    const y = pad.t + f * ch;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + cw, y); ctx.stroke();
  });

  // Zero line
  const zeroY = yp(0);
  ctx.strokeStyle = 'rgba(136,136,136,0.4)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(pad.l, zeroY); ctx.lineTo(pad.l + cw, zeroY); ctx.stroke();
  ctx.setLineDash([]);

  // Fill
  const lastVal = values[values.length - 1];
  const color = lastVal >= 0 ? '#00ff88' : '#ff4444';

  if (values.length > 1) {
    ctx.beginPath();
    ctx.moveTo(xp(0), yp(values[0]));
    values.forEach((v, i) => { if (i > 0) ctx.lineTo(xp(i), yp(v)); });
    ctx.lineTo(xp(values.length - 1), yp(0));
    ctx.lineTo(xp(0), yp(0));
    ctx.closePath();
    ctx.fillStyle = lastVal >= 0 ? 'rgba(0,255,136,0.08)' : 'rgba(255,68,68,0.08)';
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(xp(0), yp(values[0]));
    values.forEach((v, i) => { if (i > 0) ctx.lineTo(xp(i), yp(v)); });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Y axis labels
  ctx.fillStyle = '#888';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  [minV, 0, maxV].forEach(v => {
    const y = yp(v);
    ctx.fillText('$' + Math.round(v).toLocaleString(), pad.l - 4, y + 3);
  });

  // X axis labels (first, mid, last)
  ctx.textAlign = 'center';
  [[0, labels[0]], [Math.floor((labels.length-1)/2), labels[Math.floor((labels.length-1)/2)]], [labels.length-1, labels[labels.length-1]]].forEach(([i, l]) => {
    if (l) ctx.fillText(l, xp(i), H - 6);
  });
}

// ─── Main render ─────────────────────────────────────────────────────────────
function render(d) {
  // Header
  if (d.btc_price != null) {
    el('btc-price').textContent = fmt$(d.btc_price, 0);
    const chg = d.btc_change;
    el('btc-change').textContent = chg != null ? fmtPct(chg, 2) + ' 24h' : '';
    el('btc-change').className = 'stat-sub ' + (chg >= 0 ? 'ok' : 'bad');
  }
  el('equity').textContent = fmt$(d.equity, 2);
  const pnlEl = el('pnl');
  pnlEl.textContent = (d.pnl_abs >= 0 ? '+' : '') + fmt$(d.pnl_abs, 2) + ' (' + fmtPct(d.pnl_pct, 2) + ')';
  pnlEl.className = 'stat-sub ' + (d.pnl_abs >= 0 ? 'ok' : 'bad');

  // Status badge
  const badge = el('status-badge');
  if (d.system_status === 'ALL CLEAR') {
    badge.textContent = '🟢 ALL CLEAR';
    badge.className = 'status-badge badge-clear';
  } else if (d.system_status === 'WARNINGS') {
    badge.textContent = '🟡 WARNINGS';
    badge.className = 'status-badge badge-warnings';
  } else {
    badge.textContent = '🔴 ISSUES';
    badge.className = 'status-badge badge-issues';
  }

  // Alerts banner
  const alerts = [...(d.issues || []), ...(d.warnings || [])];
  const banner = el('alerts-banner');
  if (alerts.length > 0) {
    banner.style.display = 'block';
    banner.innerHTML = alerts.map(a => `<div style="background:rgba(255,68,68,0.08);border:1px solid rgba(255,68,68,0.3);border-radius:8px;padding:8px 14px;margin-bottom:6px;color:var(--red);font-size:13px">⚠️ ${a}</div>`).join('');
  } else {
    banner.style.display = 'none';
  }

  // ── Scanner ──
  const sc = d.scanner || {};
  const scanPill = el('scanner-status-pill');
  if (sc.status === 'Active') {
    scanPill.textContent = 'Active';
    scanPill.className = 'pill pill-green';
  } else {
    scanPill.textContent = 'Idle';
    scanPill.className = 'pill pill-dim';
  }
  el('scanner-last-run').textContent = sc.last_run || '—';
  el('scanner-next-run').textContent = sc.next_run || '—';
  el('scanner-last-signal').textContent = sc.last_signal || 'No signal';
  el('scanner-last-signal').className = 'card-val ' + (sc.last_signal ? 'ok' : 'dim');
  el('scanner-counts').textContent = (sc.signals_today || 0) + ' / ' + (sc.signals_week || 0);

  // ── Executor ──
  const ex = d.executor || {};
  const positions = ex.positions || [];
  const posEl = el('executor-positions');
  const exBadge = el('executor-badge');
  if (positions.length > 0) {
    exBadge.textContent = positions.length + ' Position' + (positions.length > 1 ? 's' : '');
    exBadge.className = 'pill pill-green';
    posEl.innerHTML = positions.map(p => `
      <div class="pos-card">
        <div class="pos-header ${p.upnl >= 0 ? 'ok' : 'bad'}">${p.side} ${p.leverage}x — ${fmt$(p.upnl, 2)} (${fmtPct(p.upnl_pct, 1)})</div>
        <div style="font-size:11px;color:var(--text-dim)">
          ${p.contracts} BTC @ ${fmt$(p.entry)} &nbsp;|&nbsp; Liq: ${p.liq_price ? fmt$(p.liq_price) : 'N/A'}
        </div>
      </div>
    `).join('');
  } else {
    exBadge.textContent = 'No Positions';
    exBadge.className = 'pill pill-dim';
    posEl.innerHTML = '<div class="dim" style="font-size:12px;text-align:center;padding:8px 0">No open positions</div>';
  }
  el('executor-orders').textContent = (ex.orders || []).length;
  el('executor-last-trade').textContent = ex.last_trade ? shortTs(ex.last_trade) : '—';

  // ── Risk Guard ──
  const rk = d.risk || {};
  const circEl = el('risk-circuit');
  if (rk.circuit_ok) {
    const pctFromPeak = rk.peak_equity > 0 ? ((rk.equity / rk.peak_equity - 1) * 100) : 0;
    circEl.textContent = 'OK (' + fmtPct(pctFromPeak, 1) + ' vs peak)';
    circEl.className = 'card-val ok';
  } else {
    circEl.textContent = '⚠ TRIGGERED';
    circEl.className = 'card-val bad';
  }
  const dailyEl = el('risk-daily');
  dailyEl.textContent = fmtPct(rk.daily_loss_pct, 2) + ' (' + fmt$(rk.equity) + ')';
  dailyEl.className = 'card-val ' + (rk.daily_loss_pct < -3 ? 'bad' : rk.daily_loss_pct < -1 ? 'warn' : 'ok');

  const blockEl = el('risk-block');
  if (rk.no_trade_until) {
    blockEl.textContent = 'Until ' + shortTs(rk.no_trade_until);
    blockEl.className = 'card-val bad';
  } else {
    blockEl.textContent = 'None';
    blockEl.className = 'card-val ok';
  }

  // Milestones
  const mEl = el('milestones');
  mEl.innerHTML = (rk.milestones_all || ['100x','500x','1000x']).map((m, i) => {
    const hit = (rk.milestones_hit || []).includes(m);
    return `<div class="m-dot ${hit ? 'hit' : ''}" title="${m}"></div>`;
  }).join('') + '&nbsp;<span style="font-size:11px;color:var(--text-dim)">' + (rk.milestones_hit || []).length + '/3</span>';

  const multEl = el('risk-multiple');
  const mult = rk.equity_multiple || 1;
  multEl.textContent = mult.toFixed(4) + 'x initial';
  multEl.className = 'card-val ' + (mult >= 2 ? 'ok' : mult < 0.9 ? 'bad' : '');

  // ── Briefer ──
  const br = d.briefer || {};
  el('briefer-last').textContent = br.last_briefing || 'Not yet run';
  el('briefer-next').textContent = br.next_briefing || '—';

  // ── GM ──
  const gm = d.gm || {};
  const checks = gm.health_checks || {};
  el('gm-checks').innerHTML = Object.entries(checks).map(([k, v]) =>
    `<div class="card-row">
      <span class="card-key">${k}</span>
      <span class="card-val ${v ? 'ok' : 'bad'}">${v ? '✓ OK' : '✗ FAIL'}</span>
    </div>`
  ).join('');

  const atrEl = el('gm-atr');
  if (gm.atr_current && gm.avg_atr_100) {
    const ratio = gm.atr_ratio || 1;
    const regime = ratio < 0.5 ? '⚠ Low' : ratio > 2 ? '⚡ High' : '✓ Normal';
    atrEl.textContent = `${regime} — $${Math.round(gm.atr_current)} vs avg $${Math.round(gm.avg_atr_100)} (${(ratio*100).toFixed(0)}%)`;
    atrEl.className = 'card-val ' + (ratio < 0.5 ? 'warn' : ratio > 2 ? 'ok' : 'ok');
  } else {
    atrEl.textContent = '—';
  }

  const droughtEl = el('gm-drought');
  const dd = gm.drought_days;
  if (dd == null) {
    droughtEl.textContent = 'No trades yet';
    droughtEl.className = 'card-val dim';
  } else if (dd >= 14) {
    droughtEl.textContent = dd + ' days 🚨';
    droughtEl.className = 'card-val bad';
  } else if (dd >= 7) {
    droughtEl.textContent = dd + ' days ⚠';
    droughtEl.className = 'card-val warn';
  } else {
    droughtEl.textContent = dd + ' days ✓';
    droughtEl.className = 'card-val ok';
  }

  el('gm-candle').textContent = gm.last_candle_ts || '—';
  el('gm-git').textContent = gm.last_git_commit || '—';

  // ── Performance ──
  const pf = d.performance || {};
  el('p-trades').textContent = pf.total_trades || 0;
  el('p-wins').textContent = pf.wins || 0;
  el('p-losses').textContent = pf.losses || 0;
  el('p-wr').textContent = (pf.win_rate || 0).toFixed(1) + '%';

  const pnlV = pf.total_pnl || 0;
  el('p-pnl').textContent = (pnlV >= 0 ? '+' : '') + fmt$(pnlV);
  el('p-pnl').className = 'perf-stat-val ' + (pnlV >= 0 ? 'ok' : 'bad');

  el('p-pnl-pct').textContent = fmtPct(pf.pnl_pct, 2);
  el('p-pnl-pct').className = 'perf-stat-val ' + (pf.pnl_pct >= 0 ? 'ok' : 'bad');

  el('p-best').textContent = '+' + fmt$(pf.best_trade);
  el('p-worst').textContent = fmt$(pf.worst_trade);
  el('p-equity').textContent = fmt$(pf.current_equity);

  // Chart
  drawChart(d.equity_chart || []);

  // ── Activity log ──
  const rows = (d.activity || []);
  const tbody = el('activity-log');
  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-dim);padding:20px">No log entries found. Logs will appear after agents run.</td></tr>';
  } else {
    tbody.innerHTML = rows.map(r => {
      const lvlCls = r.level === 'ERROR' ? 'log-error' : r.level === 'WARNING' ? 'log-warning' : 'log-info';
      return `<tr>
        <td style="white-space:nowrap;color:var(--text-dim)">${r.timestamp || '—'}</td>
        <td><span class="agent-badge ${agentClass(r.agent)}">${r.agent}</span></td>
        <td class="${lvlCls}" style="white-space:nowrap">${r.level}</td>
        <td>${escHtml(r.message)}</td>
      </tr>`;
    }).join('');
  }

  el('generated-at').textContent = 'Updated: ' + (d.generated_at || '—');
  el('loading').style.display = 'none';
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ─── Fetch & refresh ─────────────────────────────────────────────────────────
async function fetchData() {
  try {
    const r = await fetch('/api/data');
    const d = await r.json();
    render(d);
  } catch (e) {
    console.error('Fetch error', e);
    el('loading').style.display = 'none';
  }
}

// Auto-refresh countdown
let countdown = 60;
setInterval(() => {
  countdown--;
  el('refresh-timer').textContent = countdown;
  if (countdown <= 0) {
    countdown = 60;
    fetchData();
  }
}, 1000);

// Initial load
fetchData();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return HTML


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V20 Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default 0.0.0.0)")
    parser.add_argument("--port", default=5555, type=int, help="Port (default 5555)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  V20 Trading System — Mission Control Dashboard")
    print("=" * 60)
    print(f"  → http://localhost:{args.port}")
    print(f"  → Data: {SCRIPT_DIR}")
    print(f"  → Bybit demo: api-demo.bybit.com")
    print("=" * 60)
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)

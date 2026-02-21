#!/usr/bin/env python3
"""Performance Tracker — V20 Trade History & Statistics

Tracks every trade's outcome in scripts/.trade_history.json.

Functions:
    record_trade(signal_data, order_data)   — called by scanner after order placement
    update_trades(exchange)                 — check open trades against exchange state
    get_performance_summary()               — return stats dict

Usage (standalone):
    python scripts/gm_performance.py
"""

import json
import uuid
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional

SYDNEY_TZ = ZoneInfo("Australia/Sydney")

import ccxt

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = Path(__file__).parent
PROJECT_DIR   = SCRIPT_DIR.parent
HISTORY_FILE  = SCRIPT_DIR / ".trade_history.json"

SYMBOL         = "BTC/USDT:USDT"
INITIAL_EQUITY = 169_751.0

# ── Default structure ──────────────────────────────────────────────────────────

DEFAULT_HISTORY = {
    "trades": [],
    "summary": {
        "total_trades":   0,
        "wins":           0,
        "losses":         0,
        "liquidations":   0,
        "total_pnl":      0.0,
        "win_rate":       0.0,
        "avg_win":        0.0,
        "avg_loss":       0.0,
        "best_trade":     0.0,
        "worst_trade":    0.0,
        "start_equity":   INITIAL_EQUITY,
        "current_equity": INITIAL_EQUITY,
    }
}


# ── File I/O ───────────────────────────────────────────────────────────────────

def _load_history() -> dict:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                data = json.load(f)
            # Ensure all summary keys present (forward-compat)
            for k, v in DEFAULT_HISTORY["summary"].items():
                data.setdefault("summary", {}).setdefault(k, v)
            return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Could not read trade history ({e}), starting fresh")
    return json.loads(json.dumps(DEFAULT_HISTORY))   # deep copy


def _save_history(data: dict):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Summary Recalculation ──────────────────────────────────────────────────────

def _recalculate_summary(data: dict, current_equity: Optional[float] = None):
    """Recompute all summary stats from the trade list in-place."""
    trades  = data["trades"]
    summary = data["summary"]

    closed = [t for t in trades if t["status"] in ("closed", "liquidated")]
    wins   = [t for t in closed if (t.get("pnl") or 0) > 0]
    losses = [t for t in closed if (t.get("pnl") or 0) <= 0]
    liqs   = [t for t in trades if t["status"] == "liquidated"]

    total_pnl = sum(t.get("pnl") or 0 for t in closed)
    win_pnls  = [t["pnl"] for t in wins  if t.get("pnl") is not None]
    loss_pnls = [t["pnl"] for t in losses if t.get("pnl") is not None]

    summary["total_trades"]  = len(closed)
    summary["wins"]          = len(wins)
    summary["losses"]        = len(losses)
    summary["liquidations"]  = len(liqs)
    summary["total_pnl"]     = round(total_pnl, 2)
    summary["win_rate"]      = round(len(wins) / len(closed) * 100, 1) if closed else 0.0
    summary["avg_win"]       = round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0.0
    summary["avg_loss"]      = round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0.0
    summary["best_trade"]    = round(max(win_pnls),  2) if win_pnls  else 0.0
    summary["worst_trade"]   = round(min(loss_pnls), 2) if loss_pnls else 0.0

    if current_equity is not None:
        summary["current_equity"] = round(current_equity, 2)


# ── Public API ─────────────────────────────────────────────────────────────────

def record_trade(signal_data, order_data: dict) -> dict:
    """Record a newly placed trade in the history file.

    Args:
        signal_data: A SignalResult object (or dict-like) from live_scanner.
        order_data:  The dict returned by place_orders() in live_scanner.

    Returns:
        The newly created trade record dict.
    """
    data = _load_history()

    # Extract signal fields — handle both object attrs and dict keys
    def _get(obj, *keys, default=None):
        for k in keys:
            try:
                return getattr(obj, k)
            except AttributeError:
                pass
            if isinstance(obj, dict) and k in obj:
                return obj[k]
        return default

    entry_price = _get(signal_data, "entry_price", "entryPrice", default=0.0)
    stop_loss   = _get(signal_data, "stop_loss",   "stopLoss",   default=0.0)
    tp1         = _get(signal_data, "take_profit_1", "tp1",      default=0.0)
    tp2         = _get(signal_data, "take_profit_2", "tp2",      default=0.0)
    leverage    = _get(signal_data, "leverage",    default=0)
    side        = _get(signal_data, "side",        default="long")
    confluence  = _get(signal_data, "confluence",  default=[])
    candle_time = _get(signal_data, "candle_time", default=None)

    # Best-effort size from order data
    entry_order = order_data.get("entry", {}) or {}
    size_btc = float(entry_order.get("amount", 0) or 0)

    trade_id = str(uuid.uuid4())[:8]
    now_iso  = datetime.now(SYDNEY_TZ).strftime("%Y-%m-%dT%H:%M:%S")

    trade = {
        "id":          trade_id,
        "entry_time":  now_iso,
        "candle_time": str(candle_time) if candle_time else now_iso,
        "entry_price": float(entry_price),
        "side":        side,
        "leverage":    int(leverage),
        "size_btc":    round(size_btc, 4),
        "confluence":  list(confluence),
        "stop_loss":   round(float(stop_loss), 2),
        "tp1":         round(float(tp1), 2),
        "tp2":         round(float(tp2), 2),
        "status":      "open",
        "exit_price":  None,
        "exit_time":   None,
        "pnl":         None,
        "close_reason": None,
        # raw order IDs for cross-reference
        "entry_order_id": (order_data.get("entry") or {}).get("id"),
        "sl_order_id":    (order_data.get("stop_loss") or {}).get("id"),
        "tp1_order_id":   (order_data.get("tp1") or {}).get("id"),
    }

    data["trades"].append(trade)
    _save_history(data)

    print(f"📝 Trade recorded: {trade_id} | {side.upper()} {leverage}x @ ${entry_price:,.2f}")
    return trade


def update_trades(exchange: ccxt.Exchange) -> list[str]:
    """Check open trades against exchange positions/orders and update statuses.

    Returns a list of status messages about any closed trades found.
    """
    data   = _load_history()
    trades = data["trades"]
    msgs   = []

    open_trades = [t for t in trades if t["status"] == "open"]
    if not open_trades:
        return []

    # Fetch current state from exchange
    try:
        positions   = exchange.fetch_positions([SYMBOL])
        open_orders = exchange.fetch_open_orders(SYMBOL)
        ticker      = exchange.fetch_ticker(SYMBOL)
        current_price = float(ticker["last"])
    except Exception as e:
        msgs.append(f"⚠️  update_trades: fetch failed — {e}")
        return msgs

    # Build set of currently open order IDs
    open_order_ids = {str(o.get("id")) for o in open_orders if o.get("id")}

    # Build set of active position symbols
    active_positions = {
        p.get("symbol") for p in positions
        if abs(float(p.get("contracts", 0) or 0)) > 0
    }

    for trade in open_trades:
        entry_price = trade.get("entry_price", 0)
        sl          = trade.get("stop_loss", 0)
        tp1         = trade.get("tp1", 0)
        tp2         = trade.get("tp2", 0)
        size_btc    = trade.get("size_btc", 0) or 0
        side        = trade.get("side", "long")

        sl_id  = str(trade.get("sl_order_id", ""))
        tp1_id = str(trade.get("tp1_order_id", ""))

        # If the SL order no longer exists, it likely triggered
        sl_triggered  = sl_id  and sl_id  not in open_order_ids
        tp1_triggered = tp1_id and tp1_id not in open_order_ids

        # Also check if no position remains (position fully closed)
        no_position = SYMBOL not in active_positions

        close_reason = None
        exit_price   = None

        if sl_triggered and not tp1_triggered:
            close_reason = "stop_loss"
            exit_price   = sl
        elif tp1_triggered and not sl_triggered:
            close_reason = "tp1"
            exit_price   = tp1
        elif sl_triggered and tp1_triggered:
            # Both gone — assume TP1 was first (more conservative)
            close_reason = "tp1_then_sl"
            exit_price   = tp1
        elif no_position and size_btc > 0:
            # Position closed but we don't know why — use current price
            close_reason = "unknown"
            exit_price   = current_price

        if close_reason and exit_price:
            now_iso = datetime.now(SYDNEY_TZ).strftime("%Y-%m-%dT%H:%M:%S")

            # Compute PnL (simplified: does not account for partial closes at TP1)
            if side == "long":
                pnl = (exit_price - entry_price) * size_btc * trade.get("leverage", 1)
            else:
                pnl = (entry_price - exit_price) * size_btc * trade.get("leverage", 1)

            # Determine win/loss/liquidation
            if close_reason == "stop_loss":
                # Check if price hit stop hard (liquidation approximation)
                if sl_triggered and abs(exit_price - entry_price) / entry_price > 0.08:
                    trade["status"] = "liquidated"
                else:
                    trade["status"] = "closed"
            else:
                trade["status"] = "closed"

            trade["exit_price"]  = round(exit_price, 2)
            trade["exit_time"]   = now_iso
            trade["pnl"]         = round(pnl, 2)
            trade["close_reason"] = close_reason

            emoji = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
            msgs.append(
                f"{emoji} Trade {trade['id']} closed: {close_reason.upper()} "
                f"@ ${exit_price:,.2f} | PnL: ${pnl:+,.2f}"
            )

    # Refresh equity from exchange if possible
    try:
        bal = exchange.fetch_balance()
        current_equity = float((bal.get("USDT") or {}).get("total", 0) or 0)
    except Exception:
        current_equity = None

    _recalculate_summary(data, current_equity=current_equity)
    _save_history(data)

    return msgs


def get_performance_summary() -> dict:
    """Return the summary dict from trade history."""
    data = _load_history()
    return data["summary"]


def get_week_trades(week_start: "datetime") -> list[dict]:
    """Return trades that were entered during the given week."""
    data   = _load_history()
    trades = data["trades"]
    week_end = week_start + timedelta(days=7)

    result = []
    for t in trades:
        try:
            entry_dt = datetime.fromisoformat(t["entry_time"])
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            if week_start <= entry_dt < week_end:
                result.append(t)
        except Exception:
            pass
    return result


# ── Standalone ─────────────────────────────────────────────────────────────────

def main():
    summary = get_performance_summary()
    data    = _load_history()

    print("📊 V20 Performance Tracker")
    print("=" * 45)
    print(f"  Total trades:   {summary['total_trades']}")
    print(f"  Wins:           {summary['wins']}")
    print(f"  Losses:         {summary['losses']}")
    print(f"  Liquidations:   {summary['liquidations']}")
    print(f"  Win rate:       {summary['win_rate']:.1f}%")
    print(f"  Total PnL:      ${summary['total_pnl']:+,.2f}")
    print(f"  Avg win:        ${summary['avg_win']:+,.2f}")
    print(f"  Avg loss:       ${summary['avg_loss']:+,.2f}")
    print(f"  Best trade:     ${summary['best_trade']:+,.2f}")
    print(f"  Worst trade:    ${summary['worst_trade']:+,.2f}")
    print(f"  Start equity:   ${summary['start_equity']:,.2f}")
    print(f"  Current equity: ${summary['current_equity']:,.2f}")

    open_trades = [t for t in data["trades"] if t["status"] == "open"]
    if open_trades:
        print()
        print(f"  Open trades ({len(open_trades)}):")
        for t in open_trades:
            print(
                f"    [{t['id']}] {t['side'].upper()} {t['leverage']}x "
                f"@ ${t['entry_price']:,.2f} | "
                f"SL=${t['stop_loss']:,.2f} TP1=${t['tp1']:,.2f}"
            )


if __name__ == "__main__":
    main()

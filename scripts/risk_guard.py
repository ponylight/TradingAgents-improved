#!/usr/bin/env python3
"""Risk Guard — V20 Account Health Monitor

Checks circuit breakers, daily loss limits, profit-taking milestones,
and orphaned orders. Run after every scanner execution.

Usage:
    python scripts/risk_guard.py
"""

import os
import sys
import json
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional

SYDNEY_TZ = ZoneInfo("Australia/Sydney")

import ccxt
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
STATE_FILE = SCRIPT_DIR / ".risk_state.json"

load_dotenv(PROJECT_DIR / ".env")

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_SECRET  = os.getenv("BYBIT_SECRET", "")

# ── Constants ─────────────────────────────────────────────────────────

INITIAL_CAPITAL     = 169_751.0
CIRCUIT_BREAKER_PCT = 0.85   # close all if equity < peak * 0.85
DAILY_LOSS_PCT      = 0.97   # block trades if equity < day_start * 0.97
PROFIT_MILESTONES   = [100, 500, 1000]   # multiples of initial_capital
PROFIT_TAKE_PCT     = 0.30   # close 30% of positions at each milestone
SYMBOL              = "BTC/USDT:USDT"

# ── State File ────────────────────────────────────────────────────────

DEFAULT_STATE = {
    "peak_equity":      INITIAL_CAPITAL,
    "initial_capital":  INITIAL_CAPITAL,
    "day_start_equity": INITIAL_CAPITAL,
    "day_start_date":   datetime.now(SYDNEY_TZ).strftime("%Y-%m-%d"),
    "no_trade_until":   None,
    "milestones_hit":   [],
    "last_check":       datetime.now(SYDNEY_TZ).strftime("%Y-%m-%dT%H:%M:%S"),
}


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Could not read state file ({e}), using defaults")
    return DEFAULT_STATE.copy()


def save_state(state: dict):
    state["last_check"] = datetime.now(SYDNEY_TZ).strftime("%Y-%m-%dT%H:%M:%S")
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Exchange ──────────────────────────────────────────────────────────

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


def get_equity(exchange: ccxt.Exchange) -> Optional[float]:
    """Return total USDT equity (wallet balance) from Bybit demo."""
    try:
        bal = exchange.fetch_balance()
        # Total equity = wallet balance (includes unrealised PnL for linear perps)
        usdt = bal.get("USDT", {})
        # 'total' includes unrealised PnL; 'free' is just available margin
        equity = usdt.get("total", usdt.get("free", 0))
        return float(equity)
    except Exception as e:
        print(f"⚠️  ALERT: Failed to fetch equity: {e}")
        return None


def get_positions(exchange: ccxt.Exchange) -> list:
    """Return all open positions."""
    try:
        positions = exchange.fetch_positions([SYMBOL])
        return [p for p in positions if abs(float(p.get("contracts", 0) or 0)) > 0]
    except Exception as e:
        print(f"⚠️  ALERT: Failed to fetch positions: {e}")
        return []


def get_open_orders(exchange: ccxt.Exchange) -> list:
    """Return all open orders."""
    try:
        return exchange.fetch_open_orders(SYMBOL)
    except Exception as e:
        print(f"⚠️  ALERT: Failed to fetch open orders: {e}")
        return []


def close_all_positions(exchange: ccxt.Exchange, positions: list) -> bool:
    """Close all open positions with market orders."""
    success = True
    for pos in positions:
        try:
            contracts = abs(float(pos.get("contracts", 0) or 0))
            side = pos.get("side", "")
            if contracts <= 0:
                continue
            close_side = "sell" if side == "long" else "buy"
            exchange.create_market_order(
                SYMBOL, close_side, contracts,
                params={"reduceOnly": True, "positionIdx": 0}
            )
            print(f"  ALERT: Closed {side} {contracts} contracts (circuit breaker)")
        except Exception as e:
            print(f"  ALERT: Failed to close position: {e}")
            success = False
    return success


def close_pct_of_positions(exchange: ccxt.Exchange, positions: list, pct: float) -> bool:
    """Close a percentage of each open position."""
    success = True
    for pos in positions:
        try:
            contracts = abs(float(pos.get("contracts", 0) or 0))
            side = pos.get("side", "")
            if contracts <= 0:
                continue
            close_qty = round(contracts * pct, 3)
            if close_qty <= 0:
                continue
            close_side = "sell" if side == "long" else "buy"
            exchange.create_market_order(
                SYMBOL, close_side, close_qty,
                params={"reduceOnly": True, "positionIdx": 0}
            )
            print(f"  ALERT: Profit take — closed {pct*100:.0f}% ({close_qty} contracts) of {side}")
        except Exception as e:
            print(f"  ALERT: Failed profit take: {e}")
            success = False
    return success


def cancel_order(exchange: ccxt.Exchange, order_id: str) -> bool:
    try:
        exchange.cancel_order(order_id, SYMBOL)
        return True
    except Exception as e:
        print(f"  ALERT: Failed to cancel order {order_id}: {e}")
        return False


# ── Checks ────────────────────────────────────────────────────────────

def check_circuit_breaker(equity: float, state: dict, exchange: ccxt.Exchange, positions: list) -> str:
    """If equity dropped >15% from peak, flatten everything."""
    peak = state["peak_equity"]
    threshold = peak * CIRCUIT_BREAKER_PCT

    # Update peak
    if equity > peak:
        state["peak_equity"] = equity

    if equity < threshold:
        drop_pct = (1 - equity / peak) * 100
        print(f"🚨 ALERT: CIRCUIT BREAKER TRIGGERED — equity ${equity:,.2f} is {drop_pct:.1f}% below peak ${peak:,.2f}")
        print(f"🚨 ALERT: Closing ALL positions immediately…")
        if positions:
            close_all_positions(exchange, positions)
        else:
            print("  (No open positions to close)")
        return f"🚨 CIRCUIT BREAKER: ${equity:,.2f} vs peak ${peak:,.2f} (−{drop_pct:.1f}%)"
    else:
        pct_from_peak = (equity / peak - 1) * 100 if equity != peak else 0
        pct_str = f"+{pct_from_peak:.1f}%" if pct_from_peak >= 0 else f"{pct_from_peak:.1f}%"
        return f"✅ Circuit breaker: OK (${equity:,.2f}, {pct_str} vs peak ${peak:,.2f})"


def check_daily_loss(equity: float, state: dict) -> str:
    """If equity dropped >3% from day start, block new trades for 24h."""
    today = datetime.now(SYDNEY_TZ).strftime("%Y-%m-%d")

    # Reset day start if it's a new day (Sydney calendar day)
    if state.get("day_start_date") != today:
        state["day_start_date"]   = today
        state["day_start_equity"] = equity

    day_start = state["day_start_equity"]
    threshold = day_start * DAILY_LOSS_PCT

    if equity < threshold:
        drop_pct = (1 - equity / day_start) * 100
        # Set block for 24h if not already set
        if not state.get("no_trade_until"):
            until_dt = datetime.now(SYDNEY_TZ) + timedelta(hours=24)
            state["no_trade_until"] = until_dt.isoformat()
            print(f"🚨 ALERT: DAILY LOSS LIMIT HIT — equity ${equity:,.2f} is {drop_pct:.1f}% below day start ${day_start:,.2f}")
            print(f"🚨 ALERT: New trades BLOCKED until {state['no_trade_until']} Sydney")
        return f"⚠️  Daily loss: ${equity:,.2f} vs day start ${day_start:,.2f} (−{drop_pct:.1f}%) — TRADES BLOCKED until {state['no_trade_until']}"
    else:
        # Check if block has expired
        if state.get("no_trade_until"):
            until_dt = datetime.fromisoformat(state["no_trade_until"])
            if until_dt.tzinfo is None:
                until_dt = until_dt.replace(tzinfo=SYDNEY_TZ)
            if datetime.now(SYDNEY_TZ) >= until_dt:
                state["no_trade_until"] = None
                print("ℹ️  Daily trade block expired — trading resumed")
        return f"✅ Daily loss: OK (${equity:,.2f} vs day start ${day_start:,.2f})"


def check_milestones(equity: float, state: dict, exchange: ccxt.Exchange, positions: list) -> str:
    """Check profit-taking milestones (100x, 500x, 1000x initial capital)."""
    initial   = state["initial_capital"]
    hit_list  = state.get("milestones_hit", [])
    results   = []
    triggered = []

    for mult in PROFIT_MILESTONES:
        target = initial * mult
        label  = f"{mult}x"
        if equity >= target and label not in hit_list:
            triggered.append((label, target, mult))

    for label, target, mult in triggered:
        print(f"🎯 ALERT: PROFIT MILESTONE {label} HIT — equity ${equity:,.2f} >= ${target:,.2f}")
        print(f"  Closing {PROFIT_TAKE_PCT*100:.0f}% of all positions…")
        if positions:
            close_pct_of_positions(exchange, positions, PROFIT_TAKE_PCT)
        else:
            print("  (No open positions to close)")
        hit_list.append(label)
        results.append(f"🎯 Milestone {label}: TRIGGERED — took {PROFIT_TAKE_PCT*100:.0f}% profit")

    state["milestones_hit"] = hit_list

    if not results:
        # Report current progress toward next milestone
        hits = len(hit_list)
        current_mult = equity / initial
        next_mults = [m for m in PROFIT_MILESTONES if f"{m}x" not in hit_list]
        if next_mults:
            next_mult = next_mults[0]
            pct_to_next = (current_mult / next_mult) * 100
            return f"✅ Milestones: {hits}/{len(PROFIT_MILESTONES)} hit | {current_mult:.2f}x capital | {pct_to_next:.1f}% to {next_mult}x"
        else:
            return f"✅ Milestones: All {len(PROFIT_MILESTONES)} hit | {current_mult:.2f}x capital"

    return "\n".join(results)


def check_orphaned_orders(exchange: ccxt.Exchange, positions: list, open_orders: list) -> str:
    """Cancel open orders that have no matching position."""
    if not open_orders:
        return "✅ Orphaned orders: None (no open orders)"

    if not positions:
        # All open orders are orphaned
        orphans = open_orders
    else:
        # Only reduce-only orders without a position are orphaned
        position_symbols = {p.get("symbol") for p in positions}
        orphans = [
            o for o in open_orders
            if o.get("reduceOnly") and o.get("symbol") not in position_symbols
        ]

    if not orphans:
        return f"✅ Orphaned orders: None ({len(open_orders)} open orders all have matching positions)"

    cancelled = 0
    for order in orphans:
        oid = order.get("id", "?")
        otype = order.get("type", "?")
        print(f"🚨 ALERT: Orphaned order {oid} ({otype}) — cancelling…")
        if cancel_order(exchange, oid):
            cancelled += 1

    return f"⚠️  Orphaned orders: {len(orphans)} found, {cancelled} cancelled"


# ── Main ──────────────────────────────────────────────────────────────

def main():
    now_str = datetime.now(SYDNEY_TZ).strftime("%Y-%m-%d %H:%M %Z")
    print(f"Risk Guard — {now_str}")
    print("-" * 50)

    # Load state
    state = load_state()

    # Connect to exchange
    try:
        exchange = get_exchange()
    except Exception as e:
        print(f"🚨 ALERT: Cannot connect to Bybit: {e}")
        sys.exit(1)

    # Fetch equity
    equity = get_equity(exchange)
    if equity is None:
        print("🚨 ALERT: Could not fetch equity — skipping checks")
        save_state(state)
        sys.exit(1)

    # Fetch positions and orders
    positions    = get_positions(exchange)
    open_orders  = get_open_orders(exchange)

    print(f"  Equity:    ${equity:,.2f} USDT")
    print(f"  Positions: {len(positions)} open")
    print(f"  Orders:    {len(open_orders)} open")
    print()

    # Run checks
    results = []

    results.append(check_circuit_breaker(equity, state, exchange, positions))
    results.append(check_daily_loss(equity, state))
    results.append(check_milestones(equity, state, exchange, positions))

    # Refresh positions after any closes
    positions   = get_positions(exchange)
    open_orders = get_open_orders(exchange)
    results.append(check_orphaned_orders(exchange, positions, open_orders))

    print()
    for r in results:
        print(r)

    print()

    # Summary
    alerts = [r for r in results if r.startswith(("🚨", "⚠️"))]
    if alerts:
        print(f"STATUS: ⚠️  {len(alerts)} ISSUE(S) DETECTED")
    else:
        print("STATUS: ✅ ALL CLEAR")

    # Save updated state
    save_state(state)


if __name__ == "__main__":
    main()

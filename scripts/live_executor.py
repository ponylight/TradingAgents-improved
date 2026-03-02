#!/usr/bin/env python3
"""
TradingAgents Live Executor — Multi-agent pipeline → Bybit demo execution.

Profit-Taking Strategy (2-Target Rule + ATR Trail):
  1. TP1 at 1.5R — close 50% (pays for the trade)
  2. Remaining 50% trails with 2×ATR stop
  3. Time exit: close if no movement after 24 4H bars (4 days)

Position sizing parsed from agent output, with fallback to RISK_PCT.
"""

import ccxt
import os
import sys
import json
import re
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from tradingagents.graph.crypto_trading_graph import CryptoTradingAgentsGraph
from tradingagents.agents.utils.agent_trading_modes import extract_recommendation, get_position_transition
from tradingagents.graph.crypto_trading_graph import CRYPTO_DEFAULT_CONFIG
from tradingagents.agents.utils.agent_trading_modes import get_position_transition

# === CONFIG ===
SYMBOL = "BTC/USDT:USDT"
SPOT_SYMBOL = "BTC/USDT"
DEFAULT_ALLOC_PCT = 0.03      # 3% default margin allocation if agent doesn't specify
MAX_ALLOC_PCT = 0.12          # Never exceed 12% margin allocation
RISK_PER_TRADE = 0.01         # 1% of equity at risk per trade (professional standard)
CIRCUIT_BREAKER_PCT = 0.09
PEAK_EQUITY = 154_425
DAILY_LOSS_LIMIT = 0.02       # 2% daily loss limit — lock out after this
COLD_STREAK_THRESHOLD = 3     # After 3 consecutive losses, halve risk
COLD_STREAK_RISK_MULT = 0.5   # Multiply risk by this during cold streak
TP1_R_MULTIPLE = 1.5          # TP1 at 1.5R
TP1_CLOSE_PCT = 0.5           # Close 50% at TP1
ATR_TRAIL_MULTIPLE = 2.0      # Trail remaining with 2×ATR
TIME_EXIT_BARS = 24           # Close if flat after 24 4H bars

LOG_DIR = PROJECT_ROOT / "logs"
MEMORY_DIR = PROJECT_ROOT / "memory"
STATE_FILE = LOG_DIR / "executor_state.json"
LOG_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"executor_{datetime.now().strftime('%Y-%m-%d')}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("executor")


# === EXCHANGE ===

def get_exchange():
    is_demo = os.getenv("BYBIT_DEMO", "").lower() == "true"
    config = {
        "apiKey": os.getenv("BYBIT_API_KEY"),
        "secret": os.getenv("BYBIT_SECRET"),
        "options": {"defaultType": "linear"},
        "enableRateLimit": True,
    }
    if is_demo:
        config["urls"] = {"api": {"public": "https://api-demo.bybit.com", "private": "https://api-demo.bybit.com"}}
    exchange = ccxt.bybit(config)
    if is_demo:
        real = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})
        real.load_markets()
        exchange.markets = real.markets
        exchange.markets_by_id = real.markets_by_id
        exchange.currencies = real.currencies
        exchange.currencies_by_id = real.currencies_by_id
        log.info("🎮 DEMO MODE (api-demo.bybit.com)")
    return exchange


def get_equity(exchange):
    balance = exchange.fetch_balance({"type": "swap"})
    return float(balance["total"].get("USDT", 0))


def get_positions(exchange):
    positions = exchange.fetch_positions([SYMBOL])
    return [p for p in positions if abs(float(p["contracts"] or 0)) > 0]


def get_atr(exchange, period=14, timeframe="1d"):
    """Fetch current ATR. Default: daily candles for proper volatility measurement."""
    candles = exchange.fetch_ohlcv(SYMBOL, timeframe, limit=period + 1)
    trs = []
    for i in range(1, len(candles)):
        h, l, prev_c = candles[i][2], candles[i][3], candles[i-1][4]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    return sum(trs[-period:]) / period


def check_daily_loss_limit(executor_state, equity):
    """Check if daily losses exceed limit. Returns True if locked out."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    trades_today = [t for t in executor_state.get("trades", [])
                    if t.get("timestamp", "").startswith(today) and "close_pnl_pct" in t]
    
    daily_loss = sum(t["close_pnl_pct"] for t in trades_today if t["close_pnl_pct"] < 0)
    daily_loss_abs = abs(daily_loss) / 100  # Convert from % to fraction
    
    if daily_loss_abs >= DAILY_LOSS_LIMIT:
        log.error(f"🔒 DAILY LOSS LIMIT: {daily_loss:.2f}% today (limit: {DAILY_LOSS_LIMIT*100:.0f}%). Locked out until next session.")
        return True
    
    if trades_today:
        log.info(f"📊 Daily P&L: {daily_loss:+.2f}% | Limit: {DAILY_LOSS_LIMIT*100:.0f}%")
    return False


def get_cold_streak(executor_state):
    """Count consecutive losses. Returns streak count and risk multiplier."""
    trades = executor_state.get("trades", [])
    streak = 0
    for t in reversed(trades):
        if "close_pnl_pct" not in t:
            continue
        if t["close_pnl_pct"] < 0:
            streak += 1
        else:
            break
    
    if streak >= COLD_STREAK_THRESHOLD:
        log.warning(f"🥶 COLD STREAK: {streak} consecutive losses — risk halved to {RISK_PER_TRADE * COLD_STREAK_RISK_MULT * 100:.1f}%")
        return streak, COLD_STREAK_RISK_MULT
    return streak, 1.0


def check_circuit_breaker(equity):
    threshold = PEAK_EQUITY * (1 - CIRCUIT_BREAKER_PCT)
    if equity < threshold:
        log.error(f"🚨 CIRCUIT BREAKER: ${equity:,.0f} < ${threshold:,.0f}")
        return True
    return False


# === SIGNAL PARSING ===

def parse_trade_params(full_decision: str) -> dict:
    """Extract structured trade params. Tries ---TRADE_PLAN--- block first, falls back to regex."""
    params = {}

    # Try structured block first (from trader agent)
    block_match = re.search(r'---TRADE_PLAN---(.*?)---END_TRADE_PLAN---', full_decision, re.DOTALL)
    if block_match:
        block = block_match.group(1)
        field_map = {
            "CONFIDENCE": ("confidence", lambda v: int(v)),
            "RISK_SCORE": ("risk_score", lambda v: float(v)),
            "STOP_LOSS": ("stop_loss", lambda v: float(v.replace(",", "").replace("$", ""))),
            "TAKE_PROFIT_1": ("take_profit_1", lambda v: float(v.replace(",", "").replace("$", ""))),
            "TAKE_PROFIT_2": ("take_profit_2", lambda v: float(v.replace(",", "").replace("$", ""))),
            "POSITION_SIZE": ("risk_pct", lambda v: float(v.replace("%", "")) / 100),
            "MARGIN_ALLOCATION": ("risk_pct", lambda v: float(v.replace("%", "")) / 100),
            "RISK_REWARD": ("risk_reward", lambda v: float(v)),
        }
        for key, (param_name, converter) in field_map.items():
            m = re.search(rf'{key}:\s*(.+)', block)
            if m:
                val = m.group(1).strip()
                if val.lower() not in ("none", "n/a", ""):
                    try:
                        params[param_name] = converter(val)
                    except (ValueError, TypeError):
                        pass

        # Cap risk
        if "risk_pct" in params:
            params["risk_pct"] = min(params["risk_pct"], MAX_ALLOC_PCT)

        if params:
            return params

    # Fallback: regex parsing from prose
    text = full_decision.lower()

    size_patterns = [
        r'position\s*size[:\s]*(\d+(?:\.\d+)?)\s*%',
        r'(\d+(?:\.\d+)?)\s*%\s*(?:of\s*)?(?:portfolio|equity|capital)',
        r'allocat\w*\s*(\d+(?:\.\d+)?)\s*%',
        r'risk\w*\s*(\d+(?:\.\d+)?)\s*%\s*(?:of|per)',
    ]
    for pattern in size_patterns:
        m = re.search(pattern, text)
        if m:
            pct = float(m.group(1)) / 100
            if 0.005 <= pct <= 0.10:
                params["risk_pct"] = min(pct, MAX_ALLOC_PCT)
                break

    sl_patterns = [
        r'stop[- ]?loss[:\s]*\$?([\d,]+(?:\.\d+)?)',
        r'sl[:\s]*\$?([\d,]+(?:\.\d+)?)',
        r'stop\s*(?:at|@)\s*\$?([\d,]+(?:\.\d+)?)',
    ]
    for pattern in sl_patterns:
        m = re.search(pattern, text)
        if m:
            params["stop_loss"] = float(m.group(1).replace(",", ""))
            break

    tp_patterns = [
        r'(?:take[- ]?profit|tp)\s*(?:1|a)?[:\s]*\$?([\d,]+(?:\.\d+)?)',
        r'(?:target|tp1?)[:\s]*\$?([\d,]+(?:\.\d+)?)',
    ]
    for pattern in tp_patterns:
        m = re.search(pattern, text)
        if m:
            params["take_profit_1"] = float(m.group(1).replace(",", ""))
            break

    tp2_patterns = [
        r'(?:take[- ]?profit|tp)\s*(?:2|b)[:\s]*\$?([\d,]+(?:\.\d+)?)',
        r'(?:target|tp)\s*2[:\s]*\$?([\d,]+(?:\.\d+)?)',
    ]
    for pattern in tp2_patterns:
        m = re.search(pattern, text)
        if m:
            params["take_profit_2"] = float(m.group(1).replace(",", ""))
            break

    conf_patterns = [
        r'confidence[:\s]*(\d+)\s*/?\s*10',
        r'conviction[:\s]*(\d+)\s*/?\s*10',
        r'(\d+)\s*/\s*10\s*confidence',
    ]
    for pattern in conf_patterns:
        m = re.search(pattern, text)
        if m:
            params["confidence"] = int(m.group(1))
            break

    return params


# === POSITION MANAGEMENT ===

def close_position(exchange, position, fraction=1.0):
    """Close all or part of a position."""
    side = "sell" if position["side"] == "long" else "buy"
    total = abs(float(position["contracts"]))
    amount = float(exchange.amount_to_precision(SYMBOL, total * fraction))
    if amount <= 0:
        return None
    label = f"{fraction*100:.0f}%" if fraction < 1 else "full"
    log.info(f"📤 Closing {label} of {position['side']} ({amount}/{total} contracts)")
    order = exchange.create_order(SYMBOL, "market", side, amount, params={"reduceOnly": True})
    log.info(f"✅ Closed: {order['id']}")
    return order


def open_position(exchange, direction, equity, alloc_pct, stop_loss=None, risk_multiplier=1.0):
    """Open position sized by 1% risk target and daily ATR. Leverage auto-derived."""
    ticker = exchange.fetch_ticker(SYMBOL)
    price = ticker["last"]

    # Mechanical sizing: risk exactly RISK_PER_TRADE of equity
    atr = get_atr(exchange, period=14, timeframe="1d")
    sl_distance = 2 * atr  # 2×daily ATR stop
    effective_risk = RISK_PER_TRADE * risk_multiplier
    risk_dollars = equity * effective_risk
    notional = (risk_dollars / sl_distance) * price  # exact notional for 1% risk
    amount = float(exchange.amount_to_precision(SYMBOL, notional / price))

    if amount <= 0:
        log.error(f"❌ Amount=0. Equity=${equity:.0f}, Price=${price:.0f}")
        return None

    # Derive leverage from agent's allocation
    alloc_pct = min(alloc_pct, MAX_ALLOC_PCT)
    margin = equity * alloc_pct
    leverage = max(1, round(notional / margin))
    leverage = min(leverage, 20)  # Cap at 20x

    try:
        exchange.set_leverage(leverage, SYMBOL)
    except Exception as e:
        log.warning(f"Leverage: {e}")

    side = "buy" if direction == "BUY" else "sell"
    log.info(f"📥 {direction} {amount:.3f} BTC @ ~${price:,.0f} | ${notional:,.0f} notional | {leverage}x lev | {alloc_pct*100:.1f}% alloc | risk ${risk_dollars:,.0f} ({effective_risk*100:.1f}%)")
    log.info(f"   ATR(14d)=${atr:,.0f} | SL dist=${sl_distance:,.0f} ({sl_distance/price*100:.1f}%)")

    link_id = f"committee_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    order = exchange.create_order(SYMBOL, "market", side, amount, params={"orderLinkId": link_id})
    fill_price = float(order.get("average") or price)
    log.info(f"✅ Filled: {order['id']} @ ${fill_price:,.2f}")

    # Set stop loss on exchange
    if stop_loss and stop_loss > 0:
        try:
            sl_side = "sell" if direction == "BUY" else "buy"
            sl_params = {"triggerPrice": str(stop_loss), "reduceOnly": True}
            if direction == "BUY":
                sl_params["triggerDirection"] = 2  # trigger when price falls below
            else:
                sl_params["triggerDirection"] = 1  # trigger when price rises above
            exchange.create_order(SYMBOL, "market", sl_side, amount, params=sl_params)
            log.info(f"🛡️ Stop loss set @ ${stop_loss:,.2f}")
        except Exception as e:
            log.warning(f"⚠️ Failed to set SL on exchange: {e}")

    return {"order": order, "fill_price": fill_price, "amount": amount}


def set_take_profit(exchange, direction, amount, tp_price):
    """Set TP order for partial close."""
    try:
        tp_side = "sell" if direction == "BUY" else "buy"
        tp_amount = float(exchange.amount_to_precision(SYMBOL, amount * TP1_CLOSE_PCT))
        tp_params = {"triggerPrice": str(tp_price), "reduceOnly": True}
        if direction == "BUY":
            tp_params["triggerDirection"] = 1  # trigger when price rises above
        else:
            tp_params["triggerDirection"] = 2  # trigger when price falls below
        exchange.create_order(SYMBOL, "market", tp_side, tp_amount, params=tp_params)
        log.info(f"🎯 TP1 set @ ${tp_price:,.2f} for {tp_amount} BTC ({TP1_CLOSE_PCT*100:.0f}%)")
    except Exception as e:
        log.warning(f"⚠️ Failed to set TP on exchange: {e}")


def manage_trailing_stop(exchange, positions, state):
    """Check and update trailing stop for existing positions."""
    for p in positions:
        entry = float(p["entryPrice"])
        current = float(p["markPrice"] or p["entryPrice"])
        side = p["side"]
        contracts = abs(float(p["contracts"]))

        trade_info = state.get("active_trade", {})
        if not trade_info:
            continue

        atr = get_atr(exchange)
        trail_dist = ATR_TRAIL_MULTIPLE * atr
        opened_at = trade_info.get("opened_at", "")

        # Time-based exit: close if stale
        if opened_at:
            opened_dt = datetime.fromisoformat(opened_at)
            hours_held = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 3600
            bars_held = hours_held / 4
            if bars_held >= TIME_EXIT_BARS:
                pnl_pct = ((current - entry) / entry) if side == "long" else ((entry - current) / entry)
                if abs(pnl_pct) < 0.01:  # Less than 1% move
                    log.info(f"⏰ TIME EXIT: {bars_held:.0f} bars held, only {pnl_pct*100:.1f}% move. Closing.")
                    close_position(exchange, p)
                    returns_pct = pnl_pct * 100
                    state["pending_reflection"] = {"returns_pct": returns_pct, "direction": side, "entry": entry, "exit": current}
                    log.info(f"📊 Time exit P&L: {returns_pct:+.2f}%")
                    state.pop("active_trade", None)
                    return state

        # Update trailing stop
        if side == "long":
            new_trail = current - trail_dist
            old_trail = trade_info.get("trailing_stop", entry - trail_dist)
            if new_trail > old_trail:
                trade_info["trailing_stop"] = new_trail
                log.info(f"📈 Trail updated: ${old_trail:,.0f} → ${new_trail:,.0f} (price ${current:,.0f}, ATR ${atr:,.0f})")
            if current <= trade_info.get("trailing_stop", 0):
                log.info(f"🔔 TRAILING STOP HIT @ ${current:,.0f} (trail was ${trade_info['trailing_stop']:,.0f})")
                close_position(exchange, p)
                returns_pct = ((current - entry) / entry) * 100
                state["pending_reflection"] = {"returns_pct": returns_pct, "direction": "long", "entry": entry, "exit": current}
                log.info(f"📊 Trail close P&L: {returns_pct:+.2f}%")
                state.pop("active_trade", None)
        else:  # short
            new_trail = current + trail_dist
            old_trail = trade_info.get("trailing_stop", entry + trail_dist)
            if new_trail < old_trail:
                trade_info["trailing_stop"] = new_trail
                log.info(f"📉 Trail updated: ${old_trail:,.0f} → ${new_trail:,.0f} (price ${current:,.0f}, ATR ${atr:,.0f})")
            if current >= trade_info.get("trailing_stop", float("inf")):
                log.info(f"🔔 TRAILING STOP HIT @ ${current:,.0f} (trail was ${trade_info['trailing_stop']:,.0f})")
                close_position(exchange, p)
                returns_pct = ((entry - current) / entry) * 100
                state["pending_reflection"] = {"returns_pct": returns_pct, "direction": "short", "entry": entry, "exit": current}
                log.info(f"📊 Trail close P&L: {returns_pct:+.2f}%")
                state.pop("active_trade", None)

    return state


# === MULTI-POSITION SUPPORT (Phase 3) ===

def migrate_state(state: dict) -> dict:
    """Ensure state has positions dict. Backwards compatible."""
    if "positions" not in state:
        log.info("🔄 Migrating state to multi-position format")
        state["positions"] = {"committee": None, "green_lane": None}
        if state.get("active_trade"):
            state["positions"]["committee"] = dict(state["active_trade"])
            state["positions"]["committee"]["source"] = "committee"
            log.info("✅ Migrated active_trade → positions.committee")
    return state


def get_combined_exposure(state: dict) -> dict:
    """Calculate total exposure across all positions.
    Returns: {total_size_pct, total_leverage_est, position_count, positions_summary}
    """
    positions = state.get("positions", {})
    total_size_pct = 0.0
    position_count = 0
    positions_summary = []

    committee = positions.get("committee")
    if committee:
        size = committee.get("size_pct", DEFAULT_ALLOC_PCT * 100)
        total_size_pct += size
        position_count += 1
        positions_summary.append({"source": "committee", "side": committee.get("side", committee.get("direction", "?")), "size_pct": size})

    green_lane = positions.get("green_lane")
    if green_lane:
        size = green_lane.get("size_pct", 2.0)
        total_size_pct += size
        position_count += 1
        positions_summary.append({"source": "green_lane", "side": green_lane.get("side", "?"), "size_pct": size})

    total_leverage_est = 10.0 if total_size_pct > 0 else 0.0

    result = {
        "total_size_pct": round(total_size_pct, 2),
        "total_leverage_est": total_leverage_est,
        "position_count": position_count,
        "positions_summary": positions_summary,
    }
    log.info(f"📊 Combined exposure: {position_count} position(s), {total_size_pct:.1f}% margin")
    return result


def can_open_green_lane(state: dict, equity: float) -> tuple:
    """Check if a green lane position can be opened.
    Rules: max 1 green lane, combined size <= 10%, combined leverage <= 20x.
    Returns: (allowed: bool, reason: str)
    """
    positions = state.get("positions", {})
    if positions.get("green_lane") is not None:
        reason = "Already have an active green lane position"
        log.info(f"🚫 Cannot open green lane: {reason}")
        return False, reason

    exposure = get_combined_exposure(state)
    GREEN_LANE_SIZE_PCT = 2.0
    projected_size = exposure["total_size_pct"] + GREEN_LANE_SIZE_PCT
    if projected_size > 10.0:
        reason = f"Combined margin would be {projected_size:.1f}% (max 10%)"
        log.info(f"🚫 Cannot open green lane: {reason}")
        return False, reason

    committee = positions.get("committee")
    committee_notional = (committee.get("size_pct", DEFAULT_ALLOC_PCT * 100) * 10) if committee else 0.0
    projected_leverage = (committee_notional + GREEN_LANE_SIZE_PCT * 10) / max(projected_size, 0.01)
    if projected_leverage > 20.0:
        reason = f"Estimated combined leverage would be {projected_leverage:.1f}x (max 20x)"
        log.info(f"🚫 Cannot open green lane: {reason}")
        return False, reason

    log.info(f"✅ Green lane allowed: projected {projected_size:.1f}% margin, ~{projected_leverage:.1f}x lev")
    return True, "OK"


def open_green_lane_position(state: dict, signal, equity: float) -> dict:
    """Create a green lane position entry from a GreenLaneSignal. Default size: 2% margin."""
    GREEN_LANE_SIZE_PCT = 2.0
    if "positions" not in state:
        state["positions"] = {"committee": None, "green_lane": None}
    state["positions"]["green_lane"] = {
        "source": "green_lane",
        "side": signal.direction,
        "entry_price": signal.entry_price,
        "stop_loss": signal.stop_loss,
        "tp1": signal.tp1,
        "tp2": signal.tp2,
        "tp1_hit": False,
        "tp2_hit": False,
        "trail_stop": None,
        "trail_ema": signal.trail_ema,
        "size_pct": GREEN_LANE_SIZE_PCT,
        "opened_at": signal.timestamp,
        "quality_score": signal.quality_score,
        "pnl_pct": 0.0,
    }
    log.info(f"🟢 Opened green lane {signal.direction} @ ${signal.entry_price:,.2f} | SL ${signal.stop_loss:,.2f} | TP1 ${signal.tp1:,.2f} | TP2 ${signal.tp2:,.2f} | Quality {signal.quality_score}/10")
    return state


def check_green_lane_exits(state: dict, current_price: float, daily_ema9: float) -> list:
    """Check green lane exit conditions. Returns list of exit actions."""
    pos = state.get("positions", {}).get("green_lane")
    if not pos:
        return []
    actions = []
    side = pos.get("side", "long")
    entry = pos.get("entry_price", 0.0)
    stop_loss = pos.get("stop_loss", 0.0)
    tp1 = pos.get("tp1")
    tp2 = pos.get("tp2")
    tp1_hit = pos.get("tp1_hit", False)
    tp2_hit = pos.get("tp2_hit", False)
    trail_stop = pos.get("trail_stop")

    if side == "long":
        if current_price <= stop_loss:
            log.info(f"🔴 Green lane STOP LOSS hit: ${current_price:,.2f} <= ${stop_loss:,.2f}")
            return [{"type": "stop_loss", "price": current_price, "fraction": 1.0, "reason": "stop_loss"}]
        if tp1 and not tp1_hit and current_price >= tp1:
            log.info(f"🎯 Green lane TP1 hit: ${current_price:,.2f}")
            actions.append({"type": "tp1", "price": current_price, "fraction": 1/3, "reason": "tp1"})
            pos["tp1_hit"] = True
            pos["stop_loss"] = entry
            log.info(f"🛡️ Stop moved to breakeven: ${entry:,.2f}")
        if tp2 and not tp2_hit and current_price >= tp2:
            log.info(f"🎯 Green lane TP2 hit: ${current_price:,.2f}")
            actions.append({"type": "tp2", "price": current_price, "fraction": 1/3, "reason": "tp2"})
            pos["tp2_hit"] = True
        if tp1_hit and tp2_hit and current_price < daily_ema9:
            log.info(f"📉 Green lane trail stop hit: price ${current_price:,.2f} < EMA9 ${daily_ema9:,.2f}")
            actions.append({"type": "trail_stop", "price": current_price, "fraction": 1.0, "reason": "trail_ema9_break"})
        if trail_stop is None or daily_ema9 > trail_stop:
            pos["trail_stop"] = daily_ema9
    else:
        if current_price >= stop_loss:
            log.info(f"🔴 Green lane STOP LOSS hit (short): ${current_price:,.2f} >= ${stop_loss:,.2f}")
            return [{"type": "stop_loss", "price": current_price, "fraction": 1.0, "reason": "stop_loss"}]
        if tp1 and not tp1_hit and current_price <= tp1:
            log.info(f"🎯 Green lane TP1 hit (short): ${current_price:,.2f}")
            actions.append({"type": "tp1", "price": current_price, "fraction": 1/3, "reason": "tp1"})
            pos["tp1_hit"] = True
            pos["stop_loss"] = entry
        if tp2 and not tp2_hit and current_price <= tp2:
            log.info(f"🎯 Green lane TP2 hit (short): ${current_price:,.2f}")
            actions.append({"type": "tp2", "price": current_price, "fraction": 1/3, "reason": "tp2"})
            pos["tp2_hit"] = True
        if tp1_hit and tp2_hit and current_price > daily_ema9:
            log.info(f"📈 Green lane trail stop hit (short): price ${current_price:,.2f} > EMA9 ${daily_ema9:,.2f}")
            actions.append({"type": "trail_stop", "price": current_price, "fraction": 1.0, "reason": "trail_ema9_break"})
        if trail_stop is None or daily_ema9 < trail_stop:
            pos["trail_stop"] = daily_ema9

    if actions:
        log.info(f"📋 Green lane exit actions: {[a[chr(39)+'type'+chr(39)] for a in actions]}")
    return actions


def close_green_lane_position(state: dict, exit_price: float, reason: str) -> dict:
    """Close green lane position and move to closed_trades."""
    pos = state.get("positions", {}).get("green_lane")
    if not pos:
        log.warning("⚠️ close_green_lane_position called but no green lane position found")
        return state
    entry_price = pos.get("entry_price", exit_price)
    side = pos.get("side", "long")
    pnl_pct = ((exit_price - entry_price) / entry_price * 100) if side == "long" else ((entry_price - exit_price) / entry_price * 100)
    state.setdefault("closed_trades", []).append({
        **pos,
        "exit_price": exit_price,
        "exit_reason": reason,
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "pnl_pct": round(pnl_pct, 4),
    })
    state["positions"]["green_lane"] = None
    log.info(f"🔴 Closed green lane {side} | Entry ${entry_price:,.2f} → Exit ${exit_price:,.2f} | P&L {pnl_pct:+.2f}% | Reason: {reason}")
    return state


# === AGENTS ===

def create_agents_graph():
    """Create and return the TradingAgents graph with persistent memory."""
    config = CRYPTO_DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["deep_think_llm"] = "claude-opus-4-6"
    config["quick_think_llm"] = "claude-sonnet-4-20250514"

    ta = CryptoTradingAgentsGraph(
        selected_analysts=["market", "sentiment", "fundamentals", "news"],
        config=config,
        debug=False,
    )

    # Load persistent memories
    for mem_name, mem_obj in [
        ("bull", ta.bull_memory),
        ("bear", ta.bear_memory),
        ("trader", ta.trader_memory),
        ("invest_judge", ta.invest_judge_memory),
        ("risk_manager", ta.risk_manager_memory),
    ]:
        mem_path = str(MEMORY_DIR / f"{mem_name}_memory.json")
        if mem_obj.load(mem_path):
            log.info(f"🧠 Loaded {mem_name} memory ({len(mem_obj.documents)} entries)")

    return ta


def save_memories(ta):
    """Persist all agent memories to disk."""
    for mem_name, mem_obj in [
        ("bull", ta.bull_memory),
        ("bear", ta.bear_memory),
        ("trader", ta.trader_memory),
        ("invest_judge", ta.invest_judge_memory),
        ("risk_manager", ta.risk_manager_memory),
    ]:
        mem_path = str(MEMORY_DIR / f"{mem_name}_memory.json")
        mem_obj.save(mem_path)
    log.info("💾 Memories saved to disk")


def run_reflection(ta, returns_pct):
    """Run reflection loop — agents learn from trade outcomes."""
    if ta.curr_state is None:
        log.debug("No agent state in memory for reflection — loading last state from disk")
        # Try to load the last state from logs
        import glob
        state_files = sorted(glob.glob(str(PROJECT_ROOT / "eval_results" / "BTC_USDT" / "CryptoTradingAgents_logs" / "full_states_log_*.json")))
        if state_files:
            with open(state_files[-1]) as f:
                states = json.load(f)
            last_date = sorted(states.keys())[-1]
            last_state = states[last_date]
            # Reconstruct minimal state for reflection
            ta.curr_state = {
                "market_report": last_state.get("market_report", ""),
                "sentiment_report": last_state.get("sentiment_report", ""),
                "news_report": last_state.get("news_report", ""),
                "fundamentals_report": last_state.get("fundamentals_report", ""),
                "investment_debate_state": last_state.get("investment_debate_state", {}),
                "trader_investment_plan": last_state.get("trader_investment_decision", ""),
                "risk_debate_state": last_state.get("risk_debate_state", {}),
            }
        else:
            log.warning("⚠️ No state files found — skipping reflection")
            return

    log.info(f"🔄 Reflecting on trade outcome: {returns_pct:+.2f}%")
    ta.reflect_and_remember(f"{returns_pct:+.2f}% return on BTC/USDT position")
    save_memories(ta)
    log.info("✅ Reflection complete — agents learned from this trade")

    # Run review & optimization after reflection
    try:
        from tradingagents.review import TradeReviewer, PerformanceScorer, ParameterOptimizer
        reviewer = TradeReviewer(ta.quick_thinking_llm, str(PROJECT_ROOT))
        scorer = PerformanceScorer(str(PROJECT_ROOT))
        optimizer = ParameterOptimizer(str(PROJECT_ROOT))

        from zoneinfo import ZoneInfo
        trade_date = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d")
        agent_state = reviewer.load_state_for_date(trade_date)
        if agent_state:
            trade = {"pnl_pct": returns_pct, "side": "long" if returns_pct > 0 else "short", "status": "closed", "open_time": trade_date}
            review = reviewer.review_trade(trade, agent_state)
            scorer.record_trade(review)
            report = scorer.get_report()
            recommendations = optimizer.run_full_optimization([review], report)
            changes = recommendations.get("summary", {}).get("total_parameter_changes", 0)
            if changes > 0:
                log.info(f"🔧 {changes} optimization recommendations — review logs/optimization_recommendations.json")
            for agent in report.get("underperformers", []):
                log.warning(f"⚠️ Underperformer: {agent}")
            log.info("✅ Review & optimization complete")
        else:
            log.warning("⚠️ No agent state for review — skipping optimization")
    except Exception as e:
        log.error(f"Review/optimization failed (non-fatal): {e}")


def run_agents(ta, trade_date, portfolio_context=None):
    log.info(f"🧠 Running agents for {SPOT_SYMBOL} on {trade_date}...")

    # Inject portfolio context into the graph's initial state
    if portfolio_context:
        ta.portfolio_context = portfolio_context

    agent_state, decision = ta.propagate(SPOT_SYMBOL, trade_date)

    # Check if fund manager overrode the decision
    fund_parsed = agent_state.get("fund_manager_parsed", {})
    if fund_parsed:
        action = fund_parsed.get("action", "").upper()
        if action == "OPEN_LONG":
            decision = "BUY"
        elif action == "OPEN_SHORT":
            decision = "SELL"
        elif action in ("CLOSE", "HOLD"):
            decision = "HOLD"
        elif action in ("LONG", "SHORT", "NEUTRAL"):
            # TradingModeConfig standardized signals
            current_pos = (portfolio_context or {}).get("position", "none")
            if "long" in current_pos.lower():
                curr = "LONG"
            elif "short" in current_pos.lower():
                curr = "SHORT"
            else:
                curr = "NEUTRAL"
            transition = get_position_transition(curr, action)
            trans_action = transition["action"]
            if trans_action in ("OPEN_LONG", "REVERSE_TO_LONG"):
                decision = "BUY"
            elif trans_action in ("OPEN_SHORT", "REVERSE_TO_SHORT"):
                decision = "SELL"
            else:
                decision = "HOLD"
            log.info(f"📐 Position transition: {curr} → {action} = {trans_action}")
        log.info(f"🏛️ Fund Manager: {decision} (action={action})")
    
    # Fallback: extract from raw text if FM parsing failed
    if decision == "HOLD" and not fund_parsed:
        extracted = extract_recommendation(agent_state.get("final_trade_decision", ""), "trading")
        if extracted and extracted != "NEUTRAL":
            current_pos = (portfolio_context or {}).get("position", "none")
            if "long" in current_pos.lower():
                curr = "LONG"
            elif "short" in current_pos.lower():
                curr = "SHORT"
            else:
                curr = "NEUTRAL"
            transition = get_position_transition(curr, extracted)
            trans_action = transition["action"]
            if trans_action in ("OPEN_LONG", "REVERSE_TO_LONG"):
                decision = "BUY"
            elif trans_action in ("OPEN_SHORT", "REVERSE_TO_SHORT"):
                decision = "SELL"
            log.info(f"📐 Fallback extraction: {curr} → {extracted} = {trans_action} → {decision}")

    full_decision = agent_state.get("final_trade_decision", "")
    trade_params = parse_trade_params(full_decision)

    log.info(f"🎯 Decision: {decision}")
    log.info(f"📋 Parsed params: {trade_params}")

    # Save memories after each run (even without reflection, the propagate updates state)
    save_memories(ta)

    reports = {
        "market": agent_state.get("market_report", "")[:500],
        "sentiment": agent_state.get("sentiment_report", "")[:500],
        "news": agent_state.get("news_report", "")[:500],
        "final": full_decision[:2000],
        "decision": decision,
        "parsed_params": trade_params,
    }
    return decision, trade_params, reports


# === STATE ===

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
    else:
        state = {"last_run": None, "last_decision": None, "trades": [], "active_trade": None, "closed_trades": []}
    return migrate_state(state)


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# === MAIN ===

def main():
    log.info("=" * 60)
    log.info("🚀 TradingAgents Live Executor v2")
    log.info("=" * 60)

    exchange = get_exchange()
    equity = get_equity(exchange)
    log.info(f"💰 Equity: ${equity:,.2f}")

    if check_circuit_breaker(equity):
        return

    executor_state = load_state()

    if check_daily_loss_limit(executor_state, equity):
        return

    # Check cold streak — adjusts risk if on losing run
    cold_streak, risk_multiplier = get_cold_streak(executor_state)

    positions = get_positions(exchange)
    has_position = len(positions) > 0

    if has_position:
        for p in positions:
            pnl = float(p.get("unrealizedPnl", 0))
            log.info(f"📊 {p['side']} {p['contracts']} @ ${float(p['entryPrice']):,.0f} | PnL: ${pnl:,.2f}")

        # Manage trailing stop on existing position
        executor_state = manage_trailing_stop(exchange, positions, executor_state)
        positions = get_positions(exchange)  # Refresh after potential close
        has_position = len(positions) > 0
    else:
        log.info("📊 No open positions")

    # Create agents graph with persistent memory
    ta = create_agents_graph()

    # If a trade just closed, reflect on the outcome
    if executor_state.get("pending_reflection"):
        pr = executor_state.pop("pending_reflection")
        run_reflection(ta, pr["returns_pct"])

    # Build portfolio context for Fund Manager
    portfolio_ctx = {
        "position": "none",
        "pnl_pct": 0,
        "age_bars": 0,
        "equity": equity,
        "consecutive_same_direction": 0,
        "last_decision": executor_state.get("last_decision", "HOLD"),
    }
    if has_position and positions:
        p = positions[0]
        portfolio_ctx["position"] = p["side"]
        portfolio_ctx["pnl_pct"] = (float(p.get("unrealizedPnl", 0)) / equity) * 100
        active = executor_state.get("active_trade", {})
        if active and active.get("opened_at"):
            opened_dt = datetime.fromisoformat(active["opened_at"])
            hours = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 3600
            portfolio_ctx["age_bars"] = int(hours / 4)
    
    # Count consecutive same-direction signals
    recent_decisions = [t.get("decision") for t in executor_state.get("trades", [])[-5:]]
    if recent_decisions:
        last = recent_decisions[-1]
        count = 0
        for d in reversed(recent_decisions):
            if d == last:
                count += 1
            else:
                break
        portfolio_ctx["consecutive_same_direction"] = count

    # Run agents for new signal
    trade_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    decision, trade_params, reports = run_agents(ta, trade_date, portfolio_ctx)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "equity": equity,
        "params": trade_params,
    }

    # Map signal to position transition
    current_pos = "NEUTRAL"
    if has_position and positions:
        current_pos = positions[0]["side"].upper()  # "LONG" or "SHORT"
    
    # Normalize decision to trading mode
    signal = decision.upper()
    if signal == "BUY":
        signal = "LONG"
    elif signal == "SELL":
        signal = "SHORT"
    elif signal == "HOLD":
        signal = "NEUTRAL" if not has_position else current_pos

    transition = get_position_transition(current_pos, signal)
    action = transition["action"]
    log.info(f"📋 Position transition: {current_pos} + {signal} → {action} ({transition['description']})")
    record["transition"] = action

    if action in ("HOLD", "STAY_NEUTRAL"):
        log.info(f"⏸️  {action} — no action")

    elif action in ("CLOSE_LONG", "CLOSE_SHORT"):
        # Close without reversing
        for p in positions:
            entry_price = float(p["entryPrice"])
            close_position(exchange, p)
            exit_price = float(exchange.fetch_ticker(SYMBOL)["last"])
            if p["side"] == "long":
                returns_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                returns_pct = ((entry_price - exit_price) / entry_price) * 100
            log.info(f"📊 Closed P&L: {returns_pct:+.2f}%")
            executor_state["pending_reflection"] = {
                "returns_pct": returns_pct, "direction": p["side"],
                "entry": entry_price, "exit": exit_price,
            }
            record["closed"] = p["side"]
            record["close_pnl_pct"] = returns_pct
        has_position = False
        executor_state.pop("active_trade", None)

    elif action in ("OPEN_LONG", "OPEN_SHORT", "REVERSE_TO_LONG", "REVERSE_TO_SHORT"):
        # Flip-flop protection: require higher confidence to reverse within 12 hours
        if action in ("REVERSE_TO_LONG", "REVERSE_TO_SHORT"):
            active = executor_state.get("active_trade", {})
            opened_at = active.get("opened_at")
            if opened_at:
                try:
                    opened_dt = datetime.fromisoformat(opened_at)
                    hours_held = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 3600
                    confidence = trade_params.get("confidence", 5)
                    if hours_held < 12 and confidence < 8:
                        log.warning(f"⚠️ FLIP-FLOP BLOCKED: Position held only {hours_held:.1f}h, confidence {confidence}/10 < 8 required for reversal within 12h")
                        log.info(f"⏸️  HOLD — reversal blocked by flip-flop protection")
                        record["transition"] = "HOLD"
                        record["flip_flop_blocked"] = True
                        save_state(executor_state)
                        save_report(record)
                        return
                except (ValueError, TypeError):
                    pass
        # Close existing position first if reversing
        if action.startswith("REVERSE") and has_position:
            for p in positions:
                entry_price = float(p["entryPrice"])
                close_position(exchange, p)
                exit_price = float(exchange.fetch_ticker(SYMBOL)["last"])
                if p["side"] == "long":
                    returns_pct = ((exit_price - entry_price) / entry_price) * 100
                else:
                    returns_pct = ((entry_price - exit_price) / entry_price) * 100
                log.info(f"📊 Closed for reversal P&L: {returns_pct:+.2f}%")
                executor_state["pending_reflection"] = {
                    "returns_pct": returns_pct, "direction": p["side"],
                    "entry": entry_price, "exit": exit_price,
                }
                record["closed"] = p["side"]
                record["close_pnl_pct"] = returns_pct
            has_position = False
            executor_state.pop("active_trade", None)

        # Determine direction
        new_direction = "BUY" if "LONG" in action else "SELL"
        has_position = False  # ensure we open

    # Open new position if we don't have one
    if not has_position and action in ("OPEN_LONG", "OPEN_SHORT", "REVERSE_TO_LONG", "REVERSE_TO_SHORT"):
            alloc_pct = trade_params.get("risk_pct", DEFAULT_ALLOC_PCT)  # agent picks allocation
            stop_loss = trade_params.get("stop_loss")
            tp1 = trade_params.get("take_profit_1")

            # If no SL from agent, calculate from ATR
            if not stop_loss:
                atr = get_atr(exchange, period=14, timeframe="1d")
                ticker = exchange.fetch_ticker(SYMBOL)
                price = ticker["last"]
                if new_direction == "BUY":
                    stop_loss = price - (2 * atr)
                else:
                    stop_loss = price + (2 * atr)
                log.info(f"🔧 Auto SL from 2×daily ATR: ${stop_loss:,.0f} (ATR=${atr:,.0f})")

            # If no TP from agent, calculate at 1.5R
            result = open_position(exchange, new_direction, equity, alloc_pct, stop_loss, risk_multiplier=risk_multiplier)

            if result:
                fill_price = result["fill_price"]
                amount = result["amount"]


                # Sanity check: TP/SL values must be within 50% of current price
                # Catches parsing errors like $78 instead of $78,000
                for param_name, param_val in [('take_profit_1', tp1), ('stop_loss', stop_loss)]:
                    if param_val and abs(param_val - fill_price) / fill_price > 0.50:
                        log.warning(f'⚠️ {param_name} ${param_val:,.0f} is >50% from price ${fill_price:,.0f} — ignoring (will auto-calculate)')
                        if param_name == 'take_profit_1':
                            tp1 = None
                        elif param_name == 'stop_loss':
                            stop_loss = None
                            atr = get_atr(exchange, period=14, timeframe='1d')
                            if new_direction == 'BUY':
                                stop_loss = fill_price - (2 * atr)
                            else:
                                stop_loss = fill_price + (2 * atr)
                            log.info(f'🔧 Recalculated SL from 2×ATR: ${stop_loss:,.0f}')

                risk_dist = abs(fill_price - stop_loss)
                if not tp1:
                    if new_direction == "BUY":
                        tp1 = fill_price + (TP1_R_MULTIPLE * risk_dist)
                    else:
                        tp1 = fill_price - (TP1_R_MULTIPLE * risk_dist)
                    log.info(f"🔧 Auto TP1 at {TP1_R_MULTIPLE}R: ${tp1:,.0f}")

                # Set TP1 on exchange (close 50%)
                set_take_profit(exchange, new_direction, amount, tp1)

                # Initialize trailing stop tracking
                atr = get_atr(exchange)
                if new_direction == "BUY":
                    initial_trail = fill_price - (ATR_TRAIL_MULTIPLE * atr)
                else:
                    initial_trail = fill_price + (ATR_TRAIL_MULTIPLE * atr)

                executor_state["active_trade"] = {
                    "direction": new_direction,
                    "entry": fill_price,
                    "amount": amount,
                    "stop_loss": stop_loss,
                    "tp1": tp1,
                    "tp2": trade_params.get("take_profit_2"),
                    "trailing_stop": initial_trail,
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                    "confidence": trade_params.get("confidence"),
                    "atr_at_entry": atr,
                }

                record["action"] = f"opened_{new_direction.lower()}"
                record["price"] = fill_price
                record["stop_loss"] = stop_loss
                record["tp1"] = tp1
                record["amount"] = amount
                record["alloc_pct"] = alloc_pct

                log.info(f"📋 Trade plan: Entry ${fill_price:,.0f} | SL ${stop_loss:,.0f} | TP1 ${tp1:,.0f} | Alloc {alloc_pct*100:.1f}% | Risk 1% | Trail {ATR_TRAIL_MULTIPLE}×ATR")
            else:
                record["action"] = "failed"

    executor_state["last_run"] = record["timestamp"]
    executor_state["last_decision"] = decision
    # Save PM thesis for next run's context
    if reports.get("final"):
        import re as _re
        thesis_match = _re.search(r'THESIS:\s*(.+)', reports["final"])
        if thesis_match:
            executor_state["last_decision_reasoning"] = thesis_match.group(1).strip()
        else:
            executor_state["last_decision_reasoning"] = reports["final"][:500]
    executor_state["last_decision_time"] = record["timestamp"]
    executor_state["trades"] = executor_state.get("trades", [])
    executor_state["trades"].append(record)
    save_state(executor_state)

    with open(LOG_DIR / f"agent_reports_{trade_date}.json", "w") as f:
        json.dump(reports, f, indent=2)

    log.info("✅ Done")


if __name__ == "__main__":
    main()

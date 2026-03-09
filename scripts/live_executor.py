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
import signal
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from tradingagents.graph.crypto_trading_graph import CryptoTradingAgentsGraph
from tradingagents.scoring.objective_score import calculate_objective_score
from tradingagents.scoring.geopolitical import detect_geopolitical_events
from tradingagents.scoring.decision_validator import validate_decision
from tradingagents.scoring.outcome_tracker import OutcomeTracker
from tradingagents.scoring.risk_manager import RiskManager
from tradingagents.scoring.stability import compute_system_stability
from tradingagents.scheduling.analyst_schedule import AnalystScheduler
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
# Load peak equity from state file (was hardcoded — never updated across restarts)
def _load_peak_equity():
    try:
        import json
        state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
        return state.get('peak_equity', 154_425)
    except Exception:
        return 154_425
PEAK_EQUITY = _load_peak_equity()
DAILY_LOSS_LIMIT = 0.02       # 2% daily loss limit — lock out after this
COLD_STREAK_THRESHOLD = 3     # After 3 consecutive losses, halve risk
COLD_STREAK_RISK_MULT = 0.5   # Multiply risk by this during cold streak
COLD_STREAK_COOLDOWN_HOURS = 24  # Mandatory rest after 3 consecutive stop-losses
# Progressive risk after consecutive stops (比特皇 style):
# After 1st stop: base risk, after 2nd stop: base * 1.07, after 3rd: COOLDOWN
COLD_STREAK_RISK_PROGRESSION = {1: 1.0, 2: 1.07}
TP1_R_MULTIPLE = 1.5          # TP1 at 1.5R
TP1_CLOSE_PCT = 0.5           # Close 50% at TP1
ATR_TRAIL_MULTIPLE = 2.0      # Trail remaining with 2×ATR
TIME_EXIT_BARS = 24           # Close if flat after 24 4H bars

LOG_DIR = PROJECT_ROOT / "logs"
MEMORY_DIR = PROJECT_ROOT / "memory"
STATE_FILE = LOG_DIR / "executor_state.json"

# Module-level singletons
outcome_tracker = OutcomeTracker()

# === GRACEFUL SHUTDOWN ===
_shutdown_requested = False
_shutdown_state = {}  # Stash partial state for emergency save


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT — ONLY set flag and log. No I/O inside signal handlers.

    Writing files or calling sys.exit() inside a signal handler is dangerous — the
    signal may fire mid-write and corrupt the state file, or interrupt another syscall
    in an unrecoverable way.  The main loop polls _shutdown_requested at safe
    checkpoints and performs the actual save + exit there.
    """
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    # Use low-level write — avoids re-entering the logging machinery (which may hold locks)
    import sys as _sys
    _sys.stderr.write(f"\n⚡ Signal {sig_name} received — shutdown flag set\n")
    _sys.stderr.flush()
    _shutdown_requested = True
    # Do NOT call save_state() or sys.exit() here.
risk_mgr = RiskManager({
    "max_position_pct": MAX_ALLOC_PCT,
    "max_leverage": 20,
    "max_daily_loss_pct": 0.03,
    "max_drawdown_pct": 0.10,
    "cooldown_minutes": 30,
    "max_trades_per_day": 6,
})
analyst_scheduler = AnalystScheduler({
    "market": 4,
    "sentiment": 8,
    "news": 12,
    "fundamentals": 24,
})
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


def update_mfe_from_candles(exchange, trade_info):
    """Update max favorable excursion using actual candle highs/lows since entry.
    
    The executor only runs every 4H, so spot-price MFE misses intra-candle wicks.
    This fetches 4H candles since entry and computes MFE from actual extremes.
    """
    opened_at = trade_info.get("opened_at", "")
    entry = trade_info.get("entry", 0)
    direction = trade_info.get("direction", "")
    if not opened_at or not entry or not direction:
        return

    try:
        opened_dt = datetime.fromisoformat(opened_at)
        if opened_dt.tzinfo is None:
            opened_dt = opened_dt.replace(tzinfo=timezone.utc)
        # Fetch one candle before entry to cover partial candle containing entry
        since_ms = int((opened_dt - timedelta(hours=4)).timestamp() * 1000)
        # Dynamic limit based on trade age (4H candles)
        hours_since = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 3600
        limit = max(10, min(200, int(hours_since / 4) + 2))
        candles = exchange.fetch_ohlcv(SYMBOL, "4h", since=since_ms, limit=limit)
        if not candles:
            return

        side_lower = direction.lower()
        if side_lower in ("sell", "short"):
            # For shorts, best favorable = lowest low
            best_price = min(c[3] for c in candles)  # c[3] = low
            favorable_pct = ((entry - best_price) / entry) * 100
        else:
            # For longs, best favorable = highest high
            best_price = max(c[2] for c in candles)  # c[2] = high
            favorable_pct = ((best_price - entry) / entry) * 100

        prev_mfe = trade_info.get("max_favorable_pct", 0)
        if favorable_pct > prev_mfe:
            trade_info["max_favorable_pct"] = round(favorable_pct, 2)
            trade_info["mfe_price"] = best_price
            log.info(f"📊 MFE updated from candles: {prev_mfe:.2f}% → {favorable_pct:.2f}% "
                     f"(best price ${best_price:,.0f})")
    except Exception as e:
        log.warning(f"⚠️ Failed to update MFE from candles: {e}")


def check_daily_loss_limit(executor_state, equity):
    """Check if daily losses exceed limit. Returns True if locked out."""
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d")
    trades_today = []
    for t in executor_state.get("trades", []):
        if "close_pnl_pct" not in t:
            continue
        ts = t.get("timestamp", "")
        try:
            trade_dt = datetime.fromisoformat(ts).astimezone(ZoneInfo("Australia/Sydney"))
            if trade_dt.strftime("%Y-%m-%d") == today:
                trades_today.append(t)
        except (ValueError, TypeError):
            if ts.startswith(today):
                trades_today.append(t)
    
    daily_loss = sum(t["close_pnl_pct"] for t in trades_today if t["close_pnl_pct"] < 0)
    daily_loss_abs = abs(daily_loss) / 100  # Convert from % to fraction
    
    if daily_loss_abs >= DAILY_LOSS_LIMIT:
        log.error(f"🔒 DAILY LOSS LIMIT: {daily_loss:.2f}% today (limit: {DAILY_LOSS_LIMIT*100:.0f}%). Locked out until next session.")
    
    if trades_today:
        log.info(f"📊 Daily P&L: {daily_loss:+.2f}% | Limit: {DAILY_LOSS_LIMIT*100:.0f}%")
    return daily_loss_abs  # Numeric fraction (0.0 to 1.0)


def get_cold_streak(executor_state):
    """Count consecutive losses. Returns streak count and risk multiplier.

    Progressive risk (比特皇 style):
      - 1 consecutive loss: base risk (1×)
      - 2 consecutive losses: base risk × 1.07
      - 3+ consecutive losses: trigger mandatory cooldown, risk halved as secondary guard
    """
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
        # Only trigger NEW cooldown if we haven't already cooled down for this streak.
        # Track via last_cooldown_trade_count to prevent perma-lock loop:
        # after cooldown expires, streak is still 3+ (no new trades), so without
        # this guard it would immediately re-trigger.
        trade_count = len(executor_state.get("trades", []))
        last_trigger = executor_state.get("last_cooldown_trade_count", -1)
        cooldown_until = executor_state.get("cooldown_until")

        already_cooling = False
        if cooldown_until:
            try:
                resume_at_dt = datetime.fromisoformat(cooldown_until)
                if resume_at_dt.tzinfo is None:
                    resume_at_dt = resume_at_dt.replace(tzinfo=timezone.utc)
                already_cooling = resume_at_dt > datetime.now(timezone.utc)
            except (ValueError, TypeError):
                pass

        if not already_cooling and trade_count != last_trigger:
            resume_at = datetime.now(timezone.utc) + timedelta(hours=COLD_STREAK_COOLDOWN_HOURS)
            executor_state["cooldown_until"] = resume_at.isoformat()
            executor_state["last_cooldown_trade_count"] = trade_count
            log.error(f"🛑 MANDATORY REST: {streak} consecutive losses — cooldown until {resume_at:%Y-%m-%d %H:%M} UTC")

        log.warning(f"🥶 COLD STREAK: {streak} consecutive losses — risk halved to {RISK_PER_TRADE * COLD_STREAK_RISK_MULT * 100:.1f}%")
        return streak, COLD_STREAK_RISK_MULT

    # Progressive risk for shorter streaks
    if streak in COLD_STREAK_RISK_PROGRESSION:
        mult = COLD_STREAK_RISK_PROGRESSION[streak]
        if mult != 1.0:
            log.info(f"📉 {streak} consecutive loss(es) — risk adjusted to {RISK_PER_TRADE * mult * 100:.2f}%")
        return streak, mult

    return streak, 1.0


def check_cooldown_active(executor_state):
    """Check if mandatory cooldown is active. Returns True if trading should be skipped."""
    cooldown_until = executor_state.get("cooldown_until")
    if not cooldown_until:
        return False
    try:
        resume_at = datetime.fromisoformat(cooldown_until)
        if resume_at.tzinfo is None:
            resume_at = resume_at.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return False

    now = datetime.now(timezone.utc)
    if now < resume_at:
        remaining = resume_at - now
        hours_left = remaining.total_seconds() / 3600
        log.warning(f"⏸️ COOLDOWN ACTIVE: Skipping cycle, resuming at {resume_at:%Y-%m-%d %H:%M} UTC ({hours_left:.1f}h remaining)")
        return True

    # Cooldown expired — clear it and reset
    log.info(f"✅ Cooldown expired — resuming trading, risk reset to base")
    executor_state.pop("cooldown_until", None)
    return False


def check_circuit_breaker(equity):
    global PEAK_EQUITY
    if equity > PEAK_EQUITY:
        log.info(f"📈 New peak equity: ${equity:,.0f} (was ${PEAK_EQUITY:,.0f})")
        PEAK_EQUITY = equity
        # Persist to state file so it survives restarts
        try:
            import json
            state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
            state['peak_equity'] = equity
            STATE_FILE.write_text(json.dumps(state, indent=2))
        except Exception as e:
            log.warning(f"Failed to persist peak equity: {e}")
    threshold = PEAK_EQUITY * (1 - CIRCUIT_BREAKER_PCT)
    if equity < threshold:
        log.error(f"🚨 CIRCUIT BREAKER: ${equity:,.0f} < ${threshold:,.0f} (peak ${PEAK_EQUITY:,.0f})")
        return True
    return False


# === SIGNAL PARSING ===

def _safe_float(s: str) -> float | None:
    """Safely convert a string to float, returning None on failure."""
    if not s:
        return None
    cleaned = s.replace(",", "").replace("$", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def parse_trade_params(full_decision: str) -> dict:
    """Extract structured trade params. Tries ---TRADE_PLAN--- block first, falls back to regex."""
    params = {}

    # Try structured block first (from trader agent)
    block_match = re.search(r'---(?:TRADE_PLAN|FUND_DECISION)---(.*?)---END_(?:TRADE_PLAN|FUND_DECISION)---', full_decision, re.DOTALL)
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
    # Strip markdown bold/italic markers that interfere with regex
    text = re.sub(r'\*{1,2}', '', full_decision).lower()

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
        r'stop.?loss.*?\$?([\d,]{4,}(?:\.\d+)?)',
    ]
    for pattern in sl_patterns:
        m = re.search(pattern, text)
        if m:
            val = _safe_float(m.group(1))
            if val is not None:
                params["stop_loss"] = val
            break

    tp_patterns = [
        r'(?:take[- ]?profit|tp)\s*1[:\s]*\$?([\d,]{4,}(?:\.\d+)?)',
        r'(?:take[- ]?profit|tp)[:\s]*\$?([\d,]{4,}(?:\.\d+)?)',
        r'(?:target|tp)\s*1?[:\s]*\$?([\d,]{4,}(?:\.\d+)?)',
    ]
    for pattern in tp_patterns:
        m = re.search(pattern, text)
        if m:
            val = _safe_float(m.group(1))
            if val is not None:
                params["take_profit_1"] = val
            break

    tp2_patterns = [
        r'(?:take[- ]?profit|tp)\s*2[:\s]*\$?([\d,]{4,}(?:\.\d+)?)',
        r'(?:target|tp)\s*2[:\s]*\$?([\d,]{4,}(?:\.\d+)?)',
    ]
    for pattern in tp2_patterns:
        m = re.search(pattern, text)
        if m:
            val = _safe_float(m.group(1))
            if val is not None:
                params["take_profit_2"] = val
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

    # Entry price and type (limit vs market)
    entry_patterns = [
        r'entry[:\s]*(?:limit\s*(?:at|@)\s*)\$?([\d,]+(?:\.\d+)?)',
        r'entry[:\s]*\$?([\d,]+(?:\.\d+)?)',
        r'enter\s*(?:at|@)\s*\$?([\d,]+(?:\.\d+)?)',
        r'limit\s*(?:entry\s*)?(?:at|@)\s*\$?([\d,]+(?:\.\d+)?)',
    ]
    for pattern in entry_patterns:
        m = re.search(pattern, text)
        if m:
            val = _safe_float(m.group(1))
            if val is not None:
                params["entry_price"] = val
            break

    # Detect if entry is limit or market
    if re.search(r'limit\s*(?:at|@|entry|order)', text):
        params["entry_type"] = "limit"
    elif re.search(r'market\s*(?:entry|order)|enter\s*(?:at\s*)?market', text):
        params["entry_type"] = "market"

    # Margin allocation (e.g. "3-4% margin" or "3.5% margin")
    margin_patterns = [
        r'margin\s*allocation[:\s]*(\d+(?:\.\d+)?)\s*%',
        r'(\d+(?:\.\d+)?)\s*%\s*margin',
        r'(\d+)-(\d+)%\s*margin',
    ]
    for pattern in margin_patterns:
        m = re.search(pattern, text)
        if m:
            if m.lastindex == 2:
                # Range like "3-4%" — use midpoint
                pct = (float(m.group(1)) + float(m.group(2))) / 2 / 100
            else:
                pct = float(m.group(1)) / 100
            if 0.005 <= pct <= MAX_ALLOC_PCT:
                params["risk_pct"] = params.get("risk_pct", pct)
            break

    return params


# === POSITION MANAGEMENT ===


def open_limit_order(exchange, direction, equity, alloc_pct, limit_price, stop_loss=None, take_profit=None, risk_multiplier=1.0):
    """Place a limit order on exchange. Returns order info dict or None."""
    atr = get_atr(exchange, period=14, timeframe="1d")
    sl_distance = abs(limit_price - stop_loss) if stop_loss else 2 * atr
    effective_risk = RISK_PER_TRADE * risk_multiplier
    risk_dollars = equity * effective_risk
    notional = (risk_dollars / sl_distance) * limit_price
    amount = float(exchange.amount_to_precision(SYMBOL, notional / limit_price))

    if amount <= 0:
        log.error(f"❌ Limit order amount=0. Equity=${equity:.0f}, Price=${limit_price:.0f}")
        return None

    alloc_pct = min(alloc_pct, MAX_ALLOC_PCT)
    margin = equity * alloc_pct
    leverage = max(1, round(notional / margin))
    leverage = min(leverage, 20)

    try:
        exchange.set_leverage(leverage, SYMBOL)
    except Exception as e:
        log.warning(f"Leverage: {e}")

    side = "buy" if direction == "BUY" else "sell"
    link_id = f"limit_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    
    log.info(f"📋 LIMIT {direction} {amount:.3f} BTC @ ${limit_price:,.0f} | ${notional:,.0f} notional | {leverage}x lev")
    
    # Attach native SL/TP so position is protected from the moment it fills
    order_params = {"orderLinkId": link_id, "timeInForce": "GTC"}
    if stop_loss:
        order_params["stopLoss"] = str(round(stop_loss, 2))
        order_params["slTriggerBy"] = "MarkPrice"
    if take_profit:
        order_params["takeProfit"] = str(round(take_profit, 2))
        order_params["tpTriggerBy"] = "MarkPrice"
    
    order = exchange.create_order(
        SYMBOL, "limit", side, amount, limit_price,
        params=order_params
    )
    order_id = order["id"]
    log.info(f"✅ Limit order placed: {order_id} @ ${limit_price:,.2f}")

    return {
        "order_id": order_id,
        "link_id": link_id,
        "direction": direction,
        "amount": amount,
        "limit_price": limit_price,
        "stop_loss": stop_loss,
        "leverage": leverage,
    }


def check_pending_limit_order(exchange, executor_state):
    """Check if a pending limit order has been filled. Returns True if filled and state updated."""
    pending = executor_state.get("pending_limit_order")
    if not pending:
        return False

    order_id = pending.get("order_id")
    if not order_id:
        executor_state.pop("pending_limit_order", None)
        return False

    try:
        order = exchange.fetch_order(order_id, SYMBOL)
    except Exception as e:
        log.warning(f"\u26a0\ufe0f Could not fetch pending order {order_id}: {e}")
        return False

    status = order.get("status", "").lower()
    created_at = pending.get("created_at", "")
    
    # Check age — cancel if older than 24h
    if created_at:
        try:
            age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(created_at)).total_seconds() / 3600
            if age_h > 24 and status in ("open", "new"):
                log.info(f"\u23f3 Cancelling expired limit order {order_id} ({age_h:.0f}h old)")
                # Check for partial fills before cancelling
                filled_qty = float(order.get("filled", 0))
                try:
                    exchange.cancel_order(order_id, SYMBOL)
                except Exception as e:
                    log.warning(f"Failed to cancel order: {e}")
                if filled_qty > 0:
                    log.warning(f"\u26a0\ufe0f Expired order had partial fill: {filled_qty:.3f} BTC — will be picked up as active position")
                executor_state.pop("pending_limit_order", None)
                save_state(executor_state)
                return False
        except (ValueError, TypeError):
            pass

    if status == "closed":  # Filled
        fill_price = float(order.get("average") or pending["limit_price"])
        amount = float(order.get("filled") or pending["amount"])
        direction = pending["direction"]
        stop_loss = pending.get("stop_loss")
        tp1 = pending.get("tp1")
        
        log.info(f"\u2705 LIMIT ORDER FILLED: {direction} {amount:.3f} BTC @ ${fill_price:,.2f}")
        
        # Set SL on exchange
        if stop_loss:
            sync_stop_loss_to_exchange(exchange, direction.lower(), amount, stop_loss)
        
        # Set TP
        risk_dist = abs(fill_price - stop_loss) if stop_loss else get_atr(exchange) * 2
        if not tp1:
            if direction == "BUY":
                tp1 = fill_price + (TP1_R_MULTIPLE * risk_dist)
            else:
                tp1 = fill_price - (TP1_R_MULTIPLE * risk_dist)
            log.info(f"\U0001f527 Auto TP1 at {TP1_R_MULTIPLE}R: ${tp1:,.0f}")
        tp2 = pending.get("tp2")
        # Place protection orders — wrapped individually so partial failure is logged
        tp_orders = []
        protection_failed = False
        try:
            tp_orders = set_take_profits(exchange, direction, amount, tp1, tp2)
        except Exception as e:
            log.error(f"🚨 FILL HANDLER: set_take_profits failed — position unprotected! {e}")
            protection_failed = True

        atr = get_atr(exchange)
        try:
            set_native_trailing_stop(exchange, direction, fill_price, atr)
        except Exception as e:
            log.error(f"🚨 FILL HANDLER: set_native_trailing_stop failed: {e}")
            protection_failed = True

        if direction == "BUY":
            initial_trail = fill_price - (ATR_TRAIL_MULTIPLE * atr)
        else:
            initial_trail = fill_price + (ATR_TRAIL_MULTIPLE * atr)

        # Persist active_trade AFTER protection orders are placed (even if some failed)
        executor_state["active_trade"] = {
            "direction": direction,
            "entry": fill_price,
            "amount": amount,
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "initial_tp1": tp1,  # Immutable: for TP hard cap
            "initial_tp2": tp2,  # Immutable: for TP hard cap
            "tp_orders": tp_orders,
            "tp1_hit": False,
            "trailing_stop": initial_trail,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "confidence": pending.get("confidence"),
            "atr_at_entry": atr,
            "from_limit_order": True,
            "needs_protection_retry": protection_failed,
        }
        executor_state.pop("pending_limit_order", None)
        save_state(executor_state)
        return True
    
    elif status in ("canceled", "cancelled", "expired", "rejected"):
        filled_qty = float(order.get("filled", 0))
        if filled_qty > 0:
            # Partial fill — we have a live position that needs management!
            fill_price = float(order.get("average") or pending["limit_price"])
            direction = pending["direction"]
            stop_loss = pending.get("stop_loss")
            tp1 = pending.get("tp1")
            log.warning(f"\u26a0\ufe0f Partial fill on {status} order: {filled_qty:.3f} BTC @ ${fill_price:,.2f} — creating active_trade")
            
            atr = get_atr(exchange)
            protection_failed = False
            try:
                set_native_trailing_stop(exchange, direction, fill_price, atr)
            except Exception as e:
                log.error(f"🚨 PARTIAL FILL: set_native_trailing_stop failed: {e}")
                protection_failed = True

            if direction == "BUY":
                initial_trail = fill_price - (ATR_TRAIL_MULTIPLE * atr)
            else:
                initial_trail = fill_price + (ATR_TRAIL_MULTIPLE * atr)

            tp2 = pending.get("tp2")
            tp_orders = []
            try:
                tp_orders = set_take_profits(exchange, direction, filled_qty, tp1, tp2) if tp1 else []
            except Exception as e:
                log.error(f"🚨 PARTIAL FILL: set_take_profits failed — position unprotected! {e}")
                protection_failed = True

            executor_state["active_trade"] = {
                "direction": direction,
                "entry": fill_price,
                "amount": filled_qty,
                "stop_loss": stop_loss,
                "tp1": tp1,
                "tp2": tp2,
                "tp_orders": tp_orders,
                "tp1_hit": False,
                "trailing_stop": initial_trail,
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "confidence": pending.get("confidence"),
                "atr_at_entry": atr,
                "from_limit_order": True,
                "partial_fill": True,
                "needs_protection_retry": protection_failed,
            }
            # Ensure SL on exchange for the filled portion
            if stop_loss:
                try:
                    sync_stop_loss_to_exchange(exchange, direction.lower(), filled_qty, stop_loss)
                except Exception as e:
                    log.error(f"🚨 PARTIAL FILL: sync_stop_loss failed: {e}")
                    protection_failed = True
                    executor_state["active_trade"]["needs_protection_retry"] = True
        else:
            log.info(f"\u274c Limit order {order_id} was {status} (no fills)")
        executor_state.pop("pending_limit_order", None)
        save_state(executor_state)
        return filled_qty > 0
    
    else:  # Still open
        fill_pct = float(order.get("filled", 0)) / float(order.get("amount", 1)) * 100
        log.info(f"\u23f8\ufe0f  Limit order {order_id} still open ({status}, {fill_pct:.0f}% filled) @ ${pending['limit_price']:,.0f}")
        return False


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


_fear_greed_cache: dict = {"value": None, "fetched_at": None}  # 5-min in-process cache


def fetch_fear_greed_value() -> int | None:
    """Fetch current Fear & Greed Index value (0-100).

    Returns cached value if fetched within the last 5 minutes to reduce
    external API load.  Returns None (and logs WARNING) on any failure.
    """
    import requests

    # Return cached value if fresh (< 5 minutes old)
    cache_ttl = 300  # seconds
    cached_at = _fear_greed_cache.get("fetched_at")
    if cached_at is not None:
        age = (datetime.now(timezone.utc) - cached_at).total_seconds()
        if age < cache_ttl and _fear_greed_cache["value"] is not None:
            log.debug(f"Fear & Greed (cached, {age:.0f}s old): {_fear_greed_cache['value']}")
            return _fear_greed_cache["value"]

    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        resp.raise_for_status()  # Raise on 4xx / 5xx HTTP errors

        data = resp.json()

        # Validate expected JSON shape before indexing
        if not isinstance(data, dict) or "data" not in data:
            log.warning(f"Fear & Greed: unexpected JSON shape (missing 'data' key): {str(data)[:200]}")
            return None
        entries = data["data"]
        if not isinstance(entries, list) or len(entries) == 0:
            log.warning(f"Fear & Greed: 'data' list is empty or not a list")
            return None
        entry = entries[0]
        if "value" not in entry:
            log.warning(f"Fear & Greed: missing 'value' in first entry: {entry}")
            return None

        value = int(entry["value"])
        _fear_greed_cache["value"] = value
        _fear_greed_cache["fetched_at"] = datetime.now(timezone.utc)
        log.debug(f"Fear & Greed fetched: {value}")
        return value

    except Exception as e:
        log.warning(f"Fear & Greed fetch failed: {e}")
        return None


def open_position(exchange, direction, equity, alloc_pct, stop_loss=None, risk_multiplier=1.0):
    """Open position sized by 1% risk target and daily ATR. Leverage auto-derived."""
    ticker = exchange.fetch_ticker(SYMBOL)
    price = ticker["last"]

    # Mechanical sizing: risk exactly RISK_PER_TRADE of equity
    atr = get_atr(exchange, period=14, timeframe="1d")
    # Use actual SL distance when provided, fallback to 2×ATR
    if stop_loss:
        sl_distance = abs(price - stop_loss)
        if sl_distance < 0.001 * price:  # SL too close (<0.1%), use ATR
            log.warning(f"⚠️ SL ${stop_loss:,.0f} too close to price ${price:,.0f} — using 2×ATR")
            sl_distance = 2 * atr
    else:
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

    # Set stop loss on position (native TP/SL, not separate order)
    if stop_loss and stop_loss > 0:
        try:
            bybit_symbol = SYMBOL.replace("/", "").replace(":USDT", "")
            exchange.private_post_v5_position_trading_stop({
                "category": "linear",
                "symbol": bybit_symbol,
                "stopLoss": str(round(stop_loss, 2)),
                "slTriggerBy": "MarkPrice",
                "positionIdx": 0,
            })
            log.info(f"🛡️ Stop loss set @ ${stop_loss:,.2f} (position TP/SL)")
        except Exception as e:
            log.warning(f"⚠️ Failed to set SL on position: {e}")

    return {"order": order, "fill_price": fill_price, "amount": amount}


def set_take_profits(exchange, direction, amount, tp1_price, tp2_price=None):
    """Place multiple TP conditional orders on entry.

    TP1: closes TP1_CLOSE_PCT (50%) of position
    TP2: closes remaining (50%) if provided
    Uses conditional market orders with reduceOnly=True.

    Returns list of order dicts: [{"level": 1, "order_id": ..., "price": ..., "amount": ...}, ...]
    """
    dir_lower = str(direction).lower()
    is_short = dir_lower in ("sell", "short")
    close_side = "buy" if is_short else "sell"
    # For shorts, TP triggers when price falls → triggerDirection=2
    # For longs, TP triggers when price rises → triggerDirection=1
    trigger_dir = 2 if is_short else 1
    tp_orders = []

    # TP1 — partial close
    tp1_amount = float(exchange.amount_to_precision(SYMBOL, amount * TP1_CLOSE_PCT))
    try:
        order = exchange.create_order(
            SYMBOL, "market", close_side, tp1_amount,
            params={
                "reduceOnly": True,
                "triggerPrice": str(round(tp1_price, 2)),
                "triggerBy": "MarkPrice",
                "triggerDirection": trigger_dir,
            }
        )
        tp_orders.append({"level": 1, "order_id": order["id"], "price": tp1_price, "amount": tp1_amount})
        log.info(f"🎯 TP1 order placed @ ${tp1_price:,.2f} for {tp1_amount} BTC ({TP1_CLOSE_PCT*100:.0f}%)")
    except Exception as e:
        log.warning(f"⚠️ Failed to place TP1 order: {e}")

    # TP2 — close remainder
    if tp2_price:
        tp2_amount = float(exchange.amount_to_precision(SYMBOL, amount * (1 - TP1_CLOSE_PCT)))
        try:
            order = exchange.create_order(
                SYMBOL, "market", close_side, tp2_amount,
                params={
                    "reduceOnly": True,
                    "triggerPrice": str(round(tp2_price, 2)),
                    "triggerBy": "MarkPrice",
                    "triggerDirection": trigger_dir,
                }
            )
            tp_orders.append({"level": 2, "order_id": order["id"], "price": tp2_price, "amount": tp2_amount})
            log.info(f"🎯 TP2 order placed @ ${tp2_price:,.2f} for {tp2_amount} BTC ({(1-TP1_CLOSE_PCT)*100:.0f}%)")
        except Exception as e:
            log.warning(f"⚠️ Failed to place TP2 order: {e}")

    return tp_orders


def cancel_tp_orders(exchange, tp_orders):
    """Cancel existing TP conditional orders."""
    for tp in tp_orders:
        try:
            exchange.cancel_order(tp["order_id"], SYMBOL)
            log.info(f"🗑️ Cancelled TP{tp['level']} order {tp['order_id'][:12]}")
        except Exception as e:
            # Order might already be filled or expired
            log.debug(f"Could not cancel TP{tp['level']} order {tp['order_id'][:12]}: {e}")


def _cancel_orphan_conditionals(exchange):
    """Fallback sweep: cancel all reduce-only conditional orders for SYMBOL.
    
    Catches orphans that weren't tracked in tp_orders state (e.g. after crash,
    manual edits, or state corruption). Only cancels reduce-only orders with
    a triggerPrice to avoid nuking entry orders or other strategies.
    """
    try:
        open_orders = exchange.fetch_open_orders(SYMBOL)
        for oo in open_orders:
            trigger = oo.get("triggerPrice") or oo.get("info", {}).get("triggerPrice")
            reduce_only = oo.get("reduceOnly", oo.get("info", {}).get("reduceOnly", False))
            if not trigger:
                continue
            # Only cancel reduce-only conditionals (TP/SL orders, not entries)
            if str(reduce_only).lower() not in ("true", "1", True):
                continue
            oid = str(oo.get("id", ""))[:12]
            try:
                exchange.cancel_order(oo["id"], SYMBOL)
                log.info(f"🗑️ Cancelled orphan conditional order {oid} (trigger={trigger})")
            except Exception as e:
                err_msg = str(e).lower()
                if "not found" in err_msg or "not exist" in err_msg or "already" in err_msg:
                    log.debug(f"Orphan order {oid} already gone: {e}")
                else:
                    log.warning(f"⚠️ Failed to cancel orphan order {oid}: {e}")
    except Exception as e:
        log.warning(f"⚠️ Failed to fetch open orders for orphan cleanup: {e}")


def sync_take_profits(exchange, direction, amount, active_trade, tp1_price=None, tp2_price=None):
    """Update TP orders when agents propose new levels. Cancel old, place new.
    Skips TP1 if already hit. Skips unchanged levels to avoid churn."""
    old_orders = active_trade.get("tp_orders", [])
    tp1_hit = active_trade.get("tp1_hit", False)
    new_orders = []

    # Determine which levels need updating
    old_tp1 = next((o for o in old_orders if o["level"] == 1), None)
    old_tp2 = next((o for o in old_orders if o["level"] == 2), None)

    dir_lower = str(direction).lower()
    is_short = dir_lower in ("sell", "short")
    close_side = "buy" if is_short else "sell"
    trigger_dir = 2 if is_short else 1

    # TP1 — only update if not yet hit
    if not tp1_hit and tp1_price:
        needs_place = False
        if old_tp1 and abs(old_tp1["price"] - tp1_price) > 1.0:
            # Cancel old TP1
            try:
                exchange.cancel_order(old_tp1["order_id"], SYMBOL)
                log.info(f"🗑️ Cancelled old TP1 @ ${old_tp1['price']:,.2f}")
            except Exception as e:
                log.warning(f"⚠️ Could not cancel old TP1: {e} — placing new anyway")
            needs_place = True
        elif not old_tp1:
            # No existing TP1 order (missing/failed/restart) — place fresh
            needs_place = True
            log.info(f"🎯 No existing TP1 order — placing @ ${tp1_price:,.2f}")
        else:
            new_orders.append(old_tp1)  # Keep existing (unchanged)

        if needs_place:
            tp1_amount = float(exchange.amount_to_precision(SYMBOL, amount * TP1_CLOSE_PCT))
            if tp1_amount > 0:
                try:
                    order = exchange.create_order(
                        SYMBOL, "market", close_side, tp1_amount,
                        params={
                            "reduceOnly": True,
                            "triggerPrice": str(round(tp1_price, 2)),
                            "triggerBy": "MarkPrice",
                            "triggerDirection": trigger_dir,
                        }
                    )
                    new_orders.append({"level": 1, "order_id": order["id"], "price": tp1_price, "amount": tp1_amount})
                    log.info(f"🎯 TP1 set → ${tp1_price:,.2f} for {tp1_amount} BTC")
                except Exception as e:
                    log.warning(f"⚠️ Failed to place TP1: {e}")
            else:
                log.warning(f"⚠️ TP1 amount rounded to 0 — skipping")

    # TP2 — update if price changed or missing
    if tp2_price:
        needs_place = False
        if old_tp2 and abs(old_tp2["price"] - tp2_price) > 1.0:
            try:
                exchange.cancel_order(old_tp2["order_id"], SYMBOL)
                log.info(f"🗑️ Cancelled old TP2 @ ${old_tp2['price']:,.2f}")
            except Exception as e:
                log.warning(f"⚠️ Could not cancel old TP2: {e} — placing new anyway")
            needs_place = True
        elif not old_tp2:
            needs_place = True
            log.info(f"🎯 No existing TP2 order — placing @ ${tp2_price:,.2f}")
        else:
            new_orders.append(old_tp2)  # Keep existing

        if needs_place:
            remaining = amount * (1 - TP1_CLOSE_PCT) if not tp1_hit else amount
            tp2_amount = float(exchange.amount_to_precision(SYMBOL, remaining))
            if tp2_amount > 0:
                try:
                    order = exchange.create_order(
                        SYMBOL, "market", close_side, tp2_amount,
                        params={
                            "reduceOnly": True,
                            "triggerPrice": str(round(tp2_price, 2)),
                            "triggerBy": "MarkPrice",
                            "triggerDirection": trigger_dir,
                        }
                    )
                    new_orders.append({"level": 2, "order_id": order["id"], "price": tp2_price, "amount": tp2_amount})
                    log.info(f"🎯 TP2 set → ${tp2_price:,.2f} for {tp2_amount} BTC")
                except Exception as e:
                    log.warning(f"⚠️ Failed to place TP2: {e}")
            else:
                log.warning(f"⚠️ TP2 amount rounded to 0 — skipping")
    elif not tp2_price and old_tp2:
        new_orders.append(old_tp2)  # No TP2 proposed, keep existing

    return new_orders


# Legacy wrapper for backward compatibility
def set_take_profit(exchange, direction, amount, tp_price):
    """Legacy single-TP wrapper. Places TP1 only."""
    orders = set_take_profits(exchange, direction, amount, tp_price)
    return orders


def sync_take_profit_to_exchange(exchange, direction, amount, new_tp, tp1_hit=False):
    """Legacy single-TP sync wrapper."""
    if tp1_hit:
        log.info(f"ℹ️ TP1 already hit — skipping TP sync (remaining position is trail-only)")
        return
    # This legacy path doesn't have access to active_trade, so just log
    log.info(f"ℹ️ Use sync_take_profits() for multi-TP management")


def sync_stop_loss_to_exchange(exchange, direction, amount, new_sl):
    """Update the position stop-loss on Bybit to match our trailing stop."""
    try:
        bybit_symbol = SYMBOL.replace("/", "").replace(":USDT", "")
        exchange.private_post_v5_position_trading_stop({
            "category": "linear",
            "symbol": bybit_symbol,
            "stopLoss": str(round(new_sl, 2)),
            "slTriggerBy": "MarkPrice",
            "positionIdx": 0,
        })
        log.info(f"🛡️ Exchange SL synced to ${new_sl:,.2f} (position TP/SL)")
    except Exception as e:
        log.warning(f"⚠️ Failed to sync SL to exchange: {e}")


def set_native_trailing_stop(exchange, direction, entry_price, atr):
    """Set Bybit's native trailing stop — trails continuously on Bybit's side.

    trailingStop: distance from peak price (ATR_TRAIL_MULTIPLE × ATR)
    activePrice: only activates after trade is in profit (entry ± 0.5×ATR)

    This replaces our 4H polling-based trailing with real-time exchange trailing.
    """
    trail_distance = round(ATR_TRAIL_MULTIPLE * atr, 2)
    # Activate trailing stop only after some profit (0.5 × ATR move in our favor)
    activation_offset = round(0.5 * atr, 2)
    if direction in ("SELL", "short"):
        active_price = round(entry_price - activation_offset, 2)  # Short: activate below entry
    else:
        active_price = round(entry_price + activation_offset, 2)  # Long: activate above entry

    try:
        bybit_symbol = SYMBOL.replace("/", "").replace(":USDT", "")
        exchange.private_post_v5_position_trading_stop({
            "category": "linear",
            "symbol": bybit_symbol,
            "trailingStop": str(trail_distance),
            "activePrice": str(active_price),
            "positionIdx": 0,
        })
        log.info(f"📏 Native trailing stop set: distance=${trail_distance:,.0f} (ATR×{ATR_TRAIL_MULTIPLE}), activates @ ${active_price:,.0f}")
        return True
    except Exception as e:
        log.error(f"🚨 Native trailing stop FAILED — position may be unprotected! {e}")
        return False


def _verify_and_repair_protection(exchange, executor_state, positions):
    """Verify exchange-side protection orders match local state. Repair if missing.

    Called every cycle when we have an active_trade AND an exchange position.
    Handles:
    1. needs_protection_retry — TP/SL/trailing placement failed on fill
    2. Missing TP orders — exchange may have cancelled/expired them
    3. Missing stop loss — verify SL exists on exchange
    4. Position size mismatch — detect partial fills/manual changes
    """
    active = executor_state.get("active_trade")
    if not active:
        return

    direction = active.get("direction", "")
    amount = active.get("amount", 0)
    entry = active.get("entry", 0)
    tp1 = active.get("tp1")
    tp2 = active.get("tp2")
    stop_loss = active.get("stop_loss")
    atr = active.get("atr_at_entry", 0)
    repaired = False

    # Use live exchange position size for all protection operations
    live_amount = amount
    if positions:
        live_amount = abs(float(positions[0].get("contracts", 0))) or amount

    # 1. Retry failed protection placement
    if active.get("needs_protection_retry"):
        log.warning("🔧 PROTECTION RETRY: Previous protection placement failed — retrying")
        all_ok = True

        tp_orders = active.get("tp_orders", [])
        if not tp_orders and tp1:
            try:
                tp_orders = set_take_profits(exchange, direction, live_amount, tp1, tp2)
                active["tp_orders"] = tp_orders
                log.info(f"✅ Protection retry: TP orders placed ({len(tp_orders)} orders)")
                repaired = True
            except Exception as e:
                log.error(f"🚨 Protection retry FAILED for TP orders: {e}")
                all_ok = False

        if atr > 0:
            try:
                set_native_trailing_stop(exchange, direction, entry, atr)
                log.info("✅ Protection retry: native trailing stop set")
                repaired = True
            except Exception as e:
                log.error(f"🚨 Protection retry FAILED for trailing stop: {e}")
                all_ok = False

        if stop_loss:
            try:
                side = "long" if direction == "BUY" else "short"
                sync_stop_loss_to_exchange(exchange, side, live_amount, stop_loss)
                log.info("✅ Protection retry: SL synced")
                repaired = True
            except Exception as e:
                log.error(f"🚨 Protection retry FAILED for SL: {e}")
                all_ok = False

        # Only clear retry flag if ALL protection was placed successfully
        if all_ok:
            active["needs_protection_retry"] = False
        else:
            log.error("🚨 Protection still incomplete — will retry next cycle")
        save_state(executor_state)

    # 2. Verify TP orders still exist on exchange and repair if missing
    tp_orders = active.get("tp_orders", [])
    tp1_hit = active.get("tp1_hit", False)
    if tp_orders:
        try:
            # Fetch both active AND conditional (untriggered) orders
            open_orders = exchange.fetch_open_orders(SYMBOL)
            # Also check conditional/stop orders via Bybit API
            try:
                bybit_symbol = SYMBOL.replace("/", "").replace(":USDT", "")
                cond_resp = exchange.private_get_v5_order_realtime({
                    "category": "linear", "symbol": bybit_symbol, "orderFilter": "StopOrder"
                })
                cond_orders = cond_resp.get("result", {}).get("list", [])
                cond_ids = {o.get("orderId") for o in cond_orders}
            except Exception:
                cond_ids = set()

            all_order_ids = {o["id"] for o in open_orders} | cond_ids
            missing_tps = []

            for tp in tp_orders:
                if tp["order_id"] not in all_order_ids:
                    tp_level = tp.get("level", 0)
                    if tp_level == 1 and tp1_hit:
                        continue  # TP1 was filled, expected to be gone
                    missing_tps.append(tp)
                    log.warning(f"⚠️ TP{tp_level} order {tp['order_id'][:12]} missing from exchange")

            # Repair: re-place missing TP orders
            if missing_tps:
                log.warning(f"🔧 Re-placing {len(missing_tps)} missing TP orders")
                new_tp_orders = [tp for tp in tp_orders if tp not in missing_tps]
                for tp in missing_tps:
                    tp_level = tp.get("level", 0)
                    tp_price = tp1 if tp_level == 1 else tp2
                    tp_amount = tp.get("amount", live_amount * (TP1_CLOSE_PCT if tp_level == 1 else (1 - TP1_CLOSE_PCT)))
                    if tp_price and tp_amount > 0:
                        try:
                            dir_lower = direction.lower()
                            is_short = dir_lower in ("sell", "short")
                            close_side = "buy" if is_short else "sell"
                            trigger_dir = 2 if is_short else 1
                            order = exchange.create_order(
                                SYMBOL, "market", close_side,
                                float(exchange.amount_to_precision(SYMBOL, tp_amount)),
                                params={
                                    "reduceOnly": True,
                                    "triggerPrice": str(round(tp_price, 2)),
                                    "triggerBy": "MarkPrice",
                                    "triggerDirection": trigger_dir,
                                }
                            )
                            new_tp_orders.append({"level": tp_level, "order_id": order["id"], "price": tp_price, "amount": tp_amount})
                            log.info(f"✅ TP{tp_level} re-placed @ ${tp_price:,.2f} (order {order['id'][:12]})")
                            repaired = True
                        except Exception as e:
                            log.error(f"🚨 Failed to re-place TP{tp_level}: {e}")
                active["tp_orders"] = new_tp_orders
                save_state(executor_state)
        except Exception as e:
            log.debug(f"Could not verify TP orders: {e}")

    # 3. Verify position size matches
    if positions:
        p = positions[0]
        exchange_contracts = abs(float(p.get("contracts", 0)))
        local_amount = float(active.get("amount", 0))
        if exchange_contracts > 0 and local_amount > 0:
            size_diff_pct = abs(exchange_contracts - local_amount) / local_amount * 100
            if size_diff_pct > 5:  # More than 5% mismatch
                log.warning(
                    f"⚠️ Position size mismatch: exchange={exchange_contracts:.4f} vs local={local_amount:.4f} "
                    f"({size_diff_pct:.1f}% diff) — may indicate manual change or partial TP fill"
                )

    if repaired:
        log.info("🔧 Protection repair cycle complete")


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

        # Update MFE from actual candle data (catches intra-candle wicks)
        update_mfe_from_candles(exchange, trade_info)

        # --- TP1 hit detection ---
        # If exchange position is smaller than our tracked amount, TP1 was filled
        original_amount = trade_info.get("amount", contracts)
        if not trade_info.get("tp1_hit") and contracts < original_amount * 0.9:
            # Position shrank by >10% — TP1 partial close fired
            pnl_on_closed = 0
            closed_qty = original_amount - contracts
            if side == "long":
                tp1_price = trade_info.get("tp1", current)
                pnl_on_closed = (tp1_price - entry) * closed_qty
            else:
                tp1_price = trade_info.get("tp1", current)
                pnl_on_closed = (entry - tp1_price) * closed_qty
            trade_info["tp1_hit"] = True
            trade_info["tp1_fill_price"] = tp1_price
            trade_info["tp1_closed_qty"] = closed_qty
            trade_info["amount"] = contracts  # Update to remaining size
            log.info(f"🎯 TP1 HIT! Closed {closed_qty:.3f} BTC @ ~${tp1_price:,.0f} (${pnl_on_closed:,.2f} realized)")
            log.info(f"📊 Remaining: {contracts:.3f} BTC — TP2 + trailing stop for remainder")

            # Remove filled TP1 from tp_orders tracking
            tp_orders = trade_info.get("tp_orders", [])
            trade_info["tp_orders"] = [o for o in tp_orders if o.get("level") != 1]

            # Record partial close in closed_trades
            state.setdefault("closed_trades", []).append({
                "direction": trade_info.get("direction", side),
                "entry": entry,
                "exit": tp1_price,
                "amount": closed_qty,
                "pnl_pct": round(((tp1_price - entry) / entry * 100) if side == "long" else ((entry - tp1_price) / entry * 100), 4),
                "exit_reason": "tp1_partial",
                "opened_at": trade_info.get("opened_at", ""),
                "closed_at": datetime.now(timezone.utc).isoformat(),
            })
            save_state(state)

        # Track max adverse excursion (worst unrealized P&L during trade)
        if side == "long":
            unrealized_pct = ((current - entry) / entry) * 100
        else:
            unrealized_pct = ((entry - current) / entry) * 100
        prev_max_dd = trade_info.get("max_drawdown_pct", 0.0)
        if unrealized_pct < prev_max_dd:
            trade_info["max_drawdown_pct"] = round(unrealized_pct, 2)
            log.info(f"📉 New max drawdown: {unrealized_pct:.2f}% (was {prev_max_dd:.2f}%)")
        # Also track peak favorable excursion
        prev_peak = trade_info.get("max_favorable_pct", 0.0)
        if unrealized_pct > prev_peak:
            trade_info["max_favorable_pct"] = round(unrealized_pct, 2)

        atr = get_atr(exchange)
        trail_dist = ATR_TRAIL_MULTIPLE * atr
        # Refresh native trailing stop if ATR changed significantly (>15%)
        prev_atr = trade_info.get("atr_at_entry", atr)
        if abs(atr - prev_atr) / max(prev_atr, 1) > 0.15:
            log.info(f"📏 ATR shifted: ${prev_atr:,.0f} → ${atr:,.0f} — refreshing native trailing stop")
            if set_native_trailing_stop(exchange, side, entry, atr):
                trade_info["atr_at_entry"] = atr

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
                    cancel_tp_orders(exchange, trade_info.get("tp_orders", []))
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
                sync_stop_loss_to_exchange(exchange, "long", contracts, new_trail)
            if current <= trade_info.get("trailing_stop", 0):
                log.info(f"🔔 TRAILING STOP HIT @ ${current:,.0f} (trail was ${trade_info['trailing_stop']:,.0f})")
                cancel_tp_orders(exchange, trade_info.get("tp_orders", []))
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
                sync_stop_loss_to_exchange(exchange, "short", contracts, new_trail)
            if current >= trade_info.get("trailing_stop", float("inf")):
                log.info(f"🔔 TRAILING STOP HIT @ ${current:,.0f} (trail was ${trade_info['trailing_stop']:,.0f})")
                cancel_tp_orders(exchange, trade_info.get("tp_orders", []))
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


def can_open_green_lane(state: dict, equity: float, direction: str = None) -> tuple:
    """Check if a green lane position can be opened.
    Rules: max 1 green lane, combined size <= 10%, combined leverage <= 20x,
           no opposing direction vs committee.
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

    # Block directional conflict: green lane opposing committee position
    committee = positions.get("committee")
    if committee and direction:
        committee_side = committee.get("side", "").upper()
        gl_side = direction.upper()
        # Map BUY/SELL to LONG/SHORT for comparison
        side_map = {"BUY": "LONG", "SELL": "SHORT", "LONG": "LONG", "SHORT": "SHORT"}
        c_norm = side_map.get(committee_side, committee_side)
        g_norm = side_map.get(gl_side, gl_side)
        if c_norm and g_norm and c_norm != g_norm:
            reason = f"Directional conflict: committee is {c_norm}, green lane wants {g_norm}"
            log.info(f"🚫 Cannot open green lane: {reason}")
            return False, reason
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
        "initial_tp1": signal.tp1,
        "initial_tp2": signal.tp2,
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
        # Time-based exit for shorts
        max_hold_days = pos.get("max_hold_days", 0)
        if max_hold_days > 0:
            entry_time_str = pos.get("entry_time") or pos.get("opened_at")
            if entry_time_str:
                try:
                    from datetime import datetime, timezone
                    entry_dt = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                    held_days = (datetime.now(timezone.utc) - entry_dt).days
                    if held_days >= max_hold_days:
                        log.info(f"⏰ Green lane short time exit: held {held_days}d >= max {max_hold_days}d")
                        return [{"type": "time_exit", "price": current_price, "fraction": 1.0, "reason": f"max_hold_{max_hold_days}d"}]
                except Exception as e:
                    log.warning(f"Could not parse entry_time for time-based exit: {e}")
        # Acceleration detection: big drop in a day → take remaining profits
        daily_atr = pos.get("daily_atr", 0.0)
        prev_day_price = pos.get("prev_day_price")
        if prev_day_price and prev_day_price > 0:
            day_drop_pct = (prev_day_price - current_price) / prev_day_price * 100
            if day_drop_pct > 5.0:
                log.info(f"🚀 Short acceleration: dropped {day_drop_pct:.2f}% in a day — closing remaining")
                return [{"type": "acceleration", "price": current_price, "fraction": 1.0, "reason": "acceleration_5pct_day"}]
            if daily_atr > 0:
                atr_2x = 2 * daily_atr
                gain = prev_day_price - current_price
                if gain >= atr_2x:
                    log.info(f"🚀 Short acceleration: gained {gain:.2f} >= 2x ATR ({atr_2x:.2f}) in one day — suggesting close")
                    actions.append({"type": "acceleration", "price": current_price, "fraction": 1.0, "reason": "acceleration_2x_atr"})
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

def create_agents_graph(analysts_to_run=None):
    """Create and return the TradingAgents graph with persistent memory.
    
    Args:
        analysts_to_run: list of analyst names to actually run. Default: all.
    """
    config = CRYPTO_DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["deep_think_llm"] = "claude-opus-4-6"
    config["quick_think_llm"] = "claude-sonnet-4-20250514"
    # Fallback to OpenRouter/Minimax M2.5 on Anthropic rate limits
    config["fallback_provider"] = "openrouter"
    config["fallback_model"] = "minimax/minimax-m2.5"
    # Fund manager override — uncomment to A/B test GPT-5.4
    # config["fund_manager_llm"] = "openai/gpt-5.4"
    # config["fund_manager_llm_provider"] = "openrouter"

    # Only include analysts that need fresh runs
    selected = analysts_to_run or ["market", "sentiment", "fundamentals", "news"]
    log.info(f"\U0001f4ca Creating graph with analysts: {selected}")

    ta = CryptoTradingAgentsGraph(
        selected_analysts=selected,
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
        ("fund_manager", ta.fund_manager_memory),
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
        ("fund_manager", ta.fund_manager_memory),
    ]:
        mem_path = str(MEMORY_DIR / f"{mem_name}_memory.json")
        mem_obj.save(mem_path)
    log.info("💾 Memories saved to disk")


def run_reflection(ta, returns_pct):
    """Run reflection loop — agents learn from trade outcomes."""
    if ta.curr_state is None:
        log.debug("No agent state in memory for reflection — loading last state from disk")
        # Try to load the last state from logs (sorted by name = sorted by date)
        state_dir = PROJECT_ROOT / "eval_results" / "BTC_USDT" / "CryptoTradingAgents_logs"
        state_files = sorted(state_dir.glob("full_states_log_*.json")) if state_dir.exists() else []
        if not state_files:
            log.warning("⚠️ No state files found in %s — skipping reflection", state_dir)
            return
        try:
            with open(state_files[-1]) as f:
                states = json.load(f)
            if not states:
                log.warning("⚠️ Last state file is empty — skipping reflection")
                return
            last_date = sorted(states.keys())[-1]
            last_state = states[last_date]
            ta.curr_state = {
                "market_report": last_state.get("market_report", ""),
                "sentiment_report": last_state.get("sentiment_report", ""),
                "news_report": last_state.get("news_report", ""),
                "fundamentals_report": last_state.get("fundamentals_report", ""),
                "investment_debate_state": last_state.get("investment_debate_state", {}),
                "trader_investment_plan": last_state.get("trader_investment_decision", ""),
                "risk_debate_state": last_state.get("risk_debate_state", {}),
            }
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            log.warning(f"⚠️ Failed to parse state file {state_files[-1]}: {e} — skipping reflection")
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


def run_agents(ta, trade_date, portfolio_context=None, analysts_to_run=None, analysts_to_cache=None, analyst_scheduler=None):
    log.info(f"🧠 Running agents for {SPOT_SYMBOL} on {trade_date}...")

    # Inject portfolio context into the graph's initial state
    if portfolio_context:
        ta.portfolio_context = portfolio_context

    # Inject cached reports for analysts we're skipping
    for cached_analyst in analysts_to_cache:
        cached_report = analyst_scheduler.get_cached_report(cached_analyst)
        if cached_report:
            report_field = {
                "market": "market_report",
                "sentiment": "sentiment_report", 
                "fundamentals": "fundamentals_report",
                "news": "news_report",
            }.get(cached_analyst)
            if report_field:
                # Inject into the graph's initial state
                if not hasattr(ta, '_cached_reports'):
                    ta._cached_reports = {}
                ta._cached_reports[report_field] = cached_report
                log.info(f"\U0001f4be Injected cached {cached_analyst} report")

        # === STALE CACHE WARNING ===
        # Warn if the cached report significantly exceeds its configured TTL.
        # Don't block — just alert so operators can take action.
        try:
            cache_entry = analyst_scheduler._cache.get(cached_analyst, {})
            last_run_str = cache_entry.get("last_run")
            if last_run_str:
                last_dt = datetime.fromisoformat(last_run_str)
                # fromisoformat() may return a naive datetime (no tzinfo) if the stored
                # string has no timezone suffix.  Subtracting from an aware datetime
                # raises TypeError.  Assume UTC for naive timestamps.
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
                ttl_hours = analyst_scheduler.schedule.get(cached_analyst, 24)
                stale_threshold = ttl_hours * 1.5
                if age_hours > stale_threshold:
                    log.warning(
                        f"⚠️ STALE CACHE: {cached_analyst} report is {age_hours:.1f}h old "
                        f"(TTL: {ttl_hours}h, threshold: {stale_threshold:.1f}h) — consider refreshing"
                    )
        except Exception as _e:
            log.debug(f"Stale cache check failed for {cached_analyst}: {_e}")
    
    # Run pipeline with retry logic — LLM timeouts, API errors, parsing failures
    MAX_RETRIES = 2
    last_error = None
    agent_state = None
    decision = None
    for attempt in range(MAX_RETRIES + 1):
        if _shutdown_requested:
            raise RuntimeError("Shutdown requested — aborting agent pipeline")
        try:
            if attempt > 0:
                log.warning(f"🔄 Retry {attempt}/{MAX_RETRIES}: restarting agent pipeline...")
                time.sleep(5)  # Brief cooldown before retry
            agent_state, decision = ta.propagate(SPOT_SYMBOL, trade_date)
            break  # Success
        except Exception as e:
            last_error = e
            last_node = getattr(ta, '_last_completed_node', 'unknown')
            log.error(f"❌ Agent pipeline failed (attempt {attempt + 1}/{MAX_RETRIES + 1}) "
                      f"after node '{last_node}': {type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                log.error(f"🛑 All {MAX_RETRIES + 1} attempts failed — aborting")
                raise

    # Cache fresh reports from analysts that ran
    for analyst in analysts_to_run:
        report_field = {
            "market": "market_report",
            "sentiment": "sentiment_report",
            "fundamentals": "fundamentals_report", 
            "news": "news_report",
        }.get(analyst)
        if report_field and agent_state.get(report_field):
            analyst_scheduler.update_cache(analyst, agent_state[report_field])

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
    trader_plan = agent_state.get("trader_investment_plan", "") or agent_state.get("trader_investment_decision", "")
    
    # Parse from fund manager first (has final authority)
    trade_params = parse_trade_params(full_decision)
    
    # If fund manager said "as_proposed" or params are missing, get them from trader's plan
    if trader_plan:
        trader_params = parse_trade_params(trader_plan)
        # Fill in any missing params from trader (fund manager overrides if present)
        for key in ("stop_loss", "take_profit_1", "take_profit_2", "risk_pct", "confidence", "risk_reward"):
            if key not in trade_params and key in trader_params:
                trade_params[key] = trader_params[key]
                log.info(f"📝 Using trader's {key}: {trader_params[key]}")

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
    """Atomic state save — write to temp file then rename to prevent corruption."""
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(dir=STATE_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp_path, STATE_FILE)  # Atomic on POSIX
    except Exception as e:
        log.error(f"Failed to save state: {e}")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# === MAIN ===

def preflight_check(timeout=10):
    """Verify Bybit API is reachable before starting. Fails fast if VPN is down."""
    import urllib.request
    url = "https://api.bybit.com/v5/market/time"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "preflight"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        data = json.loads(resp.read())
        if data.get("retCode") == 0:
            log.info("✅ Preflight: Bybit API reachable")
            return True
        log.error(f"❌ Preflight: Bybit returned error: {data}")
        return False
    except Exception as e:
        log.error(f"❌ Preflight: Cannot reach Bybit API ({e}). VPN down?")
        return False


def main():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Acquire exclusive lock — prevents sentinel + cron overlap
    import fcntl
    lock_file = Path(__file__).resolve().parent.parent / "logs" / ".executor.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)  # ensure logs/ exists
    lock_fd = open(lock_file, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        log.warning("⏸️  Another executor instance is running — exiting")
        lock_fd.close()
        return
    # Lock held for duration of main() — released on process exit or explicit close

    log.info("=" * 60)
    log.info("🚀 TradingAgents Live Executor v2")
    log.info("=" * 60)

    if not preflight_check():
        log.error("🛑 Aborting: Bybit API unreachable. Check VPN connection.")
        return

    # Start CryptoMonitor background refresh (warms CII cache during LLM calls)
    try:
        from tradingagents.dataflows.crypto_monitor import start_background_refresh
        start_background_refresh(interval=1800)
    except Exception as e:
        log.warning(f"CryptoMonitor background start failed: {e}")

    exchange = get_exchange()
    equity = get_equity(exchange)
    log.info(f"💰 Equity: ${equity:,.2f}")

    if check_circuit_breaker(equity):
        return

    executor_state = load_state()
    _shutdown_state["executor_state"] = executor_state  # Stash for signal handler

    # Check mandatory cooldown FIRST — skip entire cycle if active
    if check_cooldown_active(executor_state):
        save_state(executor_state)  # Persist any cooldown_until cleanup
        return

    if check_daily_loss_limit(executor_state, equity) >= DAILY_LOSS_LIMIT:
        return

    # Check cold streak — adjusts risk if on losing run (may also trigger cooldown)
    cold_streak, risk_multiplier = get_cold_streak(executor_state)
    save_state(executor_state)  # Persist cooldown_until if newly set

    # Re-check cooldown — get_cold_streak may have just triggered it this cycle
    if check_cooldown_active(executor_state):
        return

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
        # Reconcile: clear stale active_trade if exchange has no position
        stale = executor_state.get("active_trade")
        stale_committee = executor_state.get("positions", {}).get("committee")
        # Use whichever has data (active_trade is authoritative, fall back to positions.committee)
        stale_data = stale or stale_committee
        if stale_data:
            # Normalize field names: active_trade uses direction/entry, committee uses side/entry
            stale_side = stale_data.get("direction", stale_data.get("side", "?"))
            stale_entry = float(stale_data.get("entry", stale_data.get("entry_price", 0)))
            stale_amount = float(stale_data.get("amount", stale_data.get("contracts", 0)))
            log.warning(f"🔄 RECONCILE: Position closed externally (was {stale_side} {stale_amount:.3f} BTC @ ${stale_entry:,.0f})")

            # Calculate PnL from last price
            reconcile_price = None
            try:
                last_price = float(exchange.fetch_ticker(SYMBOL)["last"])
                reconcile_price = last_price
                if stale_side.upper() in ("BUY", "LONG"):
                    pnl_pct = ((last_price - stale_entry) / stale_entry * 100) if stale_entry > 0 else 0
                else:
                    pnl_pct = ((stale_entry - last_price) / stale_entry * 100) if stale_entry > 0 else 0
                log.info(f"📊 Estimated P&L: {pnl_pct:+.2f}% (entry ${stale_entry:,.0f} → last ${last_price:,.0f})")

                # Record to closed_trades for bookkeeping
                executor_state.setdefault("closed_trades", []).append({
                    "direction": stale_side,
                    "entry": stale_entry,
                    "exit": last_price,
                    "amount": stale_amount,
                    "pnl_pct": round(pnl_pct, 4),
                    "exit_reason": "exchange_closed",
                    "opened_at": stale_data.get("opened_at", ""),
                    "closed_at": datetime.now(timezone.utc).isoformat(),
                    "confidence": stale_data.get("confidence"),
                    "tp1": stale_data.get("tp1"),
                    "tp2": stale_data.get("tp2"),
                    "stop_loss": stale_data.get("stop_loss"),
                    "max_favorable_pct": stale_data.get("max_favorable_pct", 0),
                    "max_drawdown_pct": stale_data.get("max_drawdown_pct", 0),
                })
                log.info(f"📝 Recorded closed trade: {stale_side} {stale_amount:.3f} BTC | P&L {pnl_pct:+.2f}%")

                # Set pending reflection so agents can learn from it
                executor_state["pending_reflection"] = {
                    "returns_pct": pnl_pct,
                    "direction": stale_side,
                    "entry": stale_entry,
                    "exit": last_price,
                }
            except Exception as e:
                log.warning(f"Failed to calculate reconcile PnL: {e}")

            # Record outcome for learning — position was closed externally (SL/TP/liquidation)
            ot_id = stale_data.get("outcome_tracker_id")
            if ot_id is not None:
                try:
                    exit_price = reconcile_price if reconcile_price else float(exchange.fetch_ticker(SYMBOL)["last"])
                    outcome_tracker.record_outcome(
                        record_id=ot_id,
                        exit_price=exit_price,
                        exit_reason="exchange_closed",
                        max_favorable_pct=stale_data.get("max_favorable_pct", 0),
                        max_adverse_pct=stale_data.get("max_drawdown_pct", 0),
                    )
                except Exception as e:
                    log.warning(f"Failed to record reconcile outcome: {e}")

            # Cancel any orphaned TP orders before clearing state
            tp_orders = stale_data.get("tp_orders", [])
            if tp_orders:
                log.info(f"🗑️ Cancelling {len(tp_orders)} orphaned TP orders from externally-closed position")
                cancel_tp_orders(exchange, tp_orders)
            # Always run fallback sweep — tp_orders in state may be stale/incomplete
            _cancel_orphan_conditionals(exchange)

            # Clear both active_trade AND positions.committee
            executor_state.pop("active_trade", None)
            if executor_state.get("positions", {}).get("committee") is not None:
                executor_state["positions"]["committee"] = None
                log.info("🧹 Cleared positions.committee")
            save_state(executor_state)

    # Reverse reconcile: exchange has position but we lost active_trade (crash recovery)
    if has_position and not executor_state.get("active_trade"):
        p = positions[0]
        entry_price = float(p["entryPrice"])
        contracts = abs(float(p["contracts"]))
        side = p["side"]
        log.warning(f"🔄 REVERSE RECONCILE: Exchange has {side} {contracts} BTC @ ${entry_price:,.0f} but no active_trade — reconstructing")
        atr = get_atr(exchange)
        if side == "long":
            default_sl = entry_price - (2 * atr)
            default_trail = entry_price - (ATR_TRAIL_MULTIPLE * atr)
        else:
            default_sl = entry_price + (2 * atr)
            default_trail = entry_price + (ATR_TRAIL_MULTIPLE * atr)
        executor_state["active_trade"] = {
            "direction": "BUY" if side == "long" else "SELL",
            "entry": entry_price,
            "amount": contracts,
            "stop_loss": default_sl,
            "trailing_stop": default_trail,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "confidence": 5,
            "atr_at_entry": atr,
            "reconstructed": True,
        }
        sync_stop_loss_to_exchange(exchange, side, contracts, default_sl)
        save_state(executor_state)

    # ─── Protection verification & retry ─────────────────────────────────────
    active = executor_state.get("active_trade")
    if active and has_position:
        _verify_and_repair_protection(exchange, executor_state, positions)

    # Check if a pending limit order was filled since last cycle
    if check_pending_limit_order(exchange, executor_state):
        positions = get_positions(exchange)
        has_position = len(positions) > 0
        if has_position:
            for p in positions:
                pnl = float(p.get("unrealizedPnl", 0))
                log.info(f"\U0001f4ca Limit fill: {p['side']} {p['contracts']} @ ${float(p['entryPrice']):,.0f} | PnL: ${pnl:,.2f}")

    # Check for pending entry from previous cycle (agent specified limit entry)
    pending = executor_state.get("pending_entry")
    if pending and not has_position:
        current = float(exchange.fetch_ticker(SYMBOL)["last"])
        pe_dir = pending["direction"]
        pe_price = pending["entry_price"]
        pe_age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(pending["created_at"])).total_seconds() / 3600
        
        # Expire pending entries after 24h
        if pe_age_h > 24:
            log.info(f"\u23f3 Pending entry expired ({pe_age_h:.0f}h old): {pe_dir} @ ${pe_price:,.0f}")
            executor_state.pop("pending_entry", None)
            save_state(executor_state)
        else:
            entry_ok = False
            if pe_dir == "BUY" and current <= pe_price * 1.003:
                entry_ok = True
            elif pe_dir == "SELL" and current >= pe_price * 0.997:
                entry_ok = True
            
            if entry_ok:
                log.info(f"\u2705 Pending entry triggered! {pe_dir} @ ${pe_price:,.0f} (price ${current:,.0f})")
                alloc_pct = pending.get("alloc_pct", DEFAULT_ALLOC_PCT)
                stop_loss = pending.get("stop_loss")
                tp1 = pending.get("tp1")
                risk_multiplier_pe = 1.0
                
                result = open_position(exchange, pe_dir, equity, alloc_pct, stop_loss, risk_multiplier=risk_multiplier_pe)
                if result:
                    fill_price = result["fill_price"]
                    amount = result["amount"]
                    
                    risk_dist = abs(fill_price - stop_loss) if stop_loss else get_atr(exchange) * 2
                    if not tp1:
                        if pe_dir == "BUY":
                            tp1 = fill_price + (TP1_R_MULTIPLE * risk_dist)
                        else:
                            tp1 = fill_price - (TP1_R_MULTIPLE * risk_dist)
                    
                    set_take_profit(exchange, pe_dir, amount, tp1)
                    atr = get_atr(exchange)
                    if pe_dir == "BUY":
                        initial_trail = fill_price - (ATR_TRAIL_MULTIPLE * atr)
                    else:
                        initial_trail = fill_price + (ATR_TRAIL_MULTIPLE * atr)
                    
                    tp2 = pending.get("params", {}).get("take_profit_2")
                    executor_state["active_trade"] = {
                        "direction": pe_dir,
                        "entry": fill_price,
                        "amount": amount,
                        "stop_loss": stop_loss,
                        "tp1": tp1,
                        "tp2": tp2,
                        "initial_tp1": tp1,
                        "initial_tp2": tp2,
                        "trailing_stop": initial_trail,
                        "opened_at": datetime.now(timezone.utc).isoformat(),
                        "confidence": pending.get("params", {}).get("confidence"),
                        "atr_at_entry": atr,
                        "from_pending": True,
                    }
                    executor_state.pop("pending_entry", None)
                    save_state(executor_state)
                    log.info(f"\u2705 Pending entry filled: {pe_dir} {amount} BTC @ ${fill_price:,.0f}")
                    has_position = True
            else:
                log.info(f"\u23f8\ufe0f  Pending entry still waiting: {pe_dir} @ ${pe_price:,.0f} (price ${current:,.0f}, age {pe_age_h:.1f}h)")

    # Determine which analysts need fresh runs vs cached
    analysts_to_run, analysts_to_cache = analyst_scheduler.get_analysts_to_run()
    
    # Create graph with only the analysts that need refreshing
    ta = create_agents_graph(analysts_to_run=analysts_to_run)

    # If a trade just closed, reflect on the outcome
    if executor_state.get("pending_reflection"):
        pr = executor_state.pop("pending_reflection")
        run_reflection(ta, pr["returns_pct"])

    # Build portfolio context for Fund Manager
    # Build positions summary for fund manager awareness
    positions_lines = []
    pos_state = executor_state.get("positions", {})
    for src in ("committee", "green_lane"):
        p = pos_state.get(src)
        if p:
            positions_lines.append(f"- {src}: {p.get('side', '?')} | entry ${p.get('entry_price', p.get('entry', 0)):,.0f} | size {p.get('size_pct', '?')}%")
    if not positions_lines:
        positions_lines.append("- No active positions")

    portfolio_ctx = {
        "position": "none",
        "pnl_pct": 0,
        "age_bars": 0,
        "equity": equity,
        "consecutive_same_direction": 0,
        "last_decision": executor_state.get("last_decision", "HOLD"),
        "positions_summary": "Active positions:\n" + "\n".join(positions_lines),
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
    
    # Inject performance feedback from outcome tracker
    portfolio_ctx["performance_feedback"] = outcome_tracker.get_feedback_for_agents()
    portfolio_ctx["last_decision_reasoning"] = executor_state.get("last_decision_reasoning", "No prior reasoning available.")
    portfolio_ctx["last_decision_time"] = executor_state.get("last_decision_time", "Unknown")
    
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

    # ─── Check for light layer / green lane overrides ───────────────────
    override_decision = None
    for override_name in ("green_lane_override.json", "light_override.json"):  # Green lane takes precedence
        override_file = LOG_DIR / override_name
        if override_file.exists():
            try:
                ov = json.loads(override_file.read_text())
                if "action" not in ov or "timestamp" not in ov:
                    log.warning(f"Malformed override {override_name} — quarantining")
                    override_file.rename(override_file.with_suffix(f".bad.{int(time.time())}.json"))
                    continue
                ov_age = (datetime.now(timezone.utc) - datetime.fromisoformat(ov["timestamp"])).total_seconds()
                if ov_age < 600:  # Valid if < 10 min old
                    override_decision = ov
                    log.info(f"📥 Override from {ov.get('source', override_name)}: {ov['action']} (age={ov_age:.0f}s)")
                    override_file.unlink()  # Consume it
                    break
                else:
                    log.info(f"Stale override {override_name} ({ov_age:.0f}s old) — ignoring")
                    override_file.unlink()
            except Exception as e:
                log.warning(f"Failed to read {override_name}: {e}")
                try:
                    override_file.rename(override_file.with_suffix(f".bad.{int(time.time())}.json"))
                except Exception:
                    pass

    if override_decision and override_decision["action"] in ("BUY", "SELL", "LONG", "SHORT"):
        action = override_decision["action"]
        # Normalize: LONG→BUY, SHORT→SELL
        if action == "LONG":
            action = "BUY"
        elif action == "SHORT":
            action = "SELL"
        decision = action
        trade_params = {
            "confidence": override_decision.get("confidence", override_decision.get("quality", 8)),
            "stop_loss": override_decision.get("stop_loss"),
            "take_profit_1": override_decision.get("tp1"),
            "take_profit_2": override_decision.get("tp2"),
        }
        reports = {"override_source": override_decision.get("source", "unknown"),
                   "override_reason": override_decision.get("reason", override_decision.get("reasoning", ""))}
        log.warning(f"⚡ OVERRIDE DECISION: {decision} from {reports['override_source']}")
    elif override_decision and override_decision["action"] == "CLOSE":
        decision = "CLOSE"
        trade_params = {}
        reports = {"override_source": override_decision.get("source", "unknown"),
                   "override_reason": override_decision.get("reason", "")}
        log.warning(f"⚡ OVERRIDE CLOSE from {reports['override_source']}: {reports['override_reason']}")
    else:
        # Run agents for new signal (full pipeline)
        trade_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        decision, trade_params, reports = run_agents(ta, trade_date, portfolio_ctx, analysts_to_run=analysts_to_run, analysts_to_cache=analysts_to_cache, analyst_scheduler=analyst_scheduler)

    # === SHUTDOWN CHECKPOINT (post agent pipeline) ===
    if _shutdown_requested:
        log.warning("🛑 Shutdown requested — saving state and exiting after agent pipeline")
        executor_state["_shutdown_signal"] = "post_pipeline"
        executor_state["_shutdown_at"] = datetime.now(timezone.utc).isoformat()
        save_state(executor_state)
        sys.exit(0)

    # === OBJECTIVE SCORING GUARDRAIL ===
    try:
        from tradingagents.dataflows.crypto_technical_brief import build_crypto_technical_brief
        brief = build_crypto_technical_brief(SYMBOL.replace(":USDT", "").replace("/USDT", "/USDT"))
        obj_score = calculate_objective_score(brief=brief)
    except Exception as e:
        log.warning(f"Objective scoring failed: {e}")
        from tradingagents.scoring.objective_score import ScoreBreakdown
        obj_score = ScoreBreakdown()

    # === GEOPOLITICAL EVENT DETECTION ===
    news_texts = []
    news_report = reports.get("news_report", "") if isinstance(reports, dict) else ""
    if news_report:
        news_texts = [line.strip() for line in news_report.split("\n") if len(line.strip()) > 20]
    geo_severity, geo_events = detect_geopolitical_events(news_texts)
    if geo_severity < 0:
        # Factor geopolitical risk into objective score
        obj_score.macro = round(max(-100, obj_score.macro + geo_severity * 0.5), 1)
        obj_score.overall = round(
            obj_score.technical * 0.25 + obj_score.momentum * 0.25 +
            obj_score.volume * 0.15 + obj_score.sentiment * 0.20 +
            obj_score.macro * 0.15, 1
        )

    # === DECISION VALIDATION ===
    current_pos_str = "NEUTRAL"
    if has_position and positions:
        current_pos_str = positions[0]["side"].upper()
    
    validation = validate_decision(
        agent_decision=decision,
        agent_confidence=trade_params.get("confidence", 5),
        objective_score=obj_score,
        trade_params=trade_params,
        current_position=current_pos_str,
        geopolitical_severity=geo_severity,
        geopolitical_events=geo_events,
    )
    
    # Apply validation result
    original_decision = decision
    if validation.overridden:
        decision = validation.validated_decision
        log.warning(f"\U0001f6a8 DECISION OVERRIDDEN: {original_decision} \u2192 {decision} | Reason: {validation.override_reason}")
    
    if validation.warnings:
        for w in validation.warnings:
            log.warning(f"\u26a0\ufe0f  Validation: {w}")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "original_decision": original_decision if validation.overridden else None,
        "equity": equity,
        "params": trade_params,
        "objective_score": obj_score.overall,
        "objective_signal": obj_score.signal,
        "geo_severity": geo_severity if geo_severity < 0 else None,
        "validation_warnings": validation.warnings if validation.warnings else None,
        "was_overridden": validation.overridden,
    }

    # Cancel pending limit order if agents changed direction
    pending_lo = executor_state.get("pending_limit_order")
    if pending_lo:
        old_dir = pending_lo.get("direction", "")
        new_signal = "BUY" if decision == "BUY" else "SELL" if decision == "SELL" else ""
        if new_signal and new_signal != old_dir:
            log.info(f"\U0001f504 Agents changed direction ({old_dir} \u2192 {new_signal}) \u2014 cancelling pending limit order")
            try:
                exchange.cancel_order(pending_lo["order_id"], SYMBOL)
                log.info(f"\u274c Cancelled limit order {pending_lo['order_id']}")
            except Exception as e:
                log.warning(f"Failed to cancel limit order: {e}")
            executor_state.pop("pending_limit_order", None)
            save_state(executor_state)
        elif decision == "HOLD" and not has_position:
            # Agents say HOLD with no position — keep the pending limit alive
            log.info(f"\u23f8\ufe0f  Agents say HOLD, keeping pending limit order {pending_lo.get('order_id','?')} alive")

    # Map signal to position transition
    current_pos = "NEUTRAL"
    if has_position and positions:
        current_pos = positions[0]["side"].upper()  # "LONG" or "SHORT"
    
    # Normalize decision to trading mode
    trade_signal = decision.upper()
    if trade_signal == "BUY":
        trade_signal = "LONG"
    elif trade_signal == "SELL":
        trade_signal = "SHORT"
    elif trade_signal == "HOLD":
        trade_signal = "NEUTRAL" if not has_position else current_pos

    transition = get_position_transition(current_pos, trade_signal)
    action = transition["action"]
    log.info(f"📋 Position transition: {current_pos} + {trade_signal} → {action} ({transition['description']})")
    record["transition"] = action

    if action in ("HOLD", "STAY_NEUTRAL"):
        active = executor_state.get("active_trade", {})
        if active and has_position:
            direction = active.get("direction", "")
            entry = active.get("entry", 0)
            amount = active.get("amount", 0)
            current_price = float(exchange.fetch_ticker(SYMBOL)["last"])

            # Check if agents proposed new TPs
            proposed_tp1 = trade_params.get("take_profit_1")
            proposed_tp2 = trade_params.get("take_profit_2")
            tp_changed = False

            # ═══ TP1 IS IMMUTABLE ═══
            # TP1 exists to de-risk: take 50% off the table, pay for the trade.
            # Once set at entry, it NEVER moves. Agents cannot override this.
            if proposed_tp1 and proposed_tp1 != active.get("tp1"):
                log.info(f"🔒 TP1 LOCKED: Ignoring agent proposal ${proposed_tp1:,.2f} "
                         f"(immutable at ${active.get('tp1', 0):,.2f} since entry)")

            # ═══ TP2 IS ADJUSTABLE (with guards) ═══
            # TP2 is the runner — agents can adjust it based on market conditions.
            # Guards: hard cap at 2x initial distance, ratchet when retracing.
            if proposed_tp2:
                current_tp2 = active.get("tp2")
                tp2_valid = False
                tp2_ratcheted = False

                # Basic side validation
                if direction == "BUY" and proposed_tp2 > entry:
                    tp2_valid = True
                elif direction == "SELL" and proposed_tp2 < entry:
                    tp2_valid = True

                # Hard cap: 2x initial distance max
                initial_tp2 = active.get("initial_tp2")
                if tp2_valid and initial_tp2 and entry > 0:
                    initial_dist = abs(entry - initial_tp2)
                    proposed_dist = abs(entry - proposed_tp2)
                    if proposed_dist > 2.0 * initial_dist:
                        capped = entry - 2.0 * initial_dist if direction == "SELL" else entry + 2.0 * initial_dist
                        log.warning(f"🚫 TP2 HARD CAP: ${proposed_tp2:,.0f} exceeds 2× initial distance "
                                    f"(initial ${initial_tp2:,.0f}, max ${capped:,.0f}). Capping.")
                        proposed_tp2 = round(capped, 2)

                # Ratchet: block further moves when price is retracing from MFE
                mfe = float(active.get("max_favorable_pct") or 0.0)
                if tp2_valid and current_tp2 and mfe > 0.5 and entry > 0:
                    if direction == "SELL":
                        current_favorable_pct = ((entry - current_price) / entry) * 100
                    else:
                        current_favorable_pct = ((current_price - entry) / entry) * 100
                    cf = max(0.0, current_favorable_pct)
                    retracement = min(1.0, max(0.0, 1.0 - cf / mfe)) if mfe > 0 else 0
                    if retracement > 0.5:
                        tp_moving_further = (
                            (direction == "SELL" and proposed_tp2 < current_tp2) or
                            (direction == "BUY" and proposed_tp2 > current_tp2)
                        )
                        if tp_moving_further:
                            log.warning(f"🔒 TP2 RATCHET: Blocked move further "
                                        f"${current_tp2:,.2f} → ${proposed_tp2:,.2f} "
                                        f"(MFE {mfe:.1f}%, retracement {retracement:.0%})")
                            tp2_valid = False
                            tp2_ratcheted = True

                if tp2_valid and (not current_tp2 or proposed_tp2 != current_tp2):
                    log.info(f"🎯 TP2 update: ${current_tp2 or 0:,.2f} → ${proposed_tp2:,.2f}")
                    active["tp2"] = proposed_tp2
                    record["tp2_updated"] = proposed_tp2
                    tp_changed = True
                elif not tp2_valid and not tp2_ratcheted and proposed_tp2:
                    log.warning(f"⚠️ Ignoring invalid TP2 ${proposed_tp2:,.2f} "
                                f"(wrong side of entry ${entry:,.2f} for {direction})")

            # Sync TP orders to exchange if any changed
            if tp_changed:
                new_tp_orders = sync_take_profits(
                    exchange, direction, active.get("amount", amount),
                    active,
                    tp1_price=active.get("tp1"),
                    tp2_price=active.get("tp2"),
                )
                active["tp_orders"] = new_tp_orders

            # Check if agents proposed a tighter SL
            proposed_sl = trade_params.get("stop_loss")
            if proposed_sl:
                current_trail = active.get("trailing_stop")
                if current_trail:
                    # Validate: must be tighter (closer to entry) AND price must not have breached it
                    sl_is_tighter = False
                    sl_not_breached = False
                    if direction == "BUY":
                        sl_is_tighter = proposed_sl > current_trail  # higher SL = tighter for longs
                        sl_not_breached = current_price > proposed_sl
                    elif direction == "SELL":
                        sl_is_tighter = proposed_sl < current_trail  # lower SL = tighter for shorts
                        sl_not_breached = current_price < proposed_sl
                    if sl_not_breached and proposed_sl != current_trail:
                        if not sl_is_tighter:
                            log.warning(f"⚠️ Rejecting SL widening: ${current_trail:,.2f} → ${proposed_sl:,.2f} (monotonic tightening enforced)")
                        else:
                            log.info(f"🛡️ Fund manager tighter SL: ${current_trail:,.2f} → ${proposed_sl:,.2f} (price ${current_price:,.2f})")
                            active["trailing_stop"] = proposed_sl
                            active["stop_loss"] = proposed_sl  # Keep stop_loss in sync
                            sync_stop_loss_to_exchange(exchange, direction, amount, proposed_sl)
                            record["sl_updated"] = proposed_sl
                            save_state(executor_state)  # Persist SL update immediately
                    elif not sl_not_breached:
                        log.error(f"🚨 ALERT: Fund manager SL ${proposed_sl:,.2f} already breached (price ${current_price:,.2f}), keeping ${current_trail:,.2f}")
        log.info(f"⏸️  {action} — no action")

    elif action in ("CLOSE_LONG", "CLOSE_SHORT"):
        # Cancel TP orders before closing
        active = executor_state.get("active_trade", {})
        cancel_tp_orders(exchange, active.get("tp_orders", []))
        _cancel_orphan_conditionals(exchange)  # Fallback sweep
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
            
            # Record outcome for learning
            active = executor_state.get("active_trade", {})
            ot_id = active.get("outcome_tracker_id")
            if ot_id is not None:
                try:
                    outcome_tracker.record_outcome(
                        record_id=ot_id,
                        exit_price=exit_price,
                        exit_reason="close_signal",
                        max_favorable_pct=active.get("max_favorable_pct", 0),
                        max_adverse_pct=active.get("max_drawdown_pct", 0),
                    )
                except Exception as e:
                    log.warning(f"Failed to record outcome: {e}")
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
                        executor_state["trades"] = executor_state.get("trades", [])
                        executor_state["trades"].append(record)
                        save_state(executor_state)
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
                
                # Record outcome for learning
                active = executor_state.get("active_trade", {})
                ot_id = active.get("outcome_tracker_id")
                if ot_id is not None:
                    try:
                        outcome_tracker.record_outcome(
                            record_id=ot_id,
                            exit_price=exit_price,
                            exit_reason="reversal",
                            max_favorable_pct=active.get("max_favorable_pct", 0),
                            max_adverse_pct=active.get("max_drawdown_pct", 0),
                        )
                    except Exception as e:
                        log.warning(f"Failed to record outcome: {e}")
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
            # Pre-validate TP/SL before opening (catches parsing errors like $78 instead of $78,000)
            pre_price = float(exchange.fetch_ticker(SYMBOL)["last"])
            for param_name, param_val in [('take_profit_1', tp1), ('stop_loss', stop_loss)]:
                if param_val and abs(param_val - pre_price) / pre_price > 0.50:
                    log.warning(f'⚠️ {param_name} ${param_val:,.0f} is >50% from price ${pre_price:,.0f} — ignoring (will auto-calculate)')
                    if param_name == 'take_profit_1':
                        tp1 = None
                    elif param_name == 'stop_loss':
                        stop_loss = None
                        atr = get_atr(exchange, period=14, timeframe='1d')
                        if new_direction == 'BUY':
                            stop_loss = pre_price - (2 * atr)
                        else:
                            stop_loss = pre_price + (2 * atr)
                        log.info(f'🔧 Recalculated SL from 2×ATR: ${stop_loss:,.0f}')

            # === CONSOLIDATED RISK CHECK ===
            risk_actions = risk_mgr.check_all(
                equity=equity,
                proposed_direction=new_direction,
                proposed_alloc_pct=alloc_pct,
                proposed_leverage=10,
                current_position=None,
                daily_pnl=check_daily_loss_limit(executor_state, equity),
                peak_equity=PEAK_EQUITY,
                trades_today=len([t for t in executor_state.get("trades", [])
                    if t.get("timestamp", "")[:10] == datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    and t.get("action", "").startswith("opened_")]),
                objective_score=obj_score.overall,
                geopolitical_severity=geo_severity,
                validation_result=validation,
            )
            
            blocked = [a for a in risk_actions if a.action in ("BLOCK", "FLATTEN")]
            if blocked:
                log.warning(f"\U0001f6d1 RISK BLOCKED: {blocked[0].reason}")
                record["risk_blocked"] = blocked[0].reason
                record["transition"] = "RISK_BLOCKED"
                executor_state["trades"] = executor_state.get("trades", [])
                executor_state["trades"].append(record)
                save_state(executor_state)
                return
            
            # Apply any REDUCE adjustments
            alloc_pct = risk_mgr.get_adjusted_allocation(alloc_pct, risk_actions)

            # === SENTIMENT EXTREME GATE === (REMOVED)
            # F&G is a lagging indicator and extremes can persist for months.
            # The multi-agent pipeline should make its own assessment.
            # F&G data is still available to agents via CII for informational use.

            # === SENTIMENT EXTREME OBSERVABILITY (WARNING-ONLY — does NOT block) ===
            try:
                fg_value = fetch_fear_greed_value()
                if fg_value is not None:
                    _sentiment_extreme = False
                    if fg_value <= 15 and new_direction == "SELL":
                        log.warning(
                            f"⚠️ SENTIMENT NOTE: {new_direction} at F&G={fg_value} (extreme fear) "
                            f"— proceeding but logging for review"
                        )
                        _sentiment_extreme = True
                    elif fg_value >= 85 and new_direction == "BUY":
                        log.warning(
                            f"⚠️ SENTIMENT NOTE: {new_direction} at F&G={fg_value} (extreme greed) "
                            f"— proceeding but logging for review"
                        )
                        _sentiment_extreme = True
                    if _sentiment_extreme:
                        record["sentiment_extreme"] = {"fg": fg_value, "direction": new_direction}
            except Exception as _fg_err:
                log.debug(f"Sentiment observability check failed (non-blocking): {_fg_err}")

            # Respect agent's entry price — only open if market is at or better than proposed entry
            agent_entry = trade_params.get("entry_price")
            if agent_entry:
                current = float(exchange.fetch_ticker(SYMBOL)["last"])
                entry_ok = False
                if new_direction == "BUY" and current <= agent_entry * 1.003:  # within 0.3% of limit
                    entry_ok = True
                elif new_direction == "SELL" and current >= agent_entry * 0.997:
                    entry_ok = True
                
                if not entry_ok:
                    # Place actual limit order on exchange instead of polling
                    log.info(f"\U0001f4cb Placing limit order: {new_direction} @ ${agent_entry:,.0f} (current ${current:,.0f})")
                    limit_result = open_limit_order(exchange, new_direction, equity, alloc_pct, agent_entry, stop_loss, take_profit=tp1, risk_multiplier=risk_multiplier)
                    if limit_result:
                        executor_state["pending_limit_order"] = {
                            **limit_result,
                            "tp1": tp1,
                            "tp2": trade_params.get("take_profit_2"),
                            "confidence": trade_params.get("confidence"),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                        record["transition"] = "LIMIT_PLACED"
                        record["limit_order_id"] = limit_result["order_id"]
                        record["limit_price"] = agent_entry
                    else:
                        record["transition"] = "HOLD"
                        record["limit_failed"] = True
                    executor_state["trades"] = executor_state.get("trades", [])
                    executor_state["trades"].append(record)
                    save_state(executor_state)
                    return
                else:
                    log.info(f"\u2705 Price ${current:,.0f} favorable vs agent entry ${agent_entry:,.0f} \u2014 proceeding")

            # === SHUTDOWN CHECKPOINT (before opening new position) ===
            if _shutdown_requested:
                log.warning("🛑 Shutdown requested — saving state and exiting before opening position")
                executor_state["_shutdown_signal"] = "pre_open_position"
                executor_state["_shutdown_at"] = datetime.now(timezone.utc).isoformat()
                save_state(executor_state)
                sys.exit(0)

            result = open_position(exchange, new_direction, equity, alloc_pct, stop_loss, risk_multiplier=risk_multiplier)

            if result:
                fill_price = result["fill_price"]
                amount = result["amount"]

                risk_dist = abs(fill_price - stop_loss)
                if not tp1:
                    if new_direction == "BUY":
                        tp1 = fill_price + (TP1_R_MULTIPLE * risk_dist)
                    else:
                        tp1 = fill_price - (TP1_R_MULTIPLE * risk_dist)
                    log.info(f"🔧 Auto TP1 at {TP1_R_MULTIPLE}R: ${tp1:,.0f}")

                # Set TP orders on exchange (TP1: 50%, TP2: 50% if available)
                tp2 = trade_params.get("take_profit_2")
                tp_orders = set_take_profits(exchange, new_direction, amount, tp1, tp2)

                # Initialize trailing stop — use Bybit's native trailing for real-time tracking
                atr = get_atr(exchange)
                if not set_native_trailing_stop(exchange, new_direction, fill_price, atr):
                    log.error("🚨 Native trailing stop failed on trade entry — local stop only!")
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
                    "tp2": tp2,
                    "initial_tp1": tp1,  # Immutable: for TP hard cap
                    "initial_tp2": tp2,  # Immutable: for TP hard cap
                    "tp_orders": tp_orders,
                    "tp1_hit": False,
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
                
                # Record decision for outcome tracking
                try:
                    ot_id = outcome_tracker.record_decision(
                        decision=new_direction,
                        confidence=trade_params.get("confidence", 5),
                        entry_price=fill_price,
                        stop_loss=stop_loss,
                        take_profit=tp1,
                        objective_score=obj_score.overall,
                        objective_signal=obj_score.signal,
                        was_overridden=validation.overridden,
                        override_reason=validation.override_reason if validation.overridden else "",
                        trade_params=trade_params,
                        validation_warnings=validation.warnings,
                    )
                    executor_state["active_trade"]["outcome_tracker_id"] = ot_id
                except Exception as e:
                    log.warning(f"Failed to record decision in outcome tracker: {e}")
            else:
                record["action"] = "failed"

    # Compact trades history — keep last 200 entries, archive older
    trades = executor_state.get("trades", [])
    if len(trades) > 200:
        archived = trades[:-200]
        executor_state["trades"] = trades[-200:]
        archive_file = LOG_DIR / f"trades_archive_{datetime.now(timezone.utc).strftime('%Y%m')}.json"
        try:
            existing = json.loads(archive_file.read_text()) if archive_file.exists() else []
            existing.extend(archived)
            archive_file.write_text(json.dumps(existing, indent=2, default=str))
            log.info(f"📦 Archived {len(archived)} old trade records")
        except Exception as e:
            log.warning(f"Failed to archive trades: {e}")

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

    # Sync positions state for green-lane conflict detection
    active = executor_state.get("active_trade")
    if active:
        executor_state.setdefault("positions", {})["committee"] = {
            "side": "long" if active.get("direction") == "BUY" else "short",
            "entry": active.get("entry"),
            "stop_loss": active.get("stop_loss"),
        }
    else:
        executor_state.setdefault("positions", {})["committee"] = None
    save_state(executor_state)

    with open(LOG_DIR / f"agent_reports_{trade_date}.json", "w") as f:
        json.dump(reports, f, indent=2)

    log.info("✅ Done")


if __name__ == "__main__":
    main()

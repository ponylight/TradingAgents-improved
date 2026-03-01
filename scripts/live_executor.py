#!/usr/bin/env python3
"""
TradingAgents Live Executor — Runs multi-agent pipeline and executes on Bybit demo.

Flow:
1. Run TradingAgents propagate() → BUY/SELL/HOLD
2. Execute on Bybit with position sizing
3. Manage existing positions (close on opposing signal)

Config: RISK_PCT=8%, LEVERAGE=5x, circuit breaker -9% from peak
"""

import ccxt
import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from tradingagents.graph.crypto_trading_graph import CryptoTradingAgentsGraph, CRYPTO_DEFAULT_CONFIG

# === CONFIG ===
SYMBOL = "BTC/USDT:USDT"
SPOT_SYMBOL = "BTC/USDT"
RISK_PCT = 0.08
LEVERAGE = 5
CIRCUIT_BREAKER_PCT = 0.09
PEAK_EQUITY = 154_425

LOG_DIR = PROJECT_ROOT / "logs"
STATE_FILE = LOG_DIR / "executor_state.json"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"executor_{datetime.now().strftime('%Y-%m-%d')}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("executor")


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
    balance = exchange.fetch_balance({"type": "contract"})
    return float(balance["total"].get("USDT", 0))


def get_positions(exchange):
    positions = exchange.fetch_positions([SYMBOL])
    return [p for p in positions if abs(float(p["contracts"] or 0)) > 0]


def check_circuit_breaker(equity):
    threshold = PEAK_EQUITY * (1 - CIRCUIT_BREAKER_PCT)
    if equity < threshold:
        log.error(f"🚨 CIRCUIT BREAKER: ${equity:,.0f} < ${threshold:,.0f}")
        return True
    return False


def close_position(exchange, position):
    side = "sell" if position["side"] == "long" else "buy"
    amount = abs(float(position["contracts"]))
    log.info(f"📤 Closing {position['side']} {amount} contracts")
    order = exchange.create_order(SYMBOL, "market", side, amount, params={"reduceOnly": True})
    log.info(f"✅ Closed: {order['id']}")
    return order


def open_position(exchange, direction, equity):
    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
    except Exception as e:
        log.warning(f"Leverage: {e}")

    ticker = exchange.fetch_ticker(SYMBOL)
    price = ticker["last"]
    notional = equity * RISK_PCT * LEVERAGE
    amount = float(exchange.amount_to_precision(SYMBOL, notional / price))

    if amount <= 0:
        log.error(f"❌ Amount=0. Equity=${equity:.0f}, Price=${price:.0f}")
        return None

    side = "buy" if direction == "BUY" else "sell"
    log.info(f"📥 {direction} {amount} BTC @ ~${price:,.0f} | ${notional:,.0f} notional | {LEVERAGE}x")

    order = exchange.create_order(SYMBOL, "market", side, amount)
    log.info(f"✅ Filled: {order['id']} @ ${order.get('average', 'N/A')}")
    return order


def run_agents(trade_date):
    log.info(f"🧠 Running agents for {SPOT_SYMBOL} on {trade_date}...")

    config = CRYPTO_DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["deep_think_llm"] = "claude-sonnet-4-20250514"
    config["quick_think_llm"] = "claude-sonnet-4-20250514"

    ta = CryptoTradingAgentsGraph(
        selected_analysts=["market", "sentiment", "macro", "news"],
        config=config,
        debug=False,
    )

    state, decision = ta.propagate(SPOT_SYMBOL, trade_date)
    log.info(f"🎯 Decision: {decision}")

    reports = {
        "market": state.get("market_report", "")[:500],
        "sentiment": state.get("sentiment_report", "")[:500],
        "news": state.get("news_report", "")[:500],
        "final": state.get("final_trade_decision", "")[:1000],
        "decision": decision,
    }
    return decision, reports


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_run": None, "last_decision": None, "trades": []}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def main():
    log.info("=" * 60)
    log.info("🚀 TradingAgents Live Executor")
    log.info("=" * 60)

    exchange = get_exchange()
    equity = get_equity(exchange)
    log.info(f"💰 Equity: ${equity:,.2f}")

    if check_circuit_breaker(equity):
        return

    positions = get_positions(exchange)
    has_position = len(positions) > 0

    if has_position:
        for p in positions:
            pnl = float(p.get("unrealizedPnl", 0))
            log.info(f"📊 {p['side']} {p['contracts']} @ ${float(p['entryPrice']):,.0f} | PnL: ${pnl:,.2f}")
    else:
        log.info("📊 No open positions")

    trade_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    decision, reports = run_agents(trade_date)

    state = load_state()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "equity": equity,
    }

    if decision == "HOLD":
        log.info("⏸️  HOLD — no action")

    elif decision in ("BUY", "SELL"):
        if has_position:
            for p in positions:
                opposing = (decision == "BUY" and p["side"] == "short") or \
                          (decision == "SELL" and p["side"] == "long")
                if opposing:
                    log.info(f"🔄 Closing opposing {p['side']}")
                    close_position(exchange, p)
                    record["closed"] = p["side"]
                    has_position = False
                else:
                    log.info(f"✅ Already {p['side']} — confirming hold")
                    record["action"] = "confirm"

        if not has_position:
            order = open_position(exchange, decision, equity)
            if order:
                record["action"] = f"opened_{decision.lower()}"
                record["price"] = order.get("average", order.get("price"))
            else:
                record["action"] = "failed"

    state["last_run"] = record["timestamp"]
    state["last_decision"] = decision
    state["trades"].append(record)
    save_state(state)

    with open(LOG_DIR / f"agent_reports_{trade_date}.json", "w") as f:
        json.dump(reports, f, indent=2)

    log.info("✅ Done")


if __name__ == "__main__":
    main()

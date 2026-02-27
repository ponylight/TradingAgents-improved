#!/usr/bin/env python3
"""
Polymarket Latency Scalper — BTC 5-Min Up/Down Markets

Monitors BTC price on Binance via websocket, compares with Polymarket
Up/Down odds. When Binance shows a strong directional move but Polymarket
hasn't repriced yet, places a limit order on the lagging side.

Position size: $1 max per trade. Circuit breaker after 3 consecutive losses.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import websockets
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@trade"
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
FUNDER = "0xFA5b389A03e1d79d7bDD6ef9A14c6be51a9A4D70"
SIG_TYPE = 2

MAX_POSITION_USD = 1.0
EDGE_THRESHOLD = 0.04           # 4c edge required
PRICE_MOVE_BPS = 15             # Binance move (bps in 2s) to trigger
LOOKBACK_S = 2.0
CIRCUIT_BREAKER_LOSSES = 3
POLL_INTERVAL = 0.25

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("latency_scalper")


@dataclass
class ScalperState:
    btc_prices: list = field(default_factory=list)
    consecutive_losses: int = 0
    trades_placed: int = 0
    halted: bool = False
    running: bool = True

state = ScalperState()


def current_market_slug() -> str:
    now = int(time.time())
    bucket = (now // 300) * 300
    return f"btc-updown-5m-{bucket}"


def momentum_bps(prices: list, window: float) -> float:
    now = time.time()
    cutoff = now - window
    recent = [p for ts, p in prices if ts >= cutoff]
    if len(recent) < 2:
        return 0.0
    return (recent[-1] - recent[0]) / recent[0] * 10_000


def init_clob_client() -> ClobClient:
    load_dotenv(Path.home() / "TradingAgents-improved" / ".env")
    pk = os.getenv("POLYMARKET_PRIVATE_KEY")
    if not pk:
        log.error("POLYMARKET_PRIVATE_KEY not set in .env")
        sys.exit(1)

    client = ClobClient(
        host=CLOB_HOST,
        chain_id=CHAIN_ID,
        key=pk,
        signature_type=SIG_TYPE,
        funder=FUNDER,
    )
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)
    log.info("CLOB client initialised (L2). Address: %s", client.get_address())
    return client


async def binance_listener():
    while state.running:
        try:
            async with websockets.connect(BINANCE_WS) as ws:
                log.info("Connected to Binance WS")
                async for msg in ws:
                    if not state.running:
                        break
                    data = json.loads(msg)
                    price = float(data["p"])
                    ts = time.time()
                    state.btc_prices.append((ts, price))
                    cutoff = ts - 10.0
                    state.btc_prices = [(t, p) for t, p in state.btc_prices if t >= cutoff]
        except Exception as e:
            log.warning("Binance WS error: %s — reconnecting in 1s", e)
            await asyncio.sleep(1)


async def signal_loop(client: ClobClient):
    log.info("Signal loop started (dry-run — orders built but NOT posted)")

    while state.running:
        await asyncio.sleep(POLL_INTERVAL)

        if state.halted:
            log.info("Circuit breaker active — halted.")
            await asyncio.sleep(5)
            continue

        mom = momentum_bps(state.btc_prices, LOOKBACK_S)
        if abs(mom) < PRICE_MOVE_BPS:
            continue

        direction = "UP" if mom > 0 else "DOWN"
        slug = current_market_slug()
        log.info("Signal: BTC momentum %.1f bps → %s | slug: %s", mom, direction, slug)

        # In production: resolve condition_id from slug, get token_ids, check orderbook.
        # Placeholder for dry run:
        token_id = f"<{direction}_TOKEN_FOR_{slug}>"
        price = 0.50
        size = round(MAX_POSITION_USD / price, 2)

        log.info(
            "DRY RUN — would BUY %s: token=%s price=%.2f size=%.2f ($%.2f)",
            direction, token_id, price, size, price * size,
        )
        state.trades_placed += 1

        # Circuit breaker (production):
        # state.consecutive_losses += 1
        # if state.consecutive_losses >= CIRCUIT_BREAKER_LOSSES:
        #     log.warning("CIRCUIT BREAKER: %d losses — halting", CIRCUIT_BREAKER_LOSSES)
        #     state.halted = True

        await asyncio.sleep(2)


async def main():
    client = init_clob_client()
    log.info("CLOB health: %s", client.get_ok())
    log.info("CLOB server time: %s", client.get_server_time())

    tasks = [
        asyncio.create_task(binance_listener()),
        asyncio.create_task(signal_loop(client)),
    ]

    def shutdown(*_):
        state.running = False
        for t in tasks:
            t.cancel()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    await asyncio.gather(*tasks, return_exceptions=True)
    log.info("Scalper stopped. Trades signalled: %d", state.trades_placed)


if __name__ == "__main__":
    asyncio.run(main())

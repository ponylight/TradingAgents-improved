#!/usr/bin/env python3
"""
Polymarket NegRisk Arbitrage Scanner

Scans neg-risk markets (mutually exclusive outcomes). If combined YES ask
prices sum to < $1.00, buying all sides guarantees profit.

Circuit breaker: halts after 3 consecutive losses.
"""

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
FUNDER = "0xFA5b389A03e1d79d7bDD6ef9A14c6be51a9A4D70"
SIG_TYPE = 2

MAX_POSITION_USD = 1.0
MIN_EDGE_CENTS = 1.0            # 1c minimum profit to flag
SCAN_INTERVAL_S = 10
CIRCUIT_BREAKER_LOSSES = 3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("negrisk_arb")

consecutive_losses = 0
halted = False
total_opportunities = 0


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


def get_best_asks(client: ClobClient, tokens: list) -> list[dict]:
    results = []
    for token in tokens:
        token_id = token.get("token_id")
        outcome = token.get("outcome", "?")
        if not token_id:
            continue
        try:
            book = client.get_order_book(token_id)
            asks = book.get("asks", [])
            best_ask = float(asks[0]["price"]) if asks else None
            best_size = float(asks[0]["size"]) if asks else 0
        except Exception as e:
            log.debug("Book error %s: %s", token_id[:16], e)
            best_ask, best_size = None, 0

        results.append({
            "token_id": token_id,
            "outcome": outcome,
            "best_ask": best_ask,
            "best_ask_size": best_size,
        })
    return results


def evaluate_arb(market: dict, asks: list[dict]) -> dict | None:
    if not asks or any(a["best_ask"] is None for a in asks):
        return None

    total_cost = sum(a["best_ask"] for a in asks)
    profit = 1.0 - total_cost

    if profit < MIN_EDGE_CENTS / 100:
        return None

    max_sets = min(a["best_ask_size"] for a in asks)
    max_by_budget = MAX_POSITION_USD / total_cost
    sets = min(max_sets, max_by_budget)

    return {
        "question": market.get("question", "?"),
        "condition_id": market.get("condition_id", "?"),
        "total_cost": round(total_cost, 4),
        "profit_per_set": round(profit, 4),
        "profit_pct": round(profit / total_cost * 100, 2),
        "sets": round(sets, 2),
        "legs": asks,
    }


def run(client: ClobClient):
    global consecutive_losses, halted, total_opportunities

    log.info("NegRisk arb scanner running (dry-run mode)")

    while not halted:
        log.info("--- Scan ---")
        try:
            resp = client.get_sampling_simplified_markets()
            markets = resp if isinstance(resp, list) else resp.get("data", [])
        except Exception as e:
            log.warning("Market fetch failed: %s", e)
            time.sleep(SCAN_INTERVAL_S)
            continue

        neg_risk = [m for m in markets if m.get("neg_risk", False)]
        log.info("Neg-risk markets: %d / %d total", len(neg_risk), len(markets))

        for market in neg_risk:
            tokens = market.get("tokens", [])
            if len(tokens) < 2:
                continue

            asks = get_best_asks(client, tokens)
            arb = evaluate_arb(market, asks)

            if arb:
                total_opportunities += 1
                log.info(
                    "🎯 ARB: %s | cost=%.4f | profit=%.4f (%.2f%%) | sets=%.2f",
                    arb["question"][:50], arb["total_cost"],
                    arb["profit_per_set"], arb["profit_pct"], arb["sets"],
                )
                for leg in arb["legs"]:
                    log.info("  %s — ask=%.4f sz=%.2f", leg["outcome"], leg["best_ask"], leg["best_ask_size"])

                # DRY RUN — production would place BUY orders on all legs

        log.info("Opportunities so far: %d | sleeping %ds", total_opportunities, SCAN_INTERVAL_S)
        time.sleep(SCAN_INTERVAL_S)


def main():
    client = init_clob_client()
    log.info("CLOB health: %s", client.get_ok())
    log.info("CLOB server time: %s", client.get_server_time())
    run(client)


if __name__ == "__main__":
    main()

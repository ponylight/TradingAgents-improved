#!/usr/bin/env python3
"""
Penny Strategy Scanner for Polymarket
Finds nearly-resolved markets where you can buy at 95-99c for near-guaranteed small profit.
Uses Gamma API for market discovery + CLOB for orderbook depth.
"""

import os
import json
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from py_clob_client.client import ClobClient

load_dotenv(os.path.expanduser("~/TradingAgents-improved/.env"))

CLOB_URL = "https://clob.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com/markets"

MIN_PRICE = 0.95
MAX_PRICE = 0.99
MAX_HOURS = 72
TAKER_FEE_BPS = 100
MIN_LIQUIDITY_USD = 10


def fetch_candidates():
    """Use Gamma API to find active markets ending within 72h with a 95-99c outcome."""
    now = datetime.now(timezone.utc)
    end_max = now + timedelta(hours=MAX_HOURS)
    
    all_markets = []
    offset = 0
    while True:
        r = requests.get(GAMMA_URL, params={
            'active': 'true', 'closed': 'false',
            'end_date_min': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'end_date_max': end_max.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'limit': 500, 'offset': offset,
        }, timeout=15)
        data = r.json() if r.ok else []
        if not data:
            break
        all_markets.extend(data)
        if len(data) < 500:
            break
        offset += 500
    
    print(f"  Gamma API returned {len(all_markets)} markets ending within {MAX_HOURS}h")
    
    candidates = []
    for m in all_markets:
        prices_raw = m.get("outcomePrices", "[]")
        outcomes_raw = m.get("outcomes", "[]")
        try:
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        except:
            continue
        
        end_dt = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00"))
        hours_left = (end_dt - now).total_seconds() / 3600
        
        # Get CLOB token IDs from the CLOB API for this condition
        cond_id = m.get("conditionId", "")
        
        for i, (outcome, price_str) in enumerate(zip(outcomes, prices)):
            price = float(price_str)
            if MIN_PRICE <= price <= MAX_PRICE:
                candidates.append({
                    "question": m.get("question", "?"),
                    "outcome": outcome,
                    "price": price,
                    "hours_left": round(hours_left, 1),
                    "slug": m.get("slug", ""),
                    "condition_id": cond_id,
                    "gamma_id": m.get("id"),
                    "clobTokenIds": m.get("clobTokenIds"),
                    "outcome_index": i,
                })
    
    return candidates


def check_orderbooks(candidates):
    """Check CLOB orderbooks for actual ask liquidity."""
    client = ClobClient(CLOB_URL)
    opportunities = []
    
    for i, c in enumerate(candidates):
        print(f"  Checking {i+1}/{len(candidates)}: {c['question'][:50]}...", end="\r")
        
        # Get token ID from CLOB
        token_ids = c.get("clobTokenIds")
        if token_ids:
            try:
                ids = json.loads(token_ids) if isinstance(token_ids, str) else token_ids
                token_id = ids[c["outcome_index"]] if c["outcome_index"] < len(ids) else None
            except:
                token_id = None
        else:
            token_id = None
        
        if not token_id:
            # Try to get from CLOB market endpoint
            try:
                mkt = client.get_market(c["condition_id"])
                tokens = mkt.get("tokens", [])
                for t in tokens:
                    if t.get("outcome") == c["outcome"]:
                        token_id = t["token_id"]
                        break
            except:
                continue
        
        if not token_id:
            continue
            
        try:
            book = client.get_order_book(token_id)
        except:
            continue

        asks = book.asks or []
        if not asks:
            continue

        liq = 0
        best_ask = None
        for ask in asks:
            ap = float(ask.price)
            az = float(ask.size)
            if ap <= MAX_PRICE:
                liq += az * ap
                if best_ask is None or ap < best_ask:
                    best_ask = ap

        if best_ask is None or liq < MIN_LIQUIDITY_USD:
            continue

        fee = (1 - best_ask) * (TAKER_FEE_BPS / 10000)
        net = (1 - best_ask) - fee
        if net <= 0:
            continue

        opportunities.append({
            **c,
            "best_ask": best_ask,
            "liquidity_usd": round(liq, 2),
            "net_profit_pct": round(net / best_ask * 100, 2),
            "net_per_share": round(net, 4),
            "fee_per_share": round(fee, 4),
        })

    return opportunities


def main():
    now = datetime.now(timezone.utc)
    print(f"Penny Scanner -- {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    print("\n[1/3] Finding markets ending within 72h...")
    candidates = fetch_candidates()
    print(f"  {len(candidates)} outcomes priced 95-99c")

    if not candidates:
        print("\nNo penny candidates found.")
        return

    print(f"\n[2/3] Checking orderbook liquidity...")
    opps = check_orderbooks(candidates)

    opps.sort(key=lambda x: x["net_profit_pct"], reverse=True)

    print("\n" + "=" * 70)
    print(f"PENNY OPPORTUNITIES -- {len(opps)} found")
    print("=" * 70)

    for i, o in enumerate(opps, 1):
        print(f"\n#{i} | {o['question']}")
        print(f"   Outcome: {o['outcome']} @ ${o['best_ask']:.2f}")
        print(f"   Expires in: {o['hours_left']}h")
        print(f"   Net Profit: {o['net_profit_pct']:.2f}% (${o['net_per_share']:.4f}/share)")
        print(f"   Fee: ${o['fee_per_share']:.4f}/share")
        print(f"   Ask Liquidity: ~${o['liquidity_usd']:.0f}")
        print(f"   https://polymarket.com/event/{o['slug']}")

    if not opps:
        print("\nNo opportunities with positive net profit after fees.")

    return opps


if __name__ == "__main__":
    main()

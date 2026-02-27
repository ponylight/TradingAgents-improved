#!/usr/bin/env python3
"""
Latency measurement: Binance BTC spot vs Polymarket BTC 5-min Up/Down odds.
"""

import asyncio
import json
import time
import statistics
import requests
import websockets
from dataclasses import dataclass, field


@dataclass
class Measurement:
    binance_prices: list = field(default_factory=list)
    polymarket_snapshots: list = field(default_factory=list)


def get_current_market():
    epoch = int(time.time())
    for i in range(-2, 5):
        slot = ((epoch // 300) + i) * 300
        slug = f'btc-updown-5m-{slot}'
        r = requests.get(f'https://gamma-api.polymarket.com/markets?slug={slug}', timeout=10)
        markets = r.json()
        if markets:
            m = markets[0]
            token_ids = json.loads(m['clobTokenIds'])
            print(f"Found market: {m['question']}")
            return {'up_token': token_ids[0], 'down_token': token_ids[1], 'question': m['question']}
    return None


def get_orderbook(token_id):
    try:
        r = requests.get('https://clob.polymarket.com/book', params={'token_id': token_id}, timeout=5)
        data = r.json()
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        best_bid = float(bids[0]['price']) if bids else 0.0
        best_ask = float(asks[0]['price']) if asks else 1.0
        return best_bid, best_ask, (best_bid + best_ask) / 2
    except:
        return None, None, None


async def binance_stream(m, stop):
    uri = 'wss://stream.binance.com:9443/ws/btcusdt@trade'
    print("Connecting to Binance websocket...")
    try:
        async with websockets.connect(uri) as ws:
            print("Connected to Binance")
            while not stop.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    m.binance_prices.append((time.time(), float(data['p']), data['T'] / 1000.0))
                except asyncio.TimeoutError:
                    continue
    except Exception as e:
        print(f"Binance error: {e}")


async def polymarket_poller(m, market, stop):
    print("Starting Polymarket polling (500ms)...")
    while not stop.is_set():
        ts = time.time()
        bid, ask, mid = get_orderbook(market['up_token'])
        if mid is not None:
            m.polymarket_snapshots.append((ts, mid, bid, ask))
        await asyncio.sleep(0.5)


def analyze(m):
    binance = m.binance_prices
    poly = m.polymarket_snapshots
    if not binance or not poly:
        print("No data!"); return

    print(f"\n{'='*60}")
    print("LATENCY MEASUREMENT RESULTS")
    print(f"{'='*60}")
    print(f"Binance trades: {len(binance)}")
    print(f"Polymarket snapshots: {len(poly)}")
    print(f"Duration: {binance[-1][0] - binance[0][0]:.1f}s")

    prices = [p[1] for p in binance]
    print(f"\nBinance BTC: ${min(prices):.2f} - ${max(prices):.2f} (std=${statistics.stdev(prices):.2f})")

    mids = [p[1] for p in poly]
    unique_mids = len(set(round(m, 4) for m in mids))
    print(f"Polymarket Up mid: {min(mids):.4f} - {max(mids):.4f} ({unique_mids} unique values)")

    # Bucket binance into 500ms
    start = binance[0][0]
    buckets = {}
    for ts, price, _ in binance:
        b = round((ts - start) / 0.5) * 0.5
        buckets.setdefault(b, []).append(price)
    bavg = {k: statistics.mean(v) for k, v in buckets.items()}
    sorted_b = sorted(bavg.keys())

    # Detect moves
    for threshold in [5, 10, 20, 50]:
        moves = []
        for i in range(1, len(sorted_b)):
            delta = bavg[sorted_b[i]] - bavg[sorted_b[i-1]]
            if abs(delta) >= threshold:
                moves.append((start + sorted_b[i], delta))
        print(f"  Price moves >=${threshold}: {len(moves)}")

    # Detect odds changes
    odds_changes = []
    for i in range(1, len(poly)):
        delta = poly[i][1] - poly[i-1][1]
        if abs(delta) >= 0.005:
            odds_changes.append((poly[i][0], delta))
    print(f"  Odds changes >=0.5%: {len(odds_changes)}")

    # Cross-correlate
    for threshold in [5, 10, 20]:
        moves = []
        for i in range(1, len(sorted_b)):
            delta = bavg[sorted_b[i]] - bavg[sorted_b[i-1]]
            if abs(delta) >= threshold:
                moves.append((start + sorted_b[i], 1 if delta > 0 else -1))

        latencies = []
        for mt, md in moves:
            for ot, od in odds_changes:
                od_dir = 1 if od > 0 else -1
                if od_dir == md and 0 < (ot - mt) < 10:
                    latencies.append((ot - mt) * 1000)
                    break

        if latencies:
            print(f"\n  Threshold ${threshold}: {len(latencies)} matched events")
            print(f"    Mean latency: {statistics.mean(latencies):.0f}ms")
            print(f"    Median: {statistics.median(latencies):.0f}ms")
            print(f"    Range: {min(latencies):.0f}-{max(latencies):.0f}ms")

    # Spread analysis
    spreads = [ask - bid for _, _, bid, ask in poly if bid and ask]
    if spreads:
        print(f"\nPolymarket spread: mean={statistics.mean(spreads):.4f} ({statistics.mean(spreads)*100:.1f}%)")

    # Poll interval stats
    if len(poly) >= 2:
        intervals = [(poly[i][0] - poly[i-1][0]) * 1000 for i in range(1, len(poly))]
        print(f"Poll interval: mean={statistics.mean(intervals):.0f}ms, max={max(intervals):.0f}ms")

    # Sample data
    print(f"\nSample polymarket snapshots:")
    for ts, mid, bid, ask in poly[:5]:
        print(f"  mid={mid:.4f} bid={bid:.4f} ask={ask:.4f} spread={ask-bid:.4f}")

    # Check if market maker is static
    if unique_mids <= 3:
        print(f"\n⚠️  Polymarket odds barely moved ({unique_mids} unique values).")
        print("   This suggests market makers set static prices, not reactive to spot.")
        print("   Exploitable latency arb is UNLIKELY on these markets.")


async def main():
    duration = 150
    print(f"Running latency measurement for {duration}s...\n")
    market = get_current_market()
    if not market:
        print("ERROR: No BTC 5-min market found!"); return

    bid, ask, mid = get_orderbook(market['up_token'])
    print(f"Initial orderbook: bid={bid}, ask={ask}, mid={mid}\n")

    meas = Measurement()
    stop = asyncio.Event()
    tasks = [
        asyncio.create_task(binance_stream(meas, stop)),
        asyncio.create_task(polymarket_poller(meas, market, stop)),
    ]

    await asyncio.sleep(duration)
    stop.set()
    for t in tasks: t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    analyze(meas)

if __name__ == '__main__':
    asyncio.run(main())

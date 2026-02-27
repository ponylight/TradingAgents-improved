#!/usr/bin/env python3
"""
Latency Measurement v2: Binance BTC spot vs Polymarket BTC 5-min Up/Down orderbook.

Key improvements over v1:
- Handles market rotation (recalculates slug every 5-min window)
- Tracks BOTH Up and Down token orderbooks
- Records raw event log for post-hoc analysis
- Better latency detection using price derivative vs odds derivative
- Measures HTTP round-trip time for orderbook polls
"""

import asyncio
import json
import time
import statistics
import requests
import websockets
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Event:
    ts: float
    kind: str  # 'binance' | 'poly_up' | 'poly_down' | 'market_rotate'
    data: dict = field(default_factory=dict)


@dataclass
class State:
    events: list = field(default_factory=list)
    poll_latencies_ms: list = field(default_factory=list)


def current_slot():
    return (int(time.time()) // 300) * 300


def get_market_for_slot(slot):
    slug = f'btc-updown-5m-{slot}'
    try:
        r = requests.get(f'https://gamma-api.polymarket.com/markets?slug={slug}', timeout=10)
        markets = r.json()
        if markets and markets[0].get('clobTokenIds'):
            m = markets[0]
            tokens = json.loads(m['clobTokenIds'])
            return {
                'slug': slug, 'slot': slot,
                'condition_id': m.get('conditionId', ''),
                'question': m.get('question', slug),
                'up_token': tokens[0], 'down_token': tokens[1],
                'expires': slot + 300,
            }
    except Exception as e:
        print(f"  Error fetching {slug}: {e}")
    return None


def get_orderbook(token_id):
    try:
        r = requests.get('https://clob.polymarket.com/book', params={'token_id': token_id}, timeout=5)
        data = r.json()
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        best_bid = float(bids[0]['price']) if bids else 0.0
        best_ask = float(asks[0]['price']) if asks else 1.0
        bid_size = float(bids[0]['size']) if bids else 0.0
        ask_size = float(asks[0]['size']) if asks else 0.0
        return best_bid, best_ask, (best_bid + best_ask) / 2, bid_size, ask_size, len(bids), len(asks)
    except:
        return None, None, None, 0, 0, 0, 0


def find_active_market():
    now_slot = current_slot()
    for offset in [0, 300, -300, 600]:
        slot = now_slot + offset
        m = get_market_for_slot(slot)
        if m:
            bid, ask, mid, *_ = get_orderbook(m['up_token'])
            if bid and bid > 0:
                print(f"  Found active market: {m['question']} (slot {slot})")
                print(f"    Up: {m['up_token'][:24]}...  Down: {m['down_token'][:24]}...")
                print(f"    Orderbook: bid={bid} ask={ask} mid={mid}")
                return m
            else:
                print(f"  Slot {slot}: empty orderbook, skipping...")
    return None


async def binance_stream(state, stop):
    uri = 'wss://stream.binance.com:9443/ws/btcusdt@trade'
    print("Connecting to Binance websocket...")
    try:
        async with websockets.connect(uri) as ws:
            print("  Connected to Binance")
            while not stop.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    state.events.append(Event(
                        ts=time.time(), kind='binance',
                        data={'price': float(data['p']), 'qty': float(data['q']),
                              'exchange_ts': data['T'] / 1000.0}
                    ))
                except asyncio.TimeoutError:
                    continue
    except Exception as e:
        print(f"Binance error: {e}")


async def polymarket_poller(state, stop):
    print("Finding active market...")
    market = find_active_market()
    if not market:
        print("ERROR: No active market found!"); return

    state.events.append(Event(ts=time.time(), kind='market_rotate', data={'slot': market['slot'], 'question': market['question']}))

    while not stop.is_set():
        now = time.time()
        # Check for market rotation
        new_slot = current_slot()
        if new_slot != market['slot']:
            print(f"\n  Market rotation at {datetime.now().strftime('%H:%M:%S')} (new slot {new_slot})")
            new_market = find_active_market()
            if new_market and new_market['slot'] != market['slot']:
                market = new_market
                state.events.append(Event(ts=time.time(), kind='market_rotate',
                                          data={'slot': market['slot'], 'question': market['question']}))

        # Poll Up
        t0 = time.time()
        bid, ask, mid, bsz, asz, nb, na = get_orderbook(market['up_token'])
        rtt = (time.time() - t0) * 1000
        state.poll_latencies_ms.append(rtt)
        if mid is not None:
            state.events.append(Event(ts=time.time(), kind='poly_up',
                data={'bid': bid, 'ask': ask, 'mid': mid, 'bid_size': bsz, 'ask_size': asz,
                      'n_bids': nb, 'n_asks': na, 'rtt_ms': rtt, 'slot': market['slot']}))

        # Poll Down
        t0 = time.time()
        bid2, ask2, mid2, bsz2, asz2, nb2, na2 = get_orderbook(market['down_token'])
        rtt2 = (time.time() - t0) * 1000
        state.poll_latencies_ms.append(rtt2)
        if mid2 is not None:
            state.events.append(Event(ts=time.time(), kind='poly_down',
                data={'bid': bid2, 'ask': ask2, 'mid': mid2, 'bid_size': bsz2, 'ask_size': asz2,
                      'n_bids': nb2, 'n_asks': na2, 'rtt_ms': rtt2, 'slot': market['slot']}))

        await asyncio.sleep(0.5)


def analyze(state):
    events = state.events
    if not events:
        print("No data!"); return

    binance_evts = [e for e in events if e.kind == 'binance']
    poly_up = [e for e in events if e.kind == 'poly_up']
    poly_down = [e for e in events if e.kind == 'poly_down']
    rotations = [e for e in events if e.kind == 'market_rotate']

    print(f"\n{'='*70}")
    print("LATENCY MEASUREMENT v2 — RESULTS")
    print(f"{'='*70}")
    print(f"Duration:            {events[-1].ts - events[0].ts:.1f}s")
    print(f"Binance trades:      {len(binance_evts)}")
    print(f"Polymarket Up polls: {len(poly_up)}")
    print(f"Polymarket Dn polls: {len(poly_down)}")
    print(f"Market rotations:    {len(rotations)}")

    # Binance stats
    if len(binance_evts) >= 2:
        prices = [e.data['price'] for e in binance_evts]
        ws_lats = [(e.ts - e.data['exchange_ts']) * 1000 for e in binance_evts]
        print(f"\n── Binance BTC ──")
        print(f"  Range: ${min(prices):.2f} — ${max(prices):.2f} (Δ${max(prices)-min(prices):.2f})")
        print(f"  Stdev: ${statistics.stdev(prices):.2f}")
        print(f"  WS latency: mean={statistics.mean(ws_lats):.0f}ms p50={statistics.median(ws_lats):.0f}ms "
              f"p99={sorted(ws_lats)[int(len(ws_lats)*0.99)]:.0f}ms")

    # HTTP RTT
    if state.poll_latencies_ms:
        rtts = state.poll_latencies_ms
        srt = sorted(rtts)
        print(f"\n── Polymarket HTTP RTT ──")
        print(f"  Mean={statistics.mean(rtts):.0f}ms  Median={statistics.median(rtts):.0f}ms  "
              f"p95={srt[int(len(srt)*0.95)]:.0f}ms  p99={srt[int(len(srt)*0.99)]:.0f}ms  Max={max(rtts):.0f}ms")

    # Up token analysis
    for label, evts in [('Up', poly_up), ('Down', poly_down)]:
        if not evts: continue
        mids = [e.data['mid'] for e in evts]
        unique = len(set(round(m, 4) for m in mids))
        spreads = [e.data['ask'] - e.data['bid'] for e in evts]
        print(f"\n── Polymarket {label} Token ──")
        print(f"  Mid: {min(mids):.4f} — {max(mids):.4f} ({unique} unique)")
        print(f"  Spread: mean={statistics.mean(spreads):.4f} ({statistics.mean(spreads)*100:.1f}%)")
        print(f"  Book: {evts[0].data['n_bids']} bids, {evts[0].data['n_asks']} asks")

    # Up+Down complement
    if poly_up and poly_down:
        di = 0
        pairs = []
        for ue in poly_up:
            while di < len(poly_down) - 1 and abs(poly_down[di+1].ts - ue.ts) < abs(poly_down[di].ts - ue.ts):
                di += 1
            if di < len(poly_down):
                pairs.append(ue.data['mid'] + poly_down[di].data['mid'])
        if pairs:
            print(f"\n── Up + Down Complement ──")
            print(f"  Mean: {statistics.mean(pairs):.4f} (should be ~1.0)")

    # Cross-correlation
    if len(binance_evts) >= 2 and len(poly_up) >= 2:
        print(f"\n── Cross-Correlation: BTC Price → Odds ──")
        start = binance_evts[0].ts
        buckets = {}
        for e in binance_evts:
            b = int((e.ts - start) / 0.5)
            buckets.setdefault(b, []).append(e.data['price'])
        bavg = {k: statistics.mean(v) for k, v in buckets.items()}
        sorted_b = sorted(bavg.keys())

        # Odds changes
        odds_changes = []
        for i in range(1, len(poly_up)):
            d = poly_up[i].data['mid'] - poly_up[i-1].data['mid']
            if abs(d) >= 0.003:
                odds_changes.append((poly_up[i].ts, d))

        for threshold in [5, 10, 20, 50, 100]:
            moves = []
            for i in range(1, len(sorted_b)):
                delta = bavg[sorted_b[i]] - bavg[sorted_b[i-1]]
                if abs(delta) >= threshold:
                    moves.append((start + sorted_b[i] * 0.5, delta))
            if not moves: continue

            latencies = []
            for mt, md in moves:
                direction = 1 if md > 0 else -1
                for ot, od in odds_changes:
                    lag = ot - mt
                    if (1 if od > 0 else -1) == direction and -2 < lag < 30:
                        latencies.append(lag * 1000)
                        break

            print(f"\n  Threshold ${threshold}:")
            print(f"    Price moves: {len(moves)}, Matched: {len(latencies)}")
            if latencies:
                print(f"    Latency: mean={statistics.mean(latencies):+.0f}ms median={statistics.median(latencies):+.0f}ms")
                print(f"    Range: {min(latencies):+.0f}ms to {max(latencies):+.0f}ms")
                neg = sum(1 for l in latencies if l < 0)
                if neg: print(f"    ⚡ {neg}/{len(latencies)} times Polymarket moved BEFORE Binance!")

    # Static detection
    if poly_up:
        unique = len(set(round(e.data['mid'], 4) for e in poly_up))
        if unique <= 3:
            print(f"\n⚠️  Polymarket odds barely moved ({unique} unique). Static market makers — arb unlikely.")
        elif unique <= 10:
            print(f"\n⚠️  Low odds variance ({unique} unique) — limited reactivity.")

    # Samples
    print(f"\n── Sample Snapshots ──")
    for kind, label in [('poly_up', 'Up'), ('poly_down', 'Down')]:
        evts = [e for e in events if e.kind == kind][:5]
        if evts:
            print(f"  {label}:")
            for e in evts:
                d = e.data
                t = datetime.fromtimestamp(e.ts).strftime('%H:%M:%S.%f')[:-3]
                print(f"    {t} bid={d['bid']:.4f} ask={d['ask']:.4f} mid={d['mid']:.4f} rtt={d['rtt_ms']:.0f}ms")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if state.poll_latencies_ms:
        med_rtt = statistics.median(state.poll_latencies_ms)
        print(f"• Polymarket HTTP RTT: ~{med_rtt:.0f}ms median")
    if binance_evts:
        med_ws = statistics.median([(e.ts - e.data['exchange_ts']) * 1000 for e in binance_evts])
        print(f"• Binance WS latency: ~{med_ws:.0f}ms median")
    print(f"• Minimum theoretical latency = Binance WS + Polymarket HTTP RTT + processing")
    print(f"• With 500ms polling: add 0-500ms random delay")
    if state.poll_latencies_ms and binance_evts:
        floor = med_rtt + med_ws
        print(f"• Estimated floor: ~{floor:.0f}ms (without polling delay)")
        print(f"• Estimated with polling: ~{floor + 250:.0f}ms average")


async def main():
    duration = 330  # 5.5 min
    print(f"Latency Measurement v2")
    print(f"Running for {duration}s ({duration/60:.1f} min)")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    state = State()
    stop = asyncio.Event()
    tasks = [
        asyncio.create_task(binance_stream(state, stop)),
        asyncio.create_task(polymarket_poller(state, stop)),
    ]

    await asyncio.sleep(duration)
    stop.set()
    for t in tasks: t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    analyze(state)

    # Save raw data
    outpath = f'{__file__}'.replace('.py', '_data.json')
    serializable = []
    for e in state.events:
        serializable.append({'ts': e.ts, 'kind': e.kind, 'data': e.data})
    with open(outpath, 'w') as f:
        json.dump(serializable, f)
    print(f"\nRaw data saved to {outpath} ({len(serializable)} events)")


if __name__ == '__main__':
    asyncio.run(main())

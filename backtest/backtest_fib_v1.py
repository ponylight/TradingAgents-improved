"""Fibonacci Short-Term Backtest V1 — Natural Trading Theory (自然交易理论)

4H timeframe, both long and short.

SPACE (Fibonacci Retracement):
  - Detect swing highs/lows on 4H
  - Draw Fib from recent swing low→high (for longs) or high→low (for shorts)
  - Entry zone: price reaches 0.618 or 0.382 retracement level (±1%)

TIME (Fibonacci Time):
  - Measure bars between swing points
  - 0.618 × swing duration = expected reversal bar
  - Entry has higher probability when Space + Time align (±2 bars)

ENERGY (Key K-line):
  - At the Space+Time intersection, require a "Key K-line":
    - Bullish: large body candle closing above open, OR hammer/engulfing
    - Bearish: large body candle closing below open, OR shooting star/engulfing
  - Measured as: body > 60% of range, AND range > 1.5x avg range of last 10 candles

POSITION SIZING:
  - Capital / 30 parts per trade
  - Max 2 additions at 1% intervals
  - Stop: 3% from entry
  - TP: batch exit 25% per 1% move (4 exits) OR trailing at 1.5x ATR after +2%

LEVERAGE: 5x default, 10x when Space+Time+Energy all align strongly

Both long and short trades. Direction bias from 200 SMA:
  - Price > 200 SMA: prefer longs (10x), shorts at 3x
  - Price < 200 SMA: prefer shorts (10x), longs at 3x
"""

import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backtest.indicators import compute_atr, compute_ma

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_fib_v1")

CAPITAL = 10_000.0
COMMISSION = 0.001       # 0.1%
FUNDING_PER_4H = 0.0001  # 0.01% per 8h → 0.005% per 4h, but funding settles every 8h
                          # 4H bars: 2 bars = 1 funding period. So 0.5 funding per bar.
FUNDING_PER_BAR = 0.00005  # 0.01%/8h = 0.005%/bar (each bar=4h, funding every 8h)

# Swing detection
SWING_LOOKBACK = 10       # 10 bars = 40 hours for swing detection
MIN_SWING_SIZE = 0.02    # swing must be at least 2% to count

# Fibonacci levels
FIB_LEVELS = [0.382, 0.618]
FIB_TOLERANCE = 0.01     # ±1% of the fib level

# Time alignment
TIME_TOLERANCE = 3       # ±3 bars for time alignment

# Key K-line
KEY_KLINE_BODY_RATIO = 0.55   # body > 55% of range
KEY_KLINE_RANGE_MULT = 1.3    # range > 1.3x avg range

# Position sizing
PARTS = 30               # divide capital into 30 parts
MAX_ADDS = 2
ADD_INTERVAL = 0.01      # add at 1% intervals
STOP_PCT = 0.03          # 3% stop
TP_BATCH_PCT = 0.01      # take 25% off per 1% move
TP_BATCHES = 4

# Leverage
DEFAULT_LEV = 5
STRONG_LEV = 10
COUNTER_LEV = 3

# Trailing
TRAIL_ACTIVATE_PCT = 0.02  # activate trailing after 2% profit
TRAIL_ATR_MULT = 1.5


@dataclass
class Trade:
    side: str  # "long" or "short"
    entry: float
    entry_idx: int
    entry_time: pd.Timestamp
    size: float  # BTC size (positive for long, negative for short)
    margin: float
    leverage: float
    stop: float
    fib_level: float
    time_aligned: bool
    energy_score: float
    closed: bool = False
    exit_price: float = 0
    exit_idx: int = 0
    exit_time: pd.Timestamp = None
    pnl: float = 0
    funding: float = 0
    reason: str = ""
    highest: float = 0  # for trailing (long)
    lowest: float = 999999  # for trailing (short)
    trail_active: bool = False
    trail_stop: float = 0
    tp_exits: int = 0  # how many TP batches taken
    original_size: float = 0


def detect_swings(highs, lows, lookback=SWING_LOOKBACK):
    """Detect swing highs and lows on 4H data."""
    swing_highs = []  # (index, price)
    swing_lows = []
    
    for i in range(lookback, len(highs) - lookback):
        # Swing high: highest in window
        window_h = highs[max(0, i-lookback):i+lookback+1]
        if highs[i] >= np.max(window_h):
            if not swing_highs or i - swing_highs[-1][0] >= lookback // 2:
                swing_highs.append((i, highs[i]))
        
        # Swing low: lowest in window
        window_l = lows[max(0, i-lookback):i+lookback+1]
        if lows[i] <= np.min(window_l):
            if not swing_lows or i - swing_lows[-1][0] >= lookback // 2:
                swing_lows.append((i, lows[i]))
    
    return swing_highs, swing_lows


def get_recent_swing_pair(idx, swing_highs, swing_lows, max_age=120):
    """Get the most recent swing high-low pair for fib drawing.
    max_age: max bars old (120 bars = 20 days on 4H).
    Returns (swing_low_idx, swing_low_price, swing_high_idx, swing_high_price, direction)
    direction: 'up' if low came first (draw fib for long retracement)
               'down' if high came first (draw fib for short retracement)
    """
    # Find most recent swing high and low before current bar
    recent_highs = [(i, p) for i, p in swing_highs if i < idx and idx - i <= max_age]
    recent_lows = [(i, p) for i, p in swing_lows if i < idx and idx - i <= max_age]
    
    if not recent_highs or not recent_lows:
        return None
    
    last_high = recent_highs[-1]
    last_low = recent_lows[-1]
    
    # Ensure minimum swing size
    swing_size = abs(last_high[1] - last_low[1]) / min(last_high[1], last_low[1])
    if swing_size < MIN_SWING_SIZE:
        return None
    
    if last_low[0] < last_high[0]:
        # Low came first → upswing → look for long retracement
        return (last_low[0], last_low[1], last_high[0], last_high[1], 'up')
    else:
        # High came first → downswing → look for short retracement  
        return (last_high[0], last_high[1], last_low[0], last_low[1], 'down')


def check_fib_level(price, swing_pair):
    """Check if price is at a fib retracement level.
    Returns (is_at_fib, fib_level, trade_side) or (False, 0, '')
    """
    if swing_pair is None:
        return False, 0, ''
    
    s1_idx, s1_price, s2_idx, s2_price, direction = swing_pair
    
    if direction == 'up':
        # Upswing: fib retracement for long entry
        # 0.618 retracement = swing_high - 0.618 * (swing_high - swing_low)
        move = s2_price - s1_price  # positive
        for fib in FIB_LEVELS:
            fib_price = s2_price - fib * move
            if abs(price - fib_price) / fib_price <= FIB_TOLERANCE:
                return True, fib, 'long'
    else:
        # Downswing: fib retracement for short entry
        move = s1_price - s2_price  # s1 is high, s2 is low, move is positive
        for fib in FIB_LEVELS:
            fib_price = s2_price + fib * move
            if abs(price - fib_price) / fib_price <= FIB_TOLERANCE:
                return True, fib, 'short'
    
    return False, 0, ''


def check_time_alignment(idx, swing_pair):
    """Check if current bar aligns with fibonacci time projection.
    0.618 × swing duration from the end of the swing.
    """
    if swing_pair is None:
        return False
    
    s1_idx, _, s2_idx, _, _ = swing_pair
    swing_duration = s2_idx - s1_idx
    if swing_duration < 5:
        return False
    
    # Project 0.618 of swing duration from s2
    projected_bar = s2_idx + int(swing_duration * 0.618)
    
    return abs(idx - projected_bar) <= TIME_TOLERANCE


def check_key_kline(idx, opens, highs, lows, closes, side):
    """Check if current candle is a 'Key K-line' (strong reversal candle).
    Returns energy score 0-1.
    """
    o, h, l, c = opens[idx], highs[idx], lows[idx], closes[idx]
    rng = h - l
    if rng == 0:
        return 0
    
    body = abs(c - o)
    body_ratio = body / rng
    
    # Average range of last 10 candles
    start = max(0, idx - 10)
    avg_range = np.mean(highs[start:idx] - lows[start:idx])
    if avg_range == 0:
        return 0
    range_mult = rng / avg_range
    
    # Direction check
    if side == 'long' and c <= o:
        return 0  # bearish candle can't be bullish key kline
    if side == 'short' and c >= o:
        return 0  # bullish candle can't be bearish key kline
    
    # Score
    score = 0
    if body_ratio >= KEY_KLINE_BODY_RATIO:
        score += 0.5
    if range_mult >= KEY_KLINE_RANGE_MULT:
        score += 0.5
    
    # Bonus: engulfing pattern
    if idx > 0:
        prev_body = abs(closes[idx-1] - opens[idx-1])
        if body > prev_body * 1.2:  # current body > 120% of previous
            score = min(1.0, score + 0.2)
    
    return score


def run_backtest():
    # Load data
    df = pd.read_csv("backtest/data/btc_usdt_4h_2017.csv", parse_dates=["timestamp"])
    print(f"Loaded {len(df)} 4H candles: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    
    # Indicators
    df["atr"] = compute_atr(df, period=14)
    df["sma200"] = compute_ma(df, period=200)
    
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atr = df["atr"].values
    sma200 = df["sma200"].values
    timestamps = df["timestamp"].values
    
    # Detect swings
    swing_highs, swing_lows = detect_swings(highs, lows)
    print(f"Detected {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")
    
    capital = CAPITAL
    positions = []
    closed_trades = []
    equity_curve = []
    
    stats = {
        "signals": 0, "entries": 0, "longs": 0, "shorts": 0,
        "space_only": 0, "space_time": 0, "space_time_energy": 0,
        "tp_exits": 0, "trail_exits": 0, "stop_exits": 0,
        "total_funding": 0,
    }
    
    start_idx = 200  # need SMA200
    
    for i in range(start_idx, len(df)):
        price = closes[i]
        high = highs[i]
        low = lows[i]
        ts = pd.Timestamp(timestamps[i])
        w_atr = atr[i] if not np.isnan(atr[i]) else price * 0.01
        
        # Funding on open positions
        for p in positions:
            if p.closed: continue
            notional = abs(p.size) * price
            cost = notional * FUNDING_PER_BAR
            capital -= cost
            p.funding += cost
            stats["total_funding"] += cost
        
        # Manage open positions
        for p in list(positions):
            if p.closed: continue
            
            if p.side == 'long':
                p.highest = max(p.highest, high)
                
                # Stop loss
                if low <= p.stop:
                    pnl = (p.stop - p.entry) * abs(p.size)
                    pnl -= abs(p.size) * p.stop * COMMISSION
                    p.pnl = pnl; p.exit_price = p.stop; p.exit_idx = i
                    p.exit_time = ts; p.reason = "stop"; p.closed = True
                    capital += pnl; stats["stop_exits"] += 1
                    continue
                
                # TP batches: take 25% per 1% move
                while p.tp_exits < TP_BATCHES:
                    tp_level = p.entry * (1 + TP_BATCH_PCT * (p.tp_exits + 1))
                    if high >= tp_level:
                        batch_size = p.original_size * 0.25
                        if abs(p.size) > batch_size:
                            batch_pnl = (tp_level - p.entry) * batch_size
                            batch_pnl -= batch_size * tp_level * COMMISSION
                            capital += batch_pnl
                            p.size -= batch_size
                            p.tp_exits += 1
                            stats["tp_exits"] += 1
                        else:
                            break
                    else:
                        break
                
                # Close if all TP batches taken
                if p.tp_exits >= TP_BATCHES or abs(p.size) < 0.0001:
                    if not p.closed:
                        remaining_pnl = (price - p.entry) * abs(p.size)
                        remaining_pnl -= abs(p.size) * price * COMMISSION
                        p.pnl = remaining_pnl; p.exit_price = price
                        p.exit_idx = i; p.exit_time = ts
                        p.reason = "tp_complete"; p.closed = True
                        capital += remaining_pnl
                    continue
                
                # Trailing stop (activate after 2%)
                profit_pct = (price - p.entry) / p.entry
                if profit_pct > TRAIL_ACTIVATE_PCT and not p.trail_active:
                    p.trail_active = True
                    p.trail_stop = p.highest - w_atr * TRAIL_ATR_MULT
                
                if p.trail_active:
                    p.trail_stop = max(p.trail_stop, p.highest - w_atr * TRAIL_ATR_MULT)
                    p.trail_stop = max(p.trail_stop, p.entry)  # at least breakeven
                    if low <= p.trail_stop:
                        pnl = (p.trail_stop - p.entry) * abs(p.size)
                        pnl -= abs(p.size) * p.trail_stop * COMMISSION
                        p.pnl = pnl; p.exit_price = p.trail_stop
                        p.exit_idx = i; p.exit_time = ts
                        p.reason = "trail"; p.closed = True
                        capital += pnl; stats["trail_exits"] += 1
                
            else:  # short
                p.lowest = min(p.lowest, low)
                
                # Stop loss
                if high >= p.stop:
                    pnl = (p.entry - p.stop) * abs(p.size)
                    pnl -= abs(p.size) * p.stop * COMMISSION
                    p.pnl = pnl; p.exit_price = p.stop; p.exit_idx = i
                    p.exit_time = ts; p.reason = "stop"; p.closed = True
                    capital += pnl; stats["stop_exits"] += 1
                    continue
                
                # TP batches
                while p.tp_exits < TP_BATCHES:
                    tp_level = p.entry * (1 - TP_BATCH_PCT * (p.tp_exits + 1))
                    if low <= tp_level:
                        batch_size = p.original_size * 0.25
                        if abs(p.size) > batch_size:
                            batch_pnl = (p.entry - tp_level) * batch_size
                            batch_pnl -= batch_size * tp_level * COMMISSION
                            capital += batch_pnl
                            p.size -= batch_size
                            p.tp_exits += 1
                            stats["tp_exits"] += 1
                        else:
                            break
                    else:
                        break
                
                if p.tp_exits >= TP_BATCHES or abs(p.size) < 0.0001:
                    if not p.closed:
                        remaining_pnl = (p.entry - price) * abs(p.size)
                        remaining_pnl -= abs(p.size) * price * COMMISSION
                        p.pnl = remaining_pnl; p.exit_price = price
                        p.exit_idx = i; p.exit_time = ts
                        p.reason = "tp_complete"; p.closed = True
                        capital += remaining_pnl
                    continue
                
                # Trailing
                profit_pct = (p.entry - price) / p.entry
                if profit_pct > TRAIL_ACTIVATE_PCT and not p.trail_active:
                    p.trail_active = True
                    p.trail_stop = p.lowest + w_atr * TRAIL_ATR_MULT
                
                if p.trail_active:
                    p.trail_stop = min(p.trail_stop, p.lowest + w_atr * TRAIL_ATR_MULT)
                    p.trail_stop = min(p.trail_stop, p.entry)
                    if high >= p.trail_stop:
                        pnl = (p.entry - p.trail_stop) * abs(p.size)
                        pnl -= abs(p.size) * p.trail_stop * COMMISSION
                        p.pnl = pnl; p.exit_price = p.trail_stop
                        p.exit_idx = i; p.exit_time = ts
                        p.reason = "trail"; p.closed = True
                        capital += pnl; stats["trail_exits"] += 1
        
        # Remove closed
        for p in [x for x in positions if x.closed]:
            closed_trades.append(p)
            positions.remove(p)
        
        # Check for new entry (max 1 position at a time for simplicity)
        if not positions and capital > 100:
            swing_pair = get_recent_swing_pair(i, swing_highs, swing_lows)
            if swing_pair:
                at_fib, fib_level, side = check_fib_level(price, swing_pair)
                
                if at_fib:
                    stats["signals"] += 1
                    stats["space_only"] += 1
                    
                    time_ok = check_time_alignment(i, swing_pair)
                    if time_ok:
                        stats["space_time"] += 1
                    
                    energy = check_key_kline(i, opens, highs, lows, closes, side)
                    if energy > 0.3:
                        stats["space_time_energy"] += 1
                    
                    # Entry decision: need at least Space + (Time OR Energy)
                    if time_ok or energy > 0.3:
                        # Determine leverage
                        with_trend = (side == 'long' and not np.isnan(sma200[i]) and price > sma200[i]) or \
                                     (side == 'short' and not np.isnan(sma200[i]) and price < sma200[i])
                        
                        if time_ok and energy > 0.5:
                            lev = STRONG_LEV if with_trend else DEFAULT_LEV
                        elif with_trend:
                            lev = DEFAULT_LEV
                        else:
                            lev = COUNTER_LEV
                        
                        margin = capital / PARTS
                        notional = margin * lev
                        size_btc = notional / price
                        
                        if side == 'long':
                            stop = price * (1 - STOP_PCT)
                        else:
                            stop = price * (1 + STOP_PCT)
                        
                        # Commission
                        capital -= size_btc * price * COMMISSION
                        
                        trade = Trade(
                            side=side, entry=price, entry_idx=i, entry_time=ts,
                            size=size_btc, margin=margin, leverage=lev,
                            stop=stop, fib_level=fib_level,
                            time_aligned=time_ok, energy_score=energy,
                            highest=price, lowest=price,
                            original_size=size_btc,
                        )
                        positions.append(trade)
                        stats["entries"] += 1
                        if side == 'long': stats["longs"] += 1
                        else: stats["shorts"] += 1
        
        # Equity
        unrealized = sum(
            (price - p.entry) * abs(p.size) if p.side == 'long'
            else (p.entry - price) * abs(p.size)
            for p in positions if not p.closed
        )
        equity_curve.append({"ts": ts, "eq": capital + unrealized, "price": price})
    
    # Close remaining
    for p in positions:
        if not p.closed:
            price = closes[-1]
            if p.side == 'long':
                pnl = (price - p.entry) * abs(p.size)
            else:
                pnl = (p.entry - price) * abs(p.size)
            p.pnl = pnl; p.exit_price = price; p.reason = "end"; p.closed = True
            capital += pnl
            closed_trades.append(p)
    
    # Metrics
    eq = np.array([e["eq"] for e in equity_curve])
    peak = np.maximum.accumulate(eq)
    max_dd = np.max((peak - eq) / peak * 100)
    ret = (eq[-1] - CAPITAL) / CAPITAL * 100
    
    pnls = [t.pnl for t in closed_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) * 100 if pnls else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    print(f"\n{'='*70}")
    print(f"  FIBONACCI SHORT-TERM V1 — Natural Trading Theory")
    print(f"{'='*70}")
    print(f"  Period:       {df['timestamp'].iloc[start_idx].strftime('%Y-%m')} → {df['timestamp'].iloc[-1].strftime('%Y-%m')}")
    print(f"  Capital:      ${CAPITAL:,.0f}")
    print(f"  Final Equity: ${eq[-1]:,.0f}")
    print(f"  Return:       {ret:+,.1f}%")
    print(f"  Max DD:       {max_dd:.1f}%")
    print(f"  Sharpe:       {np.mean(np.diff(eq)/eq[:-1]) / np.std(np.diff(eq)/eq[:-1]) * np.sqrt(365*6):.2f}" if len(eq)>1 else "N/A")
    print(f"  Profit Factor:{pf:.2f}")
    print(f"  Win Rate:     {wr:.1f}%")
    print(f"  Total Trades: {len(closed_trades)}")
    print(f"  Longs/Shorts: {stats['longs']}/{stats['shorts']}")
    print(f"  Avg Win:      ${avg_win:,.2f}")
    print(f"  Avg Loss:     ${avg_loss:,.2f}")
    print(f"  Avg R/R:      {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A")
    print(f"  Funding:      ${stats['total_funding']:,.0f}")
    print(f"")
    print(f"  Signal Stats:")
    print(f"    Space hits:              {stats['space_only']}")
    print(f"    Space + Time:            {stats['space_time']}")
    print(f"    Space + Time + Energy:   {stats['space_time_energy']}")
    print(f"    Entries taken:           {stats['entries']}")
    print(f"  Exit Stats:")
    print(f"    TP batch exits:          {stats['tp_exits']}")
    print(f"    Trail exits:             {stats['trail_exits']}")
    print(f"    Stop exits:              {stats['stop_exits']}")
    print(f"{'='*70}")
    
    # Top/bottom trades
    sorted_pnl = sorted(closed_trades, key=lambda x: -x.pnl)
    print(f"\n  Top 10 Trades:")
    for t in sorted_pnl[:10]:
        d = t.entry_time.strftime('%Y-%m-%d')
        print(f"    {d} {t.side:5s} ${t.entry:>8,.0f}→${t.exit_price:>8,.0f} | "
              f"{t.leverage}x | fib={t.fib_level:.3f} | time={'Y' if t.time_aligned else 'N'} | "
              f"energy={t.energy_score:.1f} | PnL=${t.pnl:>8,.0f} | {t.reason}")
    
    print(f"\n  Bottom 5 Trades:")
    for t in sorted_pnl[-5:]:
        d = t.entry_time.strftime('%Y-%m-%d')
        print(f"    {d} {t.side:5s} ${t.entry:>8,.0f}→${t.exit_price:>8,.0f} | "
              f"{t.leverage}x | PnL=${t.pnl:>8,.0f} | {t.reason}")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    metrics = {
        "system": "fib_v1",
        "total_return_pct": round(ret, 2),
        "final_equity": round(eq[-1], 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(wr, 2),
        "profit_factor": round(pf, 2),
        "total_trades": len(closed_trades),
        "longs": stats["longs"],
        "shorts": stats["shorts"],
        **stats,
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Equity plot
    eq_df = pd.DataFrame(equity_curve)
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), height_ratios=[2, 1, 1])
    fig.suptitle(f"Fibonacci V1 | Return: {ret:+,.1f}% | DD: {max_dd:.1f}% | "
                 f"WR: {wr:.0f}% | Trades: {len(closed_trades)}", fontsize=13)
    
    axes[0].plot(eq_df["ts"], eq_df["eq"], color="royalblue", linewidth=1)
    axes[0].axhline(y=CAPITAL, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Equity ($)"); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(eq_df["ts"], eq_df["price"], color="orange", linewidth=0.5, alpha=0.7)
    for t in closed_trades:
        c = "green" if t.side == "long" else "red"
        m = "^" if t.side == "long" else "v"
        axes[1].scatter(t.entry_time, t.entry, marker=m, color=c, s=20, alpha=0.5)
    axes[1].set_ylabel("BTC ($)"); axes[1].grid(True, alpha=0.3)
    
    dd_pct = (peak - eq) / peak * 100
    axes[2].fill_between([e["ts"] for e in equity_curve], 0, dd_pct, color="red", alpha=0.3)
    axes[2].set_ylabel("DD (%)"); axes[2].invert_yaxis(); axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "equity_curve.png"), dpi=150)
    plt.close()
    print(f"\n  Saved to {RESULTS_DIR}/")
    
    return metrics


if __name__ == "__main__":
    run_backtest()

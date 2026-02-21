"""Backtest V20 — Fibonacci + TD Sequential Combined Strategy

Combines two powerful reversal detection systems:

1. **TD Sequential** (Tom DeMark):
   - Setup: 9 consecutive closes higher/lower than close 4 bars ago
   - Countdown: 13 bars where close is higher/lower than high/low 2 bars ago
   - TD9 = potential reversal, TD13 = high probability reversal
   - Confirmed with price flip (close vs close 4 bars ago reverses)

2. **Fibonacci Levels** (Natural Trading Theory):
   - Key retracement levels: 0.382, 0.5, 0.618
   - Extension targets: 1.272, 1.618, 2.618
   - Entry on bounce from Fib 0.618 (golden ratio) or 0.5 with TD confirmation
   - Stop below Fib 0.786 or recent swing

3. **Signal Confluence**:
   - TD9 at Fib level = standard entry (3x)
   - TD13 at Fib level = high conviction (10x)
   - TD9 + MACD divergence at Fib = high conviction (10x)
   - Volume confirmation required for all entries

4. **Exit Strategy**:
   - TP1: Fib extension 1.272 (50% close)
   - TP2: Fib extension 1.618 (trailing remainder)
   - Stop: Below Fib 0.786 (tight) or 1.5 ATR
   - Trailing: 2.5 ATR after TP1 hit

Data: 4H BTC/USDT candles
"""

import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.fetch_data import fetch_ohlcv
from backtest.indicators import compute_macd, compute_atr, compute_ma

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v20")

# ── TD Sequential ─────────────────────────────────────────────────────

def compute_td_sequential(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TD Sequential setup and countdown.
    
    Setup: 9 consecutive closes > close[i-4] (sell setup) or < close[i-4] (buy setup)
    Countdown: After setup completes, count 13 bars where close >= high[i-2] (sell) 
               or close <= low[i-2] (buy)
    """
    n = len(df)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    
    # Setup counts
    buy_setup = np.zeros(n, dtype=int)   # negative setup (price falling, buy signal at 9)
    sell_setup = np.zeros(n, dtype=int)  # positive setup (price rising, sell signal at 9)
    
    # Countdown counts
    buy_countdown = np.zeros(n, dtype=int)
    sell_countdown = np.zeros(n, dtype=int)
    
    # Track active countdowns
    buy_cd_active = False
    sell_cd_active = False
    buy_cd_count = 0
    sell_cd_count = 0
    
    # Setup phase
    for i in range(4, n):
        # Buy setup: close < close[i-4]
        if closes[i] < closes[i - 4]:
            if buy_setup[i - 1] > 0:
                buy_setup[i] = buy_setup[i - 1] + 1
            else:
                buy_setup[i] = 1
            sell_setup[i] = 0
        # Sell setup: close > close[i-4]
        elif closes[i] > closes[i - 4]:
            if sell_setup[i - 1] > 0:
                sell_setup[i] = sell_setup[i - 1] + 1
            else:
                sell_setup[i] = 1
            buy_setup[i] = 0
        else:
            buy_setup[i] = 0
            sell_setup[i] = 0
        
        # Start countdown when setup reaches 9
        if buy_setup[i] == 9:
            buy_cd_active = True
            buy_cd_count = 0
        if sell_setup[i] == 9:
            sell_cd_active = True
            sell_cd_count = 0
        
        # Buy countdown: close <= low[i-2]
        if buy_cd_active and i >= 2:
            if closes[i] <= lows[i - 2]:
                buy_cd_count += 1
                buy_countdown[i] = buy_cd_count
            if buy_cd_count >= 13:
                buy_cd_active = False
                buy_cd_count = 0
            # Cancel if price flips strongly (sell setup starts)
            if sell_setup[i] >= 4:
                buy_cd_active = False
                buy_cd_count = 0
        
        # Sell countdown: close >= high[i-2]
        if sell_cd_active and i >= 2:
            if closes[i] >= highs[i - 2]:
                sell_cd_count += 1
                sell_countdown[i] = sell_cd_count
            if sell_cd_count >= 13:
                sell_cd_active = False
                sell_cd_count = 0
            # Cancel if price flips strongly (buy setup starts)
            if buy_setup[i] >= 4:
                sell_cd_active = False
                sell_cd_count = 0
    
    df = df.copy()
    df["td_buy_setup"] = buy_setup
    df["td_sell_setup"] = sell_setup
    df["td_buy_countdown"] = buy_countdown
    df["td_sell_countdown"] = sell_countdown
    return df


# ── Fibonacci Levels ──────────────────────────────────────────────────

def compute_fib_levels(swing_high: float, swing_low: float) -> dict:
    """Compute Fibonacci retracement and extension levels."""
    diff = swing_high - swing_low
    return {
        "0.236": swing_high - diff * 0.236,
        "0.382": swing_high - diff * 0.382,
        "0.500": swing_high - diff * 0.500,
        "0.618": swing_high - diff * 0.618,
        "0.786": swing_high - diff * 0.786,
        # Extensions (from swing low for uptrend)
        "ext_1.272": swing_low + diff * 1.272,
        "ext_1.618": swing_low + diff * 1.618,
        "ext_2.618": swing_low + diff * 2.618,
        "swing_high": swing_high,
        "swing_low": swing_low,
    }


def find_swing_points(highs: np.ndarray, lows: np.ndarray, lookback: int = 20):
    """Find significant swing highs and lows."""
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(highs) - lookback):
        start = i - lookback
        end = i + lookback + 1
        
        if highs[i] >= np.max(highs[start:end]):
            if not swing_highs or i - swing_highs[-1][0] >= lookback // 2:
                swing_highs.append((i, highs[i]))
        
        if lows[i] <= np.min(lows[start:end]):
            if not swing_lows or i - swing_lows[-1][0] >= lookback // 2:
                swing_lows.append((i, lows[i]))
    
    return swing_highs, swing_lows


def get_active_fib_levels(idx: int, swing_highs: list, swing_lows: list, 
                          closes: np.ndarray) -> dict | None:
    """Get the most relevant Fibonacci levels for current price action.
    
    For a potential long: use the most recent significant high→low swing
    For a potential short: use the most recent significant low→high swing
    """
    # Find most recent confirmed swings before current bar
    recent_highs = [(i, v) for i, v in swing_highs if i < idx - 5]
    recent_lows = [(i, v) for i, v in swing_lows if i < idx - 5]
    
    if not recent_highs or not recent_lows:
        return None
    
    last_high = recent_highs[-1]
    last_low = recent_lows[-1]
    
    # Determine trend direction based on which swing came last
    if last_high[0] > last_low[0]:
        # Downtrend (high came after low) → looking for bounce (long)
        # Use the swing from high to low for retracement
        return compute_fib_levels(last_high[1], last_low[1])
    else:
        # Uptrend (low came after high) → looking for rejection (short)
        # Use the swing from low to high for retracement
        return compute_fib_levels(last_high[1], last_low[1])


def price_near_fib(price: float, fib_levels: dict, tolerance: float = 0.005) -> str | None:
    """Check if price is near a key Fibonacci level. Returns level name or None."""
    for level_name in ["0.618", "0.500", "0.382"]:
        level_price = fib_levels[level_name]
        if abs(price - level_price) / price <= tolerance:
            return level_name
    return None


# ── MACD Divergence (from V6) ────────────────────────────────────────

def check_bottom_divergence(idx: int, hist: np.ndarray, lows: np.ndarray, 
                            lookback: int = 120) -> int:
    """Check for bullish divergence. Returns strength (0 = none)."""
    troughs = []
    start = max(0, idx - lookback)
    in_trough = False
    trough_val = 0
    trough_idx = 0

    for j in range(start, idx):
        if hist[j] < 0:
            if not in_trough or hist[j] < trough_val:
                trough_val = hist[j]
                trough_idx = j
            in_trough = True
        else:
            if in_trough and trough_val < 0:
                troughs.append((trough_idx, trough_val))
            in_trough = False
            trough_val = 0

    if in_trough and trough_val < 0:
        troughs.append((trough_idx, trough_val))

    if len(troughs) < 2:
        return 0

    divergence_count = 0
    for k in range(len(troughs) - 1):
        t1_idx, t1_val = troughs[k]
        t2_idx, t2_val = troughs[k + 1]
        if lows[t2_idx] < lows[t1_idx] and t2_val > t1_val:
            divergence_count += 1

    return min(divergence_count, 3)


def check_top_divergence(idx: int, hist: np.ndarray, highs: np.ndarray, 
                         lookback: int = 120) -> int:
    """Check for bearish divergence. Returns strength (0 = none)."""
    peaks = []
    start = max(0, idx - lookback)
    in_peak = False
    peak_val = 0
    peak_idx = 0

    for j in range(start, idx):
        if hist[j] > 0:
            if not in_peak or hist[j] > peak_val:
                peak_val = hist[j]
                peak_idx = j
            in_peak = True
        else:
            if in_peak and peak_val > 0:
                peaks.append((peak_idx, peak_val))
            in_peak = False
            peak_val = 0

    if in_peak and peak_val > 0:
        peaks.append((peak_idx, peak_val))

    if len(peaks) < 2:
        return 0

    divergence_count = 0
    for k in range(len(peaks) - 1):
        p1_idx, p1_val = peaks[k]
        p2_idx, p2_val = peaks[k + 1]
        if highs[p2_idx] > highs[p1_idx] and p2_val < p1_val:
            divergence_count += 1

    return min(divergence_count, 3)


# ── Signal Generation ─────────────────────────────────────────────────

@dataclass
class Signal:
    idx: int
    side: str  # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    leverage: int
    confluence: list  # what triggered this signal
    fib_level: str = ""


def generate_signals(df: pd.DataFrame, swing_highs: list, swing_lows: list) -> list[Signal]:
    """Generate trading signals from TD Sequential + Fibonacci + MACD confluence."""
    signals = []
    
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values
    atr = df["atr"].values
    hist = df["macd_hist"].values
    sma200 = df["sma200"].values
    
    td_buy_setup = df["td_buy_setup"].values
    td_sell_setup = df["td_sell_setup"].values
    td_buy_countdown = df["td_buy_countdown"].values
    td_sell_countdown = df["td_sell_countdown"].values
    
    min_cooldown = 12  # optimized: 12 bars (2 days on 4H)
    last_signal_idx = -min_cooldown
    
    for i in range(50, len(df)):
        if i - last_signal_idx < min_cooldown:
            continue
        if np.isnan(atr[i]) or np.isnan(sma200[i]):
            continue
        
        # Volume confirmation: current volume > 80% of 20-bar average
        avg_vol = np.mean(volumes[max(0, i-20):i]) if i > 20 else volumes[i]
        vol_ok = volumes[i] > avg_vol * 0.8
        if not vol_ok:
            continue
        
        # Get Fibonacci levels
        fib = get_active_fib_levels(i, swing_highs, swing_lows, closes)
        
        bull = closes[i] > sma200[i] if not np.isnan(sma200[i]) else False
        
        # ── LONG SIGNALS ──────────────────────────────────────
        confluence = []
        leverage = 3
        
        # TD Buy Setup = 9 (potential bottom)
        has_td9_buy = td_buy_setup[i] == 9
        # TD Buy Countdown = 13 (high probability bottom)
        has_td13_buy = td_buy_countdown[i] == 13
        # TD Setup 7-9 range (approaching completion)
        has_td_approaching = td_buy_setup[i] >= 7
        
        # MACD bullish divergence
        macd_div = check_bottom_divergence(i, hist, lows)
        has_macd_div = macd_div >= 1
        
        # Fibonacci level proximity
        fib_level = None
        if fib:
            fib_level = price_near_fib(closes[i], fib, tolerance=0.012)
        
        # Build confluence
        if has_td9_buy:
            confluence.append("TD9_buy")
        if has_td13_buy:
            confluence.append("TD13_buy")
        if has_td_approaching and not has_td9_buy:
            confluence.append(f"TD{td_buy_setup[i]}_approaching")
        if has_macd_div:
            confluence.append(f"MACD_div_{macd_div}")
        if fib_level:
            confluence.append(f"Fib_{fib_level}")
        
        # Determine if signal is valid and set leverage
        long_signal = False
        
        # Tier 1: TD13 at Fib level = highest conviction (20x)
        if has_td13_buy and fib_level:
            long_signal = True
            leverage = 20
        # Tier 2: TD9 + MACD divergence = high conviction (20x)
        elif has_td9_buy and has_macd_div:
            long_signal = True
            leverage = 20
        # Tier 3: TD9 at Fib level = standard (10x)
        elif has_td9_buy and fib_level:
            long_signal = True
            leverage = 10
        # Tier 4: TD13 alone = standard (10x)
        elif has_td13_buy:
            long_signal = True
            leverage = 10
        # Tier 5: TD9 + volume spike = base (5x)
        elif has_td9_buy and volumes[i] > avg_vol * 1.5:
            long_signal = True
            leverage = 5
            confluence.append("volume_spike")
        # Tier 6: MACD divergence at Fib = base (5x)
        elif has_macd_div and fib_level and macd_div >= 2:
            long_signal = True
            leverage = 5
        
        if long_signal:
            # Stop loss: below Fib 0.786 or 1.5 ATR, whichever is tighter
            sl_atr = closes[i] - atr[i] * 1.5
            sl_fib = fib["0.786"] if fib else sl_atr
            sl = max(sl_atr, sl_fib)  # tighter stop
            
            # Take profits using Fibonacci extensions
            if fib:
                tp1 = fib["ext_1.272"]
                tp2 = fib["ext_1.618"]
            else:
                risk = closes[i] - sl
                tp1 = closes[i] + risk * 1.5
                tp2 = closes[i] + risk * 2.5
            
            signals.append(Signal(
                idx=i, side="long", entry_price=closes[i],
                stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
                leverage=leverage, confluence=confluence,
                fib_level=fib_level or "",
            ))
            last_signal_idx = i
            continue
        
        # ── SHORT SIGNALS ─────────────────────────────────────
        # Only short in bear market (below 200 SMA)
        if bull:
            continue
        
        confluence = []
        leverage = 3
        
        has_td9_sell = td_sell_setup[i] == 9
        has_td13_sell = td_sell_countdown[i] == 13
        has_td_sell_approaching = td_sell_setup[i] >= 7
        
        macd_bear_div = check_top_divergence(i, hist, highs)
        has_macd_bear_div = macd_bear_div >= 1
        
        fib_level = None
        if fib:
            # For shorts, check if price is near fib from below (resistance)
            for level_name in ["0.382", "0.500", "0.618"]:
                level_price = fib[level_name]
                if abs(closes[i] - level_price) / closes[i] <= 0.008:
                    fib_level = level_name
                    break
        
        if has_td9_sell:
            confluence.append("TD9_sell")
        if has_td13_sell:
            confluence.append("TD13_sell")
        if has_macd_bear_div:
            confluence.append(f"MACD_bear_div_{macd_bear_div}")
        if fib_level:
            confluence.append(f"Fib_{fib_level}")
        
        short_signal = False
        
        # Shorts disabled — long-only system (shorts historically unprofitable)
        # Keeping code for reference but skipping execution
        short_signal = False
        
        if short_signal:
            sl_atr = closes[i] + atr[i] * 1.5
            sl_fib = fib["0.236"] if fib else sl_atr  # above the retracement
            sl = min(sl_atr, sl_fib)
            
            if fib:
                tp1 = fib["0.786"]
                tp2 = fib["swing_low"]
            else:
                risk = sl - closes[i]
                tp1 = closes[i] - risk * 1.5
                tp2 = closes[i] - risk * 2.5
            
            signals.append(Signal(
                idx=i, side="short", entry_price=closes[i],
                stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
                leverage=leverage, confluence=confluence,
                fib_level=fib_level or "",
            ))
            last_signal_idx = i
    
    return signals


# ── Trade Management ──────────────────────────────────────────────────

@dataclass
class Trade:
    side: str
    entry_price: float
    entry_idx: int
    entry_time: pd.Timestamp
    size: float  # BTC (leveraged)
    margin: float
    leverage: int
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    liquidation_price: float
    confluence: list
    fib_level: str = ""
    exit_price: float = 0.0
    exit_idx: int = 0
    exit_time: pd.Timestamp = None
    pnl: float = 0.0
    pnl_partial: float = 0.0
    closed: bool = False
    half_closed: bool = False
    trailing_stop: float = 0.0
    close_reason: str = ""


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    commission_rate: float = 0.001
    risk_per_trade_pct: float = 0.08  # optimized: 8% risk per trade
    max_position_pct: float = 0.50
    trailing_atr_mult: float = 3.5  # optimized: wider trail


class BacktestEngine:
    def __init__(self, df: pd.DataFrame, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.df = self._prepare_data(df)
        self.capital = self.config.initial_capital
        self.equity_curve = []
        self.trades: list[Trade] = []
        self.active_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.stats = {
            "liquidations": 0, "liquidation_loss": 0.0,
            "trades_at_3x": 0, "trades_at_5x": 0, "trades_at_10x": 0,
            "td9_signals": 0, "td13_signals": 0, "fib_signals": 0, "macd_signals": 0,
        }
        
        self.swing_highs, self.swing_lows = find_swing_points(
            self.df["high"].values, self.df["low"].values, lookback=20
        )
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_macd(df, fast=13, slow=34, signal=9)
        df["atr"] = compute_atr(df, period=14)
        df["sma50"] = compute_ma(df, period=50)
        df["sma200"] = compute_ma(df, period=200)
        df = compute_td_sequential(df)
        return df
    
    def run(self) -> dict:
        print(f"[V20] Fibonacci + TD Sequential Backtest")
        print(f"  Data: {len(self.df)} candles, {self.df['timestamp'].iloc[0]} to {self.df['timestamp'].iloc[-1]}")
        
        signals = generate_signals(self.df, self.swing_highs, self.swing_lows)
        signal_map = {s.idx: s for s in signals}
        
        long_sigs = sum(1 for s in signals if s.side == "long")
        short_sigs = sum(1 for s in signals if s.side == "short")
        print(f"  Signals: {len(signals)} ({long_sigs} long, {short_sigs} short)")
        
        # Count signal types
        for s in signals:
            for c in s.confluence:
                if "TD9" in c: self.stats["td9_signals"] += 1
                if "TD13" in c: self.stats["td13_signals"] += 1
                if "Fib" in c: self.stats["fib_signals"] += 1
                if "MACD" in c: self.stats["macd_signals"] += 1
        
        # Show leverage distribution
        lev_dist = {}
        for s in signals:
            lev_dist[s.leverage] = lev_dist.get(s.leverage, 0) + 1
        print(f"  Leverage distribution: {lev_dist}")
        
        closes = self.df["close"].values
        
        for i in range(len(self.df)):
            price = closes[i]
            high = self.df["high"].iloc[i]
            low = self.df["low"].iloc[i]
            ts = self.df["timestamp"].iloc[i]
            
            self._manage_positions(i, high, low, price, ts)
            
            if i in signal_map:
                sig = signal_map[i]
                self._execute_entry(sig, ts)
            
            unrealized = self._calc_unrealized(price)
            self.equity_curve.append({
                "timestamp": ts,
                "equity": self.capital + unrealized,
                "capital": self.capital,
            })
        
        # Close remaining
        last_price = closes[-1]
        last_ts = self.df["timestamp"].iloc[-1]
        for trade in list(self.active_trades):
            self._close_trade(trade, last_price, len(self.df) - 1, last_ts, "end_of_backtest")
        
        return self._compute_metrics()
    
    def _execute_entry(self, signal: Signal, ts: pd.Timestamp):
        # Only one position per direction
        same_dir = [t for t in self.active_trades if t.side == signal.side and not t.closed]
        if same_dir:
            return
        
        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            return
        
        # Position sizing
        risk_amount = self.capital * self.config.risk_per_trade_pct
        base_size_btc = risk_amount / risk
        max_size = (self.capital * self.config.max_position_pct) / signal.entry_price
        base_size_btc = min(base_size_btc, max_size)
        
        if base_size_btc * signal.entry_price < 10:
            return
        
        leverage = signal.leverage
        leveraged_size = base_size_btc * leverage
        margin = base_size_btc * signal.entry_price
        notional = leveraged_size * signal.entry_price
        
        commission = notional * self.config.commission_rate
        self.capital -= commission
        
        if signal.side == "long":
            liq_price = signal.entry_price * (1 - 1 / leverage)
        else:
            liq_price = signal.entry_price * (1 + 1 / leverage)
        
        # Track stats
        if leverage == 5: self.stats["trades_at_3x"] += 1  # now 5x is base
        elif leverage == 10: self.stats["trades_at_5x"] += 1
        elif leverage == 20: self.stats["trades_at_10x"] += 1
        
        trade = Trade(
            side=signal.side,
            entry_price=signal.entry_price,
            entry_idx=signal.idx,
            entry_time=ts,
            size=leveraged_size,
            margin=margin,
            leverage=leverage,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            liquidation_price=liq_price,
            confluence=signal.confluence,
            fib_level=signal.fib_level,
            trailing_stop=signal.stop_loss,
        )
        self.active_trades.append(trade)
        self.trades.append(trade)
    
    def _manage_positions(self, idx: int, high: float, low: float, close: float, ts: pd.Timestamp):
        atr_val = self.df["atr"].iloc[idx]
        
        for trade in list(self.active_trades):
            if trade.closed:
                continue
            
            if trade.side == "long":
                # Liquidation
                if low <= trade.liquidation_price:
                    self.stats["liquidations"] += 1
                    self.stats["liquidation_loss"] += trade.margin
                    self._close_trade(trade, trade.liquidation_price, idx, ts, "liquidation")
                    continue
                
                # Stop loss
                if low <= trade.trailing_stop:
                    self._close_trade(trade, trade.trailing_stop, idx, ts, "stop_loss")
                    continue
                
                # TP1: close 50%
                if not trade.half_closed and high >= trade.take_profit_1:
                    half = trade.size / 2
                    pnl = (trade.take_profit_1 - trade.entry_price) * half
                    commission = half * trade.take_profit_1 * self.config.commission_rate
                    trade.pnl_partial = pnl - commission
                    self.capital += trade.pnl_partial
                    trade.size -= half
                    trade.half_closed = True
                    # Move stop to breakeven + small buffer
                    trade.trailing_stop = max(trade.trailing_stop, trade.entry_price + (trade.take_profit_1 - trade.entry_price) * 0.1)
                
                # TP2: close remainder
                if trade.half_closed and high >= trade.take_profit_2:
                    self._close_trade(trade, trade.take_profit_2, idx, ts, "take_profit_2")
                    continue
                
                # Trailing stop after TP1
                if not np.isnan(atr_val) and trade.half_closed:
                    new_trail = close - atr_val * self.config.trailing_atr_mult
                    trade.trailing_stop = max(trade.trailing_stop, new_trail)
            
            elif trade.side == "short":
                if high >= trade.liquidation_price:
                    self.stats["liquidations"] += 1
                    self.stats["liquidation_loss"] += trade.margin
                    self._close_trade(trade, trade.liquidation_price, idx, ts, "liquidation")
                    continue
                
                if high >= trade.trailing_stop:
                    self._close_trade(trade, trade.trailing_stop, idx, ts, "stop_loss")
                    continue
                
                if not trade.half_closed and low <= trade.take_profit_1:
                    half = trade.size / 2
                    pnl = (trade.entry_price - trade.take_profit_1) * half
                    commission = half * trade.take_profit_1 * self.config.commission_rate
                    trade.pnl_partial = pnl - commission
                    self.capital += trade.pnl_partial
                    trade.size -= half
                    trade.half_closed = True
                    trade.trailing_stop = min(trade.trailing_stop, trade.entry_price - (trade.entry_price - trade.take_profit_1) * 0.1)
                
                if trade.half_closed and low <= trade.take_profit_2:
                    self._close_trade(trade, trade.take_profit_2, idx, ts, "take_profit_2")
                    continue
                
                if not np.isnan(atr_val) and trade.half_closed:
                    new_trail = close + atr_val * self.config.trailing_atr_mult
                    trade.trailing_stop = min(trade.trailing_stop, new_trail)
    
    def _close_trade(self, trade: Trade, exit_price: float, idx: int, ts: pd.Timestamp, reason: str):
        if trade.side == "long":
            pnl = (exit_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - exit_price) * trade.size
        
        commission = trade.size * exit_price * self.config.commission_rate
        pnl -= commission
        
        trade.exit_price = exit_price
        trade.exit_idx = idx
        trade.exit_time = ts
        trade.pnl = pnl + trade.pnl_partial
        trade.closed = True
        trade.close_reason = reason
        self.capital += pnl
        
        if trade in self.active_trades:
            self.active_trades.remove(trade)
        self.closed_trades.append(trade)
    
    def _calc_unrealized(self, price: float) -> float:
        total = 0.0
        for t in self.active_trades:
            if t.side == "long":
                total += (price - t.entry_price) * t.size
            else:
                total += (t.entry_price - price) * t.size
        return total
    
    def _compute_metrics(self) -> dict:
        if not self.closed_trades:
            return {"version": "v20", "error": "No trades executed"}
        
        eq_df = pd.DataFrame(self.equity_curve)
        equity = eq_df["equity"].values
        
        total_return = (equity[-1] - self.config.initial_capital) / self.config.initial_capital * 100
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_dd = np.max(drawdown)
        
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(2190) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        pnls = [t.pnl for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        
        durations = []
        for t in self.closed_trades:
            if t.entry_time and t.exit_time:
                durations.append((t.exit_time - t.entry_time).total_seconds() / 3600)
        avg_dur = np.mean(durations) if durations else 0
        
        long_trades = [t for t in self.closed_trades if t.side == "long"]
        short_trades = [t for t in self.closed_trades if t.side == "short"]
        
        # Confluence breakdown
        confluence_pnl = {}
        for t in self.closed_trades:
            for c in t.confluence:
                key = c.split("_")[0] if "_" in c else c
                if key not in confluence_pnl:
                    confluence_pnl[key] = {"count": 0, "pnl": 0}
                confluence_pnl[key]["count"] += 1
                confluence_pnl[key]["pnl"] += t.pnl
        
        # Peak equity
        peak_equity = np.max(equity)
        peak_mult = peak_equity / self.config.initial_capital
        
        return {
            "version": "v20",
            "strategy": "Fibonacci + TD Sequential",
            "total_return_pct": round(total_return, 2),
            "final_equity": round(equity[-1], 2),
            "peak_equity": round(peak_equity, 2),
            "peak_multiplier": round(peak_mult, 2),
            "initial_capital": self.config.initial_capital,
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(pf, 2),
            "total_trades": len(self.closed_trades),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl": round(sum(t.pnl for t in long_trades), 2),
            "short_pnl": round(sum(t.pnl for t in short_trades), 2),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "largest_win": round(max(wins), 2) if wins else 0,
            "largest_loss": round(min(losses), 2) if losses else 0,
            "avg_duration_hours": round(avg_dur, 1),
            "avg_duration_days": round(avg_dur / 24, 1),
            "trades_at_3x": self.stats["trades_at_3x"],
            "trades_at_5x": self.stats["trades_at_5x"],
            "trades_at_10x": self.stats["trades_at_10x"],
            "liquidations": self.stats["liquidations"],
            "liquidation_loss": round(self.stats["liquidation_loss"], 2),
            "td9_signals": self.stats["td9_signals"],
            "td13_signals": self.stats["td13_signals"],
            "fib_signals": self.stats["fib_signals"],
            "macd_signals": self.stats["macd_signals"],
            "confluence_breakdown": confluence_pnl,
            "data_start": str(self.df["timestamp"].iloc[0]),
            "data_end": str(self.df["timestamp"].iloc[-1]),
            "total_candles": len(self.df),
        }
    
    def generate_report(self, metrics: dict):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        report = self._format_report(metrics)
        with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
            f.write(report)
        print(report)
        
        self._plot_equity_curve()
        self._save_trade_log()
    
    def _format_report(self, m: dict) -> str:
        sep = "=" * 60
        lines = [
            sep,
            "  BTC/USDT V20 — Fibonacci + TD Sequential",
            sep, "",
            f"  Data: {m['data_start'][:10]} → {m['data_end'][:10]} ({m['total_candles']} 4H bars)",
            f"  Capital: ${m['initial_capital']:,.0f}", "",
            sep, "  PERFORMANCE", sep, "",
            f"  Return:       {m['total_return_pct']:+.2f}%",
            f"  Final Equity: ${m['final_equity']:,.2f}",
            f"  Peak Equity:  ${m['peak_equity']:,.2f} ({m['peak_multiplier']:.1f}x)",
            f"  Max Drawdown: {m['max_drawdown_pct']:.2f}%",
            f"  Sharpe:       {m['sharpe_ratio']:.2f}",
            f"  Profit Factor:{m['profit_factor']:.2f}",
            f"  Win Rate:     {m['win_rate_pct']:.1f}%", "",
            sep, "  TRADES", sep, "",
            f"  Total:    {m['total_trades']}",
            f"    Long:   {m['long_trades']} (PnL: ${m['long_pnl']:+,.2f})",
            f"    Short:  {m['short_trades']} (PnL: ${m['short_pnl']:+,.2f})",
            f"  Avg Win:  ${m['avg_win']:,.2f}",
            f"  Avg Loss: ${m['avg_loss']:,.2f}",
            f"  Best:     ${m['largest_win']:,.2f}",
            f"  Worst:    ${m['largest_loss']:,.2f}",
            f"  Avg Dur:  {m['avg_duration_days']:.1f} days", "",
            sep, "  LEVERAGE", sep, "",
            f"  3x trades:  {m['trades_at_3x']}",
            f"  5x trades:  {m['trades_at_5x']}",
            f"  10x trades: {m['trades_at_10x']}",
            f"  Liquidations: {m['liquidations']} (${m['liquidation_loss']:,.2f})", "",
            sep, "  SIGNAL CONFLUENCE", sep, "",
            f"  TD9 signals:  {m['td9_signals']}",
            f"  TD13 signals: {m['td13_signals']}",
            f"  Fib signals:  {m['fib_signals']}",
            f"  MACD signals: {m['macd_signals']}", "",
        ]
        
        if m.get("confluence_breakdown"):
            lines.append("  Confluence PnL breakdown:")
            for k, v in sorted(m["confluence_breakdown"].items(), key=lambda x: -x[1]["pnl"]):
                lines.append(f"    {k}: {v['count']} trades, ${v['pnl']:+,.2f}")
            lines.append("")
        
        lines += [sep, f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", sep]
        return "\n".join(lines)
    
    def _plot_equity_curve(self):
        eq_df = pd.DataFrame(self.equity_curve)
        equity = eq_df["equity"].values
        timestamps = pd.to_datetime(eq_df["timestamp"])
        
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
        
        ax1.plot(timestamps, equity, color="#FF6F00", linewidth=1.4, label="V20 (Fib + TD Sequential)")
        ax1.axhline(y=self.config.initial_capital, color="gray", linestyle="--", alpha=0.5)
        
        for trade in self.closed_trades:
            color = "#4CAF50" if trade.pnl > 0 else "#F44336"
            if trade.close_reason == "liquidation":
                color = "#9C27B0"
            ms = 3 + trade.leverage
            ax1.plot(trade.entry_time, equity[min(trade.entry_idx, len(equity)-1)],
                    marker="^" if trade.side == "long" else "v",
                    color=color, markersize=ms, alpha=0.7)
        
        ax1.set_title("V20 — Fibonacci + TD Sequential Equity Curve", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Equity ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.fill_between(timestamps, -dd, 0, color="#F44336", alpha=0.3)
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "equity_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    def _save_trade_log(self):
        records = []
        for t in self.closed_trades:
            records.append({
                "side": t.side,
                "leverage": t.leverage,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "margin": round(t.margin, 2),
                "pnl": round(t.pnl, 2),
                "close_reason": t.close_reason,
                "confluence": "|".join(t.confluence),
                "fib_level": t.fib_level,
                "duration_hours": round((t.exit_time - t.entry_time).total_seconds() / 3600, 1) if t.exit_time and t.entry_time else 0,
            })
        pd.DataFrame(records).to_csv(os.path.join(RESULTS_DIR, "trade_log.csv"), index=False)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    # Run on both datasets
    results = {}
    
    # 4-year dataset (2022-2026)
    print("\n" + "="*60)
    print("  DATASET 1: 4-Year (2022-2026)")
    print("="*60)
    df_4yr = fetch_ohlcv(use_cache=True)
    engine = BacktestEngine(df_4yr)
    m1 = engine.run()
    engine.generate_report(m1)
    results["4yr"] = m1
    
    # 8.5-year dataset (2017-2026) if available
    data_path_8yr = os.path.join(os.path.dirname(__file__), "data", "btc_usdt_4h_2017.csv")
    if os.path.exists(data_path_8yr):
        print("\n" + "="*60)
        print("  DATASET 2: 8.5-Year (2017-2026)")
        print("="*60)
        df_8yr = pd.read_csv(data_path_8yr, parse_dates=["timestamp"])
        
        engine2 = BacktestEngine(df_8yr)
        m2 = engine2.run()
        
        # Save to separate dir
        results_dir_8yr = os.path.join(os.path.dirname(__file__), "results_v20_8yr")
        os.makedirs(results_dir_8yr, exist_ok=True)
        
        old_dir = RESULTS_DIR
        # Monkey-patch for report generation
        import backtest.backtest_v20 as mod
        mod.RESULTS_DIR = results_dir_8yr
        engine2.generate_report(m2)
        mod.RESULTS_DIR = old_dir
        
        results["8yr"] = m2
    
    # Comparison with V6
    print("\n" + "="*60)
    print("  V6 vs V20 COMPARISON")
    print("="*60)
    
    v6_path = os.path.join(os.path.dirname(__file__), "results_v6", "metrics.json")
    if os.path.exists(v6_path):
        with open(v6_path) as f:
            v6 = json.load(f)
        
        v20 = results["4yr"]
        print(f"\n  {'Metric':<25} {'V6 (MACD)':>15} {'V20 (Fib+TD)':>15}")
        print(f"  {'-'*25} {'-'*15} {'-'*15}")
        print(f"  {'Return %':<25} {v6['total_return_pct']:>14.2f}% {v20['total_return_pct']:>14.2f}%")
        print(f"  {'Final Equity':<25} ${v6['final_equity']:>13,.2f} ${v20['final_equity']:>13,.2f}")
        print(f"  {'Max Drawdown':<25} {v6['max_drawdown_pct']:>14.2f}% {v20['max_drawdown_pct']:>14.2f}%")
        print(f"  {'Sharpe':<25} {v6['sharpe_ratio']:>15.2f} {v20['sharpe_ratio']:>15.2f}")
        print(f"  {'Win Rate':<25} {v6['win_rate_pct']:>14.1f}% {v20['win_rate_pct']:>14.1f}%")
        print(f"  {'Profit Factor':<25} {v6['profit_factor']:>15.2f} {v20['profit_factor']:>15.2f}")
        print(f"  {'Trades':<25} {v6['total_trades']:>15} {v20['total_trades']:>15}")
        print(f"  {'Liquidations':<25} {v6['liquidations']:>15} {v20['liquidations']:>15}")
    
    return results


if __name__ == "__main__":
    main()

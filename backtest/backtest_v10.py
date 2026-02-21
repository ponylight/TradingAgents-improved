"""Backtest V10 — V9 base + Looser Rolling Add Conditions.

Two modes run simultaneously on shared capital:

MODE 1 (V6 MACD Reversal):
  - V3 base strategy with SMA200 bull filter, SMA50 trend
  - Leveraged: 3x base, 10x if 123 Rule confirmed
  - TP1 at 1:1 R/R, double divergence (strength >= 2)
  - V10 ROLLING: After +2%, add up to 3 positions (150%/200%/250%)
    Check every 4 bars (16h). ANY condition triggers add:
      a) Price within 1 ATR of MA20
      b) 38.2% retrace of move since entry
      c) Breakout above 20-bar high
    Add margin <= 80% of floating profits.
    Each add breakeven stop at entry. Master trail 2x ATR.

MODE 2 (Graduated Leverage Bottom-Fishing):
  - Same graduated entry as V8 (5x->10x->15x->25x->50x)
  - V10 ROLLING: After +2%, add up to 3 positions (150%/200%/250%)
    Same looser conditions as Mode 1.
    Add margin <= 80% of floating profits. Master trail 2x ATR.

Starting capital: $10,000, commission: 0.1% on notional.
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v10")

# -- V6 / V3 constants -------------------------------------------------------

NEAR_BOTTOM_LOOKBACK = 2190  # 365 days in 4h bars
NEAR_BOTTOM_THRESHOLD = 0.30
SWING_LOOKBACK = 10
BASE_LEVERAGE = 3
MAX_LEVERAGE = 10

# -- V10 Rolling constants (CHANGED from V9) ---------------------------------

V10_ROLL_PROFIT_PCT = 0.02        # V10: +2% to trigger (was +3% in V9)
V10_ROLL_CHECK_BARS = 4           # V10: check every 4 bars / 16h (was 12 / 48h)
V10_ROLL_MAX_ADDS = 3
V10_ROLL_ADD_MULTS = [1.5, 2.0, 2.5]  # inverted pyramid stays same
V10_MASTER_TRAIL_ATR = 2.0        # 2x ATR master trail from highest point
V10_FLOATING_PROFIT_MULT = 0.80   # V10: add margin <= 80% of floating (was 100%)

# -- Mode 2: Graduated Bottom-Fishing constants --------------------------------

MM_GRADUATED_LEVERAGE = [5, 10, 15, 25, 50]
MM_MARGIN_PCT = 0.02
MM_MAX_ATTEMPTS = 5
MM_COOLDOWN_BARS = 180
MM_PROFIT_THRESHOLD = 0.02  # V10: lowered from 0.03 to 0.02
MM_TRAIL_ATR_MULT = 2.0
MM_LIQ_BUFFER = 1.002
MM_RSI_THRESHOLD = 30
MM_MIN_MARGIN = 20.0


# -- Data classes --------------------------------------------------------------

@dataclass
class Trade:
    """Mode 1 (V6 MACD) trade."""
    side: str
    entry_price: float
    entry_idx: int
    entry_time: pd.Timestamp
    size: float
    stop_loss: float
    take_profit_1: float
    initial_risk: float
    leverage: float = 1.0
    margin: float = 0.0
    liquidation_price: float = 0.0
    is_rolling: bool = False
    exit_price: float = 0.0
    exit_idx: int = 0
    exit_time: pd.Timestamp = None
    pnl: float = 0.0
    pnl_tp1: float = 0.0
    original_size: float = 0.0
    closed: bool = False
    half_closed: bool = False
    trailing_stop: float = 0.0
    close_reason: str = ""
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 0.0
    base_trade_idx: int = -1  # for rolling adds: links to base trade


@dataclass
class MMTrade:
    """Mode 2 (Graduated Bottom-Fishing) trade."""
    entry_price: float
    entry_idx: int
    entry_time: pd.Timestamp
    size: float
    margin: float
    leverage: float = 5.0
    liquidation_price: float = 0.0
    stop_loss: float = 0.0
    is_add: bool = False
    exit_price: float = 0.0
    exit_idx: int = 0
    exit_time: pd.Timestamp = None
    pnl: float = 0.0
    closed: bool = False
    close_reason: str = ""


@dataclass
class Signal:
    idx: int
    side: str
    entry_price: float
    stop_loss: float
    divergence_strength: float
    signal_type: str = "primary"


# -- Weekly RSI computation ----------------------------------------------------

def compute_weekly_rsi(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    ts_df = df.set_index("timestamp")["close"].copy()
    weekly_close = ts_df.resample("W").last().dropna()
    delta = weekly_close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    rsi_4h = rsi.reindex(ts_df.index, method="ffill")
    return rsi_4h.values


# -- Swing detection -----------------------------------------------------------

def detect_swings(highs: np.ndarray, lows: np.ndarray, lookback: int = SWING_LOOKBACK):
    swing_highs = []
    swing_lows = []
    for i in range(lookback, len(highs) - lookback):
        window_start = i - lookback
        window_end = i + lookback + 1
        if highs[i] >= np.max(highs[window_start:window_end]):
            if not swing_highs or i - swing_highs[-1][0] >= lookback:
                swing_highs.append((i, highs[i]))
        if lows[i] <= np.min(lows[window_start:window_end]):
            if not swing_lows or i - swing_lows[-1][0] >= lookback:
                swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


# -- 123 Rule ------------------------------------------------------------------

def check_123_rule(signal_idx, side, closes, highs, lows, atr_vals,
                   swing_highs, swing_lows):
    if side == "long":
        return _check_123_long(signal_idx, closes, highs, lows, atr_vals, swing_highs, swing_lows)
    else:
        return _check_123_short(signal_idx, closes, highs, lows, atr_vals, swing_highs, swing_lows)


def _check_123_long(signal_idx, closes, highs, lows, atr_vals, swing_highs, swing_lows):
    lb = SWING_LOOKBACK
    confirmed_lows = [(idx, val) for idx, val in swing_lows if idx + lb <= signal_idx]
    if len(confirmed_lows) < 2:
        return False
    x1, y1 = confirmed_lows[-2]
    x2, y2 = confirmed_lows[-1]
    dx = x2 - x1
    if dx == 0:
        return False
    slope = (y2 - y1) / dx
    def tl(x):
        return y1 + slope * (x - x1)
    break_bar = None
    for j in range(x2 + 1, signal_idx + 1):
        if closes[j] > tl(j):
            break_bar = j
            break
    if break_bar is None:
        return False
    retest_bar = None
    for j in range(break_bar + 1, signal_idx + 1):
        tl_j = tl(j)
        a = atr_vals[j]
        if np.isnan(a):
            continue
        if abs(lows[j] - tl_j) <= a:
            if closes[j] > tl_j:
                retest_bar = j
                break
    if retest_bar is None:
        return False
    confirmed_highs = [(idx, val) for idx, val in swing_highs if idx + lb <= signal_idx]
    if not confirmed_highs:
        return False
    _, pivot_val = confirmed_highs[-1]
    for j in range(retest_bar, signal_idx + 1):
        if closes[j] > pivot_val:
            return True
    return False


def _check_123_short(signal_idx, closes, highs, lows, atr_vals, swing_highs, swing_lows):
    lb = SWING_LOOKBACK
    confirmed_highs = [(idx, val) for idx, val in swing_highs if idx + lb <= signal_idx]
    if len(confirmed_highs) < 2:
        return False
    x1, y1 = confirmed_highs[-2]
    x2, y2 = confirmed_highs[-1]
    dx = x2 - x1
    if dx == 0:
        return False
    slope = (y2 - y1) / dx
    def tl(x):
        return y1 + slope * (x - x1)
    break_bar = None
    for j in range(x2 + 1, signal_idx + 1):
        if closes[j] < tl(j):
            break_bar = j
            break
    if break_bar is None:
        return False
    retest_bar = None
    for j in range(break_bar + 1, signal_idx + 1):
        tl_j = tl(j)
        a = atr_vals[j]
        if np.isnan(a):
            continue
        if abs(highs[j] - tl_j) <= a:
            if closes[j] < tl_j:
                retest_bar = j
                break
    if retest_bar is None:
        return False
    confirmed_lows = [(idx, val) for idx, val in swing_lows if idx + lb <= signal_idx]
    if not confirmed_lows:
        return False
    _, pivot_val = confirmed_lows[-1]
    for j in range(retest_bar, signal_idx + 1):
        if closes[j] < pivot_val:
            return True
    return False


# -- Leverage determination (V6) -----------------------------------------------

def determine_leverage(signal_idx, side, closes, highs, lows, atr_vals,
                       swing_highs, swing_lows):
    has_123 = check_123_rule(signal_idx, side, closes, highs, lows, atr_vals,
                             swing_highs, swing_lows)
    if has_123:
        return MAX_LEVERAGE, has_123
    else:
        return BASE_LEVERAGE, has_123


# -- Regime helpers (V3) -------------------------------------------------------

def is_bull_market(close, sma200):
    if np.isnan(sma200):
        return False
    return close > sma200


def is_near_market_bottom(closes, current_idx):
    lookback_start = max(0, current_idx - NEAR_BOTTOM_LOOKBACK)
    window = closes[lookback_start:current_idx + 1]
    if len(window) == 0:
        return False
    low_365 = np.min(window)
    return closes[current_idx] <= low_365 * (1.0 + NEAR_BOTTOM_THRESHOLD)


# -- V3 signal generation (identical to V6/V8/V9) -----------------------------

def compute_signals_v3(df):
    signals = []
    shorts_blocked_by_bull = 0
    hist = df["macd_hist"].values
    closes = df["close"].values
    lows = df["low"].values
    highs = df["high"].values
    atr = df["atr"].values
    sma50 = df["sma50"].values
    sma200 = df["sma200"].values
    start_idx = 205
    min_cooldown = 30
    last_signal_idx = -min_cooldown

    for i in range(start_idx, len(df)):
        if i - last_signal_idx < min_cooldown:
            continue
        if np.isnan(sma50[i]) or np.isnan(sma200[i]):
            continue
        bull = is_bull_market(closes[i], sma200[i])

        # LONG
        if hist[i] < 0 and i >= 4:
            if closes[i] <= sma50[i]:
                pass
            else:
                is_key_kline = (
                    hist[i] > hist[i - 1] and hist[i - 1] < 0
                    and hist[i - 1] < hist[i - 2] and hist[i - 2] < hist[i - 3]
                )
                min_hist_size = closes[i] * 0.001
                if is_key_kline and abs(hist[i - 1]) > min_hist_size:
                    long_div = _check_bottom_divergence(i, hist, lows, lookback=120)
                    if long_div is not None:
                        strength, peak_diff = long_div
                        if strength >= 2 and peak_diff > 0.30:
                            sl = lows[i] - atr[i] * 1.5 if not np.isnan(atr[i]) else lows[i] * 0.97
                            signals.append(Signal(
                                idx=i, side="long", entry_price=closes[i],
                                stop_loss=sl, divergence_strength=strength,
                            ))
                            last_signal_idx = i

        # SHORT
        if hist[i] > 0 and i >= 4:
            if bull:
                if closes[i] < sma50[i]:
                    is_key_kline = (
                        hist[i] < hist[i - 1] and hist[i - 1] > 0
                        and hist[i - 1] > hist[i - 2] and hist[i - 2] > hist[i - 3]
                    )
                    min_hist_size = closes[i] * 0.001
                    if is_key_kline and abs(hist[i - 1]) > min_hist_size:
                        short_div = _check_top_divergence(i, hist, highs, lookback=120)
                        if short_div is not None:
                            strength, peak_diff = short_div
                            if strength >= 2 and peak_diff > 0.30:
                                shorts_blocked_by_bull += 1
                continue
            if closes[i] >= sma50[i]:
                pass
            else:
                is_key_kline = (
                    hist[i] < hist[i - 1] and hist[i - 1] > 0
                    and hist[i - 1] > hist[i - 2] and hist[i - 2] > hist[i - 3]
                )
                min_hist_size = closes[i] * 0.001
                if is_key_kline and abs(hist[i - 1]) > min_hist_size:
                    short_div = _check_top_divergence(i, hist, highs, lookback=120)
                    if short_div is not None:
                        strength, peak_diff = short_div
                        if strength >= 2 and peak_diff > 0.30:
                            sl = highs[i] + atr[i] * 1.5 if not np.isnan(atr[i]) else highs[i] * 1.03
                            signals.append(Signal(
                                idx=i, side="short", entry_price=closes[i],
                                stop_loss=sl, divergence_strength=strength,
                            ))
                            last_signal_idx = i

    print(f"  V3 bull filter blocked {shorts_blocked_by_bull} short signals")
    return signals


def _check_bottom_divergence(current_idx, hist, lows, lookback=80):
    troughs = []
    start = max(0, current_idx - lookback)
    in_trough = False
    trough_val = 0
    trough_idx = 0
    for j in range(start, current_idx):
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
        return None
    divergence_count = 0
    for k in range(len(troughs) - 1):
        t1_idx, t1_val = troughs[k]
        t2_idx, t2_val = troughs[k + 1]
        if lows[t2_idx] < lows[t1_idx] and t2_val > t1_val:
            divergence_count += 1
    if divergence_count == 0:
        return None
    last_two = troughs[-2:]
    peak_diff = abs(abs(last_two[1][1]) - abs(last_two[0][1])) / abs(last_two[0][1])
    return (min(divergence_count, 3), peak_diff)


def _check_top_divergence(current_idx, hist, highs, lookback=80):
    peaks = []
    start = max(0, current_idx - lookback)
    in_peak = False
    peak_val = 0
    peak_idx = 0
    for j in range(start, current_idx):
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
        return None
    divergence_count = 0
    for k in range(len(peaks) - 1):
        p1_idx, p1_val = peaks[k]
        p2_idx, p2_val = peaks[k + 1]
        if highs[p2_idx] > highs[p1_idx] and p2_val < p1_val:
            divergence_count += 1
    if divergence_count == 0:
        return None
    last_two = peaks[-2:]
    peak_diff = abs(last_two[0][1] - last_two[1][1]) / last_two[0][1]
    return (min(divergence_count, 3), peak_diff)


# -- Engine --------------------------------------------------------------------

@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    commission_rate: float = 0.001
    max_position_pct: float = 0.40
    risk_per_trade_pct: float = 0.02
    max_rolling_adds: int = V10_ROLL_MAX_ADDS
    trailing_stop_atr_mult: float = 2.5
    rr_ratio: float = 1.0


class BacktestEngine:
    def __init__(self, df: pd.DataFrame, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.df = self._prepare_data(df)
        self.capital = self.config.initial_capital
        self.equity_curve = []

        # Mode 1 (V6) state
        self.trades: list[Trade] = []
        self.active_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.rolling_add_counts: dict[int, int] = {}
        self.swing_highs, self.swing_lows = detect_swings(
            self.df["high"].values, self.df["low"].values, lookback=SWING_LOOKBACK
        )
        self.leverage_stats = {
            "trades_at_3x": 0, "trades_at_10x": 0,
            "liquidations": 0, "liquidation_loss": 0.0,
        }

        # V10 rolling stats
        self.v10_stats = {
            "m1_rolling_adds": 0,
            "m1_rolling_breakeven_stops": 0,
            "m1_master_trail_closes": 0,
            "m1_rolling_pnl": 0.0,
            "m2_rolling_adds": 0,
            "m2_rolling_breakeven_stops": 0,
            "m2_rolling_pnl": 0.0,
        }

        # Mode 2 (Graduated Bottom-Fishing) state
        self.mm_base_trade: MMTrade | None = None
        self.mm_adds: list[MMTrade] = []
        self.mm_closed_trades: list[MMTrade] = []
        self.mm_attempt_count = 0
        self.mm_cooldown_until_idx = -1
        self.mm_in_profit_mode = False
        self.mm_master_trail = 0.0
        self.mm_highest_since_profit = 0.0
        self.mm_roll_add_count = 0
        self.mm_stats = {
            "total_entries": 0,
            "total_stops_phase_a": 0,
            "total_adds": 0,
            "successful_rides": 0,
            "cooldowns_triggered": 0,
            "best_ride_pnl": 0.0,
            "worst_stop_pnl": 0.0,
            "entries_per_leverage": {lev: 0 for lev in MM_GRADUATED_LEVERAGE},
            "stops_per_leverage": {lev: 0 for lev in MM_GRADUATED_LEVERAGE},
            "wins_per_leverage": {lev: 0 for lev in MM_GRADUATED_LEVERAGE},
        }

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_macd(df, fast=13, slow=34, signal=9)
        df["atr"] = compute_atr(df, period=14)
        df["ma20"] = compute_ma(df, period=20)   # V10: MA20 instead of MA30
        df["sma50"] = compute_ma(df, period=50)
        df["sma200"] = compute_ma(df, period=200)
        df["weekly_rsi"] = compute_weekly_rsi(df, period=14)
        return df

    # -- Main loop -------------------------------------------------------------

    def run(self) -> dict:
        print(f"[V10] Running backtest on {len(self.df)} candles...")
        print(f"  Date range: {self.df['timestamp'].iloc[0]} to {self.df['timestamp'].iloc[-1]}")
        print(f"  Mode 1: V6 MACD Reversal (3x/10x) + V10 Rolling")
        print(f"  Mode 2: Graduated Bottom-Fishing (5x->50x) + V10 Rolling")
        print(f"  V10 changes: 4-bar check, MA20+1ATR/Fib38.2/breakout20, +2% trigger, 80% float")
        print(f"  Pre-computed {len(self.swing_highs)} swing highs, {len(self.swing_lows)} swing lows")

        primary_signals = compute_signals_v3(self.df)
        signal_map = {s.idx: s for s in primary_signals}
        long_sigs = sum(1 for s in primary_signals if s.side == "long")
        short_sigs = sum(1 for s in primary_signals if s.side == "short")
        print(f"  Found {len(primary_signals)} primary signals ({long_sigs} long, {short_sigs} short)")

        closes = self.df["close"].values
        highs_arr = self.df["high"].values
        lows_arr = self.df["low"].values
        atr_arr = self.df["atr"].values

        for i in range(len(self.df)):
            price = self.df["close"].iloc[i]
            high = self.df["high"].iloc[i]
            low = self.df["low"].iloc[i]
            ts = self.df["timestamp"].iloc[i]
            atr_val = self.df["atr"].iloc[i]

            # Manage Mode 1 positions
            self._manage_positions(i, high, low, price, ts)

            # Manage Mode 2 positions
            self._manage_mm_positions(i, high, low, price, ts, atr_val)

            # Mode 1: V10 rolling adds
            if self.active_trades:
                self._check_mode1_v10_rolling(i, price, high, low, ts, atr_val)

            # Mode 1: primary signals
            if i in signal_map:
                sig = signal_map[i]
                self._execute_entry(sig, ts, closes, highs_arr, lows_arr, atr_arr)

            # Mode 2: check for new bottom-fish entry
            if self.mm_base_trade is None and i >= 205:
                self._check_mm_entry(i, price, ts, atr_val)

            # Record equity
            unrealized = self._calc_unrealized(price) + self._calc_mm_unrealized(price)
            self.equity_curve.append({
                "timestamp": ts,
                "equity": self.capital + unrealized,
                "capital": self.capital,
                "unrealized": unrealized,
                "num_mode1": len(self.active_trades),
                "num_mode2": (1 if self.mm_base_trade else 0) + len(self.mm_adds),
            })

        # Close remaining positions at end
        last_price = self.df["close"].iloc[-1]
        last_ts = self.df["timestamp"].iloc[-1]
        for trade in list(self.active_trades):
            self._close_trade(trade, last_price, len(self.df) - 1, last_ts, reason="end_of_backtest")
        if self.mm_base_trade or self.mm_adds:
            self._close_all_mm(last_price, len(self.df) - 1, last_ts, reason="end_of_backtest")

        return self._compute_metrics()

    # -- Mode 1: V6 entry (same as V9) ----------------------------------------

    def _execute_entry(self, signal, ts, closes, highs, lows, atr_vals):
        same_dir = [t for t in self.active_trades if t.side == signal.side and not t.is_rolling]
        if same_dir:
            return
        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            return
        risk_amount = self.capital * self.config.risk_per_trade_pct
        base_size_btc = risk_amount / risk
        max_base_size = (self.capital * self.config.max_position_pct) / signal.entry_price
        base_size_btc = min(base_size_btc, max_base_size)
        if base_size_btc * signal.entry_price < 10:
            return
        leverage, has_123 = determine_leverage(
            signal.idx, signal.side, closes, highs, lows, atr_vals,
            self.swing_highs, self.swing_lows
        )
        leveraged_size_btc = base_size_btc * leverage
        margin = base_size_btc * signal.entry_price
        notional = leveraged_size_btc * signal.entry_price
        commission = notional * self.config.commission_rate
        self.capital -= commission
        if signal.side == "long":
            liq_price = signal.entry_price * (1 - 1 / leverage)
            tp1 = signal.entry_price + risk * self.config.rr_ratio
        else:
            liq_price = signal.entry_price * (1 + 1 / leverage)
            tp1 = signal.entry_price - risk * self.config.rr_ratio
        if leverage == MAX_LEVERAGE:
            self.leverage_stats["trades_at_10x"] += 1
        else:
            self.leverage_stats["trades_at_3x"] += 1
        trade = Trade(
            side=signal.side, entry_price=signal.entry_price,
            entry_idx=signal.idx, entry_time=ts,
            size=leveraged_size_btc, stop_loss=signal.stop_loss,
            take_profit_1=tp1, initial_risk=risk,
            leverage=leverage, margin=margin,
            liquidation_price=liq_price, original_size=leveraged_size_btc,
            trailing_stop=signal.stop_loss,
            highest_since_entry=signal.entry_price,
            lowest_since_entry=signal.entry_price,
        )
        self.active_trades.append(trade)
        self.trades.append(trade)

    # -- Mode 1: V10 Rolling Check & Execution ---------------------------------

    def _check_mode1_v10_rolling(self, i, close, high, low, ts, atr_val):
        """Check for V10 rolling adds on all eligible Mode 1 base trades."""
        if np.isnan(atr_val):
            return
        ma20 = self.df["ma20"].iloc[i]
        if np.isnan(ma20):
            return

        for trade in list(self.active_trades):
            if trade.closed or trade.is_rolling:
                continue

            # V10: Check 4-bar interval (was 12 in V9)
            bars_since = i - trade.entry_idx
            if bars_since < V10_ROLL_CHECK_BARS:
                continue
            if bars_since % V10_ROLL_CHECK_BARS != 0:
                continue

            # V10: Check +2% profitable (was +3% in V9)
            if trade.side == "long":
                pnl_pct = (close - trade.entry_price) / trade.entry_price
            else:
                pnl_pct = (trade.entry_price - close) / trade.entry_price
            if pnl_pct < V10_ROLL_PROFIT_PCT:
                continue

            # Check add count
            base_key = trade.entry_idx
            add_count = self.rolling_add_counts.get(base_key, 0)
            if add_count >= V10_ROLL_MAX_ADDS:
                continue

            # V10: Looser pullback — ANY of 3 conditions (not all)
            if not self._check_pullback_v10(trade, i, close, ma20, atr_val):
                continue

            # V10: Calculate floating profits with 80% multiplier
            group_pnl = self._calc_group_unrealized(trade, close)
            if group_pnl <= 0:
                continue
            available_float = group_pnl * V10_FLOATING_PROFIT_MULT

            # Calculate add size and required margin
            add_mult = V10_ROLL_ADD_MULTS[add_count]
            add_size_btc = trade.original_size * add_mult
            add_margin = add_size_btc * close / trade.leverage

            if add_margin > available_float:
                continue  # insufficient floating profits (80% threshold)

            # Execute add
            self._execute_v10_mode1_add(trade, i, close, ts, atr_val, add_count)

    def _check_pullback_v10(self, trade, i, close, ma20, atr_val):
        """V10: Check ANY of 3 looser conditions (OR logic).

        a) Price within 1 ATR of MA20 (was MA30 with tight tolerance)
        b) 38.2% retrace of move since entry (was 50%)
        c) Breakout above 20-bar high (now available for ALL adds, not just 2+)
        """
        # Condition A: Price within 1 ATR of MA20
        if trade.side == "long":
            if close >= ma20 and (close - ma20) <= atr_val:
                return True
        else:
            if close <= ma20 and (ma20 - close) <= atr_val:
                return True

        # Condition B: 38.2% retrace of move since entry
        if trade.side == "long" and trade.highest_since_entry > trade.entry_price * 1.005:
            move = trade.highest_since_entry - trade.entry_price
            fib_382 = trade.highest_since_entry - 0.382 * move
            # Price near or below the 38.2% retrace level (within 1% tolerance)
            if close <= fib_382 * 1.01 and close >= trade.entry_price:
                return True
        elif trade.side == "short" and trade.lowest_since_entry < trade.entry_price * 0.995:
            move = trade.entry_price - trade.lowest_since_entry
            fib_382 = trade.lowest_since_entry + 0.382 * move
            if close >= fib_382 * 0.99 and close <= trade.entry_price:
                return True

        # Condition C: Breakout above 20-bar high (all adds, not just 2+)
        if trade.side == "long":
            recent_high = self.df["high"].iloc[max(0, i - 20):i].max()
            if close > recent_high:
                return True
        else:
            recent_low = self.df["low"].iloc[max(0, i - 20):i].min()
            if close < recent_low:
                return True

        return False

    def _execute_v10_mode1_add(self, base_trade, idx, close, ts, atr_val, add_count):
        """Execute a V10 rolling add for Mode 1."""
        add_mult = V10_ROLL_ADD_MULTS[add_count]
        add_size_btc = base_trade.original_size * add_mult
        leverage = base_trade.leverage
        notional = add_size_btc * close
        margin = notional / leverage
        commission = notional * self.config.commission_rate
        self.capital -= commission

        if base_trade.side == "long":
            liq_price = close * (1 - 1 / leverage)
        else:
            liq_price = close * (1 + 1 / leverage)

        # Trailing stop at entry price (breakeven stop)
        trailing_stop = close

        risk = atr_val * 1.5 if not np.isnan(atr_val) else close * 0.03
        if base_trade.side == "long":
            tp1 = close + risk
        else:
            tp1 = close - risk

        trade = Trade(
            side=base_trade.side, entry_price=close,
            entry_idx=idx, entry_time=ts,
            size=add_size_btc, stop_loss=close,
            take_profit_1=tp1, initial_risk=risk,
            leverage=leverage, margin=margin,
            liquidation_price=liq_price, original_size=add_size_btc,
            is_rolling=True, trailing_stop=trailing_stop,
            highest_since_entry=close, lowest_since_entry=close,
            base_trade_idx=base_trade.entry_idx,
        )
        self.active_trades.append(trade)
        self.trades.append(trade)

        base_key = base_trade.entry_idx
        self.rolling_add_counts[base_key] = add_count + 1

        if leverage == MAX_LEVERAGE:
            self.leverage_stats["trades_at_10x"] += 1
        else:
            self.leverage_stats["trades_at_3x"] += 1

        self.v10_stats["m1_rolling_adds"] += 1

        print(f"    [V10 Roll M1] Add #{add_count+1} ({add_mult:.0%}) at ${close:,.0f} | "
              f"{leverage}x | size={add_size_btc:.4f} BTC | breakeven stop=${close:,.0f}")

    def _calc_group_unrealized(self, base_trade, current_price):
        """Calculate unrealized PnL for a base trade and all its add positions."""
        total = 0.0
        base_key = base_trade.entry_idx
        for trade in self.active_trades:
            if trade.closed:
                continue
            is_base = (trade.entry_idx == base_key and not trade.is_rolling)
            is_add = (trade.is_rolling and trade.base_trade_idx == base_key)
            if is_base or is_add:
                if trade.side == "long":
                    total += (current_price - trade.entry_price) * trade.size
                else:
                    total += (trade.entry_price - current_price) * trade.size
        return total

    # -- Mode 1: Position Management (modified for V10) -------------------------

    def _manage_positions(self, idx, high, low, close, ts):
        # First: update highest/lowest for base trades
        for trade in self.active_trades:
            if trade.closed:
                continue
            if not trade.is_rolling:
                trade.highest_since_entry = max(trade.highest_since_entry, high)
                trade.lowest_since_entry = min(trade.lowest_since_entry, low)

        for trade in list(self.active_trades):
            if trade.closed:
                continue

            # -- Rolling add: check breakeven stop only --
            if trade.is_rolling:
                if trade.side == "long" and low <= trade.trailing_stop:
                    self._close_trade(trade, trade.trailing_stop, idx, ts, reason="add_breakeven_stop")
                    self.v10_stats["m1_rolling_breakeven_stops"] += 1
                elif trade.side == "short" and high >= trade.trailing_stop:
                    self._close_trade(trade, trade.trailing_stop, idx, ts, reason="add_breakeven_stop")
                    self.v10_stats["m1_rolling_breakeven_stops"] += 1
                continue

            # -- Base trade management --
            base_key = trade.entry_idx
            add_count = self.rolling_add_counts.get(base_key, 0)
            atr = self.df["atr"].iloc[idx]

            if trade.side == "long":
                # Liquidation check
                if low <= trade.liquidation_price:
                    self.leverage_stats["liquidations"] += 1
                    remaining_margin = trade.margin * (trade.size / trade.original_size)
                    self.leverage_stats["liquidation_loss"] += remaining_margin
                    if add_count > 0:
                        self._close_trade_group(trade, trade.liquidation_price, idx, ts, reason="liquidation")
                    else:
                        self._close_trade(trade, trade.liquidation_price, idx, ts, reason="liquidation")
                    continue

                # Master trail update (when adds exist)
                if add_count > 0 and not np.isnan(atr):
                    master_trail = trade.highest_since_entry - V10_MASTER_TRAIL_ATR * atr
                    trade.trailing_stop = max(trade.trailing_stop, master_trail)

                # Stop loss / trailing stop check
                if low <= trade.trailing_stop:
                    if add_count > 0:
                        self._close_trade_group(trade, trade.trailing_stop, idx, ts, reason="master_trail")
                        self.v10_stats["m1_master_trail_closes"] += 1
                    else:
                        self._close_trade(trade, trade.trailing_stop, idx, ts, reason="stop_loss")
                    continue

                # TP1 (existing V6 logic)
                if not trade.half_closed and high >= trade.take_profit_1:
                    half_size = trade.size / 2
                    pnl = (trade.take_profit_1 - trade.entry_price) * half_size
                    commission = half_size * trade.take_profit_1 * self.config.commission_rate
                    tp1_pnl = pnl - commission
                    self.capital += tp1_pnl
                    trade.pnl_tp1 = tp1_pnl
                    trade.size -= half_size
                    trade.half_closed = True
                    trade.trailing_stop = max(trade.trailing_stop, trade.entry_price + trade.initial_risk * 0.1)

                # Trailing stop update (only when NO adds — with adds, master trail governs)
                if not np.isnan(atr) and trade.half_closed and add_count == 0:
                    new_trail = close - atr * self.config.trailing_stop_atr_mult
                    trade.trailing_stop = max(trade.trailing_stop, new_trail)

            elif trade.side == "short":
                # Liquidation check
                if high >= trade.liquidation_price:
                    self.leverage_stats["liquidations"] += 1
                    remaining_margin = trade.margin * (trade.size / trade.original_size)
                    self.leverage_stats["liquidation_loss"] += remaining_margin
                    if add_count > 0:
                        self._close_trade_group(trade, trade.liquidation_price, idx, ts, reason="liquidation")
                    else:
                        self._close_trade(trade, trade.liquidation_price, idx, ts, reason="liquidation")
                    continue

                # Master trail update (when adds exist)
                if add_count > 0 and not np.isnan(atr):
                    master_trail = trade.lowest_since_entry + V10_MASTER_TRAIL_ATR * atr
                    trade.trailing_stop = min(trade.trailing_stop, master_trail)

                # Stop loss / trailing stop check
                if high >= trade.trailing_stop:
                    if add_count > 0:
                        self._close_trade_group(trade, trade.trailing_stop, idx, ts, reason="master_trail")
                        self.v10_stats["m1_master_trail_closes"] += 1
                    else:
                        self._close_trade(trade, trade.trailing_stop, idx, ts, reason="stop_loss")
                    continue

                # TP1
                if not trade.half_closed and low <= trade.take_profit_1:
                    half_size = trade.size / 2
                    pnl = (trade.entry_price - trade.take_profit_1) * half_size
                    commission = half_size * trade.take_profit_1 * self.config.commission_rate
                    tp1_pnl = pnl - commission
                    self.capital += tp1_pnl
                    trade.pnl_tp1 = tp1_pnl
                    trade.size -= half_size
                    trade.half_closed = True
                    trade.trailing_stop = min(trade.trailing_stop, trade.entry_price - trade.initial_risk * 0.1)

                # Trailing stop update (only when NO adds)
                if not np.isnan(atr) and trade.half_closed and add_count == 0:
                    new_trail = close + atr * self.config.trailing_stop_atr_mult
                    trade.trailing_stop = min(trade.trailing_stop, new_trail)

    def _close_trade(self, trade, exit_price, idx, ts, reason=""):
        if trade.side == "long":
            pnl = (exit_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - exit_price) * trade.size
        commission = trade.size * exit_price * self.config.commission_rate
        pnl -= commission
        trade.exit_price = exit_price
        trade.exit_idx = idx
        trade.exit_time = ts
        trade.pnl = pnl + trade.pnl_tp1
        trade.closed = True
        trade.close_reason = reason
        self.capital += pnl
        if trade in self.active_trades:
            self.active_trades.remove(trade)
        self.closed_trades.append(trade)
        if trade.is_rolling:
            self.v10_stats["m1_rolling_pnl"] += trade.pnl

    def _close_trade_group(self, base_trade, exit_price, idx, ts, reason=""):
        """Close base trade and all its rolling adds."""
        base_key = base_trade.entry_idx

        # Close base trade
        self._close_trade(base_trade, exit_price, idx, ts, reason=reason)

        # Close all linked adds
        for trade in list(self.active_trades):
            if trade.is_rolling and trade.base_trade_idx == base_key and not trade.closed:
                self._close_trade(trade, exit_price, idx, ts, reason=reason)

    def _calc_unrealized(self, current_price):
        total = 0.0
        for trade in self.active_trades:
            if trade.side == "long":
                total += (current_price - trade.entry_price) * trade.size
            else:
                total += (trade.entry_price - current_price) * trade.size
        return total

    # -- Mode 2: Graduated Bottom-Fishing --------------------------------------

    def _get_current_mm_leverage(self) -> float:
        idx = min(self.mm_attempt_count, MM_MAX_ATTEMPTS - 1)
        return MM_GRADUATED_LEVERAGE[idx]

    def _check_mm_conditions(self, idx: int, price: float) -> bool:
        if idx < 205:
            return False
        if idx < self.mm_cooldown_until_idx:
            return False
        closes = self.df["close"].values
        if not is_near_market_bottom(closes, idx):
            return False
        sma200 = self.df["sma200"].iloc[idx]
        if np.isnan(sma200) or price >= sma200:
            return False
        weekly_rsi = self.df["weekly_rsi"].iloc[idx]
        if np.isnan(weekly_rsi) or weekly_rsi >= MM_RSI_THRESHOLD:
            return False
        return True

    def _check_mm_entry(self, idx: int, price: float, ts: pd.Timestamp, atr_val: float):
        if not self._check_mm_conditions(idx, price):
            return
        self._enter_mm_trade(idx, price, ts, is_reentry=False)

    def _enter_mm_trade(self, idx: int, price: float, ts: pd.Timestamp,
                        is_reentry: bool = False):
        leverage = self._get_current_mm_leverage()
        margin = self.capital * MM_MARGIN_PCT
        if margin < MM_MIN_MARGIN:
            return
        notional = margin * leverage
        size_btc = notional / price

        commission = notional * self.config.commission_rate
        self.capital -= commission

        liq_price = price * (1 - 1 / leverage)
        stop_loss = liq_price * MM_LIQ_BUFFER

        trade = MMTrade(
            entry_price=price, entry_idx=idx, entry_time=ts,
            size=size_btc, margin=margin, leverage=leverage,
            liquidation_price=liq_price, stop_loss=stop_loss,
        )
        self.mm_base_trade = trade
        self.mm_in_profit_mode = False
        self.mm_master_trail = 0.0
        self.mm_highest_since_profit = price
        self.mm_roll_add_count = 0
        self.mm_stats["total_entries"] += 1
        self.mm_stats["entries_per_leverage"][leverage] = self.mm_stats["entries_per_leverage"].get(leverage, 0) + 1

        stop_dist_pct = (1 / leverage) * 100
        action = "re-entry" if is_reentry else "initial entry"
        attempt = self.mm_attempt_count + 1
        print(f"    [MM] {action} #{attempt} at ${price:,.0f} | "
              f"{leverage}x | margin=${margin:,.0f} | liq=${liq_price:,.0f} | "
              f"SL ~{stop_dist_pct:.1f}% away")

    def _manage_mm_positions(self, idx: int, high: float, low: float,
                             close: float, ts: pd.Timestamp, atr_val: float):
        if self.mm_base_trade is None:
            return

        base = self.mm_base_trade

        if not self.mm_in_profit_mode:
            # -- Phase A: trying to catch the bottom --
            if low <= base.liquidation_price:
                self._close_mm_single(base, base.liquidation_price, idx, ts, "liquidation")
                self._handle_mm_stop(idx, close, ts)
                return

            if low <= base.stop_loss:
                self._close_mm_single(base, base.stop_loss, idx, ts, "stop_loss")
                self._handle_mm_stop(idx, close, ts)
                return

            # V10: Check if we've reached +2% -> enter profit mode (was +3% in V9)
            gain_pct = (close - base.entry_price) / base.entry_price
            if gain_pct >= MM_PROFIT_THRESHOLD:
                self.mm_in_profit_mode = True
                self.mm_master_trail = base.entry_price
                self.mm_highest_since_profit = close
                self.mm_roll_add_count = 0
                lev = base.leverage
                self.mm_stats["wins_per_leverage"][lev] = self.mm_stats["wins_per_leverage"].get(lev, 0) + 1
                print(f"    [MM] +{gain_pct*100:.1f}% reached at ${close:,.0f} -- "
                      f"PROFIT MODE active ({lev}x) | trail=${self.mm_master_trail:,.0f}")

        if self.mm_in_profit_mode:
            # -- Phase B: riding the trend with V10 rolling --

            # Update highest price
            self.mm_highest_since_profit = max(self.mm_highest_since_profit, high)

            # Master trail: 2x ATR from highest point
            if not np.isnan(atr_val) and atr_val > 0:
                new_trail = self.mm_highest_since_profit - V10_MASTER_TRAIL_ATR * atr_val
                new_trail = max(new_trail, base.entry_price)
                self.mm_master_trail = max(self.mm_master_trail, new_trail)

            # Check individual add breakeven stops
            for add_trade in list(self.mm_adds):
                if not add_trade.closed and low <= add_trade.stop_loss:
                    self._close_mm_single(add_trade, add_trade.stop_loss, idx, ts, "add_breakeven_stop")
                    self.mm_adds.remove(add_trade)
                    self.v10_stats["m2_rolling_breakeven_stops"] += 1

            # Master trail check
            if low <= self.mm_master_trail:
                total_pnl = self._close_all_mm(self.mm_master_trail, idx, ts, reason="master_trail")
                self.mm_stats["successful_rides"] += 1
                self.mm_stats["best_ride_pnl"] = max(self.mm_stats["best_ride_pnl"], total_pnl)
                self.mm_attempt_count = 0
                return

            # V10 rolling add check every 4 bars (was 12 in V9)
            if not np.isnan(atr_val) and atr_val > 0:
                bars_since = idx - base.entry_idx
                if bars_since >= V10_ROLL_CHECK_BARS and bars_since % V10_ROLL_CHECK_BARS == 0:
                    if self.mm_roll_add_count < V10_ROLL_MAX_ADDS:
                        # V10: Check +2% still profitable
                        gain_pct = (close - base.entry_price) / base.entry_price
                        if gain_pct >= V10_ROLL_PROFIT_PCT:
                            if self._check_mm_pullback_v10(idx, close, base, atr_val):
                                unrealized = self._calc_mm_unrealized(close)
                                if unrealized > 0:
                                    available_float = unrealized * V10_FLOATING_PROFIT_MULT
                                    add_mult = V10_ROLL_ADD_MULTS[self.mm_roll_add_count]
                                    add_size = base.size * add_mult
                                    add_margin = add_size * close / base.leverage
                                    if add_margin <= available_float:
                                        self._add_mm_v10_roll(idx, close, ts, add_size, base.leverage)

    def _check_mm_pullback_v10(self, idx, close, base, atr_val):
        """V10: Check ANY of 3 looser pullback conditions for Mode 2 rolling adds."""
        ma20 = self.df["ma20"].iloc[idx]

        # Condition A: Price within 1 ATR of MA20
        if not np.isnan(ma20):
            if close >= ma20 and (close - ma20) <= atr_val:
                return True

        # Condition B: 38.2% retrace
        if self.mm_highest_since_profit > base.entry_price * 1.005:
            move = self.mm_highest_since_profit - base.entry_price
            fib_382 = self.mm_highest_since_profit - 0.382 * move
            if close <= fib_382 * 1.01 and close >= base.entry_price:
                return True

        # Condition C: Breakout above 20-bar high (all adds)
        recent_high = self.df["high"].iloc[max(0, idx - 20):idx].max()
        if close > recent_high:
            return True

        return False

    def _handle_mm_stop(self, idx: int, close: float, ts: pd.Timestamp):
        lev = MM_GRADUATED_LEVERAGE[min(self.mm_attempt_count, MM_MAX_ATTEMPTS - 1)]
        self.mm_stats["stops_per_leverage"][lev] = self.mm_stats["stops_per_leverage"].get(lev, 0) + 1
        self.mm_attempt_count += 1
        self.mm_stats["total_stops_phase_a"] += 1

        if self.mm_attempt_count >= MM_MAX_ATTEMPTS:
            self.mm_cooldown_until_idx = idx + MM_COOLDOWN_BARS
            self.mm_stats["cooldowns_triggered"] += 1
            print(f"    [MM] 5 attempts exhausted (5x->10x->15x->25x->50x) -- 30-day cooldown until bar {self.mm_cooldown_until_idx}")
            self.mm_attempt_count = 0
            self.mm_base_trade = None
        else:
            self.mm_base_trade = None
            next_lev = MM_GRADUATED_LEVERAGE[self.mm_attempt_count]
            print(f"    [MM] Stopped out -- escalating to {next_lev}x")
            self._enter_mm_trade(idx, close, ts, is_reentry=True)

    def _add_mm_v10_roll(self, idx: int, price: float, ts: pd.Timestamp,
                         add_size: float, leverage: float):
        """Add a V10 rolling position for Mode 2 using floating profits."""
        notional = add_size * price
        commission = notional * self.config.commission_rate
        self.capital -= commission

        liq_price = price * (1 - 1 / leverage)
        margin = notional / leverage

        trade = MMTrade(
            entry_price=price, entry_idx=idx, entry_time=ts,
            size=add_size, margin=margin, leverage=leverage,
            liquidation_price=liq_price, stop_loss=price,  # breakeven stop at entry
            is_add=True,
        )
        self.mm_adds.append(trade)
        self.mm_roll_add_count += 1
        self.mm_stats["total_adds"] += 1
        self.v10_stats["m2_rolling_adds"] += 1

        add_mult = V10_ROLL_ADD_MULTS[self.mm_roll_add_count - 1]
        print(f"    [MM V10 Roll] Add #{self.mm_roll_add_count} ({add_mult:.0%}) at ${price:,.0f} | "
              f"{leverage}x | margin=${margin:,.0f} (from 80% floating) | size={add_size:.4f} BTC")

    def _close_mm_single(self, trade: MMTrade, exit_price: float, idx: int,
                         ts: pd.Timestamp, reason: str):
        pnl = (exit_price - trade.entry_price) * trade.size
        commission = trade.size * exit_price * self.config.commission_rate
        pnl -= commission
        trade.exit_price = exit_price
        trade.exit_idx = idx
        trade.exit_time = ts
        trade.pnl = pnl
        trade.closed = True
        trade.close_reason = reason
        self.capital += pnl
        self.mm_closed_trades.append(trade)

        if pnl < self.mm_stats["worst_stop_pnl"]:
            self.mm_stats["worst_stop_pnl"] = pnl

        if trade.is_add:
            self.v10_stats["m2_rolling_pnl"] += pnl

    def _close_all_mm(self, exit_price: float, idx: int, ts: pd.Timestamp,
                      reason: str) -> float:
        total_pnl = 0.0
        if self.mm_base_trade and not self.mm_base_trade.closed:
            self._close_mm_single(self.mm_base_trade, exit_price, idx, ts, reason)
            total_pnl += self.mm_base_trade.pnl
        for add_t in self.mm_adds:
            if not add_t.closed:
                self._close_mm_single(add_t, exit_price, idx, ts, reason)
                total_pnl += add_t.pnl
        n_closed = 1 + len(self.mm_adds)
        print(f"    [MM] Closed {n_closed} positions at ${exit_price:,.0f} | "
              f"total PnL=${total_pnl:,.2f} | reason={reason}")
        self.mm_base_trade = None
        self.mm_adds = []
        self.mm_in_profit_mode = False
        self.mm_master_trail = 0.0
        self.mm_highest_since_profit = 0.0
        self.mm_roll_add_count = 0
        return total_pnl

    def _calc_mm_unrealized(self, current_price: float) -> float:
        total = 0.0
        if self.mm_base_trade and not self.mm_base_trade.closed:
            total += (current_price - self.mm_base_trade.entry_price) * self.mm_base_trade.size
        for add_t in self.mm_adds:
            if not add_t.closed:
                total += (current_price - add_t.entry_price) * add_t.size
        return total

    # -- Metrics & Reporting ---------------------------------------------------

    def _compute_metrics(self) -> dict:
        all_mode1 = self.closed_trades
        all_mode2 = self.mm_closed_trades

        if not all_mode1 and not all_mode2:
            return {"version": "v10", "error": "No trades executed"}

        equity_df = pd.DataFrame(self.equity_curve)
        equity = equity_df["equity"].values

        total_return = (equity[-1] - self.config.initial_capital) / self.config.initial_capital * 100
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = np.max(drawdown)

        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(2190)
        else:
            sharpe = 0.0

        all_pnls = [t.pnl for t in all_mode1] + [t.pnl for t in all_mode2]
        wins = [p for p in all_pnls if p > 0]
        losses = [p for p in all_pnls if p <= 0]
        win_rate = len(wins) / len(all_pnls) * 100 if all_pnls else 0
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Mode 1 breakdown
        m1_pnls = [t.pnl for t in all_mode1]
        m1_wins = [p for p in m1_pnls if p > 0]
        m1_losses = [p for p in m1_pnls if p <= 0]
        m1_win_rate = len(m1_wins) / len(m1_pnls) * 100 if m1_pnls else 0
        m1_long = [t for t in all_mode1 if t.side == "long"]
        m1_short = [t for t in all_mode1 if t.side == "short"]
        m1_primary = [t for t in all_mode1 if not t.is_rolling]
        m1_rolling = [t for t in all_mode1 if t.is_rolling]
        m1_durations = []
        for t in all_mode1:
            if t.entry_time is not None and t.exit_time is not None:
                m1_durations.append((t.exit_time - t.entry_time).total_seconds() / 3600)
        m1_avg_dur_h = np.mean(m1_durations) if m1_durations else 0

        # Mode 2 breakdown
        m2_pnls = [t.pnl for t in all_mode2]
        m2_wins = [p for p in m2_pnls if p > 0]
        m2_losses = [p for p in m2_pnls if p <= 0]
        m2_win_rate = len(m2_wins) / len(m2_pnls) * 100 if m2_pnls else 0
        m2_adds = [t for t in all_mode2 if t.is_add]
        m2_base = [t for t in all_mode2 if not t.is_add]
        m2_durations = []
        for t in all_mode2:
            if t.entry_time is not None and t.exit_time is not None:
                m2_durations.append((t.exit_time - t.entry_time).total_seconds() / 3600)
        m2_avg_dur_h = np.mean(m2_durations) if m2_durations else 0

        avg_leverage_m1 = np.mean([t.leverage for t in all_mode1]) if all_mode1 else 0
        avg_leverage_m2 = np.mean([t.leverage for t in all_mode2]) if all_mode2 else 0

        return {
            "version": "v10",
            # Combined
            "total_return_pct": round(total_return, 2),
            "final_equity": round(equity[-1], 2),
            "initial_capital": self.config.initial_capital,
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "total_trades": len(all_pnls),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "largest_win": round(max(wins), 2) if wins else 0,
            "largest_loss": round(min(losses), 2) if losses else 0,

            # Mode 1 (V6 MACD + V10 Rolling)
            "m1_total_trades": len(all_mode1),
            "m1_primary": len(m1_primary),
            "m1_rolling_adds": len(m1_rolling),
            "m1_long_trades": len(m1_long),
            "m1_short_trades": len(m1_short),
            "m1_long_pnl": round(sum(t.pnl for t in m1_long), 2),
            "m1_short_pnl": round(sum(t.pnl for t in m1_short), 2),
            "m1_total_pnl": round(sum(m1_pnls), 2),
            "m1_win_rate_pct": round(m1_win_rate, 2),
            "m1_avg_duration_hours": round(m1_avg_dur_h, 1),
            "m1_avg_duration_days": round(m1_avg_dur_h / 24, 1),
            "m1_avg_leverage": round(avg_leverage_m1, 2),
            "m1_trades_at_3x": self.leverage_stats["trades_at_3x"],
            "m1_trades_at_10x": self.leverage_stats["trades_at_10x"],
            "m1_liquidations": self.leverage_stats["liquidations"],
            "m1_liquidation_loss": round(self.leverage_stats["liquidation_loss"], 2),
            "rr_ratio": self.config.rr_ratio,

            # V10 Rolling Stats (Mode 1)
            "v10_m1_rolling_adds": self.v10_stats["m1_rolling_adds"],
            "v10_m1_breakeven_stops": self.v10_stats["m1_rolling_breakeven_stops"],
            "v10_m1_master_trail_closes": self.v10_stats["m1_master_trail_closes"],
            "v10_m1_rolling_pnl": round(self.v10_stats["m1_rolling_pnl"], 2),

            # Mode 2 (Graduated Bottom-Fishing + V10 Rolling)
            "m2_total_trades": len(all_mode2),
            "m2_base_entries": len(m2_base),
            "m2_adds": len(m2_adds),
            "m2_total_pnl": round(sum(m2_pnls), 2),
            "m2_win_rate_pct": round(m2_win_rate, 2),
            "m2_avg_duration_hours": round(m2_avg_dur_h, 1),
            "m2_avg_duration_days": round(m2_avg_dur_h / 24, 1),
            "m2_avg_leverage": round(avg_leverage_m2, 2),
            "m2_total_entries": self.mm_stats["total_entries"],
            "m2_stops_phase_a": self.mm_stats["total_stops_phase_a"],
            "m2_successful_rides": self.mm_stats["successful_rides"],
            "m2_cooldowns_triggered": self.mm_stats["cooldowns_triggered"],
            "m2_total_adds": self.mm_stats["total_adds"],
            "m2_best_ride_pnl": round(self.mm_stats["best_ride_pnl"], 2),
            "m2_worst_stop_pnl": round(self.mm_stats["worst_stop_pnl"], 2),
            "m2_entries_per_leverage": {str(k): v for k, v in self.mm_stats["entries_per_leverage"].items()},
            "m2_stops_per_leverage": {str(k): v for k, v in self.mm_stats["stops_per_leverage"].items()},
            "m2_wins_per_leverage": {str(k): v for k, v in self.mm_stats["wins_per_leverage"].items()},

            # V10 Rolling Stats (Mode 2)
            "v10_m2_rolling_adds": self.v10_stats["m2_rolling_adds"],
            "v10_m2_breakeven_stops": self.v10_stats["m2_rolling_breakeven_stops"],
            "v10_m2_rolling_pnl": round(self.v10_stats["m2_rolling_pnl"], 2),

            "data_start": str(self.df["timestamp"].iloc[0]),
            "data_end": str(self.df["timestamp"].iloc[-1]),
            "total_candles": len(self.df),
        }

    def generate_report(self, metrics: dict):
        os.makedirs(RESULTS_DIR, exist_ok=True)

        with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        report = self._format_report(metrics)
        with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
            f.write(report)
        print(report)

        self._plot_equity_curve()
        self._save_trade_log()
        self._generate_comparison(metrics)

    def _format_report(self, m: dict) -> str:
        sep = "=" * 65
        lines = [
            sep,
            "  BTC/USDT BACKTEST -- V10",
            "  Mode 1: V6 MACD Reversal (3x/10x) + V10 Rolling",
            "  Mode 2: Graduated Bottom-Fishing (5x->50x) + V10 Rolling",
            sep,
            "",
            f"  Data Period:     {m['data_start'][:10]} to {m['data_end'][:10]}",
            f"  Total Candles:   {m['total_candles']} (4h bars)",
            f"  Initial Capital: ${m['initial_capital']:,.2f}",
            "",
            sep,
            "  COMBINED PERFORMANCE",
            sep,
            "",
            f"  Final Equity:    ${m['final_equity']:,.2f}",
            f"  Total Return:    {m['total_return_pct']:+.2f}%",
            f"  Max Drawdown:    {m['max_drawdown_pct']:.2f}%",
            f"  Sharpe Ratio:    {m['sharpe_ratio']:.2f}",
            f"  Total Trades:    {m['total_trades']}",
            f"  Win Rate:        {m['win_rate_pct']:.1f}%",
            f"  Profit Factor:   {m['profit_factor']:.2f}",
            "",
            f"  Avg Win:         ${m['avg_win']:,.2f}",
            f"  Avg Loss:        ${m['avg_loss']:,.2f}",
            f"  Largest Win:     ${m['largest_win']:,.2f}",
            f"  Largest Loss:    ${m['largest_loss']:,.2f}",
            "",
            sep,
            "  MODE 1: V6 MACD REVERSAL + V10 ROLLING",
            sep,
            "",
            f"  Total Trades:    {m['m1_total_trades']}",
            f"    Primary:       {m['m1_primary']}",
            f"    V10 Rolling:   {m['m1_rolling_adds']}",
            f"    Long:          {m['m1_long_trades']}  (PnL: ${m['m1_long_pnl']:+,.2f})",
            f"    Short:         {m['m1_short_trades']}  (PnL: ${m['m1_short_pnl']:+,.2f})",
            f"  Total PnL:       ${m['m1_total_pnl']:+,.2f}",
            f"  Win Rate:        {m['m1_win_rate_pct']:.1f}%",
            f"  Avg Duration:    {m['m1_avg_duration_days']:.1f} days",
            "",
            f"  Avg Leverage:    {m['m1_avg_leverage']:.1f}x",
            f"  Trades at 3x:    {m['m1_trades_at_3x']}",
            f"  Trades at 10x:   {m['m1_trades_at_10x']}",
            f"  Liquidations:    {m['m1_liquidations']}",
            f"  Liquidation Loss:${m['m1_liquidation_loss']:,.2f}",
            "",
            "  V10 Rolling (Mode 1):",
            f"    Adds Executed:     {m['v10_m1_rolling_adds']}",
            f"    Breakeven Stops:   {m['v10_m1_breakeven_stops']}",
            f"    Master Trail Closes:{m['v10_m1_master_trail_closes']}",
            f"    Rolling PnL:       ${m['v10_m1_rolling_pnl']:+,.2f}",
            "",
            sep,
            "  MODE 2: GRADUATED BOTTOM-FISHING + V10 ROLLING",
            sep,
            "",
            f"  Total Trades:    {m['m2_total_trades']}",
            f"    Base Entries:   {m['m2_base_entries']}",
            f"    V10 Rolling:   {m['m2_adds']}",
            f"  Total PnL:       ${m['m2_total_pnl']:+,.2f}",
            f"  Win Rate:        {m['m2_win_rate_pct']:.1f}%",
            f"  Avg Duration:    {m['m2_avg_duration_days']:.1f} days",
            f"  Avg Leverage:    {m['m2_avg_leverage']:.1f}x",
            "",
            f"  Entry Attempts:  {m['m2_total_entries']}",
            f"  Phase A Stops:   {m['m2_stops_phase_a']}  (stopped before +2%)",
            f"  Successful Rides:{m['m2_successful_rides']}  (reached profit mode)",
            f"  Cooldowns:       {m['m2_cooldowns_triggered']}  (5 fails -> 30d disable)",
            f"  V10 Rolling Adds:{m['m2_total_adds']}",
            "",
            f"  Best Ride PnL:   ${m['m2_best_ride_pnl']:+,.2f}",
            f"  Worst Stop PnL:  ${m['m2_worst_stop_pnl']:+,.2f}",
            "",
            "  V10 Rolling (Mode 2):",
            f"    Adds Executed:     {m['v10_m2_rolling_adds']}",
            f"    Breakeven Stops:   {m['v10_m2_breakeven_stops']}",
            f"    Rolling PnL:       ${m['v10_m2_rolling_pnl']:+,.2f}",
            "",
            "  Graduated Leverage Breakdown:",
        ]

        for lev in MM_GRADUATED_LEVERAGE:
            entries = m['m2_entries_per_leverage'].get(str(lev), 0)
            stops = m['m2_stops_per_leverage'].get(str(lev), 0)
            wins = m['m2_wins_per_leverage'].get(str(lev), 0)
            stop_dist = (1/lev) * 100
            lines.append(f"    {lev:>2}x: {entries:>3} entries, {stops:>3} stops, {wins:>3} wins  (SL ~{stop_dist:.1f}% away)")

        lines += [
            "",
            sep,
            "  V10 ROLLING RULES (BOTH MODES)",
            sep,
            "",
            f"  Trigger:         Position +{V10_ROLL_PROFIT_PCT*100:.0f}% profitable",
            f"  Check Interval:  Every {V10_ROLL_CHECK_BARS} bars (16h)",
            f"  Max Adds:        {V10_ROLL_MAX_ADDS} per trade",
            f"  Add 1:           {V10_ROLL_ADD_MULTS[0]:.0%} of original size",
            f"  Add 2:           {V10_ROLL_ADD_MULTS[1]:.0%} of original size",
            f"  Add 3:           {V10_ROLL_ADD_MULTS[2]:.0%} of original size",
            f"  Funding:         {V10_FLOATING_PROFIT_MULT:.0%} of floating profits (skip if insufficient)",
            f"  Add Stop:        Breakeven (at entry price of add)",
            f"  Master Trail:    {V10_MASTER_TRAIL_ATR:.0f}x ATR from highest point (closes group)",
            "",
            "  V10 Add Conditions (ANY triggers add):",
            "    a) Price within 1 ATR of MA20",
            "    b) 38.2% Fibonacci retrace of move since entry",
            "    c) Breakout above previous 20-bar high",
            "",
            sep,
            "  V10 CHANGES FROM V9",
            sep,
            "",
            "  1. Check every 4 bars (16h) instead of 12 bars (48h)",
            "  2. Looser add conditions (ANY of 3, not all):",
            "     a) MA20 within 1 ATR (was MA30 with 1% tolerance)",
            "     b) 38.2% retrace (was 50%)",
            "     c) 20-bar high breakout (now for ALL adds, not just 2+)",
            "  3. Profit trigger at +2% (was +3%)",
            "  4. Add margin <= 80% of floating profits (was 100%)",
            "  5. Inverted pyramid stays: 150%/200%/250%",
            "  6. Each add breakeven stop, master trail 2x ATR",
            "",
            sep,
            f"  Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            sep,
        ]
        return "\n".join(lines)

    def _plot_equity_curve(self):
        equity_df = pd.DataFrame(self.equity_curve)
        equity = equity_df["equity"].values
        timestamps = pd.to_datetime(equity_df["timestamp"])

        peak = np.maximum.accumulate(equity)
        drawdown_pct = (peak - equity) / peak * 100

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 15), height_ratios=[3, 1, 1.5])

        # -- Equity curve --
        ax1.plot(timestamps, equity, color="#00E676", linewidth=1.4,
                 label="V10 Equity (V9 + Looser Rolling)")
        ax1.axhline(y=self.config.initial_capital, color="gray", linestyle="--",
                     alpha=0.5, label="Initial Capital")

        # Mode 1 trade markers
        for trade in self.closed_trades[:80]:
            if trade.close_reason == "liquidation":
                mc = "#9C27B0"
            elif trade.close_reason == "master_trail":
                mc = "#FF9800"
            elif trade.pnl > 0:
                mc = "#4CAF50"
            else:
                mc = "#F44336"
            if trade.entry_time in timestamps.values:
                eidx = timestamps[timestamps == trade.entry_time].index
                if len(eidx) > 0:
                    ms = 4 + trade.leverage
                    marker = "^" if trade.side == "long" else "v"
                    if trade.is_rolling:
                        marker = "p"  # pentagon for rolling adds
                        ms = 6
                    ax1.plot(trade.entry_time, equity[eidx[0]],
                             marker=marker, color=mc, markersize=ms, alpha=0.6)

        # Mode 2 trade markers
        for trade in self.mm_closed_trades:
            if trade.pnl > 0:
                mc = "#FF9800"
            else:
                mc = "#795548"
            if trade.entry_time in timestamps.values:
                eidx = timestamps[timestamps == trade.entry_time].index
                if len(eidx) > 0:
                    ms = 4 + trade.leverage / 5
                    marker = "D" if not trade.is_add else "p"
                    ax1.plot(trade.entry_time, equity[eidx[0]],
                             marker=marker, color=mc, markersize=ms, alpha=0.8)

        ax1.set_title("BTC/USDT V10 -- Equity Curve\n"
                       "(Mode 1: V6 MACD + V10 Rolling | Mode 2: Graduated + V10 Rolling)",
                       fontsize=13, fontweight="bold")
        ax1.set_ylabel("Equity ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # -- Drawdown --
        ax2.fill_between(timestamps, -drawdown_pct, 0, color="#F44336", alpha=0.3)
        ax2.plot(timestamps, -drawdown_pct, color="#F44336", linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Drawdown", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # -- Leverage timeline --
        mm_entries = []
        for t in self.mm_closed_trades:
            if not t.is_add:
                mm_entries.append((t.entry_time, t.exit_time, t.pnl > 0, t.leverage))

        for entry_t, exit_t, is_win, lev in mm_entries:
            color = "#FF9800" if is_win else "#795548"
            ax3.axvspan(entry_t, exit_t, alpha=0.3, color=color)
            mid_t = entry_t + (exit_t - entry_t) / 2
            ax3.annotate(f"{int(lev)}x", xy=(mid_t, lev), fontsize=6,
                        ha="center", va="bottom", color=color, alpha=0.9)

        for trade in self.closed_trades:
            if trade.is_rolling:
                color = "#00E676"
                ax3.scatter(trade.entry_time, trade.leverage, c=color, s=25, alpha=0.8,
                           zorder=4, marker="p")
            else:
                color = "#2196F3" if trade.leverage <= 3 else "#F44336"
                ax3.scatter(trade.entry_time, trade.leverage, c=color, s=20, alpha=0.7, zorder=3)

        for trade in self.mm_closed_trades:
            if not trade.is_add:
                color = "#FF9800" if trade.pnl > 0 else "#795548"
                ax3.scatter(trade.entry_time, trade.leverage, c=color, s=30, alpha=0.8,
                           zorder=4, marker="D")
            else:
                color = "#00E676" if trade.pnl > 0 else "#795548"
                ax3.scatter(trade.entry_time, trade.leverage, c=color, s=25, alpha=0.8,
                           zorder=4, marker="p")

        ax3.axhline(y=3, color="#2196F3", linestyle="--", alpha=0.3, label="3x (M1 base)")
        ax3.axhline(y=10, color="#F44336", linestyle="--", alpha=0.3, label="10x (M1 123)")
        for lev in [5, 15, 25, 50]:
            ax3.axhline(y=lev, color="#FF9800", linestyle=":", alpha=0.15)
        ax3.axhline(y=50, color="#FF9800", linestyle="--", alpha=0.3, label="50x (M2 max)")

        ax3.set_ylabel("Leverage")
        ax3.set_title("Leverage Timeline (pentagons = V10 rolling adds)", fontsize=11)
        ax3.set_ylim(0, 55)
        ax3.set_yticks([0, 3, 5, 10, 15, 25, 50])
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        fig.subplots_adjust(hspace=0.40)
        chart_path = os.path.join(RESULTS_DIR, "equity_curve.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Equity curve saved to {chart_path}")

    def _save_trade_log(self):
        records = []
        for t in self.closed_trades:
            records.append({
                "mode": "M1",
                "side": t.side,
                "type": "v10_rolling" if t.is_rolling else "primary",
                "leverage": t.leverage,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "size_btc": round(t.size, 6),
                "margin": round(t.margin, 2),
                "pnl": round(t.pnl, 2),
                "close_reason": t.close_reason,
            })
        for t in self.mm_closed_trades:
            records.append({
                "mode": "M2",
                "side": "long",
                "type": "v10_rolling" if t.is_add else "base",
                "leverage": t.leverage,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "size_btc": round(t.size, 6),
                "margin": round(t.margin, 2),
                "pnl": round(t.pnl, 2),
                "close_reason": t.close_reason,
            })
        trade_df = pd.DataFrame(records)
        trade_file = os.path.join(RESULTS_DIR, "trade_log.csv")
        trade_df.to_csv(trade_file, index=False)
        print(f"  Trade log saved to {trade_file} ({len(records)} trades)")

    def _generate_comparison(self, v10_metrics: dict):
        """Generate V6 vs V9 vs V10 comparison."""
        v6_path = os.path.join(os.path.dirname(__file__), "results_v6", "metrics.json")
        v9_path = os.path.join(os.path.dirname(__file__), "results_v9", "metrics.json")
        v6 = None
        v9 = None
        if os.path.exists(v6_path):
            with open(v6_path) as f:
                v6 = json.load(f)
        if os.path.exists(v9_path):
            with open(v9_path) as f:
                v9 = json.load(f)

        if v6 is None:
            print("  V6 results not found -- skipping comparison")
            return

        v10 = v10_metrics

        def fmt_delta(new, old, higher_is_better=True):
            diff = new - old
            if abs(diff) < 0.005:
                return ""
            arrow = "+" if diff > 0 else ""
            tag = " ^^" if (diff > 0) == higher_is_better else " vv"
            return f"({arrow}{diff:.2f}{tag})"

        def fmt_mult(new, old):
            if old == 0:
                return "(N/A)"
            ratio = new / old
            return f"({ratio:.1f}x)"

        sep = "=" * 115
        cw = 24

        if v9 is not None:
            lines = [
                sep,
                "  BACKTEST COMPARISON: V6 vs V9 vs V10 (Looser Rolling)",
                sep,
                "",
                f"  {'Metric':<28} {'V6 (MACD only)':>{cw}} {'V9 (tight rolling)':>{cw}} {'V10 (loose rolling)':>{cw}}   {'V10 vs V6':>12}",
                f"  {'-'*28} {'-'*cw} {'-'*cw} {'-'*cw}   {'-'*12}",
                "",
                f"  {'Total Return %':<28} {v6['total_return_pct']:>{cw-1}.2f}% {v9['total_return_pct']:>{cw-1}.2f}% {v10['total_return_pct']:>{cw-1}.2f}%   {fmt_delta(v10['total_return_pct'], v6['total_return_pct'])}",
                f"  {'Final Equity':<28} ${v6['final_equity']:>{cw-2},.2f} ${v9['final_equity']:>{cw-2},.2f} ${v10['final_equity']:>{cw-2},.2f}   {fmt_mult(v10['final_equity'], v6['final_equity'])}",
                f"  {'Max Drawdown %':<28} {v6['max_drawdown_pct']:>{cw-1}.2f}% {v9['max_drawdown_pct']:>{cw-1}.2f}% {v10['max_drawdown_pct']:>{cw-1}.2f}%   {fmt_delta(v10['max_drawdown_pct'], v6['max_drawdown_pct'], False)}",
                f"  {'Sharpe Ratio':<28} {v6['sharpe_ratio']:>{cw}.2f} {v9['sharpe_ratio']:>{cw}.2f} {v10['sharpe_ratio']:>{cw}.2f}   {fmt_delta(v10['sharpe_ratio'], v6['sharpe_ratio'])}",
                f"  {'Win Rate %':<28} {v6['win_rate_pct']:>{cw-1}.2f}% {v9['win_rate_pct']:>{cw-1}.2f}% {v10['win_rate_pct']:>{cw-1}.2f}%   {fmt_delta(v10['win_rate_pct'], v6['win_rate_pct'])}",
                f"  {'Profit Factor':<28} {v6['profit_factor']:>{cw}.2f} {v9['profit_factor']:>{cw}.2f} {v10['profit_factor']:>{cw}.2f}   {fmt_delta(v10['profit_factor'], v6['profit_factor'])}",
                "",
                f"  {'Total Trades':<28} {v6['total_trades']:>{cw}} {v9['total_trades']:>{cw}} {v10['total_trades']:>{cw}}",
            ]

            # V9 vs V10 rolling comparison
            lines += [
                "",
                sep,
                "  ROLLING ADDS COMPARISON: V9 (tight) vs V10 (loose)",
                sep,
                "",
                f"  {'':28} {'V9 (tight)':>{cw}} {'V10 (loose)':>{cw}}",
                f"  {'':28} {'-'*cw} {'-'*cw}",
                f"  {'Check Interval':<28} {'12 bars (48h)':>{cw}} {'4 bars (16h)':>{cw}}",
                f"  {'Profit Trigger':<28} {'+3%':>{cw}} {'+2%':>{cw}}",
                f"  {'Float Requirement':<28} {'100%':>{cw}} {'80%':>{cw}}",
                f"  {'Add Conditions':<28} {'MA30/Fib50%/breakout(2+)':>{cw}} {'MA20+1ATR/Fib38.2%/brk(all)':>{cw}}",
                "",
                f"  {'M1 Rolling Adds':<28} {v9.get('v9_m1_rolling_adds', 0):>{cw}} {v10['v10_m1_rolling_adds']:>{cw}}",
                f"  {'M1 Breakeven Stops':<28} {v9.get('v9_m1_breakeven_stops', 0):>{cw}} {v10['v10_m1_breakeven_stops']:>{cw}}",
                f"  {'M1 Master Trail Closes':<28} {v9.get('v9_m1_master_trail_closes', 0):>{cw}} {v10['v10_m1_master_trail_closes']:>{cw}}",
                f"  {'M1 Rolling PnL':<28} ${v9.get('v9_m1_rolling_pnl', 0):>{cw-2},.2f} ${v10['v10_m1_rolling_pnl']:>{cw-2},.2f}",
                "",
                f"  {'M2 Rolling Adds':<28} {v9.get('v9_m2_rolling_adds', 0):>{cw}} {v10['v10_m2_rolling_adds']:>{cw}}",
                f"  {'M2 Breakeven Stops':<28} {v9.get('v9_m2_breakeven_stops', 0):>{cw}} {v10['v10_m2_breakeven_stops']:>{cw}}",
                f"  {'M2 Rolling PnL':<28} ${v9.get('v9_m2_rolling_pnl', 0):>{cw-2},.2f} ${v10['v10_m2_rolling_pnl']:>{cw-2},.2f}",
            ]

            # Mode 2 comparison
            lines += [
                "",
                sep,
                "  MODE 2 COMPARISON: V9 vs V10",
                sep,
                "",
                f"  {'':28} {'V9':>{cw}} {'V10':>{cw}}",
                f"  {'':28} {'-'*cw} {'-'*cw}",
                f"  {'M2 Total Trades':<28} {v9['m2_total_trades']:>{cw}} {v10['m2_total_trades']:>{cw}}",
                f"  {'M2 Total PnL':<28} ${v9['m2_total_pnl']:>{cw-2},.2f} ${v10['m2_total_pnl']:>{cw-2},.2f}",
                f"  {'M2 Win Rate %':<28} {v9['m2_win_rate_pct']:>{cw-1}.2f}% {v10['m2_win_rate_pct']:>{cw-1}.2f}%",
                f"  {'M2 Entry Attempts':<28} {v9['m2_total_entries']:>{cw}} {v10['m2_total_entries']:>{cw}}",
                f"  {'M2 Phase A Stops':<28} {v9['m2_stops_phase_a']:>{cw}} {v10['m2_stops_phase_a']:>{cw}}",
                f"  {'M2 Successful Rides':<28} {v9['m2_successful_rides']:>{cw}} {v10['m2_successful_rides']:>{cw}}",
                f"  {'M2 Cooldowns':<28} {v9['m2_cooldowns_triggered']:>{cw}} {v10['m2_cooldowns_triggered']:>{cw}}",
                f"  {'M2 V-Rolling Adds':<28} {v9.get('m2_total_adds', 0):>{cw}} {v10['m2_total_adds']:>{cw}}",
                f"  {'M2 Best Ride PnL':<28} ${v9['m2_best_ride_pnl']:>{cw-2},.2f} ${v10['m2_best_ride_pnl']:>{cw-2},.2f}",
                f"  {'M2 Worst Stop PnL':<28} ${v9['m2_worst_stop_pnl']:>{cw-2},.2f} ${v10['m2_worst_stop_pnl']:>{cw-2},.2f}",
            ]
        else:
            lines = [
                sep,
                "  BACKTEST COMPARISON: V6 vs V10 (Looser Rolling)",
                sep,
                "",
                f"  {'Metric':<28} {'V6 (MACD only)':>{cw}} {'V10 (loose rolling)':>{cw}}   {'Delta':>15}",
                f"  {'-'*28} {'-'*cw} {'-'*cw}   {'-'*15}",
                "",
                f"  {'Total Return %':<28} {v6['total_return_pct']:>{cw-1}.2f}% {v10['total_return_pct']:>{cw-1}.2f}%   {fmt_delta(v10['total_return_pct'], v6['total_return_pct'])}",
                f"  {'Final Equity':<28} ${v6['final_equity']:>{cw-2},.2f} ${v10['final_equity']:>{cw-2},.2f}   {fmt_mult(v10['final_equity'], v6['final_equity'])}",
                f"  {'Max Drawdown %':<28} {v6['max_drawdown_pct']:>{cw-1}.2f}% {v10['max_drawdown_pct']:>{cw-1}.2f}%   {fmt_delta(v10['max_drawdown_pct'], v6['max_drawdown_pct'], False)}",
                f"  {'Sharpe Ratio':<28} {v6['sharpe_ratio']:>{cw}.2f} {v10['sharpe_ratio']:>{cw}.2f}   {fmt_delta(v10['sharpe_ratio'], v6['sharpe_ratio'])}",
                f"  {'Win Rate %':<28} {v6['win_rate_pct']:>{cw-1}.2f}% {v10['win_rate_pct']:>{cw-1}.2f}%   {fmt_delta(v10['win_rate_pct'], v6['win_rate_pct'])}",
                f"  {'Profit Factor':<28} {v6['profit_factor']:>{cw}.2f} {v10['profit_factor']:>{cw}.2f}   {fmt_delta(v10['profit_factor'], v6['profit_factor'])}",
                "",
                f"  {'Total Trades':<28} {v6['total_trades']:>{cw}} {v10['total_trades']:>{cw}}",
            ]

        # V10 rolling breakdown
        lines += [
            "",
            sep,
            "  V10 ROLLING ADDS SUMMARY",
            sep,
            "",
            f"  Mode 1 Rolling Adds:     {v10['v10_m1_rolling_adds']}",
            f"    Breakeven Stops:       {v10['v10_m1_breakeven_stops']}",
            f"    Master Trail Closes:   {v10['v10_m1_master_trail_closes']}",
            f"    Rolling PnL:           ${v10['v10_m1_rolling_pnl']:+,.2f}",
            "",
            f"  Mode 2 Rolling Adds:     {v10['v10_m2_rolling_adds']}",
            f"    Breakeven Stops:       {v10['v10_m2_breakeven_stops']}",
            f"    Rolling PnL:           ${v10['v10_m2_rolling_pnl']:+,.2f}",
            "",
        ]

        # Graduated leverage breakdown
        lines += [
            sep,
            "  V10 GRADUATED LEVERAGE BREAKDOWN",
            sep,
            "",
            f"  {'Leverage':<10} {'SL Distance':>12} {'Entries':>10} {'Stops':>10} {'Wins':>10}",
            f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}",
        ]
        for lev in MM_GRADUATED_LEVERAGE:
            entries = v10['m2_entries_per_leverage'].get(str(lev), 0)
            stops = v10['m2_stops_per_leverage'].get(str(lev), 0)
            wins = v10['m2_wins_per_leverage'].get(str(lev), 0)
            stop_dist = (1/lev) * 100
            lines.append(f"  {lev:>2}x        {stop_dist:>10.1f}%  {entries:>10} {stops:>10} {wins:>10}")

        lines += [
            "",
            sep,
            "  STRATEGY DIFFERENCES",
            sep,
            "",
            "  V6 (MACD only):",
            "    - V3 MACD Reversal signals only",
            "    - 3x base, 10x with 123 Rule confirmed",
            "    - TP1 at 1:1 R/R, trailing stop after",
            "",
            "  V9 = V8 + tight rolling:",
            f"    - Rolling after +3% (checks every 12 bars / 48h)",
            "    - MA30 pullback (1% tolerance) or Fib 50%",
            "    - Funded by 100% of floating profits",
            "",
            "  V10 = V9 + looser rolling:",
            f"    - Rolling after +{V10_ROLL_PROFIT_PCT*100:.0f}% (checks every {V10_ROLL_CHECK_BARS} bars / 16h)",
            "    - ANY of: MA20 within 1 ATR / Fib 38.2% / 20-bar high breakout",
            f"    - Funded by {V10_FLOATING_PROFIT_MULT:.0%} of floating profits",
            f"    - Adds at {V10_ROLL_ADD_MULTS[0]:.0%}/{V10_ROLL_ADD_MULTS[1]:.0%}/{V10_ROLL_ADD_MULTS[2]:.0%} (inverted pyramid)",
            "    - Each add: breakeven stop at entry",
            f"    - Master trail: {V10_MASTER_TRAIL_ATR:.0f}x ATR from highest (closes group)",
            "",
            sep,
        ]

        comparison = "\n".join(lines)
        comp_path = os.path.join(RESULTS_DIR, "comparison_v6_v9_v10.txt")
        with open(comp_path, "w") as f:
            f.write(comparison)
        print(f"\n{comparison}")
        print(f"\n  Comparison saved to {comp_path}")


# -- Main ----------------------------------------------------------------------

def main():
    df = fetch_ohlcv(use_cache=True)
    print(f"\nLoaded {len(df)} candles: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    config = BacktestConfig(
        initial_capital=10_000.0,
        commission_rate=0.001,
        rr_ratio=1.0,
    )

    engine = BacktestEngine(df, config)
    metrics = engine.run()
    engine.generate_report(metrics)

    print("\nBacktest V10 complete!")
    return metrics


if __name__ == "__main__":
    main()

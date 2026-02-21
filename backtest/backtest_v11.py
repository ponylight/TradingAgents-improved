"""Backtest V11 — FINAL Dual-Timeframe Strategy.

TWO INDEPENDENT SYSTEMS on separate capital ($10K total):

SYSTEM 1: LONG-TERM TREND (60% = $6,000) — WEEKLY CHARTS
  - Resample 4H data to weekly candles
  - MACD 13/34 double divergence on WEEKLY
  - Trend filter: long above weekly 50 SMA, short below
  - Bull filter: price > weekly 200 SMA = NO SHORTS
  - Leverage tiers: 5x / 10x / 25x / 50x based on conviction
  - MVRV proxy: price position in 365-day range (bottom 20% = undervalued)
  - Rolling positions (inverted pyramid): 150% → 200% → 250% adds
  - Trailing stop: 2x weekly ATR. Each add protected at breakeven.
  - Funding cost: 0.01% per 8h deducted from position value
  - NO fixed TP — ride until trailing stop hit

SYSTEM 2: SHORT-TERM TREND (40% = $4,000) — 4H CHARTS
  - V6 MACD strategy (13/34 double divergence on 4H)
  - Macro filter: price > weekly 200 SMA = LONG ONLY, below = SHORT ONLY
  - Leverage: 3x base, 10x with 123 Rule
  - Stop-loss: ATR based, max 3% price movement
  - Exit: half at 1:1 R/R, trail rest
  - Max 2 additions per trade

COMBINED: Track each system independently, sum for total equity.
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v11")

# ── Shared constants ─────────────────────────────────────────────────

TOTAL_CAPITAL = 10_000.0
LT_ALLOC = 0.60  # Long-term: 60%
ST_ALLOC = 0.40  # Short-term: 40%
COMMISSION_RATE = 0.001  # 0.1%
FUNDING_RATE_PER_8H = 0.0001  # 0.01% per 8h

# ── System 1 (Long-Term) constants ───────────────────────────────────

LT_SWING_LOOKBACK = 4  # weekly bars for swing detection
LT_MVRV_LOOKBACK = 52  # 52 weeks = ~1 year
LT_MVRV_BOTTOM_PCT = 0.20  # bottom 20% of range = undervalued
LT_TRAILING_ATR_MULT = 3.0  # wider stop for weekly positions
LT_ROLLING_SCALES = [1.5, 2.0, 2.5]  # inverted pyramid multipliers

# ── System 2 (Short-Term) constants ──────────────────────────────────

ST_SWING_LOOKBACK = 10  # 4H bars for swing detection
ST_BASE_LEVERAGE = 3
ST_MAX_LEVERAGE = 10
ST_MAX_ROLLING_ADDS = 2
ST_TRAILING_ATR_MULT = 2.5
ST_MAX_STOP_PCT = 0.03  # 3% max stop distance


# ══════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════

def detect_swings(highs: np.ndarray, lows: np.ndarray, lookback: int):
    """Detect swing highs and swing lows using bidirectional lookback."""
    swing_highs = []
    swing_lows = []
    for i in range(lookback, len(highs) - lookback):
        ws, we = i - lookback, i + lookback + 1
        if highs[i] >= np.max(highs[ws:we]):
            if not swing_highs or i - swing_highs[-1][0] >= lookback:
                swing_highs.append((i, highs[i]))
        if lows[i] <= np.min(lows[ws:we]):
            if not swing_lows or i - swing_lows[-1][0] >= lookback:
                swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def check_123_rule(signal_idx: int, side: str, closes: np.ndarray,
                   highs: np.ndarray, lows: np.ndarray, atr_vals: np.ndarray,
                   swing_highs: list, swing_lows: list,
                   lb: int = 10) -> bool:
    if side == "long":
        return _check_123_long(signal_idx, closes, highs, lows, atr_vals, swing_highs, swing_lows, lb)
    else:
        return _check_123_short(signal_idx, closes, highs, lows, atr_vals, swing_highs, swing_lows, lb)


def _check_123_long(idx, closes, highs, lows, atr_vals, swing_highs, swing_lows, lb):
    confirmed_lows = [(i, v) for i, v in swing_lows if i + lb <= idx]
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
    for j in range(x2 + 1, idx + 1):
        if closes[j] > tl(j):
            break_bar = j
            break
    if break_bar is None:
        return False

    retest_bar = None
    for j in range(break_bar + 1, idx + 1):
        tl_j = tl(j)
        a = atr_vals[j] if j < len(atr_vals) else np.nan
        if np.isnan(a):
            continue
        if abs(lows[j] - tl_j) <= a and closes[j] > tl_j:
            retest_bar = j
            break
    if retest_bar is None:
        return False

    confirmed_highs = [(i, v) for i, v in swing_highs if i + lb <= idx]
    if not confirmed_highs:
        return False
    _, pivot_val = confirmed_highs[-1]
    for j in range(retest_bar, idx + 1):
        if closes[j] > pivot_val:
            return True
    return False


def _check_123_short(idx, closes, highs, lows, atr_vals, swing_highs, swing_lows, lb):
    confirmed_highs = [(i, v) for i, v in swing_highs if i + lb <= idx]
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
    for j in range(x2 + 1, idx + 1):
        if closes[j] < tl(j):
            break_bar = j
            break
    if break_bar is None:
        return False

    retest_bar = None
    for j in range(break_bar + 1, idx + 1):
        tl_j = tl(j)
        a = atr_vals[j] if j < len(atr_vals) else np.nan
        if np.isnan(a):
            continue
        if abs(highs[j] - tl_j) <= a and closes[j] < tl_j:
            retest_bar = j
            break
    if retest_bar is None:
        return False

    confirmed_lows = [(i, v) for i, v in swing_lows if i + lb <= idx]
    if not confirmed_lows:
        return False
    _, pivot_val = confirmed_lows[-1]
    for j in range(retest_bar, idx + 1):
        if closes[j] < pivot_val:
            return True
    return False


def check_hidden_bottom_divergence(current_idx, hist, lows, lookback=100):
    """Check for hidden bullish divergence: price higher lows + MACD lower lows.
    This is a trend continuation signal in uptrends.
    Returns (strength, peak_diff) or None."""
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

    # Hidden bullish: price HIGHER lows + MACD LOWER lows
    divergence_count = 0
    for k in range(len(troughs) - 1):
        t1_idx, t1_val = troughs[k]
        t2_idx, t2_val = troughs[k + 1]
        if lows[t2_idx] > lows[t1_idx] and t2_val < t1_val:
            divergence_count += 1

    if divergence_count == 0:
        return None

    last_two = troughs[-2:]
    peak_diff = abs(abs(last_two[1][1]) - abs(last_two[0][1])) / abs(last_two[0][1])
    return (min(divergence_count, 3), peak_diff)


def check_hidden_top_divergence(current_idx, hist, highs, lookback=100):
    """Check for hidden bearish divergence: price lower highs + MACD higher highs.
    This is a trend continuation signal in downtrends.
    Returns (strength, peak_diff) or None."""
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

    # Hidden bearish: price LOWER highs + MACD HIGHER peaks
    divergence_count = 0
    for k in range(len(peaks) - 1):
        p1_idx, p1_val = peaks[k]
        p2_idx, p2_val = peaks[k + 1]
        if highs[p2_idx] < highs[p1_idx] and p2_val > p1_val:
            divergence_count += 1

    if divergence_count == 0:
        return None

    last_two = peaks[-2:]
    peak_diff = abs(last_two[0][1] - last_two[1][1]) / last_two[0][1]
    return (min(divergence_count, 3), peak_diff)


def check_bottom_divergence(current_idx, hist, lows, lookback=80):
    """Check for MACD histogram bottom divergence. Returns (strength, peak_diff) or None."""
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


def check_top_divergence(current_idx, hist, highs, lookback=80):
    """Check for MACD histogram top divergence. Returns (strength, peak_diff) or None."""
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


# ══════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════

def prepare_4h_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare 4H data with indicators for System 2."""
    df = compute_macd(df, fast=13, slow=34, signal=9)
    df["atr"] = compute_atr(df, period=14)
    df["ma30"] = compute_ma(df, period=30)
    df["sma50"] = compute_ma(df, period=50)
    df["sma200"] = compute_ma(df, period=200)
    return df


def resample_to_weekly(df_4h: pd.DataFrame) -> pd.DataFrame:
    """Resample 4H OHLCV data to weekly candles."""
    df = df_4h.set_index("timestamp")
    weekly = df.resample("W").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"]).reset_index()
    weekly.rename(columns={"timestamp": "timestamp"}, inplace=True)

    # Compute weekly indicators
    weekly = compute_macd(weekly, fast=13, slow=34, signal=9)
    weekly["atr"] = compute_atr(weekly, period=14)
    weekly["ma30"] = compute_ma(weekly, period=30)
    weekly["sma50"] = compute_ma(weekly, period=50)
    weekly["sma200"] = compute_ma(weekly, period=200)
    return weekly


def get_weekly_sma200_at_4h(df_4h: pd.DataFrame, weekly_df: pd.DataFrame) -> np.ndarray:
    """Map weekly SMA200 values back to 4H bars for the macro filter."""
    # For each 4H bar, find the most recent completed weekly bar's SMA200
    weekly_sma200 = np.full(len(df_4h), np.nan)
    w_idx = 0
    for i in range(len(df_4h)):
        ts = df_4h["timestamp"].iloc[i]
        while w_idx < len(weekly_df) - 1 and weekly_df["timestamp"].iloc[w_idx + 1] <= ts:
            w_idx += 1
        if w_idx < len(weekly_df) and weekly_df["timestamp"].iloc[w_idx] <= ts:
            weekly_sma200[i] = weekly_df["sma200"].iloc[w_idx]
    return weekly_sma200


# ══════════════════════════════════════════════════════════════════════
#  SYSTEM 1: LONG-TERM TREND (WEEKLY)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class LTTrade:
    side: str
    entry_price: float
    entry_idx: int  # weekly index
    entry_time: pd.Timestamp
    size_btc: float  # leveraged BTC size
    margin: float  # capital at risk
    leverage: float
    stop_loss: float  # trailing stop
    liquidation_price: float
    is_add: bool = False
    add_number: int = 0  # 0 = base, 1/2/3 = adds
    breakeven_stop: float = 0.0  # for adds: protected at entry
    exit_price: float = 0.0
    exit_idx: int = 0
    exit_time: pd.Timestamp = None
    pnl: float = 0.0
    funding_paid: float = 0.0
    closed: bool = False
    close_reason: str = ""
    highest_price: float = 0.0  # for trailing stop (long)
    lowest_price: float = 999999.0  # for trailing stop (short)


class LongTermSystem:
    """System 1: Weekly chart trend following with rolling positions."""

    def __init__(self, capital: float, weekly_df: pd.DataFrame):
        self.initial_capital = capital
        self.capital = capital
        self.weekly = weekly_df
        self.active_trades: list[LTTrade] = []
        self.closed_trades: list[LTTrade] = []
        self.equity_curve = []
        self.add_count = 0  # current rolling add count for active position

        # Pre-compute swings
        self.swing_highs, self.swing_lows = detect_swings(
            weekly_df["high"].values, weekly_df["low"].values,
            lookback=LT_SWING_LOOKBACK
        )

        self.stats = {
            "signals_generated": 0,
            "trades_at_5x": 0, "trades_at_10x": 0,
            "trades_at_25x": 0, "trades_at_50x": 0,
            "liquidations": 0, "liquidation_loss": 0.0,
            "total_funding_paid": 0.0,
            "rolling_adds": 0,
            "shorts_blocked_bull": 0,
        }

    def compute_mvrv_proxy(self, idx: int) -> float:
        """MVRV proxy: position in 365-day (52-week) price range.
        Returns 0.0 (at bottom) to 1.0 (at top). < 0.20 = undervalued."""
        lookback_start = max(0, idx - LT_MVRV_LOOKBACK)
        window = self.weekly["close"].values[lookback_start:idx + 1]
        if len(window) < 10:
            return 0.5
        low = np.min(window)
        high = np.max(window)
        if high == low:
            return 0.5
        return (self.weekly["close"].values[idx] - low) / (high - low)

    def determine_leverage(self, idx: int, side: str, div_strength: int) -> tuple:
        """Determine leverage tier for long-term trades.

        Returns (leverage, has_123, is_bottom_zone, mvrv_proxy).
        """
        closes = self.weekly["close"].values
        highs = self.weekly["high"].values
        lows = self.weekly["low"].values
        atr_vals = self.weekly["atr"].values

        has_123 = check_123_rule(idx, side, closes, highs, lows, atr_vals,
                                 self.swing_highs, self.swing_lows, lb=LT_SWING_LOOKBACK)
        mvrv = self.compute_mvrv_proxy(idx)
        is_bottom = mvrv < LT_MVRV_BOTTOM_PCT

        # Triple div + MVRV < 0.20 + 123 → 50x (or 25x without 123)
        if div_strength >= 3 and is_bottom and has_123:
            return 50, has_123, is_bottom, mvrv
        if div_strength >= 3 and is_bottom:
            return 25, has_123, is_bottom, mvrv

        # Double div + bottom zone → 25x with 123, 10x without
        if div_strength >= 2 and is_bottom and has_123:
            return 25, has_123, is_bottom, mvrv
        if div_strength >= 2 and is_bottom:
            return 10, has_123, is_bottom, mvrv

        # Normal setup: 5x base, 10x with 123
        if has_123:
            return 10, has_123, is_bottom, mvrv
        return 5, has_123, is_bottom, mvrv

    def generate_signals(self) -> list[dict]:
        """Generate MACD divergence signals on weekly data.

        Note: With ~209 weekly candles, SMA200 is only valid for the last ~9 bars.
        We start trading once SMA50 is available (~week 50). The SMA200 bull filter
        is only enforced when data is available — if NaN, we allow both directions.
        """
        signals = []
        hist = self.weekly["macd_hist"].values
        closes = self.weekly["close"].values
        lows = self.weekly["low"].values
        highs = self.weekly["high"].values
        atr = self.weekly["atr"].values
        sma50 = self.weekly["sma50"].values
        sma200 = self.weekly["sma200"].values

        # Start once SMA50 is available (don't wait for SMA200 — too few weekly bars)
        start_idx = 52  # ~1 year of weekly data
        for i in range(len(self.weekly)):
            if not np.isnan(sma50[i]):
                start_idx = max(start_idx, i)
                break

        min_cooldown = 3  # 3 weekly bars between signals
        last_signal_idx = -min_cooldown

        for i in range(start_idx, len(self.weekly)):
            if i - last_signal_idx < min_cooldown:
                continue
            if np.isnan(sma50[i]) or np.isnan(atr[i]):
                continue

            # Bull filter: if SMA200 available and price above it, no shorts
            bull = closes[i] > sma200[i] if not np.isnan(sma200[i]) else False

            # ── LONG SIGNAL (above weekly 50 SMA) ──
            if hist[i] < 0 and i >= 3:
                if closes[i] > sma50[i]:
                    # Key K-line: histogram turning up from trough
                    is_key_kline = (
                        hist[i] > hist[i - 1]
                        and hist[i - 1] < 0
                        and hist[i - 1] < hist[i - 2]
                    )
                    min_hist_size = closes[i] * 0.0005
                    if is_key_kline and abs(hist[i - 1]) > min_hist_size:
                        # Check classical divergence first (strongest signal)
                        div = check_bottom_divergence(i, hist, lows, lookback=80)
                        # Also check hidden bullish divergence (trend continuation)
                        # Hidden: price higher lows + MACD lower lows = uptrend continues
                        hdiv = check_hidden_bottom_divergence(i, hist, lows, lookback=100)

                        strength = 0
                        peak_diff = 0
                        if div is not None:
                            strength = div[0]
                            peak_diff = div[1]
                        elif hdiv is not None:
                            strength = hdiv[0]
                            peak_diff = hdiv[1]

                        if strength >= 1 and peak_diff > 0.10:
                            sl = lows[i] - atr[i] * LT_TRAILING_ATR_MULT
                            signals.append({
                                "idx": i, "side": "long",
                                "entry_price": closes[i], "stop_loss": sl,
                                "strength": strength,
                                "time": self.weekly["timestamp"].iloc[i],
                            })
                            last_signal_idx = i
                            self.stats["signals_generated"] += 1

            # ── SHORT SIGNAL (below weekly 50 SMA, not in bull) ──
            if hist[i] > 0 and i >= 3:
                if bull:
                    if closes[i] < sma50[i]:
                        is_key_kline = (
                            hist[i] < hist[i - 1] and hist[i - 1] > 0
                            and hist[i - 1] > hist[i - 2]
                        )
                        if is_key_kline:
                            div = check_top_divergence(i, hist, highs, lookback=80)
                            hdiv = check_hidden_top_divergence(i, hist, highs, lookback=100)
                            if (div and div[0] >= 1) or (hdiv and hdiv[0] >= 1):
                                self.stats["shorts_blocked_bull"] += 1
                    continue

                if closes[i] < sma50[i]:
                    is_key_kline = (
                        hist[i] < hist[i - 1] and hist[i - 1] > 0
                        and hist[i - 1] > hist[i - 2]
                    )
                    min_hist_size = closes[i] * 0.0005
                    if is_key_kline and abs(hist[i - 1]) > min_hist_size:
                        div = check_top_divergence(i, hist, highs, lookback=80)
                        hdiv = check_hidden_top_divergence(i, hist, highs, lookback=100)

                        strength = 0
                        peak_diff = 0
                        if div is not None:
                            strength = div[0]
                            peak_diff = div[1]
                        elif hdiv is not None:
                            strength = hdiv[0]
                            peak_diff = hdiv[1]

                        if strength >= 1 and peak_diff > 0.10:
                            sl = highs[i] + atr[i] * LT_TRAILING_ATR_MULT
                            signals.append({
                                "idx": i, "side": "short",
                                "entry_price": closes[i], "stop_loss": sl,
                                "strength": strength,
                                "time": self.weekly["timestamp"].iloc[i],
                            })
                            last_signal_idx = i
                            self.stats["signals_generated"] += 1

        return signals

    def run(self) -> dict:
        """Run the long-term system on weekly data."""
        print(f"\n[System 1: Long-Term Weekly] Running on {len(self.weekly)} weekly candles...")
        print(f"  Capital: ${self.initial_capital:,.0f}")

        signals = self.generate_signals()
        signal_map = {s["idx"]: s for s in signals}
        long_sigs = sum(1 for s in signals if s["side"] == "long")
        short_sigs = sum(1 for s in signals if s["side"] == "short")
        print(f"  Generated {len(signals)} signals ({long_sigs} long, {short_sigs} short)")

        closes = self.weekly["close"].values
        highs = self.weekly["high"].values
        lows = self.weekly["low"].values
        atr = self.weekly["atr"].values

        for i in range(len(self.weekly)):
            price = closes[i]
            high = highs[i]
            low = lows[i]
            ts = self.weekly["timestamp"].iloc[i]
            w_atr = atr[i] if not np.isnan(atr[i]) else 0

            # Deduct funding costs for active positions
            # Weekly bar = 7 days = 21 funding periods (every 8h)
            self._deduct_funding(price, periods=21)

            # Manage existing positions
            self._manage_positions(i, high, low, price, ts, w_atr)

            # Check for rolling add opportunities
            if self.active_trades:
                self._check_rolling_add(i, price, ts, w_atr)

            # New entries
            if i in signal_map:
                sig = signal_map[i]
                self._execute_entry(sig, i, ts)

            # Record equity
            unrealized = self._calc_unrealized(price)
            self.equity_curve.append({
                "timestamp": ts,
                "equity": self.capital + unrealized,
                "capital": self.capital,
            })

        # Close remaining positions at end
        last_price = closes[-1]
        last_ts = self.weekly["timestamp"].iloc[-1]
        for trade in list(self.active_trades):
            self._close_trade(trade, last_price, len(self.weekly) - 1, last_ts, "end_of_backtest")

        return self._compute_metrics()

    def _execute_entry(self, sig: dict, idx: int, ts: pd.Timestamp):
        # Don't open if already have a position in same direction
        same_dir = [t for t in self.active_trades if t.side == sig["side"] and not t.is_add]
        if same_dir:
            return

        leverage, has_123, is_bottom, mvrv = self.determine_leverage(
            idx, sig["side"], sig["strength"]
        )

        # Position sizing: conservative for weekly leveraged trades
        risk_pct = 0.04  # risk 4% of LT capital per trade
        risk_amount = self.capital * risk_pct
        risk = abs(sig["entry_price"] - sig["stop_loss"])
        if risk <= 0:
            return

        base_size_btc = risk_amount / risk
        max_base = (self.capital * 0.30) / sig["entry_price"]  # max 30% of capital as margin
        base_size_btc = min(base_size_btc, max_base)

        if base_size_btc * sig["entry_price"] < 10:
            return

        leveraged_size = base_size_btc * leverage
        margin = base_size_btc * sig["entry_price"]
        notional = leveraged_size * sig["entry_price"]

        # Commission
        commission = notional * COMMISSION_RATE
        self.capital -= commission

        # Liquidation price
        if sig["side"] == "long":
            liq_price = sig["entry_price"] * (1 - 1 / leverage)
            # Ensure stop is always closer than liquidation (with 10% buffer)
            max_stop = sig["entry_price"] - (sig["entry_price"] - liq_price) * 0.90
            sig["stop_loss"] = max(sig["stop_loss"], max_stop)
        else:
            liq_price = sig["entry_price"] * (1 + 1 / leverage)
            max_stop = sig["entry_price"] + (liq_price - sig["entry_price"]) * 0.90
            sig["stop_loss"] = min(sig["stop_loss"], max_stop)

        # Track leverage tier
        if leverage >= 50:
            self.stats["trades_at_50x"] += 1
        elif leverage >= 25:
            self.stats["trades_at_25x"] += 1
        elif leverage >= 10:
            self.stats["trades_at_10x"] += 1
        else:
            self.stats["trades_at_5x"] += 1

        trade = LTTrade(
            side=sig["side"],
            entry_price=sig["entry_price"],
            entry_idx=idx,
            entry_time=ts,
            size_btc=leveraged_size,
            margin=margin,
            leverage=leverage,
            stop_loss=sig["stop_loss"],
            liquidation_price=liq_price,
            highest_price=sig["entry_price"],
            lowest_price=sig["entry_price"],
        )
        self.active_trades.append(trade)
        self.add_count = 0

    def _check_rolling_add(self, idx: int, price: float, ts: pd.Timestamp, w_atr: float):
        """Check for rolling position add opportunities."""
        if self.add_count >= 3:
            return

        base_trades = [t for t in self.active_trades if not t.is_add and not t.closed]
        if not base_trades:
            return
        base = base_trades[0]

        # Only add when profitable (use floating profits only)
        if base.side == "long":
            unrealized = (price - base.entry_price) * base.size_btc
        else:
            unrealized = (base.entry_price - price) * base.size_btc
        if unrealized <= 0:
            return

        # Check MVRV proxy or near-bottom condition
        mvrv = self.compute_mvrv_proxy(idx)
        near_bottom = mvrv < 0.30

        ma30 = self.weekly["ma30"].values[idx] if not np.isnan(self.weekly["ma30"].values[idx]) else None
        sma50 = self.weekly["sma50"].values[idx]

        if base.side == "long":
            # Pullback to MA30 or Fib 0.5 of the move
            if ma30 is None:
                return
            fib_50 = base.entry_price + (base.highest_price - base.entry_price) * 0.5
            pullback_to_ma30 = abs(price - ma30) / price < 0.01 and price > ma30
            pullback_to_fib = price <= fib_50 and price > base.entry_price

            if not (pullback_to_ma30 or pullback_to_fib or near_bottom):
                return
        else:
            if ma30 is None:
                return
            fib_50 = base.entry_price - (base.entry_price - base.lowest_price) * 0.5
            pullback_to_ma30 = abs(price - ma30) / price < 0.01 and price < ma30
            pullback_to_fib = price >= fib_50 and price < base.entry_price

            if not (pullback_to_ma30 or pullback_to_fib or near_bottom):
                return

        # Size: inverted pyramid using floating profits only
        scale = LT_ROLLING_SCALES[self.add_count]
        base_margin_equiv = unrealized * 0.5  # use half of floating profits
        if base_margin_equiv < 50:
            return

        # Determine leverage for the add
        leverage, _, _, _ = self.determine_leverage(idx, base.side, 2)
        add_size_btc = (base_margin_equiv * leverage) / price
        margin = base_margin_equiv
        notional = add_size_btc * price

        # Commission
        commission = notional * COMMISSION_RATE
        self.capital -= commission

        if base.side == "long":
            sl = price - w_atr * LT_TRAILING_ATR_MULT if w_atr > 0 else price * 0.95
            liq_price = price * (1 - 1 / leverage)
        else:
            sl = price + w_atr * LT_TRAILING_ATR_MULT if w_atr > 0 else price * 1.05
            liq_price = price * (1 + 1 / leverage)

        add_trade = LTTrade(
            side=base.side,
            entry_price=price,
            entry_idx=idx,
            entry_time=ts,
            size_btc=add_size_btc,
            margin=margin,
            leverage=leverage,
            stop_loss=sl,
            liquidation_price=liq_price,
            is_add=True,
            add_number=self.add_count + 1,
            breakeven_stop=price,  # protected at entry
            highest_price=price,
            lowest_price=price,
        )
        self.active_trades.append(add_trade)
        self.add_count += 1
        self.stats["rolling_adds"] += 1

    def _deduct_funding(self, price: float, periods: int = 21):
        """Deduct perpetual funding cost from capital."""
        for trade in self.active_trades:
            if trade.closed:
                continue
            notional = trade.size_btc * price
            funding_cost = notional * FUNDING_RATE_PER_8H * periods
            self.capital -= funding_cost
            trade.funding_paid += funding_cost
            self.stats["total_funding_paid"] += funding_cost

    def _manage_positions(self, idx: int, high: float, low: float, price: float,
                          ts: pd.Timestamp, w_atr: float):
        for trade in list(self.active_trades):
            if trade.closed:
                continue

            if trade.side == "long":
                trade.highest_price = max(trade.highest_price, high)

                # Update trailing stop first (3x weekly ATR from highest)
                if w_atr > 0:
                    new_trail = trade.highest_price - w_atr * LT_TRAILING_ATR_MULT
                    trade.stop_loss = max(trade.stop_loss, new_trail)

                # Check stops BEFORE liquidation (stop is always closer)
                # Breakeven stop for adds
                if trade.is_add and low <= trade.breakeven_stop:
                    self._close_trade(trade, trade.breakeven_stop, idx, ts, "breakeven_stop")
                    continue

                # Trailing stop
                if low <= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, idx, ts, "trailing_stop")
                    continue

                # Liquidation only if stop somehow didn't catch it (gap through)
                if low <= trade.liquidation_price:
                    self.stats["liquidations"] += 1
                    self.stats["liquidation_loss"] += trade.margin
                    self._close_trade(trade, trade.liquidation_price, idx, ts, "liquidation")
                    continue

            else:  # short
                trade.lowest_price = min(trade.lowest_price, low)

                if w_atr > 0:
                    new_trail = trade.lowest_price + w_atr * LT_TRAILING_ATR_MULT
                    trade.stop_loss = min(trade.stop_loss, new_trail)

                if trade.is_add and high >= trade.breakeven_stop:
                    self._close_trade(trade, trade.breakeven_stop, idx, ts, "breakeven_stop")
                    continue

                if high >= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, idx, ts, "trailing_stop")
                    continue

                if high >= trade.liquidation_price:
                    self.stats["liquidations"] += 1
                    self.stats["liquidation_loss"] += trade.margin
                    self._close_trade(trade, trade.liquidation_price, idx, ts, "liquidation")
                    continue

    def _close_trade(self, trade: LTTrade, exit_price: float, idx: int,
                     ts: pd.Timestamp, reason: str):
        if trade.side == "long":
            pnl = (exit_price - trade.entry_price) * trade.size_btc
        else:
            pnl = (trade.entry_price - exit_price) * trade.size_btc

        # Exit commission
        commission = trade.size_btc * exit_price * COMMISSION_RATE
        pnl -= commission

        # Cap loss at margin (can't lose more than margin in leveraged trade)
        if reason == "liquidation":
            pnl = -trade.margin

        trade.exit_price = exit_price
        trade.exit_idx = idx
        trade.exit_time = ts
        trade.pnl = pnl
        trade.closed = True
        trade.close_reason = reason
        self.capital += pnl

        if trade in self.active_trades:
            self.active_trades.remove(trade)
        self.closed_trades.append(trade)

        # If base trade closed, close all adds too
        if not trade.is_add:
            for add_trade in list(self.active_trades):
                if add_trade.is_add and add_trade.side == trade.side and not add_trade.closed:
                    self._close_trade(add_trade, exit_price, idx, ts, "base_closed")
            self.add_count = 0

    def _calc_unrealized(self, price: float) -> float:
        total = 0.0
        for trade in self.active_trades:
            if trade.closed:
                continue
            if trade.side == "long":
                total += (price - trade.entry_price) * trade.size_btc
            else:
                total += (trade.entry_price - price) * trade.size_btc
        return total

    def _compute_metrics(self) -> dict:
        if not self.closed_trades:
            return {"system": "long_term", "error": "No trades"}

        eq = pd.DataFrame(self.equity_curve)
        equity = eq["equity"].values
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100

        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak * 100
        max_dd = np.max(dd)

        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(52) if len(returns) > 1 and np.std(returns) > 0 else 0

        pnls = [t.pnl for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

        durations = []
        for t in self.closed_trades:
            if t.entry_time and t.exit_time:
                durations.append((t.exit_time - t.entry_time).total_seconds() / 86400)

        base_trades = [t for t in self.closed_trades if not t.is_add]
        add_trades = [t for t in self.closed_trades if t.is_add]

        return {
            "system": "long_term",
            "total_return_pct": round(total_return, 2),
            "final_equity": round(equity[-1], 2),
            "initial_capital": self.initial_capital,
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_trades": len(self.closed_trades),
            "base_trades": len(base_trades),
            "rolling_adds": len(add_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "largest_win": round(max(pnls), 2) if pnls else 0,
            "largest_loss": round(min(pnls), 2) if pnls else 0,
            "avg_duration_days": round(np.mean(durations), 1) if durations else 0,
            "total_funding_paid": round(self.stats["total_funding_paid"], 2),
            "trades_at_5x": self.stats["trades_at_5x"],
            "trades_at_10x": self.stats["trades_at_10x"],
            "trades_at_25x": self.stats["trades_at_25x"],
            "trades_at_50x": self.stats["trades_at_50x"],
            "liquidations": self.stats["liquidations"],
            "liquidation_loss": round(self.stats["liquidation_loss"], 2),
            "shorts_blocked_bull": self.stats["shorts_blocked_bull"],
        }


# ══════════════════════════════════════════════════════════════════════
#  SYSTEM 2: SHORT-TERM TREND (4H) — Based on V6
# ══════════════════════════════════════════════════════════════════════

@dataclass
class STTrade:
    side: str
    entry_price: float
    entry_idx: int
    entry_time: pd.Timestamp
    size_btc: float
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


class ShortTermSystem:
    """System 2: 4H chart V6 strategy with weekly macro filter."""

    def __init__(self, capital: float, df_4h: pd.DataFrame, weekly_sma200: np.ndarray):
        self.initial_capital = capital
        self.capital = capital
        self.df = df_4h
        self.weekly_sma200 = weekly_sma200  # mapped to 4H bars
        self.active_trades: list[STTrade] = []
        self.closed_trades: list[STTrade] = []
        self.equity_curve = []
        self.rolling_add_counts: dict[int, int] = {}

        self.swing_highs, self.swing_lows = detect_swings(
            df_4h["high"].values, df_4h["low"].values, lookback=ST_SWING_LOOKBACK
        )

        self.stats = {
            "signals_generated": 0,
            "trades_at_3x": 0, "trades_at_10x": 0,
            "liquidations": 0, "liquidation_loss": 0.0,
            "longs_blocked_bear": 0, "shorts_blocked_bull": 0,
        }

    def _macro_direction(self, idx: int) -> str:
        """Determine macro direction from weekly 200 SMA.

        Returns 'bull', 'bear', or 'neutral' (when SMA200 unavailable).
        """
        w_sma = self.weekly_sma200[idx]
        if np.isnan(w_sma):
            return "neutral"  # allow both directions when data insufficient
        if self.df["close"].values[idx] > w_sma:
            return "bull"
        return "bear"

    def compute_signals(self) -> list[dict]:
        """V6 signal generation with weekly macro direction filter."""
        signals = []
        hist = self.df["macd_hist"].values
        closes = self.df["close"].values
        lows = self.df["low"].values
        highs = self.df["high"].values
        atr = self.df["atr"].values
        sma50 = self.df["sma50"].values

        start_idx = 205
        min_cooldown = 30
        last_signal_idx = -min_cooldown

        for i in range(start_idx, len(self.df)):
            if i - last_signal_idx < min_cooldown:
                continue
            if np.isnan(sma50[i]):
                continue

            macro = self._macro_direction(i)

            # ── LONG SIGNAL ──
            if hist[i] < 0 and i >= 4:
                # CRITICAL: bear market = NO LONGS (neutral allows both)
                if macro == "bear":
                    pass  # skip longs in bear
                elif closes[i] <= sma50[i]:
                    pass  # below trend filter
                else:
                    is_key_kline = (
                        hist[i] > hist[i - 1]
                        and hist[i - 1] < 0
                        and hist[i - 1] < hist[i - 2]
                        and hist[i - 2] < hist[i - 3]
                    )
                    min_hist_size = closes[i] * 0.001
                    if is_key_kline and abs(hist[i - 1]) > min_hist_size:
                        div = check_bottom_divergence(i, hist, lows, lookback=120)
                        if div is not None:
                            strength, peak_diff = div
                            if strength >= 2 and peak_diff > 0.30:
                                sl = lows[i] - atr[i] * 1.5 if not np.isnan(atr[i]) else lows[i] * 0.97
                                # Enforce 3% max stop distance
                                max_sl = closes[i] * (1 - ST_MAX_STOP_PCT)
                                sl = max(sl, max_sl)
                                signals.append({
                                    "idx": i, "side": "long",
                                    "entry_price": closes[i], "stop_loss": sl,
                                    "strength": strength,
                                })
                                last_signal_idx = i
                                self.stats["signals_generated"] += 1

            # ── SHORT SIGNAL ──
            if hist[i] > 0 and i >= 4:
                # CRITICAL: bull market = NO SHORTS (neutral allows both)
                if macro == "bull":
                    # Count blocked shorts
                    if closes[i] < sma50[i]:
                        is_key_kline = (
                            hist[i] < hist[i - 1] and hist[i - 1] > 0
                            and hist[i - 1] > hist[i - 2] and hist[i - 2] > hist[i - 3]
                        )
                        if is_key_kline:
                            div = check_top_divergence(i, hist, highs, lookback=120)
                            if div and div[0] >= 2:
                                self.stats["shorts_blocked_bull"] += 1
                    continue

                if closes[i] >= sma50[i]:
                    pass  # above trend filter for short
                else:
                    is_key_kline = (
                        hist[i] < hist[i - 1] and hist[i - 1] > 0
                        and hist[i - 1] > hist[i - 2] and hist[i - 2] > hist[i - 3]
                    )
                    min_hist_size = closes[i] * 0.001
                    if is_key_kline and abs(hist[i - 1]) > min_hist_size:
                        div = check_top_divergence(i, hist, highs, lookback=120)
                        if div is not None:
                            strength, peak_diff = div
                            if strength >= 2 and peak_diff > 0.30:
                                sl = highs[i] + atr[i] * 1.5 if not np.isnan(atr[i]) else highs[i] * 1.03
                                max_sl = closes[i] * (1 + ST_MAX_STOP_PCT)
                                sl = min(sl, max_sl)
                                signals.append({
                                    "idx": i, "side": "short",
                                    "entry_price": closes[i], "stop_loss": sl,
                                    "strength": strength,
                                })
                                last_signal_idx = i
                                self.stats["signals_generated"] += 1

        return signals

    def run(self) -> dict:
        """Run the short-term system on 4H data."""
        print(f"\n[System 2: Short-Term 4H] Running on {len(self.df)} candles...")
        print(f"  Capital: ${self.initial_capital:,.0f}")
        print(f"  Macro filter: weekly 200 SMA (LONG ONLY in bull, SHORT ONLY in bear)")

        signals = self.compute_signals()
        signal_map = {s["idx"]: s for s in signals}
        long_sigs = sum(1 for s in signals if s["side"] == "long")
        short_sigs = sum(1 for s in signals if s["side"] == "short")
        print(f"  Generated {len(signals)} signals ({long_sigs} long, {short_sigs} short)")

        closes = self.df["close"].values
        highs = self.df["high"].values
        lows = self.df["low"].values
        atr_arr = self.df["atr"].values

        for i in range(len(self.df)):
            price = closes[i]
            high = highs[i]
            low = lows[i]
            ts = self.df["timestamp"].iloc[i]

            self._manage_positions(i, high, low, price, ts)

            # Rolling adds (max 2)
            if self.active_trades:
                self._check_rolling_add(i, price, ts, closes, highs, lows, atr_arr)

            if i in signal_map:
                sig = signal_map[i]
                self._execute_entry(sig, ts, closes, highs, lows, atr_arr)

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

    def _execute_entry(self, sig: dict, ts: pd.Timestamp,
                       closes: np.ndarray, highs: np.ndarray,
                       lows: np.ndarray, atr_vals: np.ndarray):
        same_dir = [t for t in self.active_trades if t.side == sig["side"] and not t.is_rolling]
        if same_dir:
            return

        risk = abs(sig["entry_price"] - sig["stop_loss"])
        if risk <= 0:
            return

        # Position sizing
        risk_amount = self.capital * 0.02
        base_size_btc = risk_amount / risk
        max_base = (self.capital * 0.40) / sig["entry_price"]
        base_size_btc = min(base_size_btc, max_base)

        if base_size_btc * sig["entry_price"] < 10:
            return

        # Leverage: 3x base, 10x with 123 Rule
        has_123 = check_123_rule(sig["idx"], sig["side"], closes, highs, lows, atr_vals,
                                 self.swing_highs, self.swing_lows, lb=ST_SWING_LOOKBACK)
        leverage = ST_MAX_LEVERAGE if has_123 else ST_BASE_LEVERAGE

        leveraged_size = base_size_btc * leverage
        margin = base_size_btc * sig["entry_price"]
        notional = leveraged_size * sig["entry_price"]

        commission = notional * COMMISSION_RATE
        self.capital -= commission

        if sig["side"] == "long":
            liq_price = sig["entry_price"] * (1 - 1 / leverage)
            tp1 = sig["entry_price"] + risk * 1.0  # 1:1 R/R
        else:
            liq_price = sig["entry_price"] * (1 + 1 / leverage)
            tp1 = sig["entry_price"] - risk * 1.0

        if leverage == ST_MAX_LEVERAGE:
            self.stats["trades_at_10x"] += 1
        else:
            self.stats["trades_at_3x"] += 1

        trade = STTrade(
            side=sig["side"],
            entry_price=sig["entry_price"],
            entry_idx=sig["idx"],
            entry_time=ts,
            size_btc=leveraged_size,
            stop_loss=sig["stop_loss"],
            take_profit_1=tp1,
            initial_risk=risk,
            leverage=leverage,
            margin=margin,
            liquidation_price=liq_price,
            original_size=leveraged_size,
            trailing_stop=sig["stop_loss"],
        )
        self.active_trades.append(trade)

    def _check_rolling_add(self, idx: int, price: float, ts: pd.Timestamp,
                           closes: np.ndarray, highs: np.ndarray,
                           lows: np.ndarray, atr_vals: np.ndarray):
        """Rolling add check — max 2 additions per trade."""
        base_trades = [t for t in self.active_trades if not t.is_rolling and not t.closed]
        if not base_trades:
            return
        base = base_trades[0]
        base_key = base.entry_idx
        add_count = self.rolling_add_counts.get(base_key, 0)
        if add_count >= ST_MAX_ROLLING_ADDS:
            return

        ma30 = self.df["ma30"].values[idx]
        atr = atr_vals[idx]
        sma50 = self.df["sma50"].values[idx]
        if np.isnan(ma30) or np.isnan(atr) or np.isnan(sma50):
            return

        # Only with floating profits
        if base.side == "long":
            unrealized_pct = (price - base.entry_price) / base.entry_price
        else:
            unrealized_pct = (base.entry_price - price) / base.entry_price
        if unrealized_pct < 0.03:
            return

        # Near-bottom check
        lookback_start = max(0, idx - 2190)
        window = closes[lookback_start:idx + 1]
        low_365 = np.min(window)
        near_bottom = price <= low_365 * 1.30
        if not near_bottom:
            return

        macro = self._macro_direction(idx)

        if base.side == "long":
            if macro == "bear":
                return
            if price <= sma50:
                return
            near_ma30 = abs(price - ma30) / price < 0.005 and price > ma30
            recent_high = self.df["high"].values[max(0, idx - 20):idx].max()
            breakout = price > recent_high
            avg_vol = self.df["volume"].values[max(0, idx - 20):idx].mean()
            vol_confirm = self.df["volume"].values[idx] > avg_vol * 1.5
            if not ((near_ma30) or (breakout and vol_confirm)):
                return
        else:
            if macro == "bull":
                return
            if price >= sma50:
                return
            near_ma30 = abs(price - ma30) / price < 0.005 and price < ma30
            recent_low = self.df["low"].values[max(0, idx - 20):idx].min()
            breakdown = price < recent_low
            avg_vol = self.df["volume"].values[max(0, idx - 20):idx].mean()
            vol_confirm = self.df["volume"].values[idx] > avg_vol * 1.5
            if not ((near_ma30) or (breakdown and vol_confirm)):
                return

        # Size decays from base (each smaller)
        decay = 0.6 ** (add_count + 1)
        base_unlev = base.original_size / base.leverage
        add_base = base_unlev * decay
        if add_base * price > self.capital * 0.2:
            return

        risk = atr * 1.5
        max_risk = price * ST_MAX_STOP_PCT
        risk = min(risk, max_risk)
        if risk <= 0:
            return

        has_123 = check_123_rule(idx, base.side, closes, highs, lows, atr_vals,
                                 self.swing_highs, self.swing_lows, lb=ST_SWING_LOOKBACK)
        leverage = ST_MAX_LEVERAGE if has_123 else ST_BASE_LEVERAGE

        leveraged_size = add_base * leverage
        margin = add_base * price
        notional = leveraged_size * price
        commission = notional * COMMISSION_RATE
        self.capital -= commission

        if base.side == "long":
            sl = price - risk
            liq_price = price * (1 - 1 / leverage)
            tp1 = price + risk
        else:
            sl = price + risk
            liq_price = price * (1 + 1 / leverage)
            tp1 = price - risk

        trade = STTrade(
            side=base.side,
            entry_price=price,
            entry_idx=idx,
            entry_time=ts,
            size_btc=leveraged_size,
            stop_loss=sl,
            take_profit_1=tp1,
            initial_risk=risk,
            leverage=leverage,
            margin=margin,
            liquidation_price=liq_price,
            original_size=leveraged_size,
            is_rolling=True,
            trailing_stop=sl,
        )
        self.active_trades.append(trade)
        self.rolling_add_counts[base_key] = add_count + 1

    def _manage_positions(self, idx: int, high: float, low: float, price: float, ts: pd.Timestamp):
        for trade in list(self.active_trades):
            if trade.closed:
                continue

            if trade.side == "long":
                if low <= trade.liquidation_price:
                    self.stats["liquidations"] += 1
                    self.stats["liquidation_loss"] += trade.margin
                    self._close_trade(trade, trade.liquidation_price, idx, ts, "liquidation")
                    continue

                if low <= trade.trailing_stop:
                    self._close_trade(trade, trade.trailing_stop, idx, ts, "stop_loss")
                    continue

                if not trade.half_closed and high >= trade.take_profit_1:
                    half = trade.size_btc / 2
                    pnl = (trade.take_profit_1 - trade.entry_price) * half
                    comm = half * trade.take_profit_1 * COMMISSION_RATE
                    tp1_pnl = pnl - comm
                    self.capital += tp1_pnl
                    trade.pnl_tp1 = tp1_pnl
                    trade.size_btc -= half
                    trade.half_closed = True
                    trade.trailing_stop = max(trade.trailing_stop,
                                              trade.entry_price + trade.initial_risk * 0.1)

                atr = self.df["atr"].values[idx]
                if not np.isnan(atr) and trade.half_closed:
                    new_trail = price - atr * ST_TRAILING_ATR_MULT
                    trade.trailing_stop = max(trade.trailing_stop, new_trail)

            else:  # short
                if high >= trade.liquidation_price:
                    self.stats["liquidations"] += 1
                    self.stats["liquidation_loss"] += trade.margin
                    self._close_trade(trade, trade.liquidation_price, idx, ts, "liquidation")
                    continue

                if high >= trade.trailing_stop:
                    self._close_trade(trade, trade.trailing_stop, idx, ts, "stop_loss")
                    continue

                if not trade.half_closed and low <= trade.take_profit_1:
                    half = trade.size_btc / 2
                    pnl = (trade.entry_price - trade.take_profit_1) * half
                    comm = half * trade.take_profit_1 * COMMISSION_RATE
                    tp1_pnl = pnl - comm
                    self.capital += tp1_pnl
                    trade.pnl_tp1 = tp1_pnl
                    trade.size_btc -= half
                    trade.half_closed = True
                    trade.trailing_stop = min(trade.trailing_stop,
                                              trade.entry_price - trade.initial_risk * 0.1)

                atr = self.df["atr"].values[idx]
                if not np.isnan(atr) and trade.half_closed:
                    new_trail = price + atr * ST_TRAILING_ATR_MULT
                    trade.trailing_stop = min(trade.trailing_stop, new_trail)

    def _close_trade(self, trade: STTrade, exit_price: float, idx: int,
                     ts: pd.Timestamp, reason: str):
        if trade.side == "long":
            pnl = (exit_price - trade.entry_price) * trade.size_btc
        else:
            pnl = (trade.entry_price - exit_price) * trade.size_btc

        commission = trade.size_btc * exit_price * COMMISSION_RATE
        pnl -= commission

        if reason == "liquidation":
            pnl = -trade.margin

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

    def _calc_unrealized(self, price: float) -> float:
        total = 0.0
        for trade in self.active_trades:
            if trade.closed:
                continue
            if trade.side == "long":
                total += (price - trade.entry_price) * trade.size_btc
            else:
                total += (trade.entry_price - price) * trade.size_btc
        return total

    def _compute_metrics(self) -> dict:
        if not self.closed_trades:
            return {"system": "short_term", "error": "No trades"}

        eq = pd.DataFrame(self.equity_curve)
        equity = eq["equity"].values
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100

        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak * 100
        max_dd = np.max(dd)

        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(2190) if len(returns) > 1 and np.std(returns) > 0 else 0

        pnls = [t.pnl for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

        durations = []
        for t in self.closed_trades:
            if t.entry_time and t.exit_time:
                durations.append((t.exit_time - t.entry_time).total_seconds() / 3600)

        primary = [t for t in self.closed_trades if not t.is_rolling]
        rolling = [t for t in self.closed_trades if t.is_rolling]
        longs = [t for t in self.closed_trades if t.side == "long"]
        shorts = [t for t in self.closed_trades if t.side == "short"]

        return {
            "system": "short_term",
            "total_return_pct": round(total_return, 2),
            "final_equity": round(equity[-1], 2),
            "initial_capital": self.initial_capital,
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_trades": len(self.closed_trades),
            "primary_trades": len(primary),
            "rolling_adds": len(rolling),
            "long_trades": len(longs),
            "short_trades": len(shorts),
            "long_pnl": round(sum(t.pnl for t in longs), 2),
            "short_pnl": round(sum(t.pnl for t in shorts), 2),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "largest_win": round(max(pnls), 2) if pnls else 0,
            "largest_loss": round(min(pnls), 2) if pnls else 0,
            "avg_duration_hours": round(np.mean(durations), 1) if durations else 0,
            "avg_duration_days": round(np.mean(durations) / 24, 1) if durations else 0,
            "trades_at_3x": self.stats["trades_at_3x"],
            "trades_at_10x": self.stats["trades_at_10x"],
            "liquidations": self.stats["liquidations"],
            "liquidation_loss": round(self.stats["liquidation_loss"], 2),
            "shorts_blocked_bull": self.stats["shorts_blocked_bull"],
        }


# ══════════════════════════════════════════════════════════════════════
#  COMBINED REPORTING
# ══════════════════════════════════════════════════════════════════════

def build_combined_equity(lt_equity: list[dict], st_equity: list[dict],
                          lt_capital: float, st_capital: float) -> pd.DataFrame:
    """Build combined equity curve from both systems on shared timestamps."""
    # System 1 is weekly, System 2 is 4H — use 4H timestamps as base
    st_df = pd.DataFrame(st_equity)
    lt_df = pd.DataFrame(lt_equity)

    # For each 4H timestamp, get LT equity from most recent weekly
    combined = st_df[["timestamp"]].copy()
    combined["st_equity"] = st_df["equity"].values

    lt_eq = np.full(len(combined), lt_capital)
    lt_idx = 0
    for i in range(len(combined)):
        ts = combined["timestamp"].iloc[i]
        while lt_idx < len(lt_df) - 1 and lt_df["timestamp"].iloc[lt_idx + 1] <= ts:
            lt_idx += 1
        if lt_idx < len(lt_df) and lt_df["timestamp"].iloc[lt_idx] <= ts:
            lt_eq[i] = lt_df["equity"].iloc[lt_idx]

    combined["lt_equity"] = lt_eq
    combined["total_equity"] = combined["lt_equity"] + combined["st_equity"]
    return combined


def generate_report(lt_metrics: dict, st_metrics: dict, combined_eq: pd.DataFrame):
    """Generate comprehensive report with both systems + combined."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total_equity = combined_eq["total_equity"].values
    total_return = (total_equity[-1] - TOTAL_CAPITAL) / TOTAL_CAPITAL * 100

    peak = np.maximum.accumulate(total_equity)
    dd = (peak - total_equity) / peak * 100
    max_dd = np.max(dd)

    returns = np.diff(total_equity) / total_equity[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(2190) if len(returns) > 1 and np.std(returns) > 0 else 0

    combined_metrics = {
        "version": "v11",
        "total_capital": TOTAL_CAPITAL,
        "final_equity": round(total_equity[-1], 2),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "long_term": lt_metrics,
        "short_term": st_metrics,
        "data_start": str(combined_eq["timestamp"].iloc[0]),
        "data_end": str(combined_eq["timestamp"].iloc[-1]),
    }

    # Save metrics JSON
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(combined_metrics, f, indent=2)

    # Build report text
    sep = "=" * 70
    thin = "-" * 70

    lt = lt_metrics
    st = st_metrics

    lines = [
        sep,
        "  BTC/USDT DUAL-TIMEFRAME BACKTEST — V11 (FINAL)",
        sep,
        "",
        f"  Data Period:     {combined_metrics['data_start'][:10]} to {combined_metrics['data_end'][:10]}",
        f"  Total Capital:   ${TOTAL_CAPITAL:,.0f}",
        f"    Long-Term:     ${TOTAL_CAPITAL * LT_ALLOC:,.0f} (60%)",
        f"    Short-Term:    ${TOTAL_CAPITAL * ST_ALLOC:,.0f} (40%)",
        "",
        sep,
        "  COMBINED PERFORMANCE",
        sep,
        "",
        f"  Final Equity:    ${total_equity[-1]:,.2f}",
        f"  Total Return:    {total_return:+.2f}%",
        f"  Max Drawdown:    {max_dd:.2f}%",
        f"  Sharpe Ratio:    {sharpe:.2f}",
        "",
        sep,
        "  SYSTEM 1: LONG-TERM TREND (Weekly Charts, 60%)",
        sep,
        "",
    ]

    if "error" in lt:
        lines.append(f"  {lt['error']}")
    else:
        lines += [
            f"  Final Equity:    ${lt['final_equity']:,.2f}  (from ${lt['initial_capital']:,.0f})",
            f"  Total Return:    {lt['total_return_pct']:+.2f}%",
            f"  Max Drawdown:    {lt['max_drawdown_pct']:.2f}%",
            f"  Sharpe Ratio:    {lt['sharpe_ratio']:.2f}",
            f"  Win Rate:        {lt['win_rate_pct']:.1f}%",
            f"  Profit Factor:   {lt['profit_factor']:.2f}",
            "",
            f"  Total Trades:    {lt['total_trades']}",
            f"    Base Entries:  {lt['base_trades']}",
            f"    Rolling Adds:  {lt['rolling_adds']}",
            f"  Avg Win:         ${lt['avg_win']:,.2f}",
            f"  Avg Loss:        ${lt['avg_loss']:,.2f}",
            f"  Largest Win:     ${lt['largest_win']:,.2f}",
            f"  Largest Loss:    ${lt['largest_loss']:,.2f}",
            f"  Avg Duration:    {lt['avg_duration_days']:.1f} days",
            "",
            f"  Funding Paid:    ${lt['total_funding_paid']:,.2f}",
            "",
            f"  Leverage Tiers:  5x={lt['trades_at_5x']} | 10x={lt['trades_at_10x']} | 25x={lt['trades_at_25x']} | 50x={lt['trades_at_50x']}",
            f"  Liquidations:    {lt['liquidations']}  (Loss: ${lt['liquidation_loss']:,.2f})",
            f"  Shorts blocked:  {lt['shorts_blocked_bull']} (bull market filter)",
        ]

    lines += [
        "",
        sep,
        "  SYSTEM 2: SHORT-TERM TREND (4H Charts, 40%)",
        sep,
        "",
    ]

    if "error" in st:
        lines.append(f"  {st['error']}")
    else:
        lines += [
            f"  Final Equity:    ${st['final_equity']:,.2f}  (from ${st['initial_capital']:,.0f})",
            f"  Total Return:    {st['total_return_pct']:+.2f}%",
            f"  Max Drawdown:    {st['max_drawdown_pct']:.2f}%",
            f"  Sharpe Ratio:    {st['sharpe_ratio']:.2f}",
            f"  Win Rate:        {st['win_rate_pct']:.1f}%",
            f"  Profit Factor:   {st['profit_factor']:.2f}",
            "",
            f"  Total Trades:    {st['total_trades']}",
            f"    Primary:       {st['primary_trades']}",
            f"    Rolling Adds:  {st['rolling_adds']}",
            f"    Longs:         {st['long_trades']}  (PnL: ${st['long_pnl']:+,.2f})",
            f"    Shorts:        {st['short_trades']}  (PnL: ${st['short_pnl']:+,.2f})",
            f"  Avg Win:         ${st['avg_win']:,.2f}",
            f"  Avg Loss:        ${st['avg_loss']:,.2f}",
            f"  Largest Win:     ${st['largest_win']:,.2f}",
            f"  Largest Loss:    ${st['largest_loss']:,.2f}",
            f"  Avg Duration:    {st['avg_duration_days']:.1f} days ({st['avg_duration_hours']:.0f} hours)",
            "",
            f"  Leverage:        3x={st['trades_at_3x']} | 10x={st['trades_at_10x']}",
            f"  Liquidations:    {st['liquidations']}  (Loss: ${st['liquidation_loss']:,.2f})",
            f"  Shorts blocked:  {st['shorts_blocked_bull']} (bull market filter)",
        ]

    lines += [
        "",
        sep,
        "  STRATEGY RULES SUMMARY",
        sep,
        "",
        "  SYSTEM 1 (Long-Term):",
        "    - Weekly MACD 13/34 divergence (classical + hidden)",
        "    - Trend: long > weekly 50 SMA, short < weekly 50 SMA",
        "    - Bull filter: price > weekly 200 SMA = NO SHORTS",
        "    - Leverage: 5x/10x/25x/50x based on conviction + MVRV + 123 Rule",
        "    - Rolling: inverted pyramid (150%/200%/250%) on pullbacks",
        "    - Exit: trailing stop at 3x weekly ATR, no fixed TP",
        "    - Funding: 0.01% per 8h deducted",
        "",
        "  SYSTEM 2 (Short-Term):",
        "    - 4H MACD 13/34 double divergence (V6 base)",
        "    - MACRO FILTER: price > weekly 200 SMA = LONG ONLY, below = SHORT ONLY",
        "    - Leverage: 3x base, 10x with 123 Rule (cap: 10x)",
        "    - Stop: ATR-based, max 3% price movement",
        "    - Exit: half at 1:1 R/R, trail rest at 2.5x ATR",
        "    - Max 2 rolling additions per trade",
        "",
        sep,
        f"  Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
    ]

    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
        f.write(report)
    print(report)

    return combined_metrics


def generate_comparison(combined_metrics: dict):
    """Compare V11 combined vs V6 standalone."""
    v6_path = os.path.join(os.path.dirname(__file__), "results_v6", "metrics.json")
    if not os.path.exists(v6_path):
        print("  V6 results not found — skipping comparison")
        return

    with open(v6_path) as f:
        v6 = json.load(f)

    v11 = combined_metrics
    v11_lt = v11.get("long_term", {})
    v11_st = v11.get("short_term", {})

    sep = "=" * 90
    col = 18

    def delta(new, old, higher_better=True):
        d = new - old
        if abs(d) < 0.005:
            return ""
        arrow = "+" if d > 0 else ""
        tag = " ^^" if (d > 0) == higher_better else " vv"
        return f"({arrow}{d:.2f}{tag})"

    lines = [
        sep,
        "  V11 (Dual-Timeframe) vs V6 (Short-Term Only) COMPARISON",
        sep,
        "",
        f"  {'Metric':<28} {'V6 ($10K)':>{col}} {'V11 Combined':>{col}} {'V11 LT (60%)':>{col}} {'V11 ST (40%)':>{col}}",
        f"  {'-'*28} {'-'*col} {'-'*col} {'-'*col} {'-'*col}",
        "",
    ]

    v11_lt_ret = v11_lt.get("total_return_pct", 0)
    v11_st_ret = v11_st.get("total_return_pct", 0)
    v11_lt_eq = v11_lt.get("final_equity", TOTAL_CAPITAL * LT_ALLOC)
    v11_st_eq = v11_st.get("final_equity", TOTAL_CAPITAL * ST_ALLOC)
    v11_lt_dd = v11_lt.get("max_drawdown_pct", 0)
    v11_st_dd = v11_st.get("max_drawdown_pct", 0)
    v11_lt_wr = v11_lt.get("win_rate_pct", 0)
    v11_st_wr = v11_st.get("win_rate_pct", 0)
    v11_lt_pf = v11_lt.get("profit_factor", 0)
    v11_st_pf = v11_st.get("profit_factor", 0)
    v11_lt_trades = v11_lt.get("total_trades", 0)
    v11_st_trades = v11_st.get("total_trades", 0)

    lines += [
        f"  {'Starting Capital':<28} ${'10,000':>{col-1}} ${'10,000':>{col-1}} ${'6,000':>{col-1}} ${'4,000':>{col-1}}",
        f"  {'Final Equity':<28} ${v6['final_equity']:>{col-2},.2f} ${v11['final_equity']:>{col-2},.2f} ${v11_lt_eq:>{col-2},.2f} ${v11_st_eq:>{col-2},.2f}",
        f"  {'Total Return %':<28} {v6['total_return_pct']:>{col-1}.2f}% {v11['total_return_pct']:>{col-1}.2f}% {v11_lt_ret:>{col-1}.2f}% {v11_st_ret:>{col-1}.2f}%",
        f"  {'Max Drawdown %':<28} {v6['max_drawdown_pct']:>{col-1}.2f}% {v11['max_drawdown_pct']:>{col-1}.2f}% {v11_lt_dd:>{col-1}.2f}% {v11_st_dd:>{col-1}.2f}%",
        f"  {'Sharpe Ratio':<28} {v6['sharpe_ratio']:>{col}.2f} {v11['sharpe_ratio']:>{col}.2f}",
        f"  {'Win Rate %':<28} {v6['win_rate_pct']:>{col-1}.2f}% {'':>{col}} {v11_lt_wr:>{col-1}.2f}% {v11_st_wr:>{col-1}.2f}%",
        f"  {'Profit Factor':<28} {v6['profit_factor']:>{col}.2f} {'':>{col}} {v11_lt_pf:>{col}.2f} {v11_st_pf:>{col}.2f}",
        f"  {'Total Trades':<28} {v6['total_trades']:>{col}} {'':>{col}} {v11_lt_trades:>{col}} {v11_st_trades:>{col}}",
        "",
        f"  V11 Combined vs V6:  {delta(v11['total_return_pct'], v6['total_return_pct'])} return, "
        f"{delta(v11['max_drawdown_pct'], v6['max_drawdown_pct'], False)} drawdown",
    ]

    if "total_funding_paid" in v11_lt:
        lines += [
            "",
            f"  Long-Term Funding Cost:  ${v11_lt['total_funding_paid']:,.2f}",
        ]

    lines += [
        "",
        sep,
        "  KEY DIFFERENCES",
        sep,
        "",
        "  V6:  Single system, $10K, 4H charts, 3x/10x leverage",
        "  V11: Dual system — 60% weekly (5-50x) + 40% 4H (3-10x)",
        "       Macro filter: weekly 200 SMA direction controls short-term",
        "       Long-term: rolling positions, funding costs, no fixed TP",
        "       Short-term: macro-aligned (long-only in bull, short-only in bear)",
        "",
        sep,
    ]

    comparison = "\n".join(lines)
    comp_path = os.path.join(RESULTS_DIR, "comparison_v6_v11.txt")
    with open(comp_path, "w") as f:
        f.write(comparison)
    print(f"\n{comparison}")
    print(f"\n  Comparison saved to {comp_path}")


def plot_equity_curves(combined_eq: pd.DataFrame, lt_metrics: dict, st_metrics: dict):
    """Plot equity curves for both systems + combined total."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamps = pd.to_datetime(combined_eq["timestamp"])
    lt_eq = combined_eq["lt_equity"].values
    st_eq = combined_eq["st_equity"].values
    total_eq = combined_eq["total_equity"].values

    peak = np.maximum.accumulate(total_eq)
    dd_pct = (peak - total_eq) / peak * 100

    fig, axes = plt.subplots(3, 1, figsize=(18, 16), height_ratios=[3, 1.2, 1.2])

    # ── Top: Equity curves ──
    ax1 = axes[0]
    ax1.plot(timestamps, total_eq, color="#E91E63", linewidth=1.8, label="Combined Total", zorder=3)
    ax1.plot(timestamps, lt_eq, color="#2196F3", linewidth=1.2, alpha=0.8, label="System 1: Long-Term (60%)")
    ax1.plot(timestamps, st_eq, color="#FF9800", linewidth=1.2, alpha=0.8, label="System 2: Short-Term (40%)")
    ax1.axhline(y=TOTAL_CAPITAL, color="gray", linestyle="--", alpha=0.4, label=f"Initial ${TOTAL_CAPITAL:,.0f}")
    ax1.axhline(y=TOTAL_CAPITAL * LT_ALLOC, color="#2196F3", linestyle=":", alpha=0.3)
    ax1.axhline(y=TOTAL_CAPITAL * ST_ALLOC, color="#FF9800", linestyle=":", alpha=0.3)

    ax1.set_title("BTC/USDT V11 — Dual-Timeframe Strategy\n"
                   "System 1: Weekly Trend (60%, 5-50x) | System 2: 4H V6 (40%, 3-10x)",
                   fontsize=13, fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── Middle: Drawdown ──
    ax2 = axes[1]
    ax2.fill_between(timestamps, -dd_pct, 0, color="#F44336", alpha=0.3)
    ax2.plot(timestamps, -dd_pct, color="#F44336", linewidth=0.8)
    ax2.set_ylabel("Combined Drawdown (%)")
    ax2.set_title("Combined Drawdown from Peak", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── Bottom: System returns comparison ──
    ax3 = axes[2]
    lt_ret = (lt_eq / lt_eq[0] - 1) * 100
    st_ret = (st_eq / st_eq[0] - 1) * 100
    total_ret = (total_eq / total_eq[0] - 1) * 100

    ax3.plot(timestamps, total_ret, color="#E91E63", linewidth=1.5, label="Combined")
    ax3.plot(timestamps, lt_ret, color="#2196F3", linewidth=1.0, alpha=0.8, label="Long-Term")
    ax3.plot(timestamps, st_ret, color="#FF9800", linewidth=1.0, alpha=0.8, label="Short-Term")
    ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax3.set_ylabel("Return (%)")
    ax3.set_title("Percentage Returns by System", fontsize=11)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    fig.subplots_adjust(hspace=0.40)
    chart_path = os.path.join(RESULTS_DIR, "equity_curve.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Equity curve saved to {chart_path}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  BACKTEST V11 — DUAL-TIMEFRAME STRATEGY (FINAL)")
    print("=" * 60)

    # Load 4H data
    df_4h = fetch_ohlcv(use_cache=True)
    print(f"\nLoaded {len(df_4h)} 4H candles: {df_4h['timestamp'].iloc[0]} to {df_4h['timestamp'].iloc[-1]}")

    # Prepare data
    df_4h = prepare_4h_data(df_4h)

    # Resample to weekly
    weekly_df = resample_to_weekly(df_4h)
    print(f"Resampled to {len(weekly_df)} weekly candles")

    # Map weekly SMA200 back to 4H bars for macro filter
    weekly_sma200 = get_weekly_sma200_at_4h(df_4h, weekly_df)

    # Capital allocation
    lt_capital = TOTAL_CAPITAL * LT_ALLOC
    st_capital = TOTAL_CAPITAL * ST_ALLOC

    # ── Run System 1: Long-Term Weekly ──
    lt_system = LongTermSystem(lt_capital, weekly_df)
    lt_metrics = lt_system.run()

    # ── Run System 2: Short-Term 4H ──
    st_system = ShortTermSystem(st_capital, df_4h, weekly_sma200)
    st_metrics = st_system.run()

    # ── Build combined equity ──
    combined_eq = build_combined_equity(
        lt_system.equity_curve, st_system.equity_curve,
        lt_capital, st_capital
    )

    # ── Generate outputs ──
    combined_metrics = generate_report(lt_metrics, st_metrics, combined_eq)
    plot_equity_curves(combined_eq, lt_metrics, st_metrics)
    generate_comparison(combined_metrics)

    print("\n" + "=" * 60)
    print("  Backtest V11 complete!")
    print("=" * 60)

    return combined_metrics


if __name__ == "__main__":
    main()

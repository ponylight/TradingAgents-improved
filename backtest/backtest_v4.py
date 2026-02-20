"""Backtest V4 — MACD Reversal with V3 filters + 123 Rule trendline confirmation.

Changes from V3:
1. 123 Rule confirmation required before any entry:
   - Rule 1: Trendline break (price crosses above/below trendline from last 2 swing lows/highs)
   - Rule 2: Failed retest (price returns within 1 ATR of broken level but closes in new direction)
   - Rule 3: Pivot break (price exceeds the most recent swing point)
   Swing detection uses lookback=10 bars on each side for proper turning point identification.

Inherited from V3:
- No shorts in bull market (price > SMA200)
- Rolling positions only near market bottoms (within 30% of 365-day low)

Inherited from V2:
- 50 SMA trend filter: long-only when price > SMA50, short-only when price < SMA50
- Tighter TP1: 1:1 R/R
- Double divergence minimum: require divergence_strength >= 2 before entry
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

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.fetch_data import fetch_ohlcv
from backtest.indicators import compute_macd, compute_atr, compute_ma

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v4")

# V3 constants
NEAR_BOTTOM_LOOKBACK = 2190  # 365 days in 4h bars
NEAR_BOTTOM_THRESHOLD = 0.30

# V4 constants: 123 Rule
SWING_LOOKBACK = 10  # bars on each side for swing detection


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class Trade:
    side: str
    entry_price: float
    entry_idx: int
    entry_time: pd.Timestamp
    size: float
    stop_loss: float
    take_profit_1: float
    initial_risk: float
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


@dataclass
class Signal:
    idx: int
    side: str
    entry_price: float
    stop_loss: float
    divergence_strength: float
    signal_type: str = "primary"


# ── Swing detection (V4) ─────────────────────────────────────────────

def detect_swings(highs: np.ndarray, lows: np.ndarray, lookback: int = SWING_LOOKBACK):
    """Detect swing highs and swing lows using bidirectional lookback.

    A swing high at bar i: high[i] is the max within [i-lookback, i+lookback].
    A swing low at bar i: low[i] is the min within [i-lookback, i+lookback].
    Minimum spacing of `lookback` bars between consecutive swings of the same type.
    """
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(highs) - lookback):
        window_start = i - lookback
        window_end = i + lookback + 1

        # Swing high
        if highs[i] >= np.max(highs[window_start:window_end]):
            if not swing_highs or i - swing_highs[-1][0] >= lookback:
                swing_highs.append((i, highs[i]))

        # Swing low
        if lows[i] <= np.min(lows[window_start:window_end]):
            if not swing_lows or i - swing_lows[-1][0] >= lookback:
                swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


# ── 123 Rule check (V4) ──────────────────────────────────────────────

def check_123_rule(signal_idx: int, side: str, closes: np.ndarray,
                   highs: np.ndarray, lows: np.ndarray, atr_vals: np.ndarray,
                   swing_highs: list, swing_lows: list) -> bool:
    """Check if the 123 Rule pattern is confirmed before this signal."""
    if side == "long":
        return _check_123_long(signal_idx, closes, highs, lows, atr_vals, swing_highs, swing_lows)
    else:
        return _check_123_short(signal_idx, closes, highs, lows, atr_vals, swing_highs, swing_lows)


def _check_123_long(signal_idx: int, closes: np.ndarray, highs: np.ndarray,
                    lows: np.ndarray, atr_vals: np.ndarray,
                    swing_highs: list, swing_lows: list) -> bool:
    """123 Rule for long entries.

    1. Trendline from last 2 swing lows; price crosses above.
    2. Price retests within 1 ATR of trendline but closes above (failed retest).
    3. Price breaks above the most recent swing high (pivot break).
    """
    lb = SWING_LOOKBACK

    # Get confirmed swing lows (fully confirmed by signal time)
    confirmed_lows = [
        (idx, val) for idx, val in swing_lows
        if idx + lb <= signal_idx
    ]
    if len(confirmed_lows) < 2:
        return False

    # Use last 2 swing lows for trendline
    x1, y1 = confirmed_lows[-2]
    x2, y2 = confirmed_lows[-1]

    dx = x2 - x1
    if dx == 0:
        return False
    slope = (y2 - y1) / dx

    def tl(x):
        return y1 + slope * (x - x1)

    # Rule 1: Trendline break — first close above trendline after 2nd swing low
    break_bar = None
    for j in range(x2 + 1, signal_idx + 1):
        if closes[j] > tl(j):
            break_bar = j
            break
    if break_bar is None:
        return False

    # Rule 2: Failed retest — after break, price returns within 1 ATR of
    # trendline but closes above it
    retest_bar = None
    for j in range(break_bar + 1, signal_idx + 1):
        tl_j = tl(j)
        a = atr_vals[j]
        if np.isnan(a):
            continue
        # Low comes within 1 ATR of trendline (from above or touching)
        if abs(lows[j] - tl_j) <= a:
            # But close stays above trendline (failed to break back below)
            if closes[j] > tl_j:
                retest_bar = j
                break
    if retest_bar is None:
        return False

    # Rule 3: Pivot break — price exceeds the most recent confirmed swing high
    confirmed_highs = [
        (idx, val) for idx, val in swing_highs
        if idx + lb <= signal_idx
    ]
    if not confirmed_highs:
        return False

    _, pivot_val = confirmed_highs[-1]

    for j in range(retest_bar, signal_idx + 1):
        if closes[j] > pivot_val:
            return True

    return False


def _check_123_short(signal_idx: int, closes: np.ndarray, highs: np.ndarray,
                     lows: np.ndarray, atr_vals: np.ndarray,
                     swing_highs: list, swing_lows: list) -> bool:
    """123 Rule for short entries.

    1. Trendline from last 2 swing highs; price crosses below.
    2. Price retests within 1 ATR of trendline but closes below (failed retest).
    3. Price breaks below the most recent swing low (pivot break).
    """
    lb = SWING_LOOKBACK

    # Get confirmed swing highs
    confirmed_highs = [
        (idx, val) for idx, val in swing_highs
        if idx + lb <= signal_idx
    ]
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

    # Rule 1: Trendline break — first close below trendline after 2nd swing high
    break_bar = None
    for j in range(x2 + 1, signal_idx + 1):
        if closes[j] < tl(j):
            break_bar = j
            break
    if break_bar is None:
        return False

    # Rule 2: Failed retest — after break, price returns within 1 ATR of
    # trendline but closes below it
    retest_bar = None
    for j in range(break_bar + 1, signal_idx + 1):
        tl_j = tl(j)
        a = atr_vals[j]
        if np.isnan(a):
            continue
        # High comes within 1 ATR of trendline (from below or touching)
        if abs(highs[j] - tl_j) <= a:
            # But close stays below trendline (failed to break back above)
            if closes[j] < tl_j:
                retest_bar = j
                break
    if retest_bar is None:
        return False

    # Rule 3: Pivot break — price drops below the most recent confirmed swing low
    confirmed_lows = [
        (idx, val) for idx, val in swing_lows
        if idx + lb <= signal_idx
    ]
    if not confirmed_lows:
        return False

    _, pivot_val = confirmed_lows[-1]

    for j in range(retest_bar, signal_idx + 1):
        if closes[j] < pivot_val:
            return True

    return False


# ── Regime helpers (from V3) ─────────────────────────────────────────

def is_bull_market(close: float, sma200: float) -> bool:
    if np.isnan(sma200):
        return False
    return close > sma200


def is_near_market_bottom(closes: np.ndarray, current_idx: int) -> bool:
    lookback_start = max(0, current_idx - NEAR_BOTTOM_LOOKBACK)
    window = closes[lookback_start:current_idx + 1]
    if len(window) == 0:
        return False
    low_365 = np.min(window)
    current_price = closes[current_idx]
    return current_price <= low_365 * (1.0 + NEAR_BOTTOM_THRESHOLD)


# ── Strategy: signal generation with V4 filters ──────────────────────

def compute_signals_v4(df: pd.DataFrame, swing_highs: list, swing_lows: list) -> tuple[list[Signal], dict]:
    """Scan for MACD divergence signals with V4 filters (V3 + 123 Rule).

    Returns (signals, stats_dict) where stats_dict tracks filter statistics.
    """
    signals = []
    stats = {
        "shorts_blocked_by_bull": 0,
        "signals_blocked_by_123": 0,
        "signals_passed_123": 0,
        "long_blocked_123": 0,
        "short_blocked_123": 0,
    }

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

        # ── LONG SIGNAL ──
        if hist[i] < 0 and i >= 4:
            if closes[i] <= sma50[i]:
                pass
            else:
                is_key_kline = (
                    hist[i] > hist[i - 1]
                    and hist[i - 1] < 0
                    and hist[i - 1] < hist[i - 2]
                    and hist[i - 2] < hist[i - 3]
                )

                min_hist_size = closes[i] * 0.001
                if is_key_kline and abs(hist[i - 1]) > min_hist_size:
                    long_div = _check_bottom_divergence(i, hist, lows, lookback=120)
                    if long_div is not None:
                        strength, peak_diff = long_div
                        if strength >= 2 and peak_diff > 0.30:
                            # V4: 123 Rule confirmation
                            if check_123_rule(i, "long", closes, highs, lows, atr,
                                              swing_highs, swing_lows):
                                sl = lows[i] - atr[i] * 1.5 if not np.isnan(atr[i]) else lows[i] * 0.97
                                signals.append(Signal(
                                    idx=i,
                                    side="long",
                                    entry_price=closes[i],
                                    stop_loss=sl,
                                    divergence_strength=strength,
                                ))
                                last_signal_idx = i
                                stats["signals_passed_123"] += 1
                            else:
                                stats["signals_blocked_by_123"] += 1
                                stats["long_blocked_123"] += 1

        # ── SHORT SIGNAL ──
        if hist[i] > 0 and i >= 4:
            # V3: NO SHORTS IN BULL MARKET
            if bull:
                if closes[i] < sma50[i]:
                    is_key_kline = (
                        hist[i] < hist[i - 1]
                        and hist[i - 1] > 0
                        and hist[i - 1] > hist[i - 2]
                        and hist[i - 2] > hist[i - 3]
                    )
                    min_hist_size = closes[i] * 0.001
                    if is_key_kline and abs(hist[i - 1]) > min_hist_size:
                        short_div = _check_top_divergence(i, hist, highs, lookback=120)
                        if short_div is not None:
                            strength, peak_diff = short_div
                            if strength >= 2 and peak_diff > 0.30:
                                stats["shorts_blocked_by_bull"] += 1
                continue

            if closes[i] >= sma50[i]:
                pass
            else:
                is_key_kline = (
                    hist[i] < hist[i - 1]
                    and hist[i - 1] > 0
                    and hist[i - 1] > hist[i - 2]
                    and hist[i - 2] > hist[i - 3]
                )

                min_hist_size = closes[i] * 0.001
                if is_key_kline and abs(hist[i - 1]) > min_hist_size:
                    short_div = _check_top_divergence(i, hist, highs, lookback=120)
                    if short_div is not None:
                        strength, peak_diff = short_div
                        if strength >= 2 and peak_diff > 0.30:
                            # V4: 123 Rule confirmation
                            if check_123_rule(i, "short", closes, highs, lows, atr,
                                              swing_highs, swing_lows):
                                sl = highs[i] + atr[i] * 1.5 if not np.isnan(atr[i]) else highs[i] * 1.03
                                signals.append(Signal(
                                    idx=i,
                                    side="short",
                                    entry_price=closes[i],
                                    stop_loss=sl,
                                    divergence_strength=strength,
                                ))
                                last_signal_idx = i
                                stats["signals_passed_123"] += 1
                            else:
                                stats["signals_blocked_by_123"] += 1
                                stats["short_blocked_123"] += 1

    return signals, stats


def _check_bottom_divergence(
    current_idx: int, hist: np.ndarray, lows: np.ndarray, lookback: int = 80
) -> tuple[float, float] | None:
    """Check for bullish (bottom) divergence. Same logic as V1/V2/V3."""
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
        price_lower = lows[t2_idx] < lows[t1_idx]
        hist_higher = t2_val > t1_val

        if price_lower and hist_higher:
            divergence_count += 1

    if divergence_count == 0:
        return None

    last_two = troughs[-2:]
    peak_diff = abs(abs(last_two[1][1]) - abs(last_two[0][1])) / abs(last_two[0][1])
    return (min(divergence_count, 3), peak_diff)


def _check_top_divergence(
    current_idx: int, hist: np.ndarray, highs: np.ndarray, lookback: int = 80
) -> tuple[float, float] | None:
    """Check for bearish (top) divergence. Same logic as V1/V2/V3."""
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
        price_higher = highs[p2_idx] > highs[p1_idx]
        hist_lower = p2_val < p1_val

        if price_higher and hist_lower:
            divergence_count += 1

    if divergence_count == 0:
        return None

    last_two = peaks[-2:]
    peak_diff = abs(last_two[0][1] - last_two[1][1]) / last_two[0][1]
    return (min(divergence_count, 3), peak_diff)


def compute_rolling_signals_v4(
    df: pd.DataFrame, active_trades: list[Trade], current_idx: int,
    swing_highs: list, swing_lows: list
) -> list[Signal]:
    """Rolling add-on signals with V3 near-bottom filter + V4 123 Rule."""
    signals = []
    if not active_trades:
        return signals

    close = df["close"].iloc[current_idx]
    ma30 = df["ma30"].iloc[current_idx]
    atr_val = df["atr"].iloc[current_idx]
    sma50 = df["sma50"].iloc[current_idx]
    sma200 = df["sma200"].iloc[current_idx]
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atr_arr = df["atr"].values

    if np.isnan(ma30) or np.isnan(atr_val) or np.isnan(sma50) or np.isnan(sma200):
        return signals

    # V3: only allow rolling adds near market bottoms
    near_bottom = is_near_market_bottom(closes, current_idx)
    if not near_bottom:
        return signals

    bull = is_bull_market(close, sma200)

    for trade in active_trades:
        if trade.closed:
            continue

        unrealized_pnl_pct = 0
        if trade.side == "long":
            unrealized_pnl_pct = (close - trade.entry_price) / trade.entry_price
        else:
            unrealized_pnl_pct = (trade.entry_price - close) / trade.entry_price

        if unrealized_pnl_pct < 0.03:
            continue

        if trade.side == "long":
            if close <= sma50:
                continue

            # V4: 123 Rule check for rolling add
            if not check_123_rule(current_idx, "long", closes, highs, lows, atr_arr,
                                  swing_highs, swing_lows):
                continue

            near_ma30 = abs(close - ma30) / close < 0.005
            above_ma30 = close > ma30
            recent_high = df["high"].iloc[max(0, current_idx - 20):current_idx].max()
            breakout = close > recent_high
            avg_vol = df["volume"].iloc[max(0, current_idx - 20):current_idx].mean()
            vol_confirm = df["volume"].iloc[current_idx] > avg_vol * 1.5

            if (near_ma30 and above_ma30) or (breakout and vol_confirm):
                sl = close - atr_val * 1.5
                signals.append(Signal(
                    idx=current_idx,
                    side="long",
                    entry_price=close,
                    stop_loss=sl,
                    divergence_strength=1,
                    signal_type="rolling_add",
                ))

        elif trade.side == "short":
            if bull:
                continue
            if close >= sma50:
                continue

            # V4: 123 Rule check for rolling add
            if not check_123_rule(current_idx, "short", closes, highs, lows, atr_arr,
                                  swing_highs, swing_lows):
                continue

            near_ma30 = abs(close - ma30) / close < 0.005
            below_ma30 = close < ma30
            recent_low = df["low"].iloc[max(0, current_idx - 20):current_idx].min()
            breakdown = close < recent_low
            avg_vol = df["volume"].iloc[max(0, current_idx - 20):current_idx].mean()
            vol_confirm = df["volume"].iloc[current_idx] > avg_vol * 1.5

            if (near_ma30 and below_ma30) or (breakdown and vol_confirm):
                sl = close + atr_val * 1.5
                signals.append(Signal(
                    idx=current_idx,
                    side="short",
                    entry_price=close,
                    stop_loss=sl,
                    divergence_strength=1,
                    signal_type="rolling_add",
                ))

    return signals


# ── Engine ────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    commission_rate: float = 0.001
    max_position_pct: float = 0.40
    risk_per_trade_pct: float = 0.02
    pyramid_decay: float = 0.6
    max_rolling_adds: int = 3
    trailing_stop_atr_mult: float = 2.5
    rr_ratio: float = 1.0


class BacktestEngine:
    def __init__(self, df: pd.DataFrame, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.df = self._prepare_data(df)
        self.capital = self.config.initial_capital
        self.equity_curve = []
        self.trades: list[Trade] = []
        self.active_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.rolling_add_counts: dict[int, int] = {}
        self.filter_stats: dict = {}

        # Pre-compute swings for 123 Rule
        self.swing_highs, self.swing_lows = detect_swings(
            self.df["high"].values, self.df["low"].values, lookback=SWING_LOOKBACK
        )

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_macd(df, fast=13, slow=34, signal=9)
        df["atr"] = compute_atr(df, period=14)
        df["ma30"] = compute_ma(df, period=30)
        df["sma50"] = compute_ma(df, period=50)
        df["sma200"] = compute_ma(df, period=200)
        return df

    def run(self) -> dict:
        print(f"[V4] Running backtest on {len(self.df)} candles...")
        print(f"  Date range: {self.df['timestamp'].iloc[0]} to {self.df['timestamp'].iloc[-1]}")
        print(f"  V4 change: 123 Rule trendline confirmation (swing lookback={SWING_LOOKBACK})")
        print(f"  V3 inherited: SMA200 bull filter | rolling adds only near 365-day lows")
        print(f"  V2 inherited: SMA50 trend filter | TP1 at 1:{self.config.rr_ratio} R/R | double divergence")
        print(f"  Pre-computed {len(self.swing_highs)} swing highs, {len(self.swing_lows)} swing lows")

        primary_signals, self.filter_stats = compute_signals_v4(
            self.df, self.swing_highs, self.swing_lows
        )
        signal_map: dict[int, Signal] = {s.idx: s for s in primary_signals}
        long_sigs = sum(1 for s in primary_signals if s.side == "long")
        short_sigs = sum(1 for s in primary_signals if s.side == "short")
        print(f"  Found {len(primary_signals)} primary signals ({long_sigs} long, {short_sigs} short)")
        print(f"  123 Rule: {self.filter_stats['signals_passed_123']} passed, "
              f"{self.filter_stats['signals_blocked_by_123']} blocked "
              f"(long: {self.filter_stats['long_blocked_123']}, short: {self.filter_stats['short_blocked_123']})")
        print(f"  V3 bull filter blocked {self.filter_stats['shorts_blocked_by_bull']} short signals")

        for i in range(len(self.df)):
            price = self.df["close"].iloc[i]
            high = self.df["high"].iloc[i]
            low = self.df["low"].iloc[i]
            ts = self.df["timestamp"].iloc[i]

            self._manage_positions(i, high, low, price, ts)

            if self.active_trades:
                rolling_sigs = compute_rolling_signals_v4(
                    self.df, self.active_trades, i,
                    self.swing_highs, self.swing_lows
                )
                for sig in rolling_sigs:
                    self._execute_rolling_add(sig, i, ts)

            if i in signal_map:
                sig = signal_map[i]
                self._execute_entry(sig, ts)

            unrealized = self._calc_unrealized(price)
            self.equity_curve.append({
                "timestamp": ts,
                "equity": self.capital + unrealized,
                "capital": self.capital,
                "unrealized": unrealized,
                "num_positions": len(self.active_trades),
            })

        last_price = self.df["close"].iloc[-1]
        last_ts = self.df["timestamp"].iloc[-1]
        for trade in list(self.active_trades):
            self._close_trade(trade, last_price, len(self.df) - 1, last_ts, reason="end_of_backtest")

        return self._compute_metrics()

    def _execute_entry(self, signal: Signal, ts: pd.Timestamp):
        same_dir = [t for t in self.active_trades if t.side == signal.side and not t.is_rolling]
        if same_dir:
            return

        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            return

        risk_amount = self.capital * self.config.risk_per_trade_pct
        size_btc = risk_amount / risk
        max_size_btc = (self.capital * self.config.max_position_pct) / signal.entry_price
        size_btc = min(size_btc, max_size_btc)

        if size_btc * signal.entry_price < 10:
            return

        commission = size_btc * signal.entry_price * self.config.commission_rate
        self.capital -= commission

        if signal.side == "long":
            tp1 = signal.entry_price + risk * self.config.rr_ratio
        else:
            tp1 = signal.entry_price - risk * self.config.rr_ratio

        trade = Trade(
            side=signal.side,
            entry_price=signal.entry_price,
            entry_idx=signal.idx,
            entry_time=ts,
            size=size_btc,
            stop_loss=signal.stop_loss,
            take_profit_1=tp1,
            initial_risk=risk,
            original_size=size_btc,
            trailing_stop=signal.stop_loss,
        )
        self.active_trades.append(trade)
        self.trades.append(trade)

    def _execute_rolling_add(self, signal: Signal, idx: int, ts: pd.Timestamp):
        base_trades = [t for t in self.active_trades if t.side == signal.side and not t.is_rolling and not t.closed]
        if not base_trades:
            return

        base = base_trades[0]
        base_key = base.entry_idx
        add_count = self.rolling_add_counts.get(base_key, 0)
        if add_count >= self.config.max_rolling_adds:
            return

        add_size = base.size * (self.config.pyramid_decay ** (add_count + 1))
        cost = add_size * signal.entry_price
        if cost > self.capital * 0.2:
            return

        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            return

        commission = cost * self.config.commission_rate
        self.capital -= commission

        if signal.side == "long":
            tp1 = signal.entry_price + risk * self.config.rr_ratio
        else:
            tp1 = signal.entry_price - risk * self.config.rr_ratio

        trade = Trade(
            side=signal.side,
            entry_price=signal.entry_price,
            entry_idx=idx,
            entry_time=ts,
            size=add_size,
            stop_loss=signal.stop_loss,
            take_profit_1=tp1,
            initial_risk=risk,
            original_size=add_size,
            is_rolling=True,
            trailing_stop=signal.stop_loss,
        )
        self.active_trades.append(trade)
        self.trades.append(trade)
        self.rolling_add_counts[base_key] = add_count + 1

    def _manage_positions(self, idx: int, high: float, low: float, close: float, ts: pd.Timestamp):
        for trade in list(self.active_trades):
            if trade.closed:
                continue

            if trade.side == "long":
                if low <= trade.trailing_stop:
                    self._close_trade(trade, trade.trailing_stop, idx, ts, reason="stop_loss")
                    continue

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

                atr = self.df["atr"].iloc[idx]
                if not np.isnan(atr) and trade.half_closed:
                    new_trail = close - atr * self.config.trailing_stop_atr_mult
                    trade.trailing_stop = max(trade.trailing_stop, new_trail)

            elif trade.side == "short":
                if high >= trade.trailing_stop:
                    self._close_trade(trade, trade.trailing_stop, idx, ts, reason="stop_loss")
                    continue

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

                atr = self.df["atr"].iloc[idx]
                if not np.isnan(atr) and trade.half_closed:
                    new_trail = close + atr * self.config.trailing_stop_atr_mult
                    trade.trailing_stop = min(trade.trailing_stop, new_trail)

    def _close_trade(self, trade: Trade, exit_price: float, idx: int, ts: pd.Timestamp, reason: str = ""):
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
        self.capital += pnl

        if trade in self.active_trades:
            self.active_trades.remove(trade)
        self.closed_trades.append(trade)

    def _calc_unrealized(self, current_price: float) -> float:
        total = 0.0
        for trade in self.active_trades:
            if trade.side == "long":
                total += (current_price - trade.entry_price) * trade.size
            else:
                total += (trade.entry_price - current_price) * trade.size
        return total

    def _compute_metrics(self) -> dict:
        if not self.closed_trades:
            return {
                "version": "v4",
                "error": "No trades executed",
                "total_return_pct": 0.0,
                "final_equity": self.config.initial_capital,
                "initial_capital": self.config.initial_capital,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "avg_trade_duration_hours": 0,
                "avg_trade_duration_days": 0,
                "primary_trades": 0,
                "rolling_adds": 0,
                "long_trades": 0,
                "short_trades": 0,
                "long_pnl": 0,
                "short_pnl": 0,
                "rr_ratio": self.config.rr_ratio,
                "signals_passed_123": self.filter_stats.get("signals_passed_123", 0),
                "signals_blocked_123": self.filter_stats.get("signals_blocked_by_123", 0),
                "data_start": str(self.df["timestamp"].iloc[0]),
                "data_end": str(self.df["timestamp"].iloc[-1]),
                "total_candles": len(self.df),
            }

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

        pnls = [t.pnl for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        durations = []
        for t in self.closed_trades:
            if t.entry_time is not None and t.exit_time is not None:
                dur = (t.exit_time - t.entry_time).total_seconds() / 3600
                durations.append(dur)
        avg_duration_hours = np.mean(durations) if durations else 0
        avg_duration_days = avg_duration_hours / 24

        primary_trades = [t for t in self.closed_trades if not t.is_rolling]
        rolling_trades = [t for t in self.closed_trades if t.is_rolling]

        long_trades = [t for t in self.closed_trades if t.side == "long"]
        short_trades = [t for t in self.closed_trades if t.side == "short"]
        long_pnl = sum(t.pnl for t in long_trades)
        short_pnl = sum(t.pnl for t in short_trades)

        return {
            "version": "v4",
            "total_return_pct": round(total_return, 2),
            "final_equity": round(equity[-1], 2),
            "initial_capital": self.config.initial_capital,
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_trades": len(self.closed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "largest_win": round(max(wins), 2) if wins else 0,
            "largest_loss": round(min(losses), 2) if losses else 0,
            "avg_trade_duration_hours": round(avg_duration_hours, 1),
            "avg_trade_duration_days": round(avg_duration_days, 1),
            "primary_trades": len(primary_trades),
            "rolling_adds": len(rolling_trades),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl": round(long_pnl, 2),
            "short_pnl": round(short_pnl, 2),
            "rr_ratio": self.config.rr_ratio,
            "signals_passed_123": self.filter_stats.get("signals_passed_123", 0),
            "signals_blocked_123": self.filter_stats.get("signals_blocked_by_123", 0),
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
        self._generate_four_way_comparison(metrics)

    def _format_report(self, metrics: dict) -> str:
        sep = "=" * 60
        lines = [
            sep,
            "  BTC/USDT MACD REVERSAL BACKTEST — V4",
            "  V4: 123 Rule trendline confirmation filter",
            sep,
            "",
            f"  Data Period:     {metrics['data_start'][:10]} to {metrics['data_end'][:10]}",
            f"  Total Candles:   {metrics['total_candles']} (4h bars)",
            f"  Initial Capital: ${metrics['initial_capital']:,.2f}",
            "",
            sep,
            "  PERFORMANCE SUMMARY",
            sep,
            "",
            f"  Final Equity:    ${metrics['final_equity']:,.2f}",
            f"  Total Return:    {metrics['total_return_pct']:+.2f}%",
            f"  Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%",
            f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}",
            "",
            sep,
            "  TRADE STATISTICS",
            sep,
            "",
            f"  Total Trades:    {metrics['total_trades']}",
            f"    Primary:       {metrics['primary_trades']}",
            f"    Rolling Adds:  {metrics['rolling_adds']}",
            f"    Long Trades:   {metrics['long_trades']}  (PnL: ${metrics['long_pnl']:+,.2f})",
            f"    Short Trades:  {metrics['short_trades']}  (PnL: ${metrics['short_pnl']:+,.2f})",
            f"  Win Rate:        {metrics['win_rate_pct']:.1f}%",
            f"  Profit Factor:   {metrics['profit_factor']:.2f}",
            f"  R/R Ratio (TP1): 1:{metrics['rr_ratio']}",
            "",
            f"  Avg Win:         ${metrics['avg_win']:,.2f}",
            f"  Avg Loss:        ${metrics['avg_loss']:,.2f}",
            f"  Largest Win:     ${metrics['largest_win']:,.2f}",
            f"  Largest Loss:    ${metrics['largest_loss']:,.2f}",
            "",
            f"  Avg Duration:    {metrics['avg_trade_duration_days']:.1f} days ({metrics['avg_trade_duration_hours']:.0f} hours)",
            "",
            sep,
            "  V4 FILTER DETAILS — 123 RULE",
            sep,
            "",
            f"  Signals passed 123 Rule:   {metrics['signals_passed_123']}",
            f"  Signals blocked by 123:    {metrics['signals_blocked_123']}",
            "",
            "  123 Rule pattern (required before every entry):",
            f"    1. Trendline break: price crosses above/below line from",
            f"       last 2 swing lows/highs (swing lookback = {SWING_LOOKBACK} bars)",
            f"    2. Failed retest: price returns within 1 ATR of broken level",
            f"       but closes back in new trend direction",
            f"    3. Pivot break: price exceeds most recent swing point",
            "",
            "  Inherited from V3:",
            "    - No shorts in bull market (price > 200 SMA)",
            "    - Rolling adds only near market bottoms (within 30% of 365-day low)",
            "",
            "  Inherited from V2:",
            "    - 50 SMA trend filter",
            "    - TP1 at 1:1 R/R",
            "    - Double divergence (strength >= 2)",
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

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), height_ratios=[3, 1, 1.5])

        # Equity curve
        ax1.plot(timestamps, equity, color="#2196F3", linewidth=1.2, label="Equity (V4)")
        ax1.axhline(y=self.config.initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")

        ax1.set_title("BTC/USDT MACD Reversal V4 — Equity Curve\n"
                      "(123 Rule + SMA200 bull filter + near-bottom rolling + V2 filters)",
                      fontsize=13, fontweight="bold")
        ax1.set_ylabel("Equity ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        for trade in self.closed_trades[:50]:
            marker_color = "#4CAF50" if trade.pnl > 0 else "#F44336"
            if trade.entry_time in timestamps.values:
                entry_equity_idx = timestamps[timestamps == trade.entry_time].index
                if len(entry_equity_idx) > 0:
                    ax1.plot(trade.entry_time, equity[entry_equity_idx[0]],
                            marker="^" if trade.side == "long" else "v",
                            color=marker_color, markersize=6, alpha=0.8)

        # Drawdown
        ax2.fill_between(timestamps, -drawdown_pct, 0, color="#F44336", alpha=0.3)
        ax2.plot(timestamps, -drawdown_pct, color="#F44336", linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Drawdown", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # BTC price with swing points
        prices = self.df["close"].values
        sma200 = self.df["sma200"].values
        ts_all = pd.to_datetime(self.df["timestamp"])

        ax3.plot(ts_all, prices, color="#FF9800", linewidth=0.8, label="BTC Price")
        valid_sma200 = ~np.isnan(sma200)
        ax3.plot(ts_all[valid_sma200], sma200[valid_sma200], color="#9C27B0",
                linewidth=1.0, alpha=0.8, label="SMA200")

        # Plot swing points
        if self.swing_highs:
            sh_indices = [s[0] for s in self.swing_highs if s[0] < len(ts_all)]
            sh_values = [s[1] for s in self.swing_highs if s[0] < len(ts_all)]
            ax3.scatter(ts_all.iloc[sh_indices], sh_values, color="#F44336",
                       marker="v", s=8, alpha=0.3, label="Swing Highs", zorder=3)
        if self.swing_lows:
            sl_indices = [s[0] for s in self.swing_lows if s[0] < len(ts_all)]
            sl_values = [s[1] for s in self.swing_lows if s[0] < len(ts_all)]
            ax3.scatter(ts_all.iloc[sl_indices], sl_values, color="#4CAF50",
                       marker="^", s=8, alpha=0.3, label="Swing Lows", zorder=3)

        ax3.set_ylabel("BTC Price ($)")
        ax3.set_title("BTC/USDT with SMA200 + Swing Points (123 Rule)", fontsize=11)
        ax3.legend(loc="upper left", fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax3.set_ylim(0, prices.max() * 1.1)

        fig.subplots_adjust(hspace=0.40)
        chart_path = os.path.join(RESULTS_DIR, "equity_curve.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Equity curve saved to {chart_path}")

    def _save_trade_log(self):
        records = []
        for t in self.closed_trades:
            records.append({
                "side": t.side,
                "type": "rolling" if t.is_rolling else "primary",
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "size_btc": round(t.size, 6),
                "stop_loss": round(t.stop_loss, 2),
                "pnl": round(t.pnl, 2),
                "half_closed_at_tp1": t.half_closed,
                "duration_hours": round((t.exit_time - t.entry_time).total_seconds() / 3600, 1) if t.exit_time and t.entry_time else 0,
            })
        trade_df = pd.DataFrame(records)
        trade_file = os.path.join(RESULTS_DIR, "trade_log.csv")
        trade_df.to_csv(trade_file, index=False)
        print(f"  Trade log saved to {trade_file} ({len(records)} trades)")

    def _generate_four_way_comparison(self, v4_metrics: dict):
        """Generate V1 vs V2 vs V3 vs V4 side-by-side comparison."""
        base_dir = os.path.dirname(__file__)
        v1_path = os.path.join(base_dir, "results", "metrics.json")
        v2_path = os.path.join(base_dir, "results_v2", "metrics.json")
        v3_path = os.path.join(base_dir, "results_v3", "metrics.json")

        v1 = v2 = v3 = None
        if os.path.exists(v1_path):
            with open(v1_path) as f:
                v1 = json.load(f)
        if os.path.exists(v2_path):
            with open(v2_path) as f:
                v2 = json.load(f)
        if os.path.exists(v3_path):
            with open(v3_path) as f:
                v3 = json.load(f)

        v4 = v4_metrics

        if not v1 or not v2 or not v3:
            print("  V1/V2/V3 results not found — skipping four-way comparison")
            return

        def fmt_delta(new, old, higher_is_better=True):
            diff = new - old
            if abs(diff) < 0.005:
                return ""
            arrow = "+" if diff > 0 else ""
            tag = ""
            if higher_is_better:
                tag = " ^^" if diff > 0 else " vv"
            else:
                tag = " ^^" if diff < 0 else " vv"
            return f"({arrow}{diff:.2f}{tag})"

        sep = "=" * 105
        col_w = 15

        lines = [
            sep,
            "  BACKTEST COMPARISON: V1 vs V2 vs V3 vs V4",
            sep,
            "",
            f"  {'Metric':<25} {'V1':>{col_w}} {'V2':>{col_w}} {'V3':>{col_w}} {'V4':>{col_w}}   {'V4 vs V1':>12}",
            f"  {'-'*25} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}   {'-'*12}",
            "",
            f"  {'Total Return %':<25} {v1['total_return_pct']:>{col_w-1}.2f}% {v2['total_return_pct']:>{col_w-1}.2f}% {v3['total_return_pct']:>{col_w-1}.2f}% {v4['total_return_pct']:>{col_w-1}.2f}%   {fmt_delta(v4['total_return_pct'], v1['total_return_pct'])}",
            f"  {'Final Equity':<25} ${v1['final_equity']:>{col_w-2},.2f} ${v2['final_equity']:>{col_w-2},.2f} ${v3['final_equity']:>{col_w-2},.2f} ${v4['final_equity']:>{col_w-2},.2f}   {fmt_delta(v4['final_equity'], v1['final_equity'])}",
            f"  {'Max Drawdown %':<25} {v1['max_drawdown_pct']:>{col_w-1}.2f}% {v2['max_drawdown_pct']:>{col_w-1}.2f}% {v3['max_drawdown_pct']:>{col_w-1}.2f}% {v4['max_drawdown_pct']:>{col_w-1}.2f}%   {fmt_delta(v4['max_drawdown_pct'], v1['max_drawdown_pct'], False)}",
            f"  {'Sharpe Ratio':<25} {v1['sharpe_ratio']:>{col_w}.2f} {v2['sharpe_ratio']:>{col_w}.2f} {v3['sharpe_ratio']:>{col_w}.2f} {v4['sharpe_ratio']:>{col_w}.2f}   {fmt_delta(v4['sharpe_ratio'], v1['sharpe_ratio'])}",
            f"  {'Win Rate %':<25} {v1['win_rate_pct']:>{col_w-1}.2f}% {v2['win_rate_pct']:>{col_w-1}.2f}% {v3['win_rate_pct']:>{col_w-1}.2f}% {v4['win_rate_pct']:>{col_w-1}.2f}%   {fmt_delta(v4['win_rate_pct'], v1['win_rate_pct'])}",
            f"  {'Profit Factor':<25} {v1['profit_factor']:>{col_w}.2f} {v2['profit_factor']:>{col_w}.2f} {v3['profit_factor']:>{col_w}.2f} {v4['profit_factor']:>{col_w}.2f}   {fmt_delta(v4['profit_factor'], v1['profit_factor'])}",
            "",
            f"  {'Total Trades':<25} {v1['total_trades']:>{col_w}} {v2['total_trades']:>{col_w}} {v3['total_trades']:>{col_w}} {v4['total_trades']:>{col_w}}",
            f"  {'  Primary':<25} {v1['primary_trades']:>{col_w}} {v2['primary_trades']:>{col_w}} {v3['primary_trades']:>{col_w}} {v4['primary_trades']:>{col_w}}",
            f"  {'  Rolling Adds':<25} {v1['rolling_adds']:>{col_w}} {v2['rolling_adds']:>{col_w}} {v3['rolling_adds']:>{col_w}} {v4['rolling_adds']:>{col_w}}",
        ]

        # Long/short breakdown
        v1_long = v1.get("long_trades", "N/A")
        v1_short = v1.get("short_trades", "N/A")
        v1_long_pnl = v1.get("long_pnl", "N/A")
        v1_short_pnl = v1.get("short_pnl", "N/A")

        if isinstance(v1_long, int):
            lines.append(f"  {'  Long Trades':<25} {v1_long:>{col_w}} {v2['long_trades']:>{col_w}} {v3['long_trades']:>{col_w}} {v4['long_trades']:>{col_w}}")
            lines.append(f"  {'  Short Trades':<25} {v1_short:>{col_w}} {v2['short_trades']:>{col_w}} {v3['short_trades']:>{col_w}} {v4['short_trades']:>{col_w}}")
        else:
            lines.append(f"  {'  Long Trades':<25} {'N/A':>{col_w}} {v2['long_trades']:>{col_w}} {v3['long_trades']:>{col_w}} {v4['long_trades']:>{col_w}}")
            lines.append(f"  {'  Short Trades':<25} {'N/A':>{col_w}} {v2['short_trades']:>{col_w}} {v3['short_trades']:>{col_w}} {v4['short_trades']:>{col_w}}")

        if isinstance(v1_long_pnl, (int, float)):
            lines.append(f"  {'  Long PnL':<25} ${v1_long_pnl:>{col_w-2},.2f} ${v2['long_pnl']:>{col_w-2},.2f} ${v3['long_pnl']:>{col_w-2},.2f} ${v4['long_pnl']:>{col_w-2},.2f}")
            lines.append(f"  {'  Short PnL':<25} ${v1_short_pnl:>{col_w-2},.2f} ${v2['short_pnl']:>{col_w-2},.2f} ${v3['short_pnl']:>{col_w-2},.2f} ${v4['short_pnl']:>{col_w-2},.2f}")
        else:
            lines.append(f"  {'  Long PnL':<25} {'N/A':>{col_w}} ${v2['long_pnl']:>{col_w-2},.2f} ${v3['long_pnl']:>{col_w-2},.2f} ${v4['long_pnl']:>{col_w-2},.2f}")
            lines.append(f"  {'  Short PnL':<25} {'N/A':>{col_w}} ${v2['short_pnl']:>{col_w-2},.2f} ${v3['short_pnl']:>{col_w-2},.2f} ${v4['short_pnl']:>{col_w-2},.2f}")

        lines.extend([
            "",
            f"  {'Avg Win':<25} ${v1['avg_win']:>{col_w-2},.2f} ${v2['avg_win']:>{col_w-2},.2f} ${v3['avg_win']:>{col_w-2},.2f} ${v4['avg_win']:>{col_w-2},.2f}",
            f"  {'Avg Loss':<25} ${v1['avg_loss']:>{col_w-2},.2f} ${v2['avg_loss']:>{col_w-2},.2f} ${v3['avg_loss']:>{col_w-2},.2f} ${v4['avg_loss']:>{col_w-2},.2f}",
            f"  {'Largest Win':<25} ${v1['largest_win']:>{col_w-2},.2f} ${v2['largest_win']:>{col_w-2},.2f} ${v3['largest_win']:>{col_w-2},.2f} ${v4['largest_win']:>{col_w-2},.2f}",
            f"  {'Largest Loss':<25} ${v1['largest_loss']:>{col_w-2},.2f} ${v2['largest_loss']:>{col_w-2},.2f} ${v3['largest_loss']:>{col_w-2},.2f} ${v4['largest_loss']:>{col_w-2},.2f}",
            f"  {'Avg Duration (days)':<25} {v1['avg_trade_duration_days']:>{col_w}.1f} {v2['avg_trade_duration_days']:>{col_w}.1f} {v3['avg_trade_duration_days']:>{col_w}.1f} {v4['avg_trade_duration_days']:>{col_w}.1f}",
            "",
            sep,
            "  V4 — 123 RULE FILTER IMPACT",
            sep,
            "",
            f"  Signals that passed 123 Rule: {v4.get('signals_passed_123', 'N/A')}",
            f"  Signals blocked by 123 Rule:  {v4.get('signals_blocked_123', 'N/A')}",
            "",
            sep,
            "  VERSION CHANGE LOG",
            sep,
            "",
            "  V1 (baseline):",
            "    - MACD(13,34,9) divergence + Key K-line detection",
            "    - TP1 at 1:1.5 R/R, trailing stop at 2.5 ATR",
            "    - Rolling positions on breakouts / MA30 pullbacks",
            "",
            "  V2 changes:",
            "    - SMA50 trend filter: long only above SMA50, short only below",
            "    - TP1 tightened to 1:1 R/R",
            "    - Double divergence minimum (strength >= 2)",
            "",
            "  V3 changes:",
            "    - No shorts in bull market (price > 200 SMA)",
            "    - Rolling adds only near market bottoms (within 30% of 365-day low)",
            "",
            "  V4 changes:",
            "    - 123 Rule confirmation required before any entry:",
            f"      1. Trendline break (swing lookback = {SWING_LOOKBACK} bars)",
            "      2. Failed retest (within 1 ATR, closes in new direction)",
            "      3. Pivot break (exceeds most recent swing point)",
            "",
            sep,
        ])

        comparison = "\n".join(lines)
        comp_path = os.path.join(RESULTS_DIR, "comparison_all.txt")
        with open(comp_path, "w") as f:
            f.write(comparison)
        print(f"\n{comparison}")
        print(f"\n  Four-way comparison saved to {comp_path}")


# ── Main ──────────────────────────────────────────────────────────────

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

    print("\nBacktest V4 complete!")
    return metrics


if __name__ == "__main__":
    main()

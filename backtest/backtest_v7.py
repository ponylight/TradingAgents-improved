"""Backtest V7 — V6 strategy + MMCrypto Bottom-Fishing Mode.

Two modes run simultaneously on shared capital:

MODE 1 (V6 MACD Reversal):
  - V3 base strategy with SMA200 bull filter, near-bottom rolling, SMA50 trend
  - Leveraged: 3x base, 10x if 123 Rule confirmed
  - TP1 at 1:1 R/R, double divergence (strength >= 2)

MODE 2 (MMCrypto Bottom-Fishing):
  Triggers when ALL conditions met:
    - Price within 30% of 365-day low
    - Price below 200 SMA
    - RSI(14) weekly < 30

  When triggered:
    - LONG at 50x leverage, 2% of capital as margin
    - Stop-loss just above liquidation price
    - If stopped: re-enter at current price, another 2% at 50x
    - Max 5 attempts (10% total capital risk). After 5 fails: disable 30 days
    - If price moves 5% in our favor: trail stop to breakeven, start rolling
      with floating profits, pyramid adds, 2x ATR trailing stop
    - No fixed TP — ride until trailing stop hit

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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v7")

# ── V6 / V3 constants ────────────────────────────────────────────────

NEAR_BOTTOM_LOOKBACK = 2190  # 365 days in 4h bars
NEAR_BOTTOM_THRESHOLD = 0.30
SWING_LOOKBACK = 10
BASE_LEVERAGE = 3
MAX_LEVERAGE = 10

# ── MMCrypto Bottom-Fishing constants ─────────────────────────────────

MM_LEVERAGE = 50
MM_MARGIN_PCT = 0.02        # 2% of capital per attempt
MM_MAX_ATTEMPTS = 5         # max re-entries before cooldown
MM_COOLDOWN_BARS = 180      # 30 days in 4h bars (30 * 24 / 4)
MM_PROFIT_THRESHOLD = 0.05  # 5% move to enter profit mode
MM_TRAIL_ATR_MULT = 2.0     # 2x ATR trailing stop in profit mode
MM_PYRAMID_ATR_MULT = 1.5   # pyramid every 1.5 ATR
MM_PYRAMID_PROFIT_PCT = 0.30  # use 30% of unrealized for pyramid margin
MM_MAX_PYRAMIDS = 10
MM_LIQ_BUFFER = 1.002       # SL at 0.2% above liquidation price
MM_RSI_THRESHOLD = 30       # weekly RSI(14) must be below this
MM_MIN_MARGIN = 20.0        # minimum $20 margin to enter


# ── Data classes ──────────────────────────────────────────────────────

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


@dataclass
class MMTrade:
    """Mode 2 (MMCrypto Bottom-Fishing) trade."""
    entry_price: float
    entry_idx: int
    entry_time: pd.Timestamp
    size: float          # BTC (leveraged)
    margin: float        # capital backing
    leverage: float = MM_LEVERAGE
    liquidation_price: float = 0.0
    stop_loss: float = 0.0
    is_pyramid: bool = False
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


# ── Weekly RSI computation ────────────────────────────────────────────

def compute_weekly_rsi(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute weekly RSI(14) from 4h data, forward-filled back to 4h."""
    ts_df = df.set_index("timestamp")["close"].copy()

    # Resample to weekly closes
    weekly_close = ts_df.resample("W").last().dropna()

    # RSI calculation
    delta = weekly_close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)

    # Forward-fill back to 4h resolution
    rsi_4h = rsi.reindex(ts_df.index, method="ffill")
    return rsi_4h.values


# ── Swing detection (from V4/V6) ─────────────────────────────────────

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


# ── 123 Rule (from V4/V6) ────────────────────────────────────────────

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


# ── Leverage determination (V6) ──────────────────────────────────────

def determine_leverage(signal_idx, side, closes, highs, lows, atr_vals,
                       swing_highs, swing_lows):
    has_123 = check_123_rule(signal_idx, side, closes, highs, lows, atr_vals,
                             swing_highs, swing_lows)
    if has_123:
        return MAX_LEVERAGE, has_123
    else:
        return BASE_LEVERAGE, has_123


# ── Regime helpers (V3) ──────────────────────────────────────────────

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


# ── V3 signal generation (identical to V6) ───────────────────────────

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


def compute_rolling_signals_v3(df, active_trades, current_idx):
    signals = []
    if not active_trades:
        return signals
    close = df["close"].iloc[current_idx]
    ma30 = df["ma30"].iloc[current_idx]
    atr = df["atr"].iloc[current_idx]
    sma50 = df["sma50"].iloc[current_idx]
    sma200 = df["sma200"].iloc[current_idx]
    closes = df["close"].values
    if np.isnan(ma30) or np.isnan(atr) or np.isnan(sma50) or np.isnan(sma200):
        return signals
    near_bottom = is_near_market_bottom(closes, current_idx)
    if not near_bottom:
        return signals
    bull = is_bull_market(close, sma200)
    for trade in active_trades:
        if trade.closed:
            continue
        if trade.side == "long":
            unrealized_pnl_pct = (close - trade.entry_price) / trade.entry_price
        else:
            unrealized_pnl_pct = (trade.entry_price - close) / trade.entry_price
        if unrealized_pnl_pct < 0.03:
            continue
        if trade.side == "long":
            if close <= sma50:
                continue
            near_ma30 = abs(close - ma30) / close < 0.005
            above_ma30 = close > ma30
            recent_high = df["high"].iloc[max(0, current_idx - 20):current_idx].max()
            breakout = close > recent_high
            avg_vol = df["volume"].iloc[max(0, current_idx - 20):current_idx].mean()
            vol_confirm = df["volume"].iloc[current_idx] > avg_vol * 1.5
            if (near_ma30 and above_ma30) or (breakout and vol_confirm):
                sl = close - atr * 1.5
                signals.append(Signal(
                    idx=current_idx, side="long", entry_price=close,
                    stop_loss=sl, divergence_strength=1, signal_type="rolling_add",
                ))
        elif trade.side == "short":
            if bull:
                continue
            if close >= sma50:
                continue
            near_ma30 = abs(close - ma30) / close < 0.005
            below_ma30 = close < ma30
            recent_low = df["low"].iloc[max(0, current_idx - 20):current_idx].min()
            breakdown = close < recent_low
            avg_vol = df["volume"].iloc[max(0, current_idx - 20):current_idx].mean()
            vol_confirm = df["volume"].iloc[current_idx] > avg_vol * 1.5
            if (near_ma30 and below_ma30) or (breakdown and vol_confirm):
                sl = close + atr * 1.5
                signals.append(Signal(
                    idx=current_idx, side="short", entry_price=close,
                    stop_loss=sl, divergence_strength=1, signal_type="rolling_add",
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

        # Mode 2 (MMCrypto) state
        self.mm_base_trade: MMTrade | None = None
        self.mm_pyramids: list[MMTrade] = []
        self.mm_closed_trades: list[MMTrade] = []
        self.mm_attempt_count = 0
        self.mm_cooldown_until_idx = -1
        self.mm_in_profit_mode = False
        self.mm_master_trail = 0.0
        self.mm_last_pyramid_price = 0.0
        self.mm_pyramid_count = 0
        self.mm_stats = {
            "total_entries": 0,
            "total_stops_phase_a": 0,
            "total_pyramids": 0,
            "successful_rides": 0,
            "cooldowns_triggered": 0,
            "best_ride_pnl": 0.0,
            "worst_stop_pnl": 0.0,
        }

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_macd(df, fast=13, slow=34, signal=9)
        df["atr"] = compute_atr(df, period=14)
        df["ma30"] = compute_ma(df, period=30)
        df["sma50"] = compute_ma(df, period=50)
        df["sma200"] = compute_ma(df, period=200)
        df["weekly_rsi"] = compute_weekly_rsi(df, period=14)
        return df

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self) -> dict:
        print(f"[V7] Running backtest on {len(self.df)} candles...")
        print(f"  Date range: {self.df['timestamp'].iloc[0]} to {self.df['timestamp'].iloc[-1]}")
        print(f"  Mode 1: V6 MACD Reversal (3x/10x leverage)")
        print(f"  Mode 2: MMCrypto Bottom-Fishing (50x, max 5 attempts)")
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

            # Mode 1: rolling adds
            if self.active_trades:
                rolling_sigs = compute_rolling_signals_v3(self.df, self.active_trades, i)
                for sig in rolling_sigs:
                    self._execute_rolling_add(sig, i, ts, closes, highs_arr, lows_arr, atr_arr)

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
                "num_mode2": (1 if self.mm_base_trade else 0) + len(self.mm_pyramids),
            })

        # Close remaining positions at end
        last_price = self.df["close"].iloc[-1]
        last_ts = self.df["timestamp"].iloc[-1]
        for trade in list(self.active_trades):
            self._close_trade(trade, last_price, len(self.df) - 1, last_ts, reason="end_of_backtest")
        if self.mm_base_trade or self.mm_pyramids:
            self._close_all_mm(last_price, len(self.df) - 1, last_ts, reason="end_of_backtest")

        return self._compute_metrics()

    # ── Mode 1: V6 entry/exit (identical to V6) ──────────────────────

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
        )
        self.active_trades.append(trade)
        self.trades.append(trade)

    def _execute_rolling_add(self, signal, idx, ts, closes, highs, lows, atr_vals):
        base_trades = [t for t in self.active_trades if t.side == signal.side and not t.is_rolling and not t.closed]
        if not base_trades:
            return
        base = base_trades[0]
        base_key = base.entry_idx
        add_count = self.rolling_add_counts.get(base_key, 0)
        if add_count >= self.config.max_rolling_adds:
            return
        base_unleveraged_size = base.original_size / base.leverage
        add_base_size = base_unleveraged_size * (self.config.pyramid_decay ** (add_count + 1))
        base_cost = add_base_size * signal.entry_price
        if base_cost > self.capital * 0.2:
            return
        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            return
        leverage, _ = determine_leverage(
            idx, signal.side, closes, highs, lows, atr_vals,
            self.swing_highs, self.swing_lows
        )
        leveraged_size = add_base_size * leverage
        margin = add_base_size * signal.entry_price
        notional = leveraged_size * signal.entry_price
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
            entry_idx=idx, entry_time=ts,
            size=leveraged_size, stop_loss=signal.stop_loss,
            take_profit_1=tp1, initial_risk=risk,
            leverage=leverage, margin=margin,
            liquidation_price=liq_price, original_size=leveraged_size,
            is_rolling=True, trailing_stop=signal.stop_loss,
        )
        self.active_trades.append(trade)
        self.trades.append(trade)
        self.rolling_add_counts[base_key] = add_count + 1

    def _manage_positions(self, idx, high, low, close, ts):
        for trade in list(self.active_trades):
            if trade.closed:
                continue
            if trade.side == "long":
                if low <= trade.liquidation_price:
                    self.leverage_stats["liquidations"] += 1
                    remaining_margin = trade.margin * (trade.size / trade.original_size)
                    self.leverage_stats["liquidation_loss"] += remaining_margin
                    self._close_trade(trade, trade.liquidation_price, idx, ts, reason="liquidation")
                    continue
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
                if high >= trade.liquidation_price:
                    self.leverage_stats["liquidations"] += 1
                    remaining_margin = trade.margin * (trade.size / trade.original_size)
                    self.leverage_stats["liquidation_loss"] += remaining_margin
                    self._close_trade(trade, trade.liquidation_price, idx, ts, reason="liquidation")
                    continue
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

    def _calc_unrealized(self, current_price):
        total = 0.0
        for trade in self.active_trades:
            if trade.side == "long":
                total += (current_price - trade.entry_price) * trade.size
            else:
                total += (trade.entry_price - current_price) * trade.size
        return total

    # ── Mode 2: MMCrypto Bottom-Fishing ───────────────────────────────

    def _check_mm_conditions(self, idx: int, price: float) -> bool:
        """Check if all Mode 2 entry conditions are met."""
        if idx < 205:
            return False
        # On cooldown?
        if idx < self.mm_cooldown_until_idx:
            return False
        # Price within 30% of 365-day low
        closes = self.df["close"].values
        if not is_near_market_bottom(closes, idx):
            return False
        # Price below 200 SMA
        sma200 = self.df["sma200"].iloc[idx]
        if np.isnan(sma200) or price >= sma200:
            return False
        # Weekly RSI(14) < 30
        weekly_rsi = self.df["weekly_rsi"].iloc[idx]
        if np.isnan(weekly_rsi) or weekly_rsi >= MM_RSI_THRESHOLD:
            return False
        return True

    def _check_mm_entry(self, idx: int, price: float, ts: pd.Timestamp, atr_val: float):
        """Check and execute initial Mode 2 entry."""
        if not self._check_mm_conditions(idx, price):
            return
        self._enter_mm_trade(idx, price, ts, is_reentry=False)

    def _enter_mm_trade(self, idx: int, price: float, ts: pd.Timestamp,
                        is_reentry: bool = False):
        """Enter a Mode 2 bottom-fish trade at current price."""
        margin = self.capital * MM_MARGIN_PCT
        if margin < MM_MIN_MARGIN:
            return
        notional = margin * MM_LEVERAGE
        size_btc = notional / price

        # Commission on notional
        commission = notional * self.config.commission_rate
        self.capital -= commission

        # Liquidation: entry * (1 - 1/leverage)
        liq_price = price * (1 - 1 / MM_LEVERAGE)
        # Stop just above liquidation
        stop_loss = liq_price * MM_LIQ_BUFFER

        trade = MMTrade(
            entry_price=price, entry_idx=idx, entry_time=ts,
            size=size_btc, margin=margin, leverage=MM_LEVERAGE,
            liquidation_price=liq_price, stop_loss=stop_loss,
        )
        self.mm_base_trade = trade
        self.mm_in_profit_mode = False
        self.mm_master_trail = 0.0
        self.mm_last_pyramid_price = price
        self.mm_pyramid_count = 0
        self.mm_stats["total_entries"] += 1

        action = "re-entry" if is_reentry else "initial entry"
        attempt = self.mm_attempt_count + 1
        print(f"    [MM] {action} #{attempt} at ${price:,.0f} | "
              f"margin=${margin:,.0f} | 50x | liq=${liq_price:,.0f} | SL=${stop_loss:,.0f}")

    def _manage_mm_positions(self, idx: int, high: float, low: float,
                             close: float, ts: pd.Timestamp, atr_val: float):
        """Manage Mode 2 positions each bar."""
        if self.mm_base_trade is None:
            return

        base = self.mm_base_trade

        if not self.mm_in_profit_mode:
            # ── Phase A: trying to catch the bottom ──
            # Check liquidation
            if low <= base.liquidation_price:
                self._close_mm_single(base, base.liquidation_price, idx, ts, "liquidation")
                self._handle_mm_stop(idx, close, ts)
                return

            # Check stop-loss (near liquidation)
            if low <= base.stop_loss:
                self._close_mm_single(base, base.stop_loss, idx, ts, "stop_loss")
                self._handle_mm_stop(idx, close, ts)
                return

            # Check if we've reached +5% → enter profit mode
            gain_pct = (close - base.entry_price) / base.entry_price
            if gain_pct >= MM_PROFIT_THRESHOLD:
                self.mm_in_profit_mode = True
                # Trail to breakeven
                self.mm_master_trail = base.entry_price
                self.mm_last_pyramid_price = close
                print(f"    [MM] +{gain_pct*100:.1f}% reached at ${close:,.0f} — "
                      f"PROFIT MODE active | trail=${self.mm_master_trail:,.0f}")

        if self.mm_in_profit_mode:
            # ── Phase B: riding the trend ──
            # Update master trailing stop (2x ATR, only ratchets up)
            if not np.isnan(atr_val) and atr_val > 0:
                new_trail = close - MM_TRAIL_ATR_MULT * atr_val
                # Never move below breakeven
                new_trail = max(new_trail, base.entry_price)
                self.mm_master_trail = max(self.mm_master_trail, new_trail)

            # Check if trailing stop hit (applies to ALL mm trades)
            if low <= self.mm_master_trail:
                # Close everything — this is a successful ride
                total_pnl = self._close_all_mm(self.mm_master_trail, idx, ts, reason="trailing_stop")
                self.mm_stats["successful_rides"] += 1
                self.mm_stats["best_ride_pnl"] = max(self.mm_stats["best_ride_pnl"], total_pnl)
                # Reset attempt count on success (we caught the bottom!)
                self.mm_attempt_count = 0
                return

            # Check for pyramid add (using floating profits)
            if not np.isnan(atr_val) and atr_val > 0:
                dist_from_last = close - self.mm_last_pyramid_price
                if dist_from_last >= MM_PYRAMID_ATR_MULT * atr_val:
                    if self.mm_pyramid_count < MM_MAX_PYRAMIDS:
                        self._add_mm_pyramid(idx, close, ts)

    def _handle_mm_stop(self, idx: int, close: float, ts: pd.Timestamp):
        """Handle a Mode 2 stop-out in Phase A (attempting to catch bottom)."""
        self.mm_attempt_count += 1
        self.mm_stats["total_stops_phase_a"] += 1

        if self.mm_attempt_count >= MM_MAX_ATTEMPTS:
            # Max attempts reached — cooldown
            self.mm_cooldown_until_idx = idx + MM_COOLDOWN_BARS
            self.mm_stats["cooldowns_triggered"] += 1
            print(f"    [MM] 5 attempts exhausted — 30-day cooldown until bar {self.mm_cooldown_until_idx}")
            self.mm_attempt_count = 0
            self.mm_base_trade = None
        else:
            # Re-enter immediately at current close
            self.mm_base_trade = None
            self._enter_mm_trade(idx, close, ts, is_reentry=True)

    def _add_mm_pyramid(self, idx: int, price: float, ts: pd.Timestamp):
        """Add a pyramid position using floating profits."""
        # Calculate total unrealized from all active MM trades
        unrealized = self._calc_mm_unrealized(price)
        if unrealized < 50:  # minimum $50 unrealized for pyramiding
            return

        # Use portion of floating profits as margin
        pyramid_margin = unrealized * MM_PYRAMID_PROFIT_PCT
        if pyramid_margin < MM_MIN_MARGIN:
            return

        notional = pyramid_margin * MM_LEVERAGE
        size_btc = notional / price
        commission = notional * self.config.commission_rate
        self.capital -= commission

        liq_price = price * (1 - 1 / MM_LEVERAGE)

        trade = MMTrade(
            entry_price=price, entry_idx=idx, entry_time=ts,
            size=size_btc, margin=pyramid_margin, leverage=MM_LEVERAGE,
            liquidation_price=liq_price, stop_loss=self.mm_master_trail,
            is_pyramid=True,
        )
        self.mm_pyramids.append(trade)
        self.mm_last_pyramid_price = price
        self.mm_pyramid_count += 1
        self.mm_stats["total_pyramids"] += 1

        print(f"    [MM] Pyramid #{self.mm_pyramid_count} at ${price:,.0f} | "
              f"margin=${pyramid_margin:,.0f} (from floating) | size={size_btc:.4f} BTC")

    def _close_mm_single(self, trade: MMTrade, exit_price: float, idx: int,
                         ts: pd.Timestamp, reason: str):
        """Close a single MM trade and credit P&L."""
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

    def _close_all_mm(self, exit_price: float, idx: int, ts: pd.Timestamp,
                      reason: str) -> float:
        """Close base trade + all pyramids. Returns total P&L."""
        total_pnl = 0.0
        if self.mm_base_trade and not self.mm_base_trade.closed:
            self._close_mm_single(self.mm_base_trade, exit_price, idx, ts, reason)
            total_pnl += self.mm_base_trade.pnl
        for pyr in self.mm_pyramids:
            if not pyr.closed:
                self._close_mm_single(pyr, exit_price, idx, ts, reason)
                total_pnl += pyr.pnl
        n_closed = 1 + len(self.mm_pyramids)
        print(f"    [MM] Closed {n_closed} positions at ${exit_price:,.0f} | "
              f"total PnL=${total_pnl:,.2f} | reason={reason}")
        # Reset state
        self.mm_base_trade = None
        self.mm_pyramids = []
        self.mm_in_profit_mode = False
        self.mm_master_trail = 0.0
        self.mm_pyramid_count = 0
        return total_pnl

    def _calc_mm_unrealized(self, current_price: float) -> float:
        total = 0.0
        if self.mm_base_trade and not self.mm_base_trade.closed:
            total += (current_price - self.mm_base_trade.entry_price) * self.mm_base_trade.size
        for pyr in self.mm_pyramids:
            if not pyr.closed:
                total += (current_price - pyr.entry_price) * pyr.size
        return total

    # ── Metrics & Reporting ───────────────────────────────────────────

    def _compute_metrics(self) -> dict:
        all_mode1 = self.closed_trades
        all_mode2 = self.mm_closed_trades

        if not all_mode1 and not all_mode2:
            return {"version": "v7", "error": "No trades executed"}

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

        # Combined P&L
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
        m2_pyramids = [t for t in all_mode2 if t.is_pyramid]
        m2_base = [t for t in all_mode2 if not t.is_pyramid]
        m2_durations = []
        for t in all_mode2:
            if t.entry_time is not None and t.exit_time is not None:
                m2_durations.append((t.exit_time - t.entry_time).total_seconds() / 3600)
        m2_avg_dur_h = np.mean(m2_durations) if m2_durations else 0

        avg_leverage_m1 = np.mean([t.leverage for t in all_mode1]) if all_mode1 else 0

        return {
            "version": "v7",
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

            # Mode 1 (V6 MACD)
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

            # Mode 2 (MMCrypto)
            "m2_total_trades": len(all_mode2),
            "m2_base_entries": len(m2_base),
            "m2_pyramids": len(m2_pyramids),
            "m2_total_pnl": round(sum(m2_pnls), 2),
            "m2_win_rate_pct": round(m2_win_rate, 2),
            "m2_avg_duration_hours": round(m2_avg_dur_h, 1),
            "m2_avg_duration_days": round(m2_avg_dur_h / 24, 1),
            "m2_total_entries": self.mm_stats["total_entries"],
            "m2_stops_phase_a": self.mm_stats["total_stops_phase_a"],
            "m2_successful_rides": self.mm_stats["successful_rides"],
            "m2_cooldowns_triggered": self.mm_stats["cooldowns_triggered"],
            "m2_total_pyramid_adds": self.mm_stats["total_pyramids"],
            "m2_best_ride_pnl": round(self.mm_stats["best_ride_pnl"], 2),
            "m2_worst_stop_pnl": round(self.mm_stats["worst_stop_pnl"], 2),

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
            "  BTC/USDT BACKTEST — V7",
            "  Mode 1: V6 MACD Reversal (3x/10x)",
            "  Mode 2: MMCrypto Bottom-Fishing (50x, max 5 attempts)",
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
            "  MODE 1: V6 MACD REVERSAL",
            sep,
            "",
            f"  Total Trades:    {m['m1_total_trades']}",
            f"    Primary:       {m['m1_primary']}",
            f"    Rolling Adds:  {m['m1_rolling_adds']}",
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
            sep,
            "  MODE 2: MMCRYPTO BOTTOM-FISHING",
            sep,
            "",
            f"  Total Trades:    {m['m2_total_trades']}",
            f"    Base Entries:   {m['m2_base_entries']}",
            f"    Pyramid Adds:  {m['m2_pyramids']}",
            f"  Total PnL:       ${m['m2_total_pnl']:+,.2f}",
            f"  Win Rate:        {m['m2_win_rate_pct']:.1f}%",
            f"  Avg Duration:    {m['m2_avg_duration_days']:.1f} days",
            "",
            f"  Entry Attempts:  {m['m2_total_entries']}",
            f"  Phase A Stops:   {m['m2_stops_phase_a']}  (stopped before +5%)",
            f"  Successful Rides:{m['m2_successful_rides']}  (reached profit mode)",
            f"  Cooldowns:       {m['m2_cooldowns_triggered']}  (5 fails → 30d disable)",
            f"  Pyramid Adds:    {m['m2_total_pyramid_adds']}",
            "",
            f"  Best Ride PnL:   ${m['m2_best_ride_pnl']:+,.2f}",
            f"  Worst Stop PnL:  ${m['m2_worst_stop_pnl']:+,.2f}",
            "",
            "  Mode 2 Rules:",
            "    Entry: price within 30% of 365d low + below SMA200 + wRSI<30",
            "    50x leverage, 2% capital as margin per attempt",
            "    SL just above liquidation price (~2% from entry)",
            "    Max 5 attempts (10% capital risk), then 30-day cooldown",
            f"    Profit mode at +{MM_PROFIT_THRESHOLD*100:.0f}%: trail to breakeven",
            f"    {MM_TRAIL_ATR_MULT:.0f}x ATR trailing stop, pyramid with floating profits",
            "    No fixed TP — ride until trailing stop hit",
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

        # ── Equity curve ──
        ax1.plot(timestamps, equity, color="#E91E63", linewidth=1.4, label="V7 Equity (Mode 1 + Mode 2)")
        ax1.axhline(y=self.config.initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")

        # Mode 1 trade markers
        for trade in self.closed_trades[:80]:
            if trade.close_reason == "liquidation":
                mc = "#9C27B0"
            elif trade.pnl > 0:
                mc = "#4CAF50"
            else:
                mc = "#F44336"
            if trade.entry_time in timestamps.values:
                eidx = timestamps[timestamps == trade.entry_time].index
                if len(eidx) > 0:
                    ms = 4 + trade.leverage
                    ax1.plot(trade.entry_time, equity[eidx[0]],
                             marker="^" if trade.side == "long" else "v",
                             color=mc, markersize=ms, alpha=0.6)

        # Mode 2 trade markers (diamonds)
        for trade in self.mm_closed_trades:
            if trade.pnl > 0:
                mc = "#FF9800"  # orange for MM wins
            else:
                mc = "#795548"  # brown for MM losses
            if trade.entry_time in timestamps.values:
                eidx = timestamps[timestamps == trade.entry_time].index
                if len(eidx) > 0:
                    ms = 8 if not trade.is_pyramid else 5
                    ax1.plot(trade.entry_time, equity[eidx[0]],
                             marker="D", color=mc, markersize=ms, alpha=0.8)

        ax1.set_title("BTC/USDT V7 — Equity Curve\n"
                       "(Mode 1: V6 MACD 3x/10x | Mode 2: MMCrypto Bottom-Fish 50x)",
                       fontsize=13, fontweight="bold")
        ax1.set_ylabel("Equity ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # ── Drawdown ──
        ax2.fill_between(timestamps, -drawdown_pct, 0, color="#F44336", alpha=0.3)
        ax2.plot(timestamps, -drawdown_pct, color="#F44336", linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Drawdown", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # ── Mode 2 activity timeline ──
        # Show MM entries as vertical spans
        mm_entries = []
        mm_exits = []
        for t in self.mm_closed_trades:
            if not t.is_pyramid:
                mm_entries.append((t.entry_time, t.exit_time, t.pnl > 0))

        for entry_t, exit_t, is_win in mm_entries:
            color = "#FF9800" if is_win else "#795548"
            ax3.axvspan(entry_t, exit_t, alpha=0.3, color=color)

        # Show Mode 1 leverage scatter
        for trade in self.closed_trades:
            color = "#2196F3" if trade.leverage <= 3 else "#F44336"
            ax3.scatter(trade.entry_time, trade.leverage, c=color, s=20, alpha=0.7, zorder=3)

        ax3.axhline(y=3, color="#2196F3", linestyle="--", alpha=0.3, label="3x (M1 base)")
        ax3.axhline(y=10, color="#F44336", linestyle="--", alpha=0.3, label="10x (M1 123)")
        ax3.axhline(y=50, color="#FF9800", linestyle="--", alpha=0.3, label="50x (M2 bottom)")

        ax3.set_ylabel("Leverage")
        ax3.set_title("Mode 1 Leverage + Mode 2 Activity (orange/brown spans)", fontsize=11)
        ax3.set_ylim(0, 55)
        ax3.set_yticks([0, 3, 10, 50])
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
        # Mode 1 trades
        for t in self.closed_trades:
            records.append({
                "mode": "M1",
                "side": t.side,
                "type": "rolling" if t.is_rolling else "primary",
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
        # Mode 2 trades
        for t in self.mm_closed_trades:
            records.append({
                "mode": "M2",
                "side": "long",
                "type": "pyramid" if t.is_pyramid else "base",
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

    def _generate_comparison(self, v7_metrics: dict):
        """Generate V6 vs V7 comparison."""
        v6_path = os.path.join(os.path.dirname(__file__), "results_v6", "metrics.json")
        v6 = None
        if os.path.exists(v6_path):
            with open(v6_path) as f:
                v6 = json.load(f)

        if v6 is None:
            print("  V6 results not found — skipping comparison")
            return

        v7 = v7_metrics

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

        sep = "=" * 95
        cw = 20

        lines = [
            sep,
            "  BACKTEST COMPARISON: V6 (MACD 3x/10x) vs V7 (V6 + MMCrypto Bottom-Fishing)",
            sep,
            "",
            f"  {'Metric':<28} {'V6':>{cw}} {'V7':>{cw}}   {'Delta':>15}",
            f"  {'-'*28} {'-'*cw} {'-'*cw}   {'-'*15}",
            "",
            f"  {'Total Return %':<28} {v6['total_return_pct']:>{cw-1}.2f}% {v7['total_return_pct']:>{cw-1}.2f}%   {fmt_delta(v7['total_return_pct'], v6['total_return_pct'])}",
            f"  {'Final Equity':<28} ${v6['final_equity']:>{cw-2},.2f} ${v7['final_equity']:>{cw-2},.2f}   {fmt_mult(v7['final_equity'], v6['final_equity'])}",
            f"  {'Max Drawdown %':<28} {v6['max_drawdown_pct']:>{cw-1}.2f}% {v7['max_drawdown_pct']:>{cw-1}.2f}%   {fmt_delta(v7['max_drawdown_pct'], v6['max_drawdown_pct'], False)}",
            f"  {'Sharpe Ratio':<28} {v6['sharpe_ratio']:>{cw}.2f} {v7['sharpe_ratio']:>{cw}.2f}   {fmt_delta(v7['sharpe_ratio'], v6['sharpe_ratio'])}",
            f"  {'Win Rate %':<28} {v6['win_rate_pct']:>{cw-1}.2f}% {v7['win_rate_pct']:>{cw-1}.2f}%   {fmt_delta(v7['win_rate_pct'], v6['win_rate_pct'])}",
            f"  {'Profit Factor':<28} {v6['profit_factor']:>{cw}.2f} {v7['profit_factor']:>{cw}.2f}   {fmt_delta(v7['profit_factor'], v6['profit_factor'])}",
            "",
            f"  {'Total Trades':<28} {v6['total_trades']:>{cw}} {v7['total_trades']:>{cw}}",
        ]

        # V7 mode breakdown
        lines += [
            "",
            sep,
            "  V7 MODE BREAKDOWN",
            sep,
            "",
            f"  {'':28} {'Mode 1 (MACD)':>{cw}} {'Mode 2 (MM)':>{cw}}",
            f"  {'':28} {'-'*cw} {'-'*cw}",
            f"  {'Total Trades':<28} {v7['m1_total_trades']:>{cw}} {v7['m2_total_trades']:>{cw}}",
            f"  {'Total PnL':<28} ${v7['m1_total_pnl']:>{cw-2},.2f} ${v7['m2_total_pnl']:>{cw-2},.2f}",
            f"  {'Win Rate %':<28} {v7['m1_win_rate_pct']:>{cw-1}.2f}% {v7['m2_win_rate_pct']:>{cw-1}.2f}%",
            f"  {'Avg Duration (days)':<28} {v7['m1_avg_duration_days']:>{cw}.1f} {v7['m2_avg_duration_days']:>{cw}.1f}",
        ]

        # Mode 2 details
        lines += [
            "",
            sep,
            "  MODE 2 DETAILS (MMCrypto Bottom-Fishing)",
            sep,
            "",
            f"  Entry Attempts:          {v7['m2_total_entries']}",
            f"  Phase A Stops:           {v7['m2_stops_phase_a']}  (stopped before +5%)",
            f"  Successful Rides:        {v7['m2_successful_rides']}  (caught the bottom)",
            f"  Cooldowns Triggered:     {v7['m2_cooldowns_triggered']}",
            f"  Pyramid Adds:            {v7['m2_total_pyramid_adds']}",
            f"  Best Ride PnL:           ${v7['m2_best_ride_pnl']:+,.2f}",
            f"  Worst Stop PnL:          ${v7['m2_worst_stop_pnl']:+,.2f}",
        ]

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
            "  V7 = V6 + MMCrypto Bottom-Fishing:",
            "    - Mode 1: identical to V6",
            "    - Mode 2: 50x bottom-fishing when price in oversold zone",
            "      * Entry: within 30% of 365d low + below SMA200 + wRSI<30",
            "      * 2% capital per attempt, max 5 attempts (10% risk)",
            "      * 30-day cooldown after 5 consecutive fails",
            "      * Profit mode at +5%: breakeven trail, pyramid adds",
            "      * 2x ATR trailing stop, no fixed TP",
            "",
            sep,
        ]

        comparison = "\n".join(lines)
        comp_path = os.path.join(RESULTS_DIR, "comparison_v6_v7.txt")
        with open(comp_path, "w") as f:
            f.write(comparison)
        print(f"\n{comparison}")
        print(f"\n  Comparison saved to {comp_path}")


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

    print("\nBacktest V7 complete!")
    return metrics


if __name__ == "__main__":
    main()

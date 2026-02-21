"""Backtest V6 — V3 base strategy + SIMPLIFIED LEVERAGE (3x / 10x).

Base strategy: V3 (MACD Reversal with SMA200 bull filter + near-bottom rolling).

Leverage rules (simplified from V5):
1. Base leverage: 3x for all entries (no 123 Rule)
2. If 123 Rule confirmed (trendline break + failed retest + pivot break): 10x
   - Removed the "within 20% of 365-day low" requirement for 10x
   - Removed the 5x tier entirely

Mechanics:
- Leverage multiplies the BTC position size (e.g. $1000 margin at 3x = $3000 notional)
- Liquidation check: if unrealised loss exceeds margin (notional / leverage), force close
- Starting capital: $10,000, commission: 0.1% on notional

Inherited from V3:
- No shorts in bull market (price > SMA200)
- Rolling positions only near market bottoms (within 30% of 365-day low)
- 50 SMA trend filter, 1:1 TP1, double divergence (strength >= 2)
"""

import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.fetch_data import fetch_ohlcv
from backtest.indicators import compute_macd, compute_atr, compute_ma

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v6")

# V3 constants
NEAR_BOTTOM_LOOKBACK = 2190  # 365 days in 4h bars
NEAR_BOTTOM_THRESHOLD = 0.30  # for rolling adds (V3)

# V6 constants — simplified leverage
SWING_LOOKBACK = 10  # for 123 Rule swing detection
BASE_LEVERAGE = 3
MAX_LEVERAGE = 10  # 123 Rule confirmed = 10x (no near-bottom requirement)


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class Trade:
    side: str
    entry_price: float
    entry_idx: int
    entry_time: pd.Timestamp
    size: float  # BTC size (leveraged)
    stop_loss: float
    take_profit_1: float
    initial_risk: float
    leverage: float = 1.0
    margin: float = 0.0  # capital backing (notional / leverage)
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
class Signal:
    idx: int
    side: str
    entry_price: float
    stop_loss: float
    divergence_strength: float
    signal_type: str = "primary"


# ── Swing detection (from V4) ────────────────────────────────────────

def detect_swings(highs: np.ndarray, lows: np.ndarray, lookback: int = SWING_LOOKBACK):
    """Detect swing highs and swing lows using bidirectional lookback."""
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


# ── 123 Rule check (from V4) ─────────────────────────────────────────

def check_123_rule(signal_idx: int, side: str, closes: np.ndarray,
                   highs: np.ndarray, lows: np.ndarray, atr_vals: np.ndarray,
                   swing_highs: list, swing_lows: list) -> bool:
    if side == "long":
        return _check_123_long(signal_idx, closes, highs, lows, atr_vals, swing_highs, swing_lows)
    else:
        return _check_123_short(signal_idx, closes, highs, lows, atr_vals, swing_highs, swing_lows)


def _check_123_long(signal_idx: int, closes: np.ndarray, highs: np.ndarray,
                    lows: np.ndarray, atr_vals: np.ndarray,
                    swing_highs: list, swing_lows: list) -> bool:
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


def _check_123_short(signal_idx: int, closes: np.ndarray, highs: np.ndarray,
                     lows: np.ndarray, atr_vals: np.ndarray,
                     swing_highs: list, swing_lows: list) -> bool:
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


# ── Leverage determination (V6 simplified) ────────────────────────────

def determine_leverage(signal_idx: int, side: str, closes: np.ndarray,
                       highs: np.ndarray, lows: np.ndarray, atr_vals: np.ndarray,
                       swing_highs: list, swing_lows: list) -> tuple[int, bool]:
    """Determine leverage for a trade.

    V6 simplified: 3x base, 10x if 123 Rule confirmed.
    Returns (leverage, has_123).
    """
    has_123 = check_123_rule(signal_idx, side, closes, highs, lows, atr_vals,
                             swing_highs, swing_lows)

    if has_123:
        return MAX_LEVERAGE, has_123
    else:
        return BASE_LEVERAGE, has_123


# ── Regime helpers (from V3) ──────────────────────────────────────────

def is_bull_market(close: float, sma200: float) -> bool:
    if np.isnan(sma200):
        return False
    return close > sma200


def is_near_market_bottom(closes: np.ndarray, current_idx: int) -> bool:
    """Within 30% of 365-day low (V3 rolling add filter)."""
    lookback_start = max(0, current_idx - NEAR_BOTTOM_LOOKBACK)
    window = closes[lookback_start:current_idx + 1]
    if len(window) == 0:
        return False
    low_365 = np.min(window)
    return closes[current_idx] <= low_365 * (1.0 + NEAR_BOTTOM_THRESHOLD)


# ── Strategy: V3 signal generation (unchanged) ───────────────────────

def compute_signals_v3(df: pd.DataFrame) -> list[Signal]:
    """V3 signal generation — same as backtest_v3.py."""
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
                            sl = lows[i] - atr[i] * 1.5 if not np.isnan(atr[i]) else lows[i] * 0.97
                            signals.append(Signal(
                                idx=i, side="long", entry_price=closes[i],
                                stop_loss=sl, divergence_strength=strength,
                            ))
                            last_signal_idx = i

        # ── SHORT SIGNAL ──
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


def _check_bottom_divergence(
    current_idx: int, hist: np.ndarray, lows: np.ndarray, lookback: int = 80
) -> tuple[float, float] | None:
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


def _check_top_divergence(
    current_idx: int, hist: np.ndarray, highs: np.ndarray, lookback: int = 80
) -> tuple[float, float] | None:
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


def compute_rolling_signals_v3(
    df: pd.DataFrame, active_trades: list[Trade], current_idx: int
) -> list[Signal]:
    """Rolling add-on signals with V3 near-bottom filter."""
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
        self.trades: list[Trade] = []
        self.active_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.rolling_add_counts: dict[int, int] = {}

        # Pre-compute swings for 123 Rule / leverage determination
        self.swing_highs, self.swing_lows = detect_swings(
            self.df["high"].values, self.df["low"].values, lookback=SWING_LOOKBACK
        )

        # Leverage stats (V6: only 3x and 10x)
        self.leverage_stats = {
            "trades_at_3x": 0, "trades_at_10x": 0,
            "liquidations": 0, "liquidation_loss": 0.0,
        }

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_macd(df, fast=13, slow=34, signal=9)
        df["atr"] = compute_atr(df, period=14)
        df["ma30"] = compute_ma(df, period=30)
        df["sma50"] = compute_ma(df, period=50)
        df["sma200"] = compute_ma(df, period=200)
        return df

    def run(self) -> dict:
        print(f"[V6] Running backtest on {len(self.df)} candles...")
        print(f"  Date range: {self.df['timestamp'].iloc[0]} to {self.df['timestamp'].iloc[-1]}")
        print(f"  V6: SIMPLIFIED LEVERAGE (3x base / 10x with 123 Rule)")
        print(f"  Base strategy: V3 (SMA200 bull filter | near-bottom rolling | SMA50 trend)")
        print(f"  Pre-computed {len(self.swing_highs)} swing highs, {len(self.swing_lows)} swing lows")

        primary_signals = compute_signals_v3(self.df)
        signal_map: dict[int, Signal] = {s.idx: s for s in primary_signals}
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

            self._manage_positions(i, high, low, price, ts)

            if self.active_trades:
                rolling_sigs = compute_rolling_signals_v3(self.df, self.active_trades, i)
                for sig in rolling_sigs:
                    self._execute_rolling_add(sig, i, ts, closes, highs_arr, lows_arr, atr_arr)

            if i in signal_map:
                sig = signal_map[i]
                self._execute_entry(sig, ts, closes, highs_arr, lows_arr, atr_arr)

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

    def _execute_entry(self, signal: Signal, ts: pd.Timestamp,
                       closes: np.ndarray, highs: np.ndarray,
                       lows: np.ndarray, atr_vals: np.ndarray):
        same_dir = [t for t in self.active_trades if t.side == signal.side and not t.is_rolling]
        if same_dir:
            return

        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            return

        # Base position sizing (same as V3 — unleveraged)
        risk_amount = self.capital * self.config.risk_per_trade_pct
        base_size_btc = risk_amount / risk
        max_base_size = (self.capital * self.config.max_position_pct) / signal.entry_price
        base_size_btc = min(base_size_btc, max_base_size)

        if base_size_btc * signal.entry_price < 10:
            return

        # Determine leverage (V6: 3x or 10x)
        leverage, has_123 = determine_leverage(
            signal.idx, signal.side, closes, highs, lows, atr_vals,
            self.swing_highs, self.swing_lows
        )

        # Apply leverage
        leveraged_size_btc = base_size_btc * leverage
        margin = base_size_btc * signal.entry_price  # unleveraged position value
        notional = leveraged_size_btc * signal.entry_price

        # Commission on notional
        commission = notional * self.config.commission_rate
        self.capital -= commission

        # Liquidation price
        if signal.side == "long":
            liq_price = signal.entry_price * (1 - 1 / leverage)
            tp1 = signal.entry_price + risk * self.config.rr_ratio
        else:
            liq_price = signal.entry_price * (1 + 1 / leverage)
            tp1 = signal.entry_price - risk * self.config.rr_ratio

        # Track leverage usage
        if leverage == MAX_LEVERAGE:
            self.leverage_stats["trades_at_10x"] += 1
        else:
            self.leverage_stats["trades_at_3x"] += 1

        trade = Trade(
            side=signal.side,
            entry_price=signal.entry_price,
            entry_idx=signal.idx,
            entry_time=ts,
            size=leveraged_size_btc,
            stop_loss=signal.stop_loss,
            take_profit_1=tp1,
            initial_risk=risk,
            leverage=leverage,
            margin=margin,
            liquidation_price=liq_price,
            original_size=leveraged_size_btc,
            trailing_stop=signal.stop_loss,
        )
        self.active_trades.append(trade)
        self.trades.append(trade)

    def _execute_rolling_add(self, signal: Signal, idx: int, ts: pd.Timestamp,
                             closes: np.ndarray, highs: np.ndarray,
                             lows: np.ndarray, atr_vals: np.ndarray):
        base_trades = [t for t in self.active_trades if t.side == signal.side and not t.is_rolling and not t.closed]
        if not base_trades:
            return

        base = base_trades[0]
        base_key = base.entry_idx
        add_count = self.rolling_add_counts.get(base_key, 0)
        if add_count >= self.config.max_rolling_adds:
            return

        # Base size decays from the base trade's unleveraged size
        base_unleveraged_size = base.original_size / base.leverage
        add_base_size = base_unleveraged_size * (self.config.pyramid_decay ** (add_count + 1))
        base_cost = add_base_size * signal.entry_price
        if base_cost > self.capital * 0.2:
            return

        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            return

        # Determine leverage for rolling add (V6: 3x or 10x)
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
            side=signal.side,
            entry_price=signal.entry_price,
            entry_idx=idx,
            entry_time=ts,
            size=leveraged_size,
            stop_loss=signal.stop_loss,
            take_profit_1=tp1,
            initial_risk=risk,
            leverage=leverage,
            margin=margin,
            liquidation_price=liq_price,
            original_size=leveraged_size,
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
                # Liquidation check first (takes priority over stop loss)
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
                # Liquidation check first
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
        trade.close_reason = reason
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
            return {"version": "v6", "error": "No trades executed"}

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

        # Leverage breakdown
        avg_leverage = np.mean([t.leverage for t in self.closed_trades]) if self.closed_trades else 0

        return {
            "version": "v6",
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
            # V6 leverage metrics
            "avg_leverage": round(avg_leverage, 2),
            "trades_at_3x": self.leverage_stats["trades_at_3x"],
            "trades_at_10x": self.leverage_stats["trades_at_10x"],
            "liquidations": self.leverage_stats["liquidations"],
            "liquidation_loss": round(self.leverage_stats["liquidation_loss"], 2),
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

    def _format_report(self, metrics: dict) -> str:
        sep = "=" * 60
        lines = [
            sep,
            "  BTC/USDT MACD REVERSAL BACKTEST — V6",
            "  V6: SIMPLIFIED LEVERAGE (3x / 10x) on V3 base strategy",
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
            "  LEVERAGE DETAILS",
            sep,
            "",
            f"  Avg Leverage:    {metrics['avg_leverage']:.1f}x",
            f"  Trades at 3x:    {metrics['trades_at_3x']}  (base — no 123 Rule)",
            f"  Trades at 10x:   {metrics['trades_at_10x']}  (123 Rule confirmed)",
            "",
            f"  Liquidations:    {metrics['liquidations']}",
            f"  Liquidation Loss:${metrics['liquidation_loss']:,.2f}",
            "",
            "  Leverage rules (V6 simplified):",
            "    Base:    3x for all entries",
            "    10x:     if 123 Rule confirmed (trendline break +",
            "             failed retest + pivot break)",
            "    No 5x tier — either 3x or 10x",
            "    No near-bottom requirement for 10x",
            "",
            "  Liquidation: force close when unrealised loss",
            "    exceeds margin (notional / leverage)",
            "",
            sep,
            "  BASE STRATEGY: V3",
            sep,
            "",
            "  - No shorts in bull market (price > 200 SMA)",
            "  - Rolling adds only near market bottoms (within 30% of 365d low)",
            "  - 50 SMA trend filter",
            "  - TP1 at 1:1 R/R, double divergence (strength >= 2)",
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
        ax1.plot(timestamps, equity, color="#E91E63", linewidth=1.4, label="V6 (3x/10x Leveraged)")
        ax1.axhline(y=self.config.initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")

        ax1.set_title("BTC/USDT MACD Reversal V6 — Equity Curve (Simplified Leverage)\n"
                      "(3x base / 10x with 123 Rule — no near-bottom requirement)",
                      fontsize=13, fontweight="bold")
        ax1.set_ylabel("Equity ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # Trade markers with leverage coloring
        for trade in self.closed_trades[:80]:
            if trade.close_reason == "liquidation":
                marker_color = "#9C27B0"  # purple for liquidations
            elif trade.pnl > 0:
                marker_color = "#4CAF50"
            else:
                marker_color = "#F44336"

            if trade.entry_time in timestamps.values:
                entry_equity_idx = timestamps[timestamps == trade.entry_time].index
                if len(entry_equity_idx) > 0:
                    ms = 4 + trade.leverage  # bigger marker for higher leverage
                    ax1.plot(trade.entry_time, equity[entry_equity_idx[0]],
                            marker="^" if trade.side == "long" else "v",
                            color=marker_color, markersize=ms, alpha=0.7)

        # Drawdown
        ax2.fill_between(timestamps, -drawdown_pct, 0, color="#F44336", alpha=0.3)
        ax2.plot(timestamps, -drawdown_pct, color="#F44336", linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Drawdown", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # Leverage usage over time
        lev_by_trade = []
        for trade in self.closed_trades:
            lev_by_trade.append((trade.entry_time, trade.leverage))

        if lev_by_trade:
            lev_times, lev_vals = zip(*lev_by_trade)
            colors = []
            for lv in lev_vals:
                if lv >= 10:
                    colors.append("#F44336")  # red for 10x
                else:
                    colors.append("#2196F3")  # blue for 3x
            ax3.scatter(lev_times, lev_vals, c=colors, s=25, alpha=0.8, zorder=3)
            ax3.axhline(y=3, color="#2196F3", linestyle="--", alpha=0.3, label="3x (base)")
            ax3.axhline(y=10, color="#F44336", linestyle="--", alpha=0.3, label="10x (123 Rule)")

        ax3.set_ylabel("Leverage")
        ax3.set_title("Leverage per Trade", fontsize=11)
        ax3.set_ylim(0, 12)
        ax3.set_yticks([0, 3, 10])
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
                "side": t.side,
                "type": "rolling" if t.is_rolling else "primary",
                "leverage": t.leverage,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "size_btc": round(t.size, 6),
                "margin": round(t.margin, 2),
                "liquidation_price": round(t.liquidation_price, 2),
                "stop_loss": round(t.stop_loss, 2),
                "pnl": round(t.pnl, 2),
                "close_reason": t.close_reason,
                "half_closed_at_tp1": t.half_closed,
                "duration_hours": round((t.exit_time - t.entry_time).total_seconds() / 3600, 1) if t.exit_time and t.entry_time else 0,
            })
        trade_df = pd.DataFrame(records)
        trade_file = os.path.join(RESULTS_DIR, "trade_log.csv")
        trade_df.to_csv(trade_file, index=False)
        print(f"  Trade log saved to {trade_file} ({len(records)} trades)")

    def _generate_comparison(self, v6_metrics: dict):
        """Generate V3 vs V5 vs V6 comparison."""
        v3_path = os.path.join(os.path.dirname(__file__), "results_v3", "metrics.json")
        v5_path = os.path.join(os.path.dirname(__file__), "results_v5", "metrics.json")

        v3 = None
        v5 = None
        if os.path.exists(v3_path):
            with open(v3_path) as f:
                v3 = json.load(f)
        if os.path.exists(v5_path):
            with open(v5_path) as f:
                v5 = json.load(f)

        if v3 is None:
            print("  V3 results not found — skipping comparison")
            return

        v6 = v6_metrics

        def fmt_delta(new, old, higher_is_better=True):
            diff = new - old
            if abs(diff) < 0.005:
                return ""
            arrow = "+" if diff > 0 else ""
            if higher_is_better:
                tag = " ^^" if diff > 0 else " vv"
            else:
                tag = " ^^" if diff < 0 else " vv"
            return f"({arrow}{diff:.2f}{tag})"

        def fmt_mult(new, old):
            if old == 0:
                return "(N/A)"
            ratio = new / old
            return f"({ratio:.1f}x)"

        sep = "=" * 95
        col_w = 18

        # Build header and rows depending on V5 availability
        if v5 is not None:
            lines = [
                sep,
                "  BACKTEST COMPARISON: V3 (Unleveraged) vs V5 (3x/5x/10x) vs V6 (3x/10x)",
                sep,
                "",
                f"  {'Metric':<25} {'V3 (1x)':>{col_w}} {'V5 (3/5/10x)':>{col_w}} {'V6 (3/10x)':>{col_w}}   {'V6 vs V3':>12}",
                f"  {'-'*25} {'-'*col_w} {'-'*col_w} {'-'*col_w}   {'-'*12}",
                "",
                f"  {'Total Return %':<25} {v3['total_return_pct']:>{col_w-1}.2f}% {v5['total_return_pct']:>{col_w-1}.2f}% {v6['total_return_pct']:>{col_w-1}.2f}%   {fmt_delta(v6['total_return_pct'], v3['total_return_pct'])}",
                f"  {'Final Equity':<25} ${v3['final_equity']:>{col_w-2},.2f} ${v5['final_equity']:>{col_w-2},.2f} ${v6['final_equity']:>{col_w-2},.2f}   {fmt_mult(v6['final_equity'], v3['final_equity'])}",
                f"  {'Max Drawdown %':<25} {v3['max_drawdown_pct']:>{col_w-1}.2f}% {v5['max_drawdown_pct']:>{col_w-1}.2f}% {v6['max_drawdown_pct']:>{col_w-1}.2f}%   {fmt_delta(v6['max_drawdown_pct'], v3['max_drawdown_pct'], False)}",
                f"  {'Sharpe Ratio':<25} {v3['sharpe_ratio']:>{col_w}.2f} {v5['sharpe_ratio']:>{col_w}.2f} {v6['sharpe_ratio']:>{col_w}.2f}   {fmt_delta(v6['sharpe_ratio'], v3['sharpe_ratio'])}",
                f"  {'Win Rate %':<25} {v3['win_rate_pct']:>{col_w-1}.2f}% {v5['win_rate_pct']:>{col_w-1}.2f}% {v6['win_rate_pct']:>{col_w-1}.2f}%   {fmt_delta(v6['win_rate_pct'], v3['win_rate_pct'])}",
                f"  {'Profit Factor':<25} {v3['profit_factor']:>{col_w}.2f} {v5['profit_factor']:>{col_w}.2f} {v6['profit_factor']:>{col_w}.2f}   {fmt_delta(v6['profit_factor'], v3['profit_factor'])}",
                "",
                f"  {'Total Trades':<25} {v3['total_trades']:>{col_w}} {v5['total_trades']:>{col_w}} {v6['total_trades']:>{col_w}}",
                f"  {'  Primary':<25} {v3['primary_trades']:>{col_w}} {v5['primary_trades']:>{col_w}} {v6['primary_trades']:>{col_w}}",
                f"  {'  Rolling Adds':<25} {v3['rolling_adds']:>{col_w}} {v5['rolling_adds']:>{col_w}} {v6['rolling_adds']:>{col_w}}",
                f"  {'  Long Trades':<25} {v3['long_trades']:>{col_w}} {v5['long_trades']:>{col_w}} {v6['long_trades']:>{col_w}}",
                f"  {'  Short Trades':<25} {v3['short_trades']:>{col_w}} {v5['short_trades']:>{col_w}} {v6['short_trades']:>{col_w}}",
                "",
                f"  {'Long PnL':<25} ${v3['long_pnl']:>{col_w-2},.2f} ${v5['long_pnl']:>{col_w-2},.2f} ${v6['long_pnl']:>{col_w-2},.2f}   {fmt_mult(v6['long_pnl'], v3['long_pnl'])}",
                f"  {'Short PnL':<25} ${v3['short_pnl']:>{col_w-2},.2f} ${v5['short_pnl']:>{col_w-2},.2f} ${v6['short_pnl']:>{col_w-2},.2f}   {fmt_mult(v6['short_pnl'], v3['short_pnl'])}",
                "",
                f"  {'Avg Win':<25} ${v3['avg_win']:>{col_w-2},.2f} ${v5['avg_win']:>{col_w-2},.2f} ${v6['avg_win']:>{col_w-2},.2f}   {fmt_mult(v6['avg_win'], v3['avg_win'])}",
                f"  {'Avg Loss':<25} ${v3['avg_loss']:>{col_w-2},.2f} ${v5['avg_loss']:>{col_w-2},.2f} ${v6['avg_loss']:>{col_w-2},.2f}   {fmt_mult(abs(v6['avg_loss']), abs(v3['avg_loss']))}",
                f"  {'Largest Win':<25} ${v3['largest_win']:>{col_w-2},.2f} ${v5['largest_win']:>{col_w-2},.2f} ${v6['largest_win']:>{col_w-2},.2f}   {fmt_mult(v6['largest_win'], v3['largest_win'])}",
                f"  {'Largest Loss':<25} ${v3['largest_loss']:>{col_w-2},.2f} ${v5['largest_loss']:>{col_w-2},.2f} ${v6['largest_loss']:>{col_w-2},.2f}   {fmt_mult(abs(v6['largest_loss']), abs(v3['largest_loss']))}",
                f"  {'Avg Duration (days)':<25} {v3['avg_trade_duration_days']:>{col_w}.1f} {v5['avg_trade_duration_days']:>{col_w}.1f} {v6['avg_trade_duration_days']:>{col_w}.1f}",
                "",
                sep,
                "  LEVERAGE BREAKDOWN",
                sep,
                "",
                f"  {'':25} {'V5':>{col_w}} {'V6':>{col_w}}",
                f"  {'':25} {'-'*col_w} {'-'*col_w}",
                f"  {'Avg Leverage':<25} {v5['avg_leverage']:>{col_w}.1f} {v6['avg_leverage']:>{col_w}.1f}",
                f"  {'Trades at 3x':<25} {v5['trades_at_3x']:>{col_w}} {v6['trades_at_3x']:>{col_w}}",
                f"  {'Trades at 5x':<25} {v5.get('trades_at_5x', 0):>{col_w}} {'N/A':>{col_w}}",
                f"  {'Trades at 10x':<25} {v5['trades_at_10x']:>{col_w}} {v6['trades_at_10x']:>{col_w}}",
                f"  {'Liquidations':<25} {v5['liquidations']:>{col_w}} {v6['liquidations']:>{col_w}}",
                f"  {'Liquidation Loss':<25} ${v5['liquidation_loss']:>{col_w-2},.2f} ${v6['liquidation_loss']:>{col_w-2},.2f}",
            ]
        else:
            lines = [
                sep,
                "  BACKTEST COMPARISON: V3 (Unleveraged) vs V6 (3x/10x)",
                sep,
                "",
                f"  {'Metric':<25} {'V3 (1x)':>{col_w}} {'V6 (3/10x)':>{col_w}}   {'Delta':>15}",
                f"  {'-'*25} {'-'*col_w} {'-'*col_w}   {'-'*15}",
                "",
                f"  {'Total Return %':<25} {v3['total_return_pct']:>{col_w-1}.2f}% {v6['total_return_pct']:>{col_w-1}.2f}%   {fmt_delta(v6['total_return_pct'], v3['total_return_pct'])}",
                f"  {'Final Equity':<25} ${v3['final_equity']:>{col_w-2},.2f} ${v6['final_equity']:>{col_w-2},.2f}   {fmt_mult(v6['final_equity'], v3['final_equity'])}",
                f"  {'Max Drawdown %':<25} {v3['max_drawdown_pct']:>{col_w-1}.2f}% {v6['max_drawdown_pct']:>{col_w-1}.2f}%   {fmt_delta(v6['max_drawdown_pct'], v3['max_drawdown_pct'], False)}",
                f"  {'Sharpe Ratio':<25} {v3['sharpe_ratio']:>{col_w}.2f} {v6['sharpe_ratio']:>{col_w}.2f}   {fmt_delta(v6['sharpe_ratio'], v3['sharpe_ratio'])}",
                f"  {'Win Rate %':<25} {v3['win_rate_pct']:>{col_w-1}.2f}% {v6['win_rate_pct']:>{col_w-1}.2f}%   {fmt_delta(v6['win_rate_pct'], v3['win_rate_pct'])}",
                f"  {'Profit Factor':<25} {v3['profit_factor']:>{col_w}.2f} {v6['profit_factor']:>{col_w}.2f}   {fmt_delta(v6['profit_factor'], v3['profit_factor'])}",
                "",
                f"  {'Total Trades':<25} {v3['total_trades']:>{col_w}} {v6['total_trades']:>{col_w}}",
            ]

        lines += [
            "",
            sep,
            "  STRATEGY DIFFERENCES",
            sep,
            "",
            "  V3 (baseline, unleveraged):",
            "    - Spot trading (1x), no leverage",
            "    - Same signal generation, same filters",
            "",
            "  V5 (3-tier leverage):",
            "    - 3x base leverage on all entries",
            "    - 5x when 123 Rule pattern confirmed",
            "    - 10x when 123 Rule + price within 20% of 365-day low",
            "",
            "  V6 (simplified 2-tier leverage):",
            "    - 3x base leverage on all entries",
            "    - 10x when 123 Rule pattern confirmed (ANY price level)",
            "    - No 5x tier — binary 3x or 10x",
            "    - Removed near-bottom requirement for max leverage",
            "",
            sep,
        ]

        comparison = "\n".join(lines)
        comp_path = os.path.join(RESULTS_DIR, "comparison_v3_v5_v6.txt")
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

    print("\nBacktest V6 complete!")
    return metrics


if __name__ == "__main__":
    main()

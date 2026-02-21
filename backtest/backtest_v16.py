"""Backtest V16 — 肥宅 (FEIZHAI) ROLLING STRATEGY.

Based on V15. LONG ONLY on weekly resampled data.

CHANGES FROM V15:
  1. PYRAMID SIZING (each add SMALLER, not bigger):
     - Base: 100% of normal entry margin
     - Add 1: 75% of base margin
     - Add 2: 50% of base margin
     - Add 3: 25% of base margin
     Max 3 adds (diminishing pyramid prevents avg cost rising too fast)

  2. ATR-BASED STOPS ON ADDS (NOT breakeven):
     - Each add stop = add entry price - 1.5x weekly ATR
     - Master trailing stop at 3x weekly ATR from highest point (unchanged)

  3. FAKEOUT MANAGEMENT:
     - If price falls below add entry price within 3 weekly candles → close ONLY the add
     - Base position keeps its own trailing stop independently

  4. ADD TIMING (肥宅 style):
     - Check every week when profitable
     - Must be in uptrend: price > 20-week SMA
     - Add on ONE of:
       a) Pullback to 20-week MA (price touched SMA20 this week: low <= sma20)
       b) Fib 0.5-0.618 retracement from swing low to current high
       c) Breakout from consolidation (price > previous 4-week high + volume increase)

  5. FLOATING PROFITS ONLY for adds (肥宅 style, no additional capital):
     - Add margin comes ONLY from unrealized profit * 50%
     - No extra capital injection at all

ENTRY — ANY ONE condition sufficient (same as V14/V15):
  1. MVRV proxy < 0.8
  2. Price within 30% of 365-day low AND Fear proxy active
  3. Weekly MACD 13/34 single divergence
  4. Price crosses ABOVE 20-week SMA from below
  5. Price within 20% of 52-week low

EXIT:
  - Master trailing stop: 3x weekly ATR from highest point (base position)
  - Individual add stop: 1.5x weekly ATR from add entry price
  - Fakeout close: price < add entry within 3 weeks → close add only
  - Scale out when MVRV proxy > 2.5 (take 20% off)
  - Aggressive exit when MVRV proxy > 3.5

FUNDING: 0.01% per 8h (21 periods per weekly bar)
Starting capital: $6,000. Commission: 0.1%.
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v16")

# ── Constants ─────────────────────────────────────────────────────────

CAPITAL = 6_000.0
COMMISSION_RATE = 0.001          # 0.1%
FUNDING_RATE_PER_8H = 0.0001    # 0.01% per 8h

SWING_LOOKBACK = 10              # weekly bars for swing detection
MVRV_LOOKBACK = 52               # 52 weeks ≈ 365 days
MVRV_UNDERVALUED = 0.8
FEAR_PROXIMITY_PCT = 0.30
SOPR_DECLINE_WEEKS = 3
FIFTY_TWO_WEEK_PROXIMITY = 0.20

TRAILING_ATR_MULT = 3.0          # 3x weekly ATR trailing stop (base/master)
ADD_ATR_MULT = 1.5               # 1.5x weekly ATR stop for adds
ENTRY_LEVERAGE = 5               # always enter at 5x
ENTRY_MARGIN_PCT = 0.05          # 5% of capital as test position

MVRV_SCALEOUT = 2.5
MVRV_AGGRESSIVE_EXIT = 3.5

# V16: 肥宅 pyramid sizing (fraction of base margin)
PYRAMID_SIZES = [0.75, 0.50, 0.25]  # Add 1, 2, 3 as fraction of base margin
MAX_ADDS = 3

# V16: Fakeout management
FAKEOUT_WINDOW = 3               # close add if price < add entry within 3 weeks

# V16: Add timing parameters
CONSOLIDATION_LOOKBACK = 4       # breakout above 4-week high
FIB_RETRACE_LOW = 0.50           # Fibonacci retracement range
FIB_RETRACE_HIGH = 0.618

# V16: Floating profit only for adds
ADD_PROFIT_SHARE = 0.50          # use 50% of floating profit for add margin
ADD_MIN_SIZE_BTC = 0.001

# Swing low safety
SWING_LOW_LOOKBACK = 10


# ══════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════

def resample_to_weekly(df_4h: pd.DataFrame) -> pd.DataFrame:
    """Resample 4H OHLCV to weekly candles with indicators."""
    df = df_4h.set_index("timestamp")
    weekly = df.resample("W").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"]).reset_index()

    weekly = compute_macd(weekly, fast=13, slow=34, signal=9)
    weekly["atr"] = compute_atr(weekly, period=14)
    weekly["sma20"] = compute_ma(weekly, period=20)
    weekly["sma50"] = compute_ma(weekly, period=50)
    return weekly


# ══════════════════════════════════════════════════════════════════════
#  SWING DETECTION
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


# ══════════════════════════════════════════════════════════════════════
#  DIVERGENCE DETECTION
# ══════════════════════════════════════════════════════════════════════

def check_bottom_divergence(current_idx: int, hist: np.ndarray,
                            lows: np.ndarray, lookback: int = 80):
    """Classical bullish divergence: price lower lows + MACD higher lows."""
    troughs = []
    start = max(0, current_idx - lookback)
    in_trough = False
    trough_val = 0.0
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
            trough_val = 0.0

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


# ══════════════════════════════════════════════════════════════════════
#  TRADE DATA STRUCTURE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_price: float
    entry_idx: int
    entry_time: pd.Timestamp
    size_btc: float           # leveraged BTC size
    margin: float             # capital at risk
    leverage: float
    stop_loss: float          # trailing stop (base) or ATR stop (add)
    liquidation_price: float
    conditions_met: int
    condition_flags: str
    is_add: bool = False
    add_number: int = 0       # 0=base, 1-3 = add number
    exit_price: float = 0.0
    exit_idx: int = 0
    exit_time: pd.Timestamp = None
    pnl: float = 0.0
    funding_paid: float = 0.0
    closed: bool = False
    close_reason: str = ""
    highest_price: float = 0.0
    scale_out_done: bool = False


# ══════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════

class LongTermWeeklyBacktest:
    """V16: 肥宅 rolling strategy — pyramid sizing, ATR stops, fakeout mgmt."""

    def __init__(self, weekly_df: pd.DataFrame, capital: float = CAPITAL):
        self.initial_capital = capital
        self.capital = capital
        self.max_capital = capital
        self.w = weekly_df
        self.active_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.equity_curve = []
        self.trade_log = []

        # Add tracking
        self.base_entry_idx = -1
        self.base_margin = 0.0       # remember base margin for pyramid sizing
        self.add_count = 0

        # Pre-compute swings
        self.swing_highs, self.swing_lows = detect_swings(
            weekly_df["high"].values, weekly_df["low"].values,
            lookback=SWING_LOOKBACK
        )

        self.stats = {
            "signals_generated": 0,
            "liquidations": 0, "liquidation_loss": 0.0,
            "total_funding_paid": 0.0,
            "pyramid_adds": 0,
            "adds_attempted": 0,
            "adds_skipped_no_profit": 0,
            "adds_skipped_no_uptrend": 0,
            "adds_skipped_no_signal": 0,
            "adds_skipped_liq_unsafe": 0,
            "adds_skipped_too_small": 0,
            "adds_skipped_max_reached": 0,
            "adds_skipped_insufficient_margin": 0,
            "adds_via_pullback_sma20": 0,
            "adds_via_fib_retracement": 0,
            "adds_via_consolidation_breakout": 0,
            "fakeout_closes": 0,
            "total_add_margin_from_profit": 0.0,
            "scale_outs": 0,
            "aggressive_exits": 0,
        }

    # ── Entry condition checks ────────────────────────────────────────

    def _mvrv_proxy(self, idx: int) -> float:
        start = max(0, idx - MVRV_LOOKBACK)
        window = self.w["close"].values[start:idx + 1]
        if len(window) < 10:
            return 1.0
        avg_365 = np.mean(window)
        if avg_365 == 0:
            return 1.0
        return self.w["close"].values[idx] / avg_365

    def _check_mvrv_undervalued(self, idx: int) -> bool:
        return self._mvrv_proxy(idx) < MVRV_UNDERVALUED

    def _check_fear_and_proximity(self, idx: int) -> bool:
        start = max(0, idx - MVRV_LOOKBACK)
        window = self.w["low"].values[start:idx + 1]
        if len(window) < 10:
            return False
        low_365 = np.min(window)
        price = self.w["close"].values[idx]
        if price > low_365 * (1 + FEAR_PROXIMITY_PCT):
            return False
        if idx < SOPR_DECLINE_WEEKS:
            return False
        closes = self.w["close"].values
        sma50 = self.w["sma50"].values
        if np.isnan(sma50[idx]):
            return False
        if closes[idx] >= sma50[idx]:
            return False
        for k in range(1, SOPR_DECLINE_WEEKS + 1):
            if idx - k < 0:
                return False
            if closes[idx - k + 1] >= closes[idx - k]:
                return False
        return True

    def _check_macd_divergence(self, idx: int) -> bool:
        hist = self.w["macd_hist"].values
        lows = self.w["low"].values
        if idx < 3:
            return False
        if hist[idx] >= 0 and hist[idx - 1] >= 0:
            return False
        div = check_bottom_divergence(idx, hist, lows, lookback=80)
        if div is not None and div[0] >= 1:
            return True
        return False

    def _check_sma20_crossover(self, idx: int) -> bool:
        if idx < 1:
            return False
        closes = self.w["close"].values
        sma20 = self.w["sma20"].values
        if np.isnan(sma20[idx]) or np.isnan(sma20[idx - 1]):
            return False
        return closes[idx - 1] < sma20[idx - 1] and closes[idx] > sma20[idx]

    def _check_52w_low_proximity(self, idx: int) -> bool:
        start = max(0, idx - MVRV_LOOKBACK)
        window = self.w["low"].values[start:idx + 1]
        if len(window) < 10:
            return False
        low_52w = np.min(window)
        price = self.w["close"].values[idx]
        return price <= low_52w * (1 + FIFTY_TWO_WEEK_PROXIMITY)

    def _evaluate_entry(self, idx: int) -> tuple[int, str]:
        conditions = []
        flags = []
        if self._check_mvrv_undervalued(idx):
            conditions.append(1); flags.append("MVRV<0.8")
        if self._check_fear_and_proximity(idx):
            conditions.append(2); flags.append("fear_30%+decline")
        if self._check_macd_divergence(idx):
            conditions.append(3); flags.append("MACD_div")
        if self._check_sma20_crossover(idx):
            conditions.append(4); flags.append("SMA20_cross")
        if self._check_52w_low_proximity(idx):
            conditions.append(5); flags.append("52w_low_20%")
        return len(conditions), "|".join(flags)

    # ── 肥宅 Add Timing Signals ──────────────────────────────────────

    def _check_add_signal(self, idx: int, price: float) -> tuple[bool, str]:
        """Check if any of the 3 肥宅-style add timing conditions are met.

        Returns (signal_fired, signal_type).
        """
        sma20 = self.w["sma20"].values
        lows = self.w["low"].values
        highs = self.w["high"].values
        closes = self.w["close"].values
        volumes = self.w["volume"].values

        # Must be in uptrend: price > 20-week SMA
        if np.isnan(sma20[idx]) or price <= sma20[idx]:
            return False, ""

        # Signal A: Pullback to 20-week MA (low touched SMA20 this week)
        if lows[idx] <= sma20[idx]:
            return True, "pullback_sma20"

        # Signal B: Fib 0.5-0.618 retracement
        # Find the most recent swing low before position was opened
        # and use the highest high since entry as the swing high
        if self.base_entry_idx >= 0:
            # Swing low: lowest low in 20 weeks before base entry
            swing_low_start = max(0, self.base_entry_idx - 20)
            swing_low = np.min(lows[swing_low_start:self.base_entry_idx + 1])
            # Swing high: highest high from entry to now
            swing_high = np.max(highs[self.base_entry_idx:idx + 1])

            if swing_high > swing_low:
                move = swing_high - swing_low
                fib_50 = swing_high - move * FIB_RETRACE_LOW
                fib_618 = swing_high - move * FIB_RETRACE_HIGH

                # Price pulled back into the 0.5-0.618 zone
                if fib_618 <= price <= fib_50:
                    return True, "fib_retracement"

        # Signal C: Breakout from consolidation
        # Price > previous 4-week high AND volume increase
        if idx >= CONSOLIDATION_LOOKBACK:
            prev_4w_high = np.max(highs[idx - CONSOLIDATION_LOOKBACK:idx])
            if price > prev_4w_high:
                # Volume increase: current week volume > avg of previous 4 weeks
                avg_vol = np.mean(volumes[idx - CONSOLIDATION_LOOKBACK:idx])
                if avg_vol > 0 and volumes[idx] > avg_vol:
                    return True, "consolidation_breakout"

        return False, ""

    # ── Execution ─────────────────────────────────────────────────────

    def run(self) -> dict:
        print(f"\n[V16 肥宅 Rolling] Running on {len(self.w)} weekly candles...")
        print(f"  Capital: ${self.initial_capital:,.0f}")

        closes = self.w["close"].values
        highs = self.w["high"].values
        lows = self.w["low"].values
        atr = self.w["atr"].values

        start_idx = max(52, SWING_LOOKBACK * 2 + 1)
        for i in range(start_idx, len(self.w)):
            if not np.isnan(self.w["sma50"].values[i]):
                start_idx = i
                break

        print(f"  Starting at week index {start_idx} "
              f"({self.w['timestamp'].iloc[start_idx].strftime('%Y-%m-%d')})")

        min_cooldown = 4

        for i in range(len(self.w)):
            price = closes[i]
            high = highs[i]
            low = lows[i]
            ts = self.w["timestamp"].iloc[i]
            w_atr = atr[i] if not np.isnan(atr[i]) else 0

            # Deduct funding for active positions
            self._deduct_funding(price, periods=21)

            # Manage existing positions (trailing stop, fakeout, scale-out)
            self._manage_positions(i, high, low, price, ts, w_atr)

            # Check 肥宅-style adds every week
            if self.active_trades and i >= start_idx:
                self._check_feizhai_add(i, price, low, ts, w_atr)

            # Check for new entry — ANY 1 condition
            if i >= start_idx and not self.active_trades:
                cond_count, cond_flags = self._evaluate_entry(i)
                if cond_count >= 1:
                    last_close_idx = max(
                        (t.exit_idx for t in self.closed_trades), default=-999
                    )
                    if i - last_close_idx >= min_cooldown:
                        self._execute_entry(i, price, ts, w_atr, cond_count, cond_flags)

            # Record equity
            unrealized = self._calc_unrealized(price)
            self.equity_curve.append({
                "timestamp": ts,
                "equity": self.capital + unrealized,
                "capital": self.capital,
                "price": price,
            })

        # Close remaining at end
        if self.active_trades:
            last_price = closes[-1]
            last_ts = self.w["timestamp"].iloc[-1]
            for trade in list(self.active_trades):
                self._close_trade(trade, last_price, len(self.w) - 1,
                                  last_ts, "end_of_backtest")

        return self._compute_metrics()

    def _execute_entry(self, idx: int, price: float, ts: pd.Timestamp,
                       w_atr: float, cond_count: int, cond_flags: str):
        """Base entry at 5x with 5% of capital as margin."""
        leverage = ENTRY_LEVERAGE
        self.stats["signals_generated"] += 1

        margin = self.capital * ENTRY_MARGIN_PCT
        if margin < 10:
            return

        notional = margin * leverage
        size_btc = notional / price

        # Stop at 3x ATR
        stop_distance = w_atr * TRAILING_ATR_MULT if w_atr > 0 else price * 0.08
        stop_loss = price - stop_distance

        # Commission
        commission = notional * COMMISSION_RATE
        self.capital -= commission

        # Liquidation price (long)
        liq_price = price * (1 - 1 / leverage)

        # Ensure stop above liquidation with buffer
        min_stop = liq_price + (price - liq_price) * 0.10
        stop_loss = max(stop_loss, min_stop)

        trade = Trade(
            entry_price=price,
            entry_idx=idx,
            entry_time=ts,
            size_btc=size_btc,
            margin=margin,
            leverage=leverage,
            stop_loss=stop_loss,
            liquidation_price=liq_price,
            conditions_met=cond_count,
            condition_flags=cond_flags,
            highest_price=price,
        )
        self.active_trades.append(trade)

        # Reset add tracking
        self.base_entry_idx = idx
        self.base_margin = margin
        self.add_count = 0

        self.trade_log.append(
            f"[{ts.strftime('%Y-%m-%d')}] ENTRY: ${price:,.0f} | "
            f"5x (base) | margin=${margin:,.0f} | "
            f"size={size_btc:.4f} BTC | "
            f"conds={cond_count} ({cond_flags}) | SL=${stop_loss:,.0f} | "
            f"liq=${liq_price:,.0f}"
        )

    def _check_feizhai_add(self, idx: int, price: float, current_low: float,
                            ts: pd.Timestamp, w_atr: float):
        """肥宅-style pyramid add — checked every week.

        Requirements:
        1. Max 3 adds (diminishing pyramid)
        2. Position must be profitable
        3. Add timing signal must fire (pullback/fib/breakout)
        4. Margin from floating profits only (50%)
        5. Pyramid sizing: add 1=75%, add 2=50%, add 3=25% of base margin
        6. Liquidation safety check
        """
        if self.base_entry_idx < 0:
            return

        self.stats["adds_attempted"] += 1

        # Step 1: Max adds check
        if self.add_count >= MAX_ADDS:
            self.stats["adds_skipped_max_reached"] += 1
            return

        # Step 2: Position must be profitable
        agg_pnl = sum(
            (price - t.entry_price) * t.size_btc
            for t in self.active_trades if not t.closed
        )
        if agg_pnl <= 0:
            self.stats["adds_skipped_no_profit"] += 1
            return

        # Step 3: Check add timing signal (肥宅 style)
        signal_fired, signal_type = self._check_add_signal(idx, price)
        if not signal_fired:
            self.stats["adds_skipped_no_signal"] += 1
            return

        # Step 4: Calculate margin from floating profits only
        available_margin = agg_pnl * ADD_PROFIT_SHARE

        # Step 5: Pyramid sizing — cap by pyramid fraction of base margin
        pyramid_fraction = PYRAMID_SIZES[self.add_count]  # 0.75, 0.50, 0.25
        pyramid_max_margin = self.base_margin * pyramid_fraction
        add_margin = min(available_margin, pyramid_max_margin)

        if add_margin <= 0:
            self.stats["adds_skipped_insufficient_margin"] += 1
            return

        leverage = ENTRY_LEVERAGE
        add_notional = add_margin * leverage
        add_btc = add_notional / price

        # Step 6: Liquidation safety check
        lookback_start = max(0, idx - SWING_LOW_LOOKBACK)
        recent_lows = self.w["low"].values[lookback_start:idx + 1]
        if len(recent_lows) == 0:
            self.stats["adds_skipped_liq_unsafe"] += 1
            return
        lowest_swing_low = np.min(recent_lows)

        liq_price = price * (1 - 1 / leverage)
        if liq_price >= lowest_swing_low:
            self.stats["adds_skipped_liq_unsafe"] += 1
            return

        # Check combined position liq safety
        total_margin = sum(t.margin for t in self.active_trades if not t.closed)
        total_notional = sum(t.size_btc * t.entry_price for t in self.active_trades if not t.closed)
        total_btc = sum(t.size_btc for t in self.active_trades if not t.closed)

        new_total_margin = total_margin + add_margin
        new_total_notional = total_notional + add_notional
        new_total_btc = total_btc + add_btc
        new_avg_entry = new_total_notional / new_total_btc if new_total_btc > 0 else price
        new_eff_lev = new_total_notional / new_total_margin if new_total_margin > 0 else leverage
        new_combined_liq = new_avg_entry * (1 - 1 / new_eff_lev) if new_eff_lev > 1 else 0

        if new_combined_liq >= lowest_swing_low:
            # Scale down to fit
            lo, hi = 0.0, add_margin
            safe_margin = 0.0
            for _ in range(30):
                mid = (lo + hi) / 2
                m_notional = mid * leverage
                m_btc = m_notional / price
                m_total_margin = total_margin + mid
                m_total_notional = total_notional + m_notional
                m_total_btc = total_btc + m_btc
                m_avg_entry = m_total_notional / m_total_btc if m_total_btc > 0 else price
                m_eff_lev = m_total_notional / m_total_margin if m_total_margin > 0 else leverage
                m_liq = m_avg_entry * (1 - 1 / m_eff_lev) if m_eff_lev > 1 else 0
                if m_liq < lowest_swing_low:
                    safe_margin = mid
                    lo = mid
                else:
                    hi = mid
            add_margin = safe_margin
            add_notional = add_margin * leverage
            add_btc = add_notional / price

        # Minimum size check
        if add_btc < ADD_MIN_SIZE_BTC:
            self.stats["adds_skipped_too_small"] += 1
            return

        # Commission
        commission = add_notional * COMMISSION_RATE
        self.capital -= commission

        # ATR-based stop for this add (1.5x weekly ATR from add entry)
        add_stop = price - w_atr * ADD_ATR_MULT if w_atr > 0 else price * 0.95

        # Ensure stop above liquidation
        min_stop = liq_price + (price - liq_price) * 0.10
        add_stop = max(add_stop, min_stop)

        self.stats["total_add_margin_from_profit"] += add_margin

        # Track signal type
        if signal_type == "pullback_sma20":
            self.stats["adds_via_pullback_sma20"] += 1
        elif signal_type == "fib_retracement":
            self.stats["adds_via_fib_retracement"] += 1
        elif signal_type == "consolidation_breakout":
            self.stats["adds_via_consolidation_breakout"] += 1

        # Get base trade for condition info
        base_trades = [t for t in self.active_trades if not t.is_add and not t.closed]
        base = base_trades[0] if base_trades else self.active_trades[0]

        self.add_count += 1

        add_trade = Trade(
            entry_price=price,
            entry_idx=idx,
            entry_time=ts,
            size_btc=add_btc,
            margin=add_margin,
            leverage=leverage,
            stop_loss=add_stop,
            liquidation_price=liq_price,
            conditions_met=base.conditions_met,
            condition_flags=base.condition_flags,
            is_add=True,
            add_number=self.add_count,
            highest_price=price,
        )
        self.active_trades.append(add_trade)
        self.stats["pyramid_adds"] += 1

        # Log
        new_total_btc = sum(t.size_btc for t in self.active_trades if not t.closed)
        new_total_margin = sum(t.margin for t in self.active_trades if not t.closed)
        new_total_notional = sum(t.size_btc * price for t in self.active_trades if not t.closed)
        eff_leverage = new_total_notional / new_total_margin if new_total_margin > 0 else leverage

        self.trade_log.append(
            f"[{ts.strftime('%Y-%m-%d')}] PYRAMID ADD #{self.add_count} ({signal_type}): "
            f"${price:,.0f} | {leverage}x | "
            f"margin=${add_margin:,.0f} ({pyramid_fraction:.0%} of base) | "
            f"size={add_btc:.4f} BTC | agg_pnl=${agg_pnl:,.0f} | "
            f"total_btc={new_total_btc:.4f} | eff_lev={eff_leverage:.1f}x | "
            f"add_stop=${add_stop:,.0f}"
        )

    def _deduct_funding(self, price: float, periods: int = 21):
        for trade in self.active_trades:
            if trade.closed:
                continue
            notional = trade.size_btc * price
            cost = notional * FUNDING_RATE_PER_8H * periods
            self.capital -= cost
            trade.funding_paid += cost
            self.stats["total_funding_paid"] += cost

    def _manage_positions(self, idx: int, high: float, low: float,
                          price: float, ts: pd.Timestamp, w_atr: float):
        mvrv = self._mvrv_proxy(idx)

        for trade in list(self.active_trades):
            if trade.closed:
                continue

            trade.highest_price = max(trade.highest_price, high)

            if trade.is_add:
                # ── ADD POSITION MANAGEMENT ──

                # Fakeout check: if price < add entry within 3 weeks → close add only
                weeks_since_add = idx - trade.entry_idx
                if weeks_since_add <= FAKEOUT_WINDOW and low <= trade.entry_price:
                    self.stats["fakeout_closes"] += 1
                    self._close_trade(trade, trade.entry_price, idx, ts, "fakeout_close")
                    continue

                # ATR-based stop for add (fixed at 1.5x ATR from entry, doesn't trail)
                if low <= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, idx, ts, "add_atr_stop")
                    continue

                # Liquidation
                if low <= trade.liquidation_price:
                    self.stats["liquidations"] += 1
                    self.stats["liquidation_loss"] += trade.margin
                    self._close_trade(trade, trade.liquidation_price, idx, ts, "liquidation")
                    continue

                # Scale out for adds too
                if mvrv > MVRV_SCALEOUT and not trade.scale_out_done:
                    scale_size = trade.size_btc * 0.20
                    scale_pnl = (price - trade.entry_price) * scale_size
                    commission = scale_size * price * COMMISSION_RATE
                    scale_pnl -= commission
                    self.capital += scale_pnl
                    trade.size_btc -= scale_size
                    trade.scale_out_done = True
                    self.stats["scale_outs"] += 1
                    self.trade_log.append(
                        f"[{ts.strftime('%Y-%m-%d')}] SCALE OUT 20% (add#{trade.add_number}): "
                        f"${price:,.0f} | MVRV={mvrv:.2f} | pnl=${scale_pnl:,.0f}"
                    )

                # Aggressive exit
                if mvrv > MVRV_AGGRESSIVE_EXIT:
                    self.stats["aggressive_exits"] += 1
                    self._close_trade(trade, price, idx, ts, "mvrv_aggressive_exit")
                    continue

            else:
                # ── BASE POSITION MANAGEMENT ──

                # Update master trailing stop (3x weekly ATR from highest)
                if w_atr > 0:
                    new_trail = trade.highest_price - w_atr * TRAILING_ATR_MULT
                    trade.stop_loss = max(trade.stop_loss, new_trail)

                # Scale out: MVRV proxy > 2.5 → take 20% off
                if mvrv > MVRV_SCALEOUT and not trade.scale_out_done:
                    scale_size = trade.size_btc * 0.20
                    scale_pnl = (price - trade.entry_price) * scale_size
                    commission = scale_size * price * COMMISSION_RATE
                    scale_pnl -= commission
                    self.capital += scale_pnl
                    trade.size_btc -= scale_size
                    trade.scale_out_done = True
                    self.stats["scale_outs"] += 1
                    self.trade_log.append(
                        f"[{ts.strftime('%Y-%m-%d')}] SCALE OUT 20% (base): "
                        f"${price:,.0f} | MVRV={mvrv:.2f} | pnl=${scale_pnl:,.0f}"
                    )

                # Aggressive exit: MVRV > 3.5
                if mvrv > MVRV_AGGRESSIVE_EXIT:
                    self.stats["aggressive_exits"] += 1
                    self._close_trade(trade, price, idx, ts, "mvrv_aggressive_exit")
                    continue

                # Trailing stop
                if low <= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, idx, ts, "trailing_stop")
                    continue

                # Liquidation
                if low <= trade.liquidation_price:
                    self.stats["liquidations"] += 1
                    self.stats["liquidation_loss"] += trade.margin
                    self._close_trade(trade, trade.liquidation_price, idx, ts, "liquidation")
                    continue

    def _close_trade(self, trade: Trade, exit_price: float, idx: int,
                     ts: pd.Timestamp, reason: str):
        pnl = (exit_price - trade.entry_price) * trade.size_btc
        commission = trade.size_btc * exit_price * COMMISSION_RATE
        pnl -= commission

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

        self.trade_log.append(
            f"[{ts.strftime('%Y-%m-%d')}] EXIT ({reason}): ${exit_price:,.0f} | "
            f"pnl=${pnl:,.0f} | funding=${trade.funding_paid:,.0f} | "
            f"{trade.leverage}x | add={trade.add_number}"
        )

        # Base trade closed → close all adds and reset tracking
        if not trade.is_add:
            for add_trade in list(self.active_trades):
                if add_trade.is_add and not add_trade.closed:
                    self._close_trade(add_trade, exit_price, idx, ts, "base_closed")
            self.base_entry_idx = -1
            self.base_margin = 0.0
            self.add_count = 0

    def _calc_unrealized(self, price: float) -> float:
        return sum(
            (price - t.entry_price) * t.size_btc
            for t in self.active_trades if not t.closed
        )

    def _compute_metrics(self) -> dict:
        eq = pd.DataFrame(self.equity_curve)
        if eq.empty:
            return {"error": "No data"}

        equity = eq["equity"].values
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100

        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak * 100
        max_dd = np.max(dd) if len(dd) > 0 else 0

        returns = np.diff(equity) / equity[:-1]
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(52)
                  if len(returns) > 1 and np.std(returns) > 0 else 0)

        if not self.closed_trades:
            return {
                "system": "v16_feizhai_rolling",
                "total_return_pct": round(total_return, 2),
                "final_equity": round(equity[-1], 2),
                "initial_capital": self.initial_capital,
                "max_drawdown_pct": round(max_dd, 2),
                "sharpe_ratio": round(sharpe, 2),
                "total_trades": 0,
                "note": "No trades generated",
                **self.stats,
            }

        pnls = [t.pnl for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100
        profit_factor = (sum(wins) / abs(sum(losses))
                         if losses and sum(losses) != 0 else float("inf"))

        durations = []
        for t in self.closed_trades:
            if t.entry_time and t.exit_time:
                durations.append(
                    (t.exit_time - t.entry_time).total_seconds() / 86400
                )

        base_trades = [t for t in self.closed_trades if not t.is_add]
        add_trades = [t for t in self.closed_trades if t.is_add]

        net_pnl = sum(pnls)
        net_after_funding = net_pnl - self.stats["total_funding_paid"]

        return {
            "system": "v16_feizhai_rolling",
            "total_return_pct": round(total_return, 2),
            "final_equity": round(equity[-1], 2),
            "initial_capital": self.initial_capital,
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_trades": len(self.closed_trades),
            "base_trades": len(base_trades),
            "pyramid_adds": len(add_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "largest_win": round(max(pnls), 2) if pnls else 0,
            "largest_loss": round(min(pnls), 2) if pnls else 0,
            "avg_duration_days": round(np.mean(durations), 1) if durations else 0,
            "gross_pnl": round(net_pnl, 2),
            "total_funding_cost": round(self.stats["total_funding_paid"], 2),
            "net_pnl_after_funding": round(net_after_funding, 2),
            **self.stats,
        }


# ══════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════

def generate_report(metrics: dict, trade_log: list, closed_trades: list) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  BACKTEST V16 — 肥宅 (FEIZHAI) ROLLING STRATEGY")
    lines.append("=" * 70)
    lines.append("")
    lines.append("STRATEGY: Long-only weekly, 5 entry conditions (ANY 1 sufficient)")
    lines.append("CONDITIONS:")
    lines.append("  1. MVRV proxy < 0.8")
    lines.append("  2. Price within 30% of 365d low + fear proxy")
    lines.append("  3. Weekly MACD 13/34 single divergence")
    lines.append("  4. Price crosses above 20-week SMA from below")
    lines.append("  5. Price within 20% of 52-week low")
    lines.append("")
    lines.append("ADD LOGIC (肥宅 pyramid, checked every week):")
    lines.append("  PYRAMID SIZING (each add SMALLER):")
    lines.append("    - Base: 100% of entry margin")
    lines.append("    - Add 1: 75% of base margin")
    lines.append("    - Add 2: 50% of base margin")
    lines.append("    - Add 3: 25% of base margin (max 3 adds)")
    lines.append("  ADD TIMING:")
    lines.append("    - Must be in uptrend: price > 20-week SMA")
    lines.append("    - Pullback to 20-week MA (low touched SMA20)")
    lines.append("    - OR Fib 0.5-0.618 retracement")
    lines.append("    - OR Consolidation breakout (> 4-week high + volume)")
    lines.append("  MARGIN: Floating profits only (50% of unrealized PnL)")
    lines.append("  STOPS: Each add = entry - 1.5x weekly ATR")
    lines.append("  FAKEOUT: Price < add entry within 3 weeks → close add only")
    lines.append("")
    lines.append("MASTER TRAILING STOP: 3x weekly ATR (base position)")
    lines.append(f"CAPITAL:  ${metrics.get('initial_capital', 0):,.0f}")
    lines.append(f"LEVERAGE: {ENTRY_LEVERAGE}x fixed for all positions")
    lines.append(f"PERIOD:   ~4 years of BTC/USDT weekly data")
    lines.append("")

    lines.append("── PERFORMANCE ──────────────────────────────────────────")
    lines.append(f"  Final Equity:     ${metrics.get('final_equity', 0):,.2f}")
    lines.append(f"  Total Return:     {metrics.get('total_return_pct', 0):+.2f}%")
    lines.append(f"  Max Drawdown:     {metrics.get('max_drawdown_pct', 0):.2f}%")
    lines.append(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}")
    lines.append(f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
    lines.append(f"  Win Rate:         {metrics.get('win_rate_pct', 0):.1f}%")
    lines.append("")

    lines.append("── TRADE SUMMARY ────────────────────────────────────────")
    lines.append(f"  Total Trades:     {metrics.get('total_trades', 0)}")
    lines.append(f"  Base Trades:      {metrics.get('base_trades', 0)}")
    lines.append(f"  Pyramid Adds:     {metrics.get('pyramid_adds', 0)}")
    lines.append(f"  Winning:          {metrics.get('winning_trades', 0)}")
    lines.append(f"  Losing:           {metrics.get('losing_trades', 0)}")
    lines.append(f"  Avg Win:          ${metrics.get('avg_win', 0):,.2f}")
    lines.append(f"  Avg Loss:         ${metrics.get('avg_loss', 0):,.2f}")
    lines.append(f"  Largest Win:      ${metrics.get('largest_win', 0):,.2f}")
    lines.append(f"  Largest Loss:     ${metrics.get('largest_loss', 0):,.2f}")
    lines.append(f"  Avg Duration:     {metrics.get('avg_duration_days', 0):.1f} days")
    lines.append("")

    lines.append("── PYRAMID ADD STATS ────────────────────────────────────")
    lines.append(f"  Pyramid Adds:     {metrics.get('pyramid_adds', 0)}")
    lines.append(f"  Attempts:         {metrics.get('adds_attempted', 0)}")
    lines.append(f"  Skip (no profit): {metrics.get('adds_skipped_no_profit', 0)}")
    lines.append(f"  Skip (no uptrend):{metrics.get('adds_skipped_no_uptrend', 0)}")
    lines.append(f"  Skip (no signal): {metrics.get('adds_skipped_no_signal', 0)}")
    lines.append(f"  Skip (max adds):  {metrics.get('adds_skipped_max_reached', 0)}")
    lines.append(f"  Skip (liq risk):  {metrics.get('adds_skipped_liq_unsafe', 0)}")
    lines.append(f"  Skip (too small): {metrics.get('adds_skipped_too_small', 0)}")
    lines.append(f"  Skip (no margin): {metrics.get('adds_skipped_insufficient_margin', 0)}")
    lines.append(f"  Via SMA20 pullbk: {metrics.get('adds_via_pullback_sma20', 0)}")
    lines.append(f"  Via Fib retrace:  {metrics.get('adds_via_fib_retracement', 0)}")
    lines.append(f"  Via breakout:     {metrics.get('adds_via_consolidation_breakout', 0)}")
    lines.append(f"  Fakeout closes:   {metrics.get('fakeout_closes', 0)}")
    lines.append(f"  Margin from profit:${metrics.get('total_add_margin_from_profit', 0):,.2f}")
    lines.append("")

    lines.append("── COSTS & RISK ─────────────────────────────────────────")
    lines.append(f"  Total Funding:    ${metrics.get('total_funding_cost', 0):,.2f}")
    lines.append(f"  Gross PnL:        ${metrics.get('gross_pnl', 0):,.2f}")
    lines.append(f"  Net (after fund): ${metrics.get('net_pnl_after_funding', 0):,.2f}")
    lines.append(f"  Liquidations:     {metrics.get('liquidations', 0)}")
    lines.append(f"  Liquidation Loss: ${metrics.get('liquidation_loss', 0):,.2f}")
    lines.append(f"  Scale Outs:       {metrics.get('scale_outs', 0)}")
    lines.append(f"  Aggressive Exits: {metrics.get('aggressive_exits', 0)}")
    lines.append("")

    lines.append("── TRADE LOG ────────────────────────────────────────────")
    for entry in trade_log:
        lines.append(f"  {entry}")
    lines.append("")

    lines.append("── TRADE DETAILS ────────────────────────────────────────")
    for i, t in enumerate(closed_trades, 1):
        add_str = f" (Add #{t.add_number})" if t.is_add else ""
        lines.append(
            f"  #{i}{add_str}: {t.entry_time.strftime('%Y-%m-%d')} → "
            f"{t.exit_time.strftime('%Y-%m-%d') if t.exit_time else '?'} | "
            f"${t.entry_price:,.0f} → ${t.exit_price:,.0f} | "
            f"{t.leverage}x | PnL=${t.pnl:,.2f} | "
            f"funding=${t.funding_paid:,.2f} | {t.close_reason} | "
            f"conds: {t.condition_flags}"
        )
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def plot_equity_curve(equity_data: list, closed_trades: list, save_path: str):
    eq = pd.DataFrame(equity_data)
    if eq.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[2, 1, 1])
    fig.suptitle("V16 — 肥宅 (Feizhai) Rolling Strategy", fontsize=14, y=0.98)

    # Panel 1: Equity curve
    ax1 = axes[0]
    ax1.plot(eq["timestamp"], eq["equity"], color="royalblue", linewidth=1.5,
             label="Equity")
    ax1.axhline(y=CAPITAL, color="gray", linestyle="--", alpha=0.5,
                label=f"Starting ${CAPITAL:,.0f}")
    ax1.fill_between(eq["timestamp"], CAPITAL, eq["equity"],
                     where=eq["equity"] >= CAPITAL, alpha=0.15, color="green")
    ax1.fill_between(eq["timestamp"], CAPITAL, eq["equity"],
                     where=eq["equity"] < CAPITAL, alpha=0.15, color="red")

    for t in closed_trades:
        if not t.is_add:
            ax1.axvline(x=t.entry_time, color="green", alpha=0.3, linewidth=0.8)
        if t.exit_time and not t.is_add:
            color = "red" if t.pnl < 0 else "blue"
            ax1.axvline(x=t.exit_time, color=color, alpha=0.3, linewidth=0.8)

    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: BTC price with entry/exit markers
    ax2 = axes[1]
    ax2.plot(eq["timestamp"], eq["price"], color="orange", linewidth=1,
             alpha=0.8, label="BTC Price")
    for t in closed_trades:
        if not t.is_add:
            ax2.scatter(t.entry_time, t.entry_price, marker="^", color="green",
                        s=80, zorder=5)
        else:
            ax2.scatter(t.entry_time, t.entry_price, marker="^", color="lime",
                        s=40, zorder=5, alpha=0.7)
        if t.exit_time:
            marker_color = "red" if t.pnl < 0 else "blue"
            ax2.scatter(t.exit_time, t.exit_price, marker="v",
                        color=marker_color, s=60, zorder=5)
    ax2.set_ylabel("BTC Price ($)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Drawdown
    ax3 = axes[2]
    equity = eq["equity"].values
    peak = np.maximum.accumulate(equity)
    dd_pct = (peak - equity) / peak * 100
    ax3.fill_between(eq["timestamp"], 0, dd_pct, color="red", alpha=0.3)
    ax3.plot(eq["timestamp"], dd_pct, color="red", linewidth=0.8)
    ax3.set_ylabel("Drawdown (%)")
    ax3.set_xlabel("Date")
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved equity curve to {save_path}")


def plot_comparison_3way(v14_path: str, v15_path: str, v16_metrics: dict, save_path: str):
    """Generate V14 vs V15 vs V16 comparison chart."""
    with open(v14_path) as f:
        v14 = json.load(f)
    with open(v15_path) as f:
        v15 = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("V14 (Time-Based) vs V15 (Profit+Safety) vs V16 (肥宅 Pyramid)",
                 fontsize=13)

    # Panel 1: Bar comparison
    ax1 = axes[0]
    labels = ["Return %", "Max DD %", "Sharpe", "Win Rate %", "Trades"]
    v14_vals = [v14.get("total_return_pct", 0), -v14.get("max_drawdown_pct", 0),
                v14.get("sharpe_ratio", 0), v14.get("win_rate_pct", 0),
                v14.get("total_trades", 0)]
    v15_vals = [v15.get("total_return_pct", 0), -v15.get("max_drawdown_pct", 0),
                v15.get("sharpe_ratio", 0), v15.get("win_rate_pct", 0),
                v15.get("total_trades", 0)]
    v16_vals = [v16_metrics.get("total_return_pct", 0), -v16_metrics.get("max_drawdown_pct", 0),
                v16_metrics.get("sharpe_ratio", 0), v16_metrics.get("win_rate_pct", 0),
                v16_metrics.get("total_trades", 0)]

    x = np.arange(len(labels))
    width = 0.25
    bars1 = ax1.bar(x - width, v14_vals, width, label="V14 (time adds)",
                    color="steelblue", alpha=0.8)
    bars2 = ax1.bar(x, v15_vals, width, label="V15 (profit adds)",
                    color="darkorange", alpha=0.8)
    bars3 = ax1.bar(x + width, v16_vals, width, label="V16 (肥宅 pyramid)",
                    color="forestgreen", alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_title("Key Metrics Comparison")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h,
                     f'{h:.1f}', ha='center', va='bottom' if h >= 0 else 'top',
                     fontsize=7)

    # Panel 2: Table
    ax2 = axes[1]
    ax2.axis("off")
    table_data = [
        ["Metric", "V14 (time)", "V15 (profit)", "V16 (肥宅)"],
        ["Final Equity",
         f"${v14.get('final_equity', 0):,.2f}",
         f"${v15.get('final_equity', 0):,.2f}",
         f"${v16_metrics.get('final_equity', 0):,.2f}"],
        ["Return",
         f"{v14.get('total_return_pct', 0):+.2f}%",
         f"{v15.get('total_return_pct', 0):+.2f}%",
         f"{v16_metrics.get('total_return_pct', 0):+.2f}%"],
        ["Max Drawdown",
         f"{v14.get('max_drawdown_pct', 0):.2f}%",
         f"{v15.get('max_drawdown_pct', 0):.2f}%",
         f"{v16_metrics.get('max_drawdown_pct', 0):.2f}%"],
        ["Sharpe",
         f"{v14.get('sharpe_ratio', 0):.2f}",
         f"{v15.get('sharpe_ratio', 0):.2f}",
         f"{v16_metrics.get('sharpe_ratio', 0):.2f}"],
        ["Total Trades",
         str(v14.get("total_trades", 0)),
         str(v15.get("total_trades", 0)),
         str(v16_metrics.get("total_trades", 0))],
        ["Win Rate",
         f"{v14.get('win_rate_pct', 0):.1f}%",
         f"{v15.get('win_rate_pct', 0):.1f}%",
         f"{v16_metrics.get('win_rate_pct', 0):.1f}%"],
        ["Profit Factor",
         f"{v14.get('profit_factor', 0):.2f}",
         f"{v15.get('profit_factor', 0):.2f}",
         f"{v16_metrics.get('profit_factor', 0):.2f}"],
        ["Funding Cost",
         f"${v14.get('total_funding_cost', 0):,.2f}",
         f"${v15.get('total_funding_cost', 0):,.2f}",
         f"${v16_metrics.get('total_funding_cost', 0):,.2f}"],
        ["Gross PnL",
         f"${v14.get('gross_pnl', 0):,.2f}",
         f"${v15.get('gross_pnl', 0):,.2f}",
         f"${v16_metrics.get('gross_pnl', 0):,.2f}"],
        ["Liquidations",
         str(v14.get("liquidations", 0)),
         str(v15.get("liquidations", 0)),
         str(v16_metrics.get("liquidations", 0))],
        ["Adds",
         str(v14.get("rolling_adds", 0)),
         str(v15.get("profit_adds", 0)),
         str(v16_metrics.get("pyramid_adds", 0))],
        ["Add Method", "time milestones", "profit+safety", "肥宅 pyramid"],
        ["Add Sizing", "increasing", "uncapped", "75/50/25%"],
        ["Add Stops", "breakeven", "breakeven", "1.5x ATR"],
        ["Fakeout Mgmt", "none", "none", "3-week close"],
        ["Master Stop", "3x ATR", "3x ATR", "3x ATR"],
        ["Leverage", "5x→50x", "5x fixed", "5x fixed"],
    ]

    table = ax2.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)

    for j in range(4):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(table_data)):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(4):
            table[i, j].set_facecolor(color)

    ax2.set_title("Detailed 3-Way Comparison", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved 3-way comparison to {save_path}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load and resample data
    df_4h = fetch_ohlcv(use_cache=True)
    print(f"Loaded {len(df_4h)} 4H candles: "
          f"{df_4h['timestamp'].iloc[0]} → {df_4h['timestamp'].iloc[-1]}")

    weekly = resample_to_weekly(df_4h)
    print(f"Resampled to {len(weekly)} weekly candles: "
          f"{weekly['timestamp'].iloc[0]} → {weekly['timestamp'].iloc[-1]}")

    # Run backtest
    bt = LongTermWeeklyBacktest(weekly, capital=CAPITAL)
    metrics = bt.run()

    # Save results
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Saved metrics to {metrics_path}")

    report = generate_report(metrics, bt.trade_log, bt.closed_trades)
    report_path = os.path.join(RESULTS_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved report to {report_path}")

    plot_equity_curve(bt.equity_curve, bt.closed_trades,
                      os.path.join(RESULTS_DIR, "equity_curve.png"))

    # V14 vs V15 vs V16 comparison
    v14_path = os.path.join(os.path.dirname(__file__), "results_v14", "metrics.json")
    v15_path = os.path.join(os.path.dirname(__file__), "results_v15", "metrics.json")
    if os.path.exists(v14_path) and os.path.exists(v15_path):
        plot_comparison_3way(v14_path, v15_path, metrics,
                            os.path.join(RESULTS_DIR, "comparison_v14_v15_v16.png"))

    # Print summary
    print(f"\n{'='*50}")
    print(f"  V16 RESULTS (肥宅 ROLLING STRATEGY)")
    print(f"{'='*50}")
    print(f"  Return:       {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"  Final Equity: ${metrics.get('final_equity', 0):,.2f}")
    print(f"  Max DD:       {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Sharpe:       {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Trades:       {metrics.get('total_trades', 0)}")
    print(f"  Win Rate:     {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"  Pyramid Adds: {metrics.get('pyramid_adds', 0)}")
    print(f"  Fakeout Cls:  {metrics.get('fakeout_closes', 0)}")
    print(f"  Funding Cost: ${metrics.get('total_funding_cost', 0):,.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

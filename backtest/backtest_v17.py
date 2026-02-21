"""Backtest V17 — UNLEASHED ROLLING STRATEGY.

Based on V16 with AGGRESSIVE changes per Master's directive.

KEY CHANGES FROM V16 (conservative → aggressive):
  1. POSITION SIZE: 20% of capital as margin (was 5%)
  2. LEVERAGE: 10x base, 15x on adds (was 5x fixed)
  3. ADD MARGIN: From additional capital + floating profits (was floating only)
     - Uses 50% of TOTAL available capital (not just profits)
  4. FAKEOUT WINDOW: Removed entirely (was 3 weeks auto-close)
     - Adds use their own ATR stop instead
  5. LIQUIDATION SAFETY: Softened — only checks individual add liq, not combined
  6. ADD SIGNALS: Loosened — removed uptrend requirement for pullback adds
     - Pullback to SMA20 works even if price is below SMA20 (catching the bounce)
     - Breakout only needs > 3-week high (was 4)
  7. MAX ADDS: 5 (was 3)
  8. PYRAMID SIZING: Flatter — 80/60/50/40/30% (was 75/50/25)
  9. ADD STOPS: 2x ATR (was 1.5x) — more breathing room
  10. TRAILING STOP: 2.5x ATR (was 3x) — lock profits faster on base

ENTRY — ANY ONE condition sufficient (same as V16):
  1. MVRV proxy < 0.8
  2. Price within 30% of 365-day low AND Fear proxy active
  3. Weekly MACD 13/34 single divergence
  4. Price crosses ABOVE 20-week SMA from below
  5. Price within 20% of 52-week low

EXIT:
  - Master trailing stop: 2.5x weekly ATR from highest point (base position)
  - Individual add stop: 2x weekly ATR from add entry price
  - Scale out when MVRV proxy > 2.5 (take 20% off)
  - Aggressive exit when MVRV proxy > 3.5

FUNDING: 0.01% per 8h (21 periods per weekly bar)
Starting capital: $10,000. Commission: 0.1%.
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v17")

# ── Constants ─────────────────────────────────────────────────────────

CAPITAL = 10_000.0
COMMISSION_RATE = 0.001          # 0.1%
FUNDING_RATE_PER_8H = 0.0001    # 0.01% per 8h

SWING_LOOKBACK = 10
MVRV_LOOKBACK = 52
MVRV_UNDERVALUED = 0.8
FEAR_PROXIMITY_PCT = 0.30
SOPR_DECLINE_WEEKS = 3
FIFTY_TWO_WEEK_PROXIMITY = 0.20

# AGGRESSIVE: tighter trailing, wider add stops
TRAILING_ATR_MULT = 2.5          # 2.5x weekly ATR trailing stop (base) — lock profits
ADD_ATR_MULT = 2.0               # 2x weekly ATR stop for adds — breathing room
ENTRY_LEVERAGE = 10              # 10x base leverage
ADD_LEVERAGE = 15                # 15x on adds — compound harder
ENTRY_MARGIN_PCT = 0.20          # 20% of capital as initial margin

MVRV_SCALEOUT = 2.5
MVRV_AGGRESSIVE_EXIT = 3.5

# AGGRESSIVE pyramid sizing (flatter = more firepower)
PYRAMID_SIZES = [0.80, 0.60, 0.50, 0.40, 0.30]  # Add 1-5
MAX_ADDS = 5

# ADD: use capital + profits, not just profits
ADD_CAPITAL_SHARE = 0.50         # 50% of available capital for each add
ADD_MIN_SIZE_BTC = 0.001

# Loosened add signals
CONSOLIDATION_LOOKBACK = 3       # 3-week high breakout (was 4)
FIB_RETRACE_LOW = 0.382          # Wider fib range: 0.382-0.618 (was 0.5-0.618)
FIB_RETRACE_HIGH = 0.618

SWING_LOW_LOOKBACK = 10

# Cooldown between base entries
MIN_COOLDOWN = 2                 # weeks (was 4)


# ══════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════

def resample_to_weekly(df_4h: pd.DataFrame) -> pd.DataFrame:
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

def detect_swings(highs, lows, lookback):
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

def check_bottom_divergence(current_idx, hist, lows, lookback=80):
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
    return (min(divergence_count, 3), 0)


# ══════════════════════════════════════════════════════════════════════
#  TRADE DATA STRUCTURE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_price: float
    entry_idx: int
    entry_time: pd.Timestamp
    size_btc: float
    margin: float
    leverage: float
    stop_loss: float
    liquidation_price: float
    conditions_met: int
    condition_flags: str
    is_add: bool = False
    add_number: int = 0
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

class UnleashedBacktest:
    """V17: Unleashed rolling — bigger positions, higher leverage, looser guards."""

    def __init__(self, weekly_df, capital=CAPITAL):
        self.initial_capital = capital
        self.capital = capital
        self.max_capital = capital
        self.w = weekly_df
        self.active_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.equity_curve = []
        self.trade_log = []

        self.base_entry_idx = -1
        self.base_margin = 0.0
        self.add_count = 0

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
            "adds_skipped_no_signal": 0,
            "adds_skipped_liq_unsafe": 0,
            "adds_skipped_too_small": 0,
            "adds_skipped_max_reached": 0,
            "adds_skipped_insufficient_margin": 0,
            "adds_via_pullback_sma20": 0,
            "adds_via_fib_retracement": 0,
            "adds_via_consolidation_breakout": 0,
            "fakeout_closes": 0,
            "total_add_margin": 0.0,
            "scale_outs": 0,
            "aggressive_exits": 0,
        }

    # ── Entry conditions (same as V16) ───────────────────────────────

    def _mvrv_proxy(self, idx):
        start = max(0, idx - MVRV_LOOKBACK)
        window = self.w["close"].values[start:idx + 1]
        if len(window) < 10:
            return 1.0
        avg = np.mean(window)
        return self.w["close"].values[idx] / avg if avg > 0 else 1.0

    def _check_mvrv_undervalued(self, idx):
        return self._mvrv_proxy(idx) < MVRV_UNDERVALUED

    def _check_fear_and_proximity(self, idx):
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
        if np.isnan(sma50[idx]) or closes[idx] >= sma50[idx]:
            return False
        for k in range(1, SOPR_DECLINE_WEEKS + 1):
            if idx - k < 0 or closes[idx - k + 1] >= closes[idx - k]:
                return False
        return True

    def _check_macd_divergence(self, idx):
        hist = self.w["macd_hist"].values
        lows = self.w["low"].values
        if idx < 3 or (hist[idx] >= 0 and hist[idx - 1] >= 0):
            return False
        div = check_bottom_divergence(idx, hist, lows, lookback=80)
        return div is not None and div[0] >= 1

    def _check_sma20_crossover(self, idx):
        if idx < 1:
            return False
        closes = self.w["close"].values
        sma20 = self.w["sma20"].values
        if np.isnan(sma20[idx]) or np.isnan(sma20[idx - 1]):
            return False
        return closes[idx - 1] < sma20[idx - 1] and closes[idx] > sma20[idx]

    def _check_52w_low_proximity(self, idx):
        start = max(0, idx - MVRV_LOOKBACK)
        window = self.w["low"].values[start:idx + 1]
        if len(window) < 10:
            return False
        low_52w = np.min(window)
        price = self.w["close"].values[idx]
        return price <= low_52w * (1 + FIFTY_TWO_WEEK_PROXIMITY)

    def _evaluate_entry(self, idx):
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

    # ── LOOSENED Add Signals ──────────────────────────────────────────

    def _check_add_signal(self, idx, price):
        """Loosened add signal checks — removed strict uptrend requirement."""
        sma20 = self.w["sma20"].values
        lows = self.w["low"].values
        highs = self.w["high"].values
        volumes = self.w["volume"].values

        # Signal A: Pullback to 20-week MA — works even below SMA20
        # (catching the bounce back up to moving average)
        if not np.isnan(sma20[idx]):
            distance_to_sma = abs(price - sma20[idx]) / sma20[idx]
            if distance_to_sma < 0.03:  # within 3% of SMA20
                return True, "pullback_sma20"
            # Also trigger if low touched SMA20
            if lows[idx] <= sma20[idx] <= price:
                return True, "pullback_sma20"

        # Signal B: Fib 0.382-0.618 retracement (wider range)
        if self.base_entry_idx >= 0:
            swing_low_start = max(0, self.base_entry_idx - 20)
            swing_low = np.min(lows[swing_low_start:self.base_entry_idx + 1])
            swing_high = np.max(highs[self.base_entry_idx:idx + 1])
            if swing_high > swing_low:
                move = swing_high - swing_low
                fib_382 = swing_high - move * FIB_RETRACE_LOW
                fib_618 = swing_high - move * FIB_RETRACE_HIGH
                if fib_618 <= price <= fib_382:
                    return True, "fib_retracement"

        # Signal C: Breakout > 3-week high with volume
        if idx >= CONSOLIDATION_LOOKBACK:
            prev_high = np.max(highs[idx - CONSOLIDATION_LOOKBACK:idx])
            if price > prev_high:
                avg_vol = np.mean(volumes[idx - CONSOLIDATION_LOOKBACK:idx])
                if avg_vol > 0 and volumes[idx] > avg_vol * 0.8:  # 80% of avg vol OK
                    return True, "consolidation_breakout"

        return False, ""

    # ── Execution ─────────────────────────────────────────────────────

    def run(self):
        print(f"\n[V17 UNLEASHED] Running on {len(self.w)} weekly candles...")
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

        for i in range(len(self.w)):
            price = closes[i]
            high = highs[i]
            low = lows[i]
            ts = self.w["timestamp"].iloc[i]
            w_atr = atr[i] if not np.isnan(atr[i]) else 0

            self._deduct_funding(price, periods=21)
            self._manage_positions(i, high, low, price, ts, w_atr)

            # Check adds
            if self.active_trades and i >= start_idx:
                self._check_add(i, price, low, ts, w_atr)

            # New entry
            if i >= start_idx and not self.active_trades:
                cond_count, cond_flags = self._evaluate_entry(i)
                if cond_count >= 1:
                    last_close_idx = max(
                        (t.exit_idx for t in self.closed_trades), default=-999
                    )
                    if i - last_close_idx >= MIN_COOLDOWN:
                        self._execute_entry(i, price, ts, w_atr, cond_count, cond_flags)

            # Record equity
            unrealized = self._calc_unrealized(price)
            self.equity_curve.append({
                "timestamp": ts,
                "equity": self.capital + unrealized,
                "capital": self.capital,
                "price": price,
            })

        # Close remaining
        if self.active_trades:
            last_price = closes[-1]
            last_ts = self.w["timestamp"].iloc[-1]
            for trade in list(self.active_trades):
                self._close_trade(trade, last_price, len(self.w) - 1,
                                  last_ts, "end_of_backtest")

        return self._compute_metrics()

    def _execute_entry(self, idx, price, ts, w_atr, cond_count, cond_flags):
        """Base entry at 10x with 20% of capital."""
        leverage = ENTRY_LEVERAGE
        self.stats["signals_generated"] += 1

        margin = self.capital * ENTRY_MARGIN_PCT
        if margin < 50:
            return

        notional = margin * leverage
        size_btc = notional / price

        stop_distance = w_atr * TRAILING_ATR_MULT if w_atr > 0 else price * 0.06
        stop_loss = price - stop_distance

        commission = notional * COMMISSION_RATE
        self.capital -= commission

        liq_price = price * (1 - 1 / leverage)
        min_stop = liq_price + (price - liq_price) * 0.10
        stop_loss = max(stop_loss, min_stop)

        trade = Trade(
            entry_price=price, entry_idx=idx, entry_time=ts,
            size_btc=size_btc, margin=margin, leverage=leverage,
            stop_loss=stop_loss, liquidation_price=liq_price,
            conditions_met=cond_count, condition_flags=cond_flags,
            highest_price=price,
        )
        self.active_trades.append(trade)

        self.base_entry_idx = idx
        self.base_margin = margin
        self.add_count = 0

        self.trade_log.append(
            f"[{ts.strftime('%Y-%m-%d')}] ENTRY: ${price:,.0f} | "
            f"{leverage}x | margin=${margin:,.0f} | "
            f"size={size_btc:.4f} BTC | "
            f"conds={cond_count} ({cond_flags}) | SL=${stop_loss:,.0f} | "
            f"liq=${liq_price:,.0f}"
        )

    def _check_add(self, idx, price, current_low, ts, w_atr):
        """Aggressive pyramid add — uses capital + profits, no fakeout auto-close."""
        if self.base_entry_idx < 0:
            return

        self.stats["adds_attempted"] += 1

        if self.add_count >= MAX_ADDS:
            self.stats["adds_skipped_max_reached"] += 1
            return

        # Must have SOME profit (at least breakeven)
        agg_pnl = sum(
            (price - t.entry_price) * t.size_btc
            for t in self.active_trades if not t.closed
        )
        if agg_pnl < 0:
            self.stats["adds_skipped_no_profit"] += 1
            return

        # Check add signal
        signal_fired, signal_type = self._check_add_signal(idx, price)
        if not signal_fired:
            self.stats["adds_skipped_no_signal"] += 1
            return

        # AGGRESSIVE: Available margin = portion of remaining capital + floating profit
        remaining_capital = self.capital - sum(
            t.margin for t in self.active_trades if not t.closed
        )
        available = max(0, remaining_capital * ADD_CAPITAL_SHARE + max(0, agg_pnl) * 0.5)

        # Pyramid cap
        pyramid_fraction = PYRAMID_SIZES[self.add_count]
        pyramid_max = self.base_margin * pyramid_fraction
        add_margin = min(available, pyramid_max)

        if add_margin < 50:
            self.stats["adds_skipped_insufficient_margin"] += 1
            return

        leverage = ADD_LEVERAGE
        add_notional = add_margin * leverage
        add_btc = add_notional / price

        # Simplified liq check — only individual position
        liq_price = price * (1 - 1 / leverage)

        if add_btc < ADD_MIN_SIZE_BTC:
            self.stats["adds_skipped_too_small"] += 1
            return

        commission = add_notional * COMMISSION_RATE
        self.capital -= commission

        add_stop = price - w_atr * ADD_ATR_MULT if w_atr > 0 else price * 0.93
        min_stop = liq_price + (price - liq_price) * 0.10
        add_stop = max(add_stop, min_stop)

        self.stats["total_add_margin"] += add_margin
        if signal_type == "pullback_sma20":
            self.stats["adds_via_pullback_sma20"] += 1
        elif signal_type == "fib_retracement":
            self.stats["adds_via_fib_retracement"] += 1
        elif signal_type == "consolidation_breakout":
            self.stats["adds_via_consolidation_breakout"] += 1

        base_trades = [t for t in self.active_trades if not t.is_add and not t.closed]
        base = base_trades[0] if base_trades else self.active_trades[0]

        self.add_count += 1

        add_trade = Trade(
            entry_price=price, entry_idx=idx, entry_time=ts,
            size_btc=add_btc, margin=add_margin, leverage=leverage,
            stop_loss=add_stop, liquidation_price=liq_price,
            conditions_met=base.conditions_met, condition_flags=base.condition_flags,
            is_add=True, add_number=self.add_count,
            highest_price=price,
        )
        self.active_trades.append(add_trade)
        self.stats["pyramid_adds"] += 1

        total_btc = sum(t.size_btc for t in self.active_trades if not t.closed)
        total_margin = sum(t.margin for t in self.active_trades if not t.closed)
        total_notional = sum(t.size_btc * price for t in self.active_trades if not t.closed)
        eff_lev = total_notional / total_margin if total_margin > 0 else leverage

        self.trade_log.append(
            f"[{ts.strftime('%Y-%m-%d')}] ADD #{self.add_count} ({signal_type}): "
            f"${price:,.0f} | {leverage}x | "
            f"margin=${add_margin:,.0f} ({pyramid_fraction:.0%} cap) | "
            f"size={add_btc:.4f} BTC | pnl=${agg_pnl:,.0f} | "
            f"total_btc={total_btc:.4f} | eff_lev={eff_lev:.1f}x | "
            f"stop=${add_stop:,.0f}"
        )

    def _deduct_funding(self, price, periods=21):
        for trade in self.active_trades:
            if trade.closed:
                continue
            notional = trade.size_btc * price
            cost = notional * FUNDING_RATE_PER_8H * periods
            self.capital -= cost
            trade.funding_paid += cost
            self.stats["total_funding_paid"] += cost

    def _manage_positions(self, idx, high, low, price, ts, w_atr):
        mvrv = self._mvrv_proxy(idx)

        for trade in list(self.active_trades):
            if trade.closed:
                continue

            trade.highest_price = max(trade.highest_price, high)

            # Liquidation check first
            if low <= trade.liquidation_price:
                self.stats["liquidations"] += 1
                self.stats["liquidation_loss"] += trade.margin
                self._close_trade(trade, trade.liquidation_price, idx, ts, "liquidation")
                continue

            if trade.is_add:
                # Add stop (ATR-based, no fakeout auto-close)
                # Trail the add stop too when profitable
                if w_atr > 0:
                    new_trail = trade.highest_price - w_atr * ADD_ATR_MULT
                    trade.stop_loss = max(trade.stop_loss, new_trail)

                if low <= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, idx, ts, "add_stop")
                    continue
            else:
                # Base trailing stop
                if w_atr > 0:
                    new_trail = trade.highest_price - w_atr * TRAILING_ATR_MULT
                    trade.stop_loss = max(trade.stop_loss, new_trail)

                if low <= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, idx, ts, "trailing_stop")
                    continue

            # Scale out: MVRV > 2.5
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
                    f"[{ts.strftime('%Y-%m-%d')}] SCALE OUT 20%: "
                    f"${price:,.0f} | MVRV={mvrv:.2f} | pnl=${scale_pnl:,.0f} | "
                    f"add={trade.add_number}"
                )

            # Aggressive exit: MVRV > 3.5
            if mvrv > MVRV_AGGRESSIVE_EXIT:
                self.stats["aggressive_exits"] += 1
                self._close_trade(trade, price, idx, ts, "mvrv_aggressive_exit")

    def _close_trade(self, trade, exit_price, idx, ts, reason):
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

        # Base closed → close all adds
        if not trade.is_add:
            for add_trade in list(self.active_trades):
                if add_trade.is_add and not add_trade.closed:
                    self._close_trade(add_trade, exit_price, idx, ts, "base_closed")
            self.base_entry_idx = -1
            self.base_margin = 0.0
            self.add_count = 0

    def _calc_unrealized(self, price):
        return sum(
            (price - t.entry_price) * t.size_btc
            for t in self.active_trades if not t.closed
        )

    def _compute_metrics(self):
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
                "system": "v17_unleashed",
                "total_return_pct": round(total_return, 2),
                "final_equity": round(equity[-1], 2),
                "initial_capital": self.initial_capital,
                "max_drawdown_pct": round(max_dd, 2),
                "sharpe_ratio": round(sharpe, 2),
                "total_trades": 0,
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
                durations.append((t.exit_time - t.entry_time).total_seconds() / 86400)

        base_trades = [t for t in self.closed_trades if not t.is_add]
        add_trades = [t for t in self.closed_trades if t.is_add]

        return {
            "system": "v17_unleashed",
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
            "gross_pnl": round(sum(pnls), 2),
            "total_funding_cost": round(self.stats["total_funding_paid"], 2),
            "net_pnl_after_funding": round(sum(pnls) - self.stats["total_funding_paid"], 2),
            **self.stats,
        }


# ══════════════════════════════════════════════════════════════════════
#  REPORTING & MAIN
# ══════════════════════════════════════════════════════════════════════

def generate_report(metrics, trade_log, closed_trades):
    lines = []
    lines.append("=" * 70)
    lines.append("  BACKTEST V17 — UNLEASHED ROLLING STRATEGY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"CAPITAL:     ${metrics.get('initial_capital', 0):,.0f}")
    lines.append(f"BASE LEV:    {ENTRY_LEVERAGE}x | ADD LEV: {ADD_LEVERAGE}x")
    lines.append(f"MARGIN:      {ENTRY_MARGIN_PCT:.0%} of capital per entry")
    lines.append(f"ADD SOURCE:  Capital + Profits (50% available)")
    lines.append(f"PYRAMID:     {PYRAMID_SIZES} (max {MAX_ADDS} adds)")
    lines.append(f"TRAIL STOP:  {TRAILING_ATR_MULT}x ATR (base) | {ADD_ATR_MULT}x ATR (add)")
    lines.append(f"FAKEOUT:     DISABLED (adds use ATR stops)")
    lines.append("")

    lines.append("── PERFORMANCE ──────────────────────────────────────────")
    for k in ["final_equity", "total_return_pct", "max_drawdown_pct",
              "sharpe_ratio", "profit_factor", "win_rate_pct"]:
        v = metrics.get(k, 0)
        if "pct" in k:
            lines.append(f"  {k:20s}: {v:+.2f}%")
        elif "equity" in k or "capital" in k:
            lines.append(f"  {k:20s}: ${v:,.2f}")
        else:
            lines.append(f"  {k:20s}: {v:.2f}")
    lines.append("")

    lines.append("── TRADE SUMMARY ────────────────────────────────────────")
    for k in ["total_trades", "base_trades", "pyramid_adds",
              "winning_trades", "losing_trades", "avg_win", "avg_loss",
              "largest_win", "largest_loss", "avg_duration_days"]:
        v = metrics.get(k, 0)
        if isinstance(v, float):
            lines.append(f"  {k:20s}: ${v:,.2f}" if "win" in k or "loss" in k
                         else f"  {k:20s}: {v:.1f}")
        else:
            lines.append(f"  {k:20s}: {v}")
    lines.append("")

    lines.append("── ADD STATS ────────────────────────────────────────────")
    for k in ["pyramid_adds", "adds_attempted", "adds_skipped_no_profit",
              "adds_skipped_no_signal", "adds_skipped_max_reached",
              "adds_skipped_liq_unsafe", "adds_skipped_too_small",
              "adds_skipped_insufficient_margin",
              "adds_via_pullback_sma20", "adds_via_fib_retracement",
              "adds_via_consolidation_breakout", "fakeout_closes"]:
        lines.append(f"  {k:30s}: {metrics.get(k, 0)}")
    lines.append(f"  {'total_add_margin':30s}: ${metrics.get('total_add_margin', 0):,.2f}")
    lines.append("")

    lines.append("── COSTS ────────────────────────────────────────────────")
    lines.append(f"  Funding:     ${metrics.get('total_funding_cost', 0):,.2f}")
    lines.append(f"  Gross PnL:   ${metrics.get('gross_pnl', 0):,.2f}")
    lines.append(f"  Net PnL:     ${metrics.get('net_pnl_after_funding', 0):,.2f}")
    lines.append(f"  Liquidations:{metrics.get('liquidations', 0)}")
    lines.append(f"  Liq Loss:    ${metrics.get('liquidation_loss', 0):,.2f}")
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
            f"{t.condition_flags}"
        )
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def plot_equity_curve(equity_data, closed_trades, save_path):
    eq = pd.DataFrame(equity_data)
    if eq.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[2, 1, 1])
    fig.suptitle("V17 — UNLEASHED Rolling Strategy", fontsize=14, y=0.98)

    ax1 = axes[0]
    ax1.plot(eq["timestamp"], eq["equity"], color="royalblue", linewidth=1.5, label="Equity")
    ax1.axhline(y=CAPITAL, color="gray", linestyle="--", alpha=0.5, label=f"Start ${CAPITAL:,.0f}")
    ax1.fill_between(eq["timestamp"], CAPITAL, eq["equity"],
                     where=eq["equity"] >= CAPITAL, alpha=0.15, color="green")
    ax1.fill_between(eq["timestamp"], CAPITAL, eq["equity"],
                     where=eq["equity"] < CAPITAL, alpha=0.15, color="red")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(eq["timestamp"], eq["price"], color="orange", linewidth=1, alpha=0.8, label="BTC")
    for t in closed_trades:
        color = "green" if not t.is_add else "lime"
        size = 80 if not t.is_add else 40
        ax2.scatter(t.entry_time, t.entry_price, marker="^", color=color, s=size, zorder=5)
        if t.exit_time:
            mc = "red" if t.pnl < 0 else "blue"
            ax2.scatter(t.exit_time, t.exit_price, marker="v", color=mc, s=60, zorder=5)
    ax2.set_ylabel("BTC Price ($)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_4h = fetch_ohlcv(use_cache=True)
    print(f"Loaded {len(df_4h)} 4H candles: "
          f"{df_4h['timestamp'].iloc[0]} → {df_4h['timestamp'].iloc[-1]}")

    weekly = resample_to_weekly(df_4h)
    print(f"Resampled to {len(weekly)} weekly candles")

    bt = UnleashedBacktest(weekly, capital=CAPITAL)
    metrics = bt.run()

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    report = generate_report(metrics, bt.trade_log, bt.closed_trades)
    with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
        f.write(report)
    print(f"  Saved report")

    plot_equity_curve(bt.equity_curve, bt.closed_trades,
                      os.path.join(RESULTS_DIR, "equity_curve.png"))

    print(f"\n{'='*50}")
    print(f"  V17 UNLEASHED RESULTS")
    print(f"{'='*50}")
    print(f"  Return:       {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"  Final Equity: ${metrics.get('final_equity', 0):,.2f}")
    print(f"  Max DD:       {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Sharpe:       {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Trades:       {metrics.get('total_trades', 0)}")
    print(f"  Win Rate:     {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"  Adds:         {metrics.get('pyramid_adds', 0)}")
    print(f"  Liquidations: {metrics.get('liquidations', 0)}")
    print(f"  Funding:      ${metrics.get('total_funding_cost', 0):,.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

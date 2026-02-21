"""Backtest V13 — LONG-TERM WEEKLY with LOOSENED entry conditions.

Based on V12. LONG ONLY on weekly resampled data.

ENTRY — ANY ONE condition is sufficient to open long:
  1. MVRV proxy < 0.8 (price / 365-day-avg ratio)
  2. Price within 30% of 365-day low AND Fear proxy active
     (price below 50-week SMA declining 3+ weeks)
  3. Weekly MACD 13/34 single divergence (histogram higher low
     while price makes lower low)
  4. Price crosses ABOVE 20-week SMA from below (trend reversal)

LEVERAGE scales with how many signals align:
  1 signal  → 5x
  2 signals → 10x
  3 signals → 25x
  4 signals → 50x

ROLLING POSITIONS (inverted pyramid):
  - After +5% profitable, check every 2 weekly bars
  - Add on pullback to 20-week MA or Fib 0.382-0.618
  - Sizes: 150% → 200% → 250% of base
  - Can use floating profits + additional capital (up to $6K allocated)
  - Liquidation must stay below previous swing low
  - Each add gets breakeven stop at entry

EXIT:
  - Master trailing stop: 2x weekly ATR from highest point
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v13")

# ── Constants ─────────────────────────────────────────────────────────

CAPITAL = 6_000.0
COMMISSION_RATE = 0.001          # 0.1%
FUNDING_RATE_PER_8H = 0.0001    # 0.01% per 8h

SWING_LOOKBACK = 10              # weekly bars for swing detection
MVRV_LOOKBACK = 52               # 52 weeks ≈ 365 days
MVRV_UNDERVALUED = 0.8           # price/365d-avg ratio threshold
FEAR_PROXIMITY_PCT = 0.30        # within 30% of 365-day low (loosened from 15%)
SOPR_DECLINE_WEEKS = 3           # consecutive declining weeks below SMA50

TRAILING_ATR_MULT = 2.0          # 2x weekly ATR trailing stop
ROLLING_PROFIT_TRIGGER = 0.05    # +5% to trigger add check
ROLLING_CHECK_INTERVAL = 2       # every 2 weekly bars
ROLLING_SCALES = [1.5, 2.0, 2.5] # inverted pyramid multipliers

MVRV_SCALEOUT = 2.5              # take 20% off
MVRV_AGGRESSIVE_EXIT = 3.5       # aggressive exit

# V13: 1 signal minimum, leverage tiers for 1-4 conditions
LEVERAGE_TIERS = {1: 5, 2: 10, 3: 25, 4: 50}


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
    """Classical bullish divergence: price lower lows + MACD higher lows.
    Returns (strength, peak_diff) or None."""
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
        # Price lower low + MACD higher low = bullish divergence
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
    stop_loss: float          # trailing stop
    liquidation_price: float
    conditions_met: int       # how many entry conditions were met
    condition_flags: str      # which conditions triggered
    is_add: bool = False
    add_number: int = 0       # 0=base, 1/2/3=adds
    breakeven_stop: float = 0.0
    exit_price: float = 0.0
    exit_idx: int = 0
    exit_time: pd.Timestamp = None
    pnl: float = 0.0
    funding_paid: float = 0.0
    closed: bool = False
    close_reason: str = ""
    highest_price: float = 0.0
    scale_out_done: bool = False  # already took 20% off at MVRV > 2.5


# ══════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════

class LongTermWeeklyBacktest:
    """V13: Long-only weekly accumulation — loosened entry (ANY 1 signal)."""

    def __init__(self, weekly_df: pd.DataFrame, capital: float = CAPITAL):
        self.initial_capital = capital
        self.capital = capital
        self.max_capital = capital  # track allocated capital ceiling
        self.w = weekly_df
        self.active_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.equity_curve = []
        self.add_count = 0
        self.last_add_idx = -999  # for 2-bar interval check
        self.trade_log = []

        # Pre-compute swings
        self.swing_highs, self.swing_lows = detect_swings(
            weekly_df["high"].values, weekly_df["low"].values,
            lookback=SWING_LOOKBACK
        )

        self.stats = {
            "signals_generated": 0,
            "trades_at_5x": 0, "trades_at_10x": 0,
            "trades_at_25x": 0, "trades_at_50x": 0,
            "liquidations": 0, "liquidation_loss": 0.0,
            "total_funding_paid": 0.0,
            "rolling_adds": 0,
            "scale_outs": 0,
            "aggressive_exits": 0,
        }

    # ── Entry condition checks ────────────────────────────────────────

    def _mvrv_proxy(self, idx: int) -> float:
        """MVRV proxy: price / 365-day average price.
        Below 0.8 = undervalued (market cap below realized)."""
        start = max(0, idx - MVRV_LOOKBACK)
        window = self.w["close"].values[start:idx + 1]
        if len(window) < 10:
            return 1.0
        avg_365 = np.mean(window)
        if avg_365 == 0:
            return 1.0
        return self.w["close"].values[idx] / avg_365

    def _mvrv_range_position(self, idx: int) -> float:
        """Position in 365-day range: 0.0 = bottom, 1.0 = top."""
        start = max(0, idx - MVRV_LOOKBACK)
        window = self.w["close"].values[start:idx + 1]
        if len(window) < 10:
            return 0.5
        lo = np.min(window)
        hi = np.max(window)
        if hi == lo:
            return 0.5
        return (self.w["close"].values[idx] - lo) / (hi - lo)

    def _check_mvrv_undervalued(self, idx: int) -> bool:
        """Condition 1: MVRV proxy < 0.8."""
        return self._mvrv_proxy(idx) < MVRV_UNDERVALUED

    def _check_fear_and_proximity(self, idx: int) -> bool:
        """Condition 2: Price within 30% of 365-day low AND Fear proxy active.
        Fear proxy = price below 50-week SMA declining 3+ weeks."""
        # Check proximity to 365-day low (30%)
        start = max(0, idx - MVRV_LOOKBACK)
        window = self.w["low"].values[start:idx + 1]
        if len(window) < 10:
            return False
        low_365 = np.min(window)
        price = self.w["close"].values[idx]
        if price > low_365 * (1 + FEAR_PROXIMITY_PCT):
            return False

        # Check fear proxy: below SMA50 and declining 3+ weeks
        if idx < SOPR_DECLINE_WEEKS:
            return False
        closes = self.w["close"].values
        sma50 = self.w["sma50"].values

        if np.isnan(sma50[idx]):
            return False
        if closes[idx] >= sma50[idx]:
            return False

        # Check consecutive declining weeks
        for k in range(1, SOPR_DECLINE_WEEKS + 1):
            if idx - k < 0:
                return False
            if closes[idx - k + 1] >= closes[idx - k]:
                return False
        return True

    def _check_macd_divergence(self, idx: int) -> bool:
        """Condition 3: Weekly MACD 13/34 single divergence.
        Histogram making higher low while price makes lower low."""
        hist = self.w["macd_hist"].values
        lows = self.w["low"].values

        if idx < 3:
            return False
        if hist[idx] >= 0 and hist[idx - 1] >= 0:
            return False

        # Check for bullish divergence (classical)
        div = check_bottom_divergence(idx, hist, lows, lookback=80)
        if div is not None and div[0] >= 1:
            return True

        return False

    def _check_sma20_crossover(self, idx: int) -> bool:
        """Condition 4: Price crosses ABOVE 20-week SMA from below.
        Current close > SMA20 AND previous close < SMA20."""
        if idx < 1:
            return False
        closes = self.w["close"].values
        sma20 = self.w["sma20"].values

        if np.isnan(sma20[idx]) or np.isnan(sma20[idx - 1]):
            return False

        # Cross from below: previous below, current above
        return closes[idx - 1] < sma20[idx - 1] and closes[idx] > sma20[idx]

    def _evaluate_entry(self, idx: int) -> tuple[int, str]:
        """Evaluate all 4 entry conditions. Returns (count, flags_string).
        V13: ANY 1 condition is sufficient."""
        conditions = []
        flags = []

        if self._check_mvrv_undervalued(idx):
            conditions.append(1)
            flags.append("MVRV<0.8")

        if self._check_fear_and_proximity(idx):
            conditions.append(2)
            flags.append("fear_30%+decline")

        if self._check_macd_divergence(idx):
            conditions.append(3)
            flags.append("MACD_div")

        if self._check_sma20_crossover(idx):
            conditions.append(4)
            flags.append("SMA20_cross")

        return len(conditions), "|".join(flags)

    def _get_leverage(self, conditions_met: int) -> float:
        """Map condition count to leverage tier. V13: 1→5x, 2→10x, 3→25x, 4→50x."""
        if conditions_met >= 4:
            return 50
        return LEVERAGE_TIERS.get(conditions_met, 0)

    # ── Execution ─────────────────────────────────────────────────────

    def run(self) -> dict:
        print(f"\n[V13 Long-Term Weekly LOOSENED] Running on {len(self.w)} weekly candles...")
        print(f"  Capital: ${self.initial_capital:,.0f}")

        closes = self.w["close"].values
        highs = self.w["high"].values
        lows = self.w["low"].values
        atr = self.w["atr"].values

        # Need at least 52 weeks of data for MVRV lookback
        start_idx = max(52, SWING_LOOKBACK * 2 + 1)
        # Also need SMA50 to be available
        for i in range(start_idx, len(self.w)):
            if not np.isnan(self.w["sma50"].values[i]):
                start_idx = i
                break

        print(f"  Starting at week index {start_idx} "
              f"({self.w['timestamp'].iloc[start_idx].strftime('%Y-%m-%d')})")

        min_cooldown = 4  # minimum 4 weeks between new base entries

        for i in range(len(self.w)):
            price = closes[i]
            high = highs[i]
            low = lows[i]
            ts = self.w["timestamp"].iloc[i]
            w_atr = atr[i] if not np.isnan(atr[i]) else 0

            # Deduct funding for active positions (21 × 8h periods per week)
            self._deduct_funding(price, periods=21)

            # Manage existing positions (trailing stop, scale-out, etc.)
            self._manage_positions(i, high, low, price, ts, w_atr)

            # Check rolling add opportunities
            if self.active_trades and i >= start_idx:
                self._check_rolling_add(i, price, ts, w_atr)

            # Check for new entry — V13: only need 1+ condition
            if i >= start_idx and not self.active_trades:
                cond_count, cond_flags = self._evaluate_entry(i)
                if cond_count >= 1:
                    # Cooldown: check last closed trade
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
        leverage = self._get_leverage(cond_count)
        if leverage == 0:
            return

        self.stats["signals_generated"] += 1

        # Position sizing: risk 5% of capital, stop at 2x ATR
        stop_distance = w_atr * TRAILING_ATR_MULT if w_atr > 0 else price * 0.05
        stop_loss = price - stop_distance

        risk_amount = self.capital * 0.05
        base_size_btc = risk_amount / stop_distance
        max_margin = self.capital * 0.30  # max 30% as initial margin
        max_size = (max_margin * leverage) / price
        base_size_btc = min(base_size_btc, max_size)

        notional = base_size_btc * price
        margin = notional / leverage

        if margin < 10:
            return

        # Commission
        commission = notional * COMMISSION_RATE
        self.capital -= commission

        # Liquidation price (long)
        liq_price = price * (1 - 1 / leverage)

        # Ensure stop is above liquidation with buffer
        min_stop = liq_price + (price - liq_price) * 0.10
        stop_loss = max(stop_loss, min_stop)

        # Track leverage tier
        if leverage >= 50:
            self.stats["trades_at_50x"] += 1
        elif leverage >= 25:
            self.stats["trades_at_25x"] += 1
        elif leverage >= 10:
            self.stats["trades_at_10x"] += 1
        else:
            self.stats["trades_at_5x"] += 1

        trade = Trade(
            entry_price=price,
            entry_idx=idx,
            entry_time=ts,
            size_btc=base_size_btc,
            margin=margin,
            leverage=leverage,
            stop_loss=stop_loss,
            liquidation_price=liq_price,
            conditions_met=cond_count,
            condition_flags=cond_flags,
            highest_price=price,
        )
        self.active_trades.append(trade)
        self.add_count = 0
        self.last_add_idx = -999

        self.trade_log.append(
            f"[{ts.strftime('%Y-%m-%d')}] ENTRY: ${price:,.0f} | "
            f"{leverage}x | margin=${margin:,.0f} | "
            f"conds={cond_count} ({cond_flags}) | SL=${stop_loss:,.0f} | "
            f"liq=${liq_price:,.0f}"
        )

    def _check_rolling_add(self, idx: int, price: float, ts: pd.Timestamp,
                           w_atr: float):
        if self.add_count >= 3:
            return

        # 2-bar interval between adds
        if idx - self.last_add_idx < ROLLING_CHECK_INTERVAL:
            return

        base_trades = [t for t in self.active_trades if not t.is_add and not t.closed]
        if not base_trades:
            return
        base = base_trades[0]

        # Must be +5% profitable
        pct_profit = (price - base.entry_price) / base.entry_price
        if pct_profit < ROLLING_PROFIT_TRIGGER:
            return

        # Check pullback condition: near 20-week MA or Fib retracement
        sma20 = self.w["sma20"].values[idx]
        if np.isnan(sma20):
            return

        # Fib 0.382-0.618 of the move from entry to highest
        move = base.highest_price - base.entry_price
        if move <= 0:
            return
        fib_382 = base.highest_price - move * 0.382
        fib_618 = base.highest_price - move * 0.618

        near_sma20 = abs(price - sma20) / price < 0.015 and price >= sma20
        in_fib_zone = fib_618 <= price <= fib_382

        if not (near_sma20 or in_fib_zone):
            return

        # Determine add size using inverted pyramid
        scale = ROLLING_SCALES[self.add_count]

        # Can use floating profits + additional capital
        floating_pnl = (price - base.entry_price) * base.size_btc
        for t in self.active_trades:
            if t.is_add and not t.closed:
                floating_pnl += (price - t.entry_price) * t.size_btc

        # Available capital for adds: floating profits + remaining allocated capital
        available_extra = max(0, self.max_capital - self.capital)
        add_budget = max(0, floating_pnl * 0.5) + min(available_extra, self.capital * 0.20)
        if add_budget < 50:
            return

        # Size the add
        leverage = self._get_leverage(base.conditions_met)
        add_margin = add_budget * scale / sum(ROLLING_SCALES[:self.add_count + 1])
        add_margin = min(add_margin, add_budget)
        add_size_btc = (add_margin * leverage) / price
        notional = add_size_btc * price

        if notional < 50:
            return

        # Liquidation check: must be below previous swing low
        liq_price = price * (1 - 1 / leverage)
        confirmed_lows = [(i, v) for i, v in self.swing_lows
                          if i + SWING_LOOKBACK <= idx]
        if confirmed_lows:
            prev_swing_low = confirmed_lows[-1][1]
            if liq_price >= prev_swing_low:
                return  # liquidation too close to swing low — skip

        # Commission
        commission = notional * COMMISSION_RATE
        self.capital -= commission

        stop_loss = price - w_atr * TRAILING_ATR_MULT if w_atr > 0 else price * 0.95

        add_trade = Trade(
            entry_price=price,
            entry_idx=idx,
            entry_time=ts,
            size_btc=add_size_btc,
            margin=add_margin,
            leverage=leverage,
            stop_loss=stop_loss,
            liquidation_price=liq_price,
            conditions_met=base.conditions_met,
            condition_flags=base.condition_flags,
            is_add=True,
            add_number=self.add_count + 1,
            breakeven_stop=price,  # protected at entry
            highest_price=price,
        )
        self.active_trades.append(add_trade)
        self.add_count += 1
        self.last_add_idx = idx
        self.stats["rolling_adds"] += 1

        self.trade_log.append(
            f"[{ts.strftime('%Y-%m-%d')}] ADD #{self.add_count}: ${price:,.0f} | "
            f"{leverage}x | margin=${add_margin:,.0f} | "
            f"size={add_size_btc:.4f} BTC | liq=${liq_price:,.0f}"
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

            # Update trailing stop (2x weekly ATR from highest)
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
                    f"[{ts.strftime('%Y-%m-%d')}] SCALE OUT 20%: ${price:,.0f} | "
                    f"MVRV={mvrv:.2f} | pnl=${scale_pnl:,.0f}"
                )

            # Aggressive exit: MVRV > 3.5 → close entire position
            if mvrv > MVRV_AGGRESSIVE_EXIT:
                self.stats["aggressive_exits"] += 1
                self._close_trade(trade, price, idx, ts, "mvrv_aggressive_exit")
                continue

            # Breakeven stop for adds
            if trade.is_add and low <= trade.breakeven_stop:
                self._close_trade(trade, trade.breakeven_stop, idx, ts, "breakeven_stop")
                continue

            # Trailing stop
            if low <= trade.stop_loss:
                self._close_trade(trade, trade.stop_loss, idx, ts, "trailing_stop")
                continue

            # Liquidation (gap through stop)
            if low <= trade.liquidation_price:
                self.stats["liquidations"] += 1
                self.stats["liquidation_loss"] += trade.margin
                self._close_trade(trade, trade.liquidation_price, idx, ts, "liquidation")
                continue

    def _close_trade(self, trade: Trade, exit_price: float, idx: int,
                     ts: pd.Timestamp, reason: str):
        pnl = (exit_price - trade.entry_price) * trade.size_btc

        # Exit commission
        commission = trade.size_btc * exit_price * COMMISSION_RATE
        pnl -= commission

        # Cap loss at margin for liquidation
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
            f"add={trade.add_number}"
        )

        # Base trade closed → close all adds
        if not trade.is_add:
            for add_trade in list(self.active_trades):
                if add_trade.is_add and not add_trade.closed:
                    self._close_trade(add_trade, exit_price, idx, ts, "base_closed")
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
                "system": "v13_loosened_entry",
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
            "system": "v13_loosened_entry",
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
    lines.append("  BACKTEST V13 — LONG-TERM WEEKLY (LOOSENED ENTRY)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("STRATEGY: Long-only weekly, 4 entry conditions (ANY 1 sufficient)")
    lines.append("CONDITIONS:")
    lines.append("  1. MVRV proxy < 0.8")
    lines.append("  2. Price within 30% of 365d low + fear proxy (below SMA50 declining 3w)")
    lines.append("  3. Weekly MACD 13/34 single divergence")
    lines.append("  4. Price crosses above 20-week SMA from below")
    lines.append(f"CAPITAL:  ${metrics.get('initial_capital', 0):,.0f}")
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
    lines.append(f"  Rolling Adds:     {metrics.get('rolling_adds', 0)}")
    lines.append(f"  Winning:          {metrics.get('winning_trades', 0)}")
    lines.append(f"  Losing:           {metrics.get('losing_trades', 0)}")
    lines.append(f"  Avg Win:          ${metrics.get('avg_win', 0):,.2f}")
    lines.append(f"  Avg Loss:         ${metrics.get('avg_loss', 0):,.2f}")
    lines.append(f"  Largest Win:      ${metrics.get('largest_win', 0):,.2f}")
    lines.append(f"  Largest Loss:     ${metrics.get('largest_loss', 0):,.2f}")
    lines.append(f"  Avg Duration:     {metrics.get('avg_duration_days', 0):.1f} days")
    lines.append("")

    lines.append("── LEVERAGE DISTRIBUTION ─────────────────────────────────")
    lines.append(f"  5x trades:        {metrics.get('trades_at_5x', 0)}")
    lines.append(f"  10x trades:       {metrics.get('trades_at_10x', 0)}")
    lines.append(f"  25x trades:       {metrics.get('trades_at_25x', 0)}")
    lines.append(f"  50x trades:       {metrics.get('trades_at_50x', 0)}")
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

    # Per-trade detail
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
    fig.suptitle("V13 — Long-Term Weekly (Loosened Entry)", fontsize=14, y=0.98)

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

    # Mark entries and exits
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
                        s=50, zorder=5)
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


def plot_comparison(v12_metrics_path: str, v13_metrics: dict,
                    v13_equity_data: list, save_path: str):
    """Generate V12 vs V13 comparison chart."""
    # Load V12 metrics
    with open(v12_metrics_path) as f:
        v12 = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("V12 vs V13 — Loosened Entry Comparison", fontsize=14)

    # Panel 1: Bar comparison of key metrics
    ax1 = axes[0]
    labels = ["Return %", "Max DD %", "Sharpe", "Win Rate %", "Trades"]
    v12_vals = [
        v12.get("total_return_pct", 0),
        -v12.get("max_drawdown_pct", 0),
        v12.get("sharpe_ratio", 0),
        v12.get("win_rate_pct", 0),
        v12.get("total_trades", 0),
    ]
    v13_vals = [
        v13_metrics.get("total_return_pct", 0),
        -v13_metrics.get("max_drawdown_pct", 0),
        v13_metrics.get("sharpe_ratio", 0),
        v13_metrics.get("win_rate_pct", 0),
        v13_metrics.get("total_trades", 0),
    ]

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax1.bar(x - width/2, v12_vals, width, label="V12 (2+ conds)",
                    color="steelblue", alpha=0.8)
    bars2 = ax1.bar(x + width/2, v13_vals, width, label="V13 (1+ cond)",
                    color="darkorange", alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_title("Key Metrics Comparison")

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h,
                 f'{h:.1f}', ha='center', va='bottom' if h >= 0 else 'top',
                 fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h,
                 f'{h:.1f}', ha='center', va='bottom' if h >= 0 else 'top',
                 fontsize=8)

    # Panel 2: Financial summary table
    ax2 = axes[1]
    ax2.axis("off")
    table_data = [
        ["Metric", "V12 (2+ conds)", "V13 (1+ cond)"],
        ["Final Equity", f"${v12.get('final_equity', 0):,.2f}",
         f"${v13_metrics.get('final_equity', 0):,.2f}"],
        ["Return", f"{v12.get('total_return_pct', 0):+.2f}%",
         f"{v13_metrics.get('total_return_pct', 0):+.2f}%"],
        ["Max Drawdown", f"{v12.get('max_drawdown_pct', 0):.2f}%",
         f"{v13_metrics.get('max_drawdown_pct', 0):.2f}%"],
        ["Sharpe", f"{v12.get('sharpe_ratio', 0):.2f}",
         f"{v13_metrics.get('sharpe_ratio', 0):.2f}"],
        ["Total Trades", str(v12.get("total_trades", 0)),
         str(v13_metrics.get("total_trades", 0))],
        ["Win Rate", f"{v12.get('win_rate_pct', 0):.1f}%",
         f"{v13_metrics.get('win_rate_pct', 0):.1f}%"],
        ["Profit Factor", f"{v12.get('profit_factor', 0):.2f}",
         f"{v13_metrics.get('profit_factor', 0):.2f}"],
        ["Funding Cost", f"${v12.get('total_funding_cost', 0):,.2f}",
         f"${v13_metrics.get('total_funding_cost', 0):,.2f}"],
        ["Gross PnL", f"${v12.get('gross_pnl', 0):,.2f}",
         f"${v13_metrics.get('gross_pnl', 0):,.2f}"],
        ["Liquidations", str(v12.get("liquidations", 0)),
         str(v13_metrics.get("liquidations", 0))],
        ["Entry Conditions", "5 (need 2+)", "4 (need 1+)"],
    ]

    table = ax2.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Style header row
    for j in range(3):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(3):
            table[i, j].set_facecolor(color)

    ax2.set_title("Detailed Comparison", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison to {save_path}")


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

    # V12 vs V13 comparison
    v12_metrics_path = os.path.join(os.path.dirname(__file__),
                                     "results_v12", "metrics.json")
    if os.path.exists(v12_metrics_path):
        plot_comparison(v12_metrics_path, metrics, bt.equity_curve,
                        os.path.join(RESULTS_DIR, "comparison_v12_v13.png"))

    # Print summary
    print(f"\n{'='*50}")
    print(f"  V13 RESULTS (LOOSENED ENTRY)")
    print(f"{'='*50}")
    print(f"  Return:       {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"  Final Equity: ${metrics.get('final_equity', 0):,.2f}")
    print(f"  Max DD:       {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Sharpe:       {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Trades:       {metrics.get('total_trades', 0)}")
    print(f"  Win Rate:     {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"  Funding Cost: ${metrics.get('total_funding_cost', 0):,.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

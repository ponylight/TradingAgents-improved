"""Backtest V21 — V20 Signals + V19 "20 Bullets" Position Sizing

HYPOTHESIS: Combine V20's superior signal detection (TD Sequential + Fibonacci + MACD
confluence, 85 trades, 50% WR) with V19's aggressive compounding (20 bullets, 20x→5x
decreasing leverage) to produce returns that massively outperform both individually.

V20 had 3,954% (4yr) / 153,854% (8.5yr) with 0 liquidations at 3-10x.
V19 had 6,851% (8.5yr) with 91% liquidation rate but massive winners.
V21 expects many liquidations at 20x, but the few winning signals should compound
to far exceed V20's moderate leverage results.

Signal Generation (from V20):
  - TD Sequential setup (9 bars) and countdown (13 bars)
  - Fibonacci retracement/extension levels (0.382, 0.5, 0.618)
  - MACD divergence confirmation
  - Volume confirmation

Position Sizing (from V19 — "20 Bullets"):
  - Each signal = 1 bullet = 5% of INITIAL capital at 20x leverage
  - No stop-loss on initial entry → let it liquidate or catch a trend
  - Rolling adds when floating PnL > 80% of initial_margin:
      Add 1: 15x leverage, margin = 100% of initial_margin
      Add 2: 10x leverage, margin = 75% of initial_margin
      Add 3: 5x  leverage, margin = 50% of initial_margin
      Add 4: 5x  leverage, margin = 25% of initial_margin
  - Trailing stop: ONLY after 100% profit on base, set at 3x ATR
  - Max 4 adds per bullet

Data: 4H BTC/USDT candles (both 4yr and 8.5yr datasets)
"""

import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.fetch_data import fetch_ohlcv
from backtest.indicators import compute_macd, compute_atr, compute_ma

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v21")

# ══════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION (copied from V20 — do not modify)
# ══════════════════════════════════════════════════════════════════════

def compute_td_sequential(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TD Sequential setup and countdown."""
    n = len(df)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    buy_setup = np.zeros(n, dtype=int)
    sell_setup = np.zeros(n, dtype=int)
    buy_countdown = np.zeros(n, dtype=int)
    sell_countdown = np.zeros(n, dtype=int)

    buy_cd_active = False
    sell_cd_active = False
    buy_cd_count = 0
    sell_cd_count = 0

    for i in range(4, n):
        if closes[i] < closes[i - 4]:
            buy_setup[i] = buy_setup[i - 1] + 1 if buy_setup[i - 1] > 0 else 1
            sell_setup[i] = 0
        elif closes[i] > closes[i - 4]:
            sell_setup[i] = sell_setup[i - 1] + 1 if sell_setup[i - 1] > 0 else 1
            buy_setup[i] = 0
        else:
            buy_setup[i] = 0
            sell_setup[i] = 0

        if buy_setup[i] == 9:
            buy_cd_active = True
            buy_cd_count = 0
        if sell_setup[i] == 9:
            sell_cd_active = True
            sell_cd_count = 0

        if buy_cd_active and i >= 2:
            if closes[i] <= lows[i - 2]:
                buy_cd_count += 1
                buy_countdown[i] = buy_cd_count
            if buy_cd_count >= 13:
                buy_cd_active = False
                buy_cd_count = 0
            if sell_setup[i] >= 4:
                buy_cd_active = False
                buy_cd_count = 0

        if sell_cd_active and i >= 2:
            if closes[i] >= highs[i - 2]:
                sell_cd_count += 1
                sell_countdown[i] = sell_cd_count
            if sell_cd_count >= 13:
                sell_cd_active = False
                sell_cd_count = 0
            if buy_setup[i] >= 4:
                sell_cd_active = False
                sell_cd_count = 0

    df = df.copy()
    df["td_buy_setup"] = buy_setup
    df["td_sell_setup"] = sell_setup
    df["td_buy_countdown"] = buy_countdown
    df["td_sell_countdown"] = sell_countdown
    return df


def compute_fib_levels(swing_high: float, swing_low: float) -> dict:
    diff = swing_high - swing_low
    return {
        "0.236": swing_high - diff * 0.236,
        "0.382": swing_high - diff * 0.382,
        "0.500": swing_high - diff * 0.500,
        "0.618": swing_high - diff * 0.618,
        "0.786": swing_high - diff * 0.786,
        "ext_1.272": swing_low + diff * 1.272,
        "ext_1.618": swing_low + diff * 1.618,
        "ext_2.618": swing_low + diff * 2.618,
        "swing_high": swing_high,
        "swing_low": swing_low,
    }


def find_swing_points(highs: np.ndarray, lows: np.ndarray, lookback: int = 20):
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
                          closes: np.ndarray) -> Optional[dict]:
    recent_highs = [(i, v) for i, v in swing_highs if i < idx - 5]
    recent_lows = [(i, v) for i, v in swing_lows if i < idx - 5]

    if not recent_highs or not recent_lows:
        return None

    last_high = recent_highs[-1]
    last_low = recent_lows[-1]
    return compute_fib_levels(last_high[1], last_low[1])


def price_near_fib(price: float, fib_levels: dict, tolerance: float = 0.005) -> Optional[str]:
    for level_name in ["0.618", "0.500", "0.382"]:
        level_price = fib_levels[level_name]
        if abs(price - level_price) / price <= tolerance:
            return level_name
    return None


def check_bottom_divergence(idx: int, hist: np.ndarray, lows: np.ndarray,
                            lookback: int = 120) -> int:
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


@dataclass
class Signal:
    idx: int
    side: str
    entry_price: float
    confluence: list
    fib_level: str = ""
    # Note: stop_loss / take_profit not used in V21 (bullet sizing handles exits)


def generate_signals(df: pd.DataFrame, swing_highs: list, swing_lows: list) -> list:
    """Generate trading signals — identical logic to V20, but leverage is overridden by bullet sizing."""
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

    min_cooldown = 18  # same as V20: 3 days on 4H
    last_signal_idx = -min_cooldown

    for i in range(50, len(df)):
        if i - last_signal_idx < min_cooldown:
            continue
        if np.isnan(atr[i]) or np.isnan(sma200[i]):
            continue

        avg_vol = np.mean(volumes[max(0, i-20):i]) if i > 20 else volumes[i]
        vol_ok = volumes[i] > avg_vol * 0.8
        if not vol_ok:
            continue

        fib = get_active_fib_levels(i, swing_highs, swing_lows, closes)
        bull = closes[i] > sma200[i] if not np.isnan(sma200[i]) else False

        # ── LONG SIGNALS ──────────────────────────────────────
        confluence = []

        has_td9_buy = td_buy_setup[i] == 9
        has_td13_buy = td_buy_countdown[i] == 13
        macd_div = check_bottom_divergence(i, hist, lows)
        has_macd_div = macd_div >= 1

        fib_level = None
        if fib:
            fib_level = price_near_fib(closes[i], fib, tolerance=0.008)

        if has_td9_buy:
            confluence.append("TD9_buy")
        if has_td13_buy:
            confluence.append("TD13_buy")
        if has_macd_div:
            confluence.append(f"MACD_div_{macd_div}")
        if fib_level:
            confluence.append(f"Fib_{fib_level}")

        long_signal = False

        # Tier 1: TD13 at Fib = highest conviction
        if has_td13_buy and fib_level:
            long_signal = True
        # Tier 2: TD9 + MACD divergence
        elif has_td9_buy and has_macd_div:
            long_signal = True
        # Tier 3: TD9 at Fib
        elif has_td9_buy and fib_level:
            long_signal = True
        # Tier 4: TD13 alone
        elif has_td13_buy:
            long_signal = True
        # Tier 5: TD9 + volume spike
        elif has_td9_buy and volumes[i] > avg_vol * 1.5:
            long_signal = True
            confluence.append("volume_spike")
        # Tier 6: MACD divergence at Fib (strong)
        elif has_macd_div and fib_level and macd_div >= 2:
            long_signal = True

        if long_signal:
            signals.append(Signal(
                idx=i, side="long", entry_price=closes[i],
                confluence=confluence, fib_level=fib_level or "",
            ))
            last_signal_idx = i
            continue

        # ── SHORT SIGNALS ─────────────────────────────────────
        # Only short in bear market
        if bull:
            continue

        confluence = []

        has_td9_sell = td_sell_setup[i] == 9
        has_td13_sell = td_sell_countdown[i] == 13
        macd_bear_div = check_top_divergence(i, hist, highs)
        has_macd_bear_div = macd_bear_div >= 1

        fib_level = None
        if fib:
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

        if has_td13_sell and fib_level:
            short_signal = True
        elif has_td9_sell and has_macd_bear_div:
            short_signal = True
        elif has_td9_sell and fib_level:
            short_signal = True
        elif has_td13_sell:
            short_signal = True
        elif has_td9_sell and volumes[i] > avg_vol * 1.5:
            short_signal = True
            confluence.append("volume_spike")
        elif has_macd_bear_div and fib_level and macd_bear_div >= 2:
            short_signal = True

        if short_signal:
            signals.append(Signal(
                idx=i, side="short", entry_price=closes[i],
                confluence=confluence, fib_level=fib_level or "",
            ))
            last_signal_idx = i

    return signals


# ══════════════════════════════════════════════════════════════════════
# V19 BULLET POSITION SIZING ENGINE
# ══════════════════════════════════════════════════════════════════════

# Decreasing leverage for adds (V19 design)
ADD_LEVERAGES = [15, 10, 5, 5]  # Add 1, 2, 3, 4
# Pyramid sizing: each add margin = fraction of available floating profit
# Available = 80% of current floating PnL; then apply pyramid cap
ADD_AVAILABLE_FRAC = 0.80       # use 80% of floating pnl as "available" margin
ADD_PYRAMID_CAPS = [1.0, 0.75, 0.50, 0.25]  # caps on available per add
# Floating profit threshold to trigger next add (as fraction of initial_margin)
ADD_PROFIT_THRESHOLD = 0.80
# Base profit threshold to activate trailing stop (100% of initial_margin)
TRAILING_ACTIVATE_PROFIT = 1.0
# Trailing stop: 3x ATR
TRAILING_ATR_MULT = 3.0
# Base entry leverage
BASE_LEVERAGE = 20
# Bullet size = 5% of INITIAL capital
BULLET_MARGIN_FRAC = 0.05
# Commission rate
COMMISSION_RATE = 0.001


@dataclass
class BulletLeg:
    """A single leveraged position (base or add) within a bullet."""
    leg_id: int  # 0 = base, 1-4 = adds
    entry_price: float
    leverage: int
    size_btc: float        # leveraged BTC position
    margin: float          # USD margin committed
    liquidation_price: float
    closed: bool = False
    realized_pnl: float = 0.0
    close_reason: str = ""
    close_price: float = 0.0


@dataclass
class Bullet:
    """A complete bullet: base position + up to 4 rolling adds."""
    bullet_id: int
    side: str              # "long" or "short"
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    initial_margin: float  # 5% of initial capital
    confluence: list
    fib_level: str = ""

    legs: list = field(default_factory=list)
    n_adds: int = 0                    # adds fired so far
    trailing_stop_active: bool = False
    trailing_stop: float = 0.0
    closed: bool = False
    close_reason: str = ""
    exit_time: Optional[pd.Timestamp] = None
    total_pnl: float = 0.0             # final realized PnL (all legs)
    peak_floating_pnl: float = 0.0     # for tracking 100% activation
    # Threshold tracking: next add fires when cumulative floating PnL exceeds this
    next_add_threshold: float = 0.0

    def active_legs(self):
        return [l for l in self.legs if not l.closed]

    def calc_floating_pnl(self, price: float) -> float:
        total = 0.0
        for leg in self.active_legs():
            if self.side == "long":
                total += (price - leg.entry_price) * leg.size_btc
            else:
                total += (leg.entry_price - price) * leg.size_btc
        return total


class V21Engine:
    """Backtest engine using V20 signals with V19 bullet position sizing."""

    def __init__(self, df: pd.DataFrame, initial_capital: float = 10_000.0,
                 results_dir: str = RESULTS_DIR):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.results_dir = results_dir
        self.df = self._prepare_data(df)
        self.bullets: list[Bullet] = []
        self.active_bullets: list[Bullet] = []
        self.closed_bullets: list[Bullet] = []
        self.equity_curve = []
        self._bullet_id = 0
        self.stats = {
            "total_bullets": 0,
            "liquidated_bullets": 0,
            "trailing_stop_bullets": 0,
            "end_of_backtest_bullets": 0,
            "total_legs": 0,
            "liquidated_legs": 0,
            "total_adds_fired": 0,
            "bullets_with_adds": 0,
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

    def _new_bullet_id(self) -> int:
        self._bullet_id += 1
        return self._bullet_id

    def _open_bullet(self, signal: Signal, ts: pd.Timestamp):
        """Open a new bullet with base position at 20x leverage."""
        margin = self.initial_capital * BULLET_MARGIN_FRAC
        if margin > self.capital:
            return  # not enough capital to fire bullet

        price = signal.entry_price
        leverage = BASE_LEVERAGE
        size_btc = (margin * leverage) / price

        if signal.side == "long":
            liq_price = price * (1 - 1 / leverage)
        else:
            liq_price = price * (1 + 1 / leverage)

        commission = size_btc * price * COMMISSION_RATE
        self.capital -= (margin + commission)

        base_leg = BulletLeg(
            leg_id=0,
            entry_price=price,
            leverage=leverage,
            size_btc=size_btc,
            margin=margin,
            liquidation_price=liq_price,
        )

        bullet = Bullet(
            bullet_id=self._new_bullet_id(),
            side=signal.side,
            entry_idx=signal.idx,
            entry_time=ts,
            entry_price=price,
            initial_margin=margin,
            confluence=signal.confluence,
            fib_level=signal.fib_level,
            legs=[base_leg],
            next_add_threshold=margin * ADD_PROFIT_THRESHOLD,  # 80% of initial margin
        )

        self.active_bullets.append(bullet)
        self.bullets.append(bullet)
        self.stats["total_bullets"] += 1
        self.stats["total_legs"] += 1

    def _fire_add(self, bullet: Bullet, price: float, floating_pnl: float, ts: pd.Timestamp):
        """Add to a winning bullet using decreasing leverage.
        
        CRITICAL: Add margin scales with floating profit (V19 design).
        80% of current floating PnL is 'available'; apply pyramid cap per add.
        This is what makes V19 compound massively — a $500 bullet with $50K
        floating PnL can fuel an add with $40K margin at 10x = $400K exposure.
        """
        add_num = bullet.n_adds  # 0-based index into ADD_LEVERAGES
        if add_num >= 4:
            return

        add_leverage = ADD_LEVERAGES[add_num]
        
        # Add margin = pyramid_cap * (80% of floating PnL)
        # This is the V19 design: use floating profit as collateral
        available = floating_pnl * ADD_AVAILABLE_FRAC
        add_margin = available * ADD_PYRAMID_CAPS[add_num]
        
        # Minimum meaningful add
        if add_margin < bullet.initial_margin * 0.5:
            # Too small to bother — bump threshold and move on
            bullet.next_add_threshold *= 2
            return

        # Fund from capital (floating gains serve as effective collateral)
        if add_margin > self.capital:
            add_margin = self.capital * 0.8  # use whatever we have
            if add_margin < bullet.initial_margin * 0.5:
                return

        size_btc = (add_margin * add_leverage) / price

        if bullet.side == "long":
            liq_price = price * (1 - 1 / add_leverage)
        else:
            liq_price = price * (1 + 1 / add_leverage)

        commission = size_btc * price * COMMISSION_RATE
        self.capital -= (add_margin + commission)

        leg = BulletLeg(
            leg_id=add_num + 1,
            entry_price=price,
            leverage=add_leverage,
            size_btc=size_btc,
            margin=add_margin,
            liquidation_price=liq_price,
        )
        bullet.legs.append(leg)
        bullet.n_adds += 1

        # Update next add threshold: next add fires when floating PnL doubles again
        # (meaning the bullet needs to grow another ~2x to trigger the next add)
        bullet.next_add_threshold = floating_pnl * 2.0

        self.stats["total_legs"] += 1
        self.stats["total_adds_fired"] += 1
        if bullet.n_adds == 1:
            self.stats["bullets_with_adds"] += 1

    def _close_leg(self, leg: BulletLeg, price: float, reason: str):
        """Close a single leg and return capital."""
        if leg.closed:
            return 0.0

        # Calculate PnL for this leg's remaining size
        # Note: leg.size_btc is the full leveraged size
        # PnL = (exit - entry) * size_btc for long
        pnl_raw = 0.0
        # (direction handled by caller — but we need the bullet's side)
        leg.close_price = price
        leg.close_reason = reason
        leg.closed = True
        # PnL computed by caller, this just marks it closed
        return pnl_raw

    def _close_bullet(self, bullet: Bullet, price: float, ts: pd.Timestamp, reason: str):
        """Close all remaining legs of a bullet."""
        total_pnl = 0.0

        for leg in bullet.active_legs():
            if bullet.side == "long":
                pnl_raw = (price - leg.entry_price) * leg.size_btc
            else:
                pnl_raw = (leg.entry_price - price) * leg.size_btc

            commission = leg.size_btc * price * COMMISSION_RATE
            realized = pnl_raw - commission

            leg.realized_pnl = realized
            leg.close_price = price
            leg.close_reason = reason
            leg.closed = True

            # Return margin + realized PnL to capital
            self.capital += leg.margin + realized
            total_pnl += leg.margin + realized

        bullet.total_pnl = total_pnl - bullet.initial_margin  # net PnL (subtract base margin already deducted)
        # Recalculate more carefully: total_pnl is already margin-inclusive above,
        # but margin was already deducted at entry. So net P&L is:
        bullet.total_pnl = sum(l.realized_pnl for l in bullet.legs)
        bullet.closed = True
        bullet.close_reason = reason
        bullet.exit_time = ts

        if bullet in self.active_bullets:
            self.active_bullets.remove(bullet)
        self.closed_bullets.append(bullet)

    def _process_bullets(self, i: int, high: float, low: float, close: float,
                         ts: pd.Timestamp, atr_val: float):
        """Process all active bullets for the current bar."""

        for bullet in list(self.active_bullets):
            if bullet.closed:
                continue

            # --- Step 1: Check individual leg liquidations ---
            for leg in bullet.active_legs():
                liquidated = False
                if bullet.side == "long" and low <= leg.liquidation_price:
                    liquidated = True
                elif bullet.side == "short" and high >= leg.liquidation_price:
                    liquidated = True

                if liquidated:
                    # Leg is liquidated: margin is lost
                    leg.realized_pnl = -leg.margin
                    leg.close_price = leg.liquidation_price
                    leg.close_reason = "liquidation"
                    leg.closed = True
                    # Capital: margin was already deducted at entry, nothing returned
                    self.stats["liquidated_legs"] += 1

            # Check if all legs liquidated
            active = bullet.active_legs()
            if not active:
                bullet.total_pnl = sum(l.realized_pnl for l in bullet.legs)
                bullet.closed = True
                bullet.close_reason = "liquidation"
                bullet.exit_time = ts
                if bullet in self.active_bullets:
                    self.active_bullets.remove(bullet)
                self.closed_bullets.append(bullet)
                self.stats["liquidated_bullets"] += 1
                continue

            # --- Step 2: Compute floating PnL ---
            floating_pnl = bullet.calc_floating_pnl(close)
            bullet.peak_floating_pnl = max(bullet.peak_floating_pnl, floating_pnl)

            # --- Step 3: Check trailing stop ---
            if bullet.trailing_stop_active:
                hit = False
                if bullet.side == "long" and low <= bullet.trailing_stop:
                    hit = True
                    exit_price = min(close, bullet.trailing_stop)  # realistic fill
                elif bullet.side == "short" and high >= bullet.trailing_stop:
                    hit = True
                    exit_price = max(close, bullet.trailing_stop)

                if hit:
                    self._close_bullet(bullet, exit_price, ts, "trailing_stop")
                    self.stats["trailing_stop_bullets"] += 1
                    continue

            # --- Step 4: Update trailing stop if active ---
            if bullet.trailing_stop_active and not np.isnan(atr_val) and atr_val > 0:
                if bullet.side == "long":
                    new_trail = close - TRAILING_ATR_MULT * atr_val
                    bullet.trailing_stop = max(bullet.trailing_stop, new_trail)
                else:
                    new_trail = close + TRAILING_ATR_MULT * atr_val
                    bullet.trailing_stop = min(bullet.trailing_stop, new_trail)

            # --- Step 5: Activate trailing stop at 100% profit ---
            if not bullet.trailing_stop_active:
                profit_threshold = bullet.initial_margin * TRAILING_ACTIVATE_PROFIT
                if floating_pnl >= profit_threshold and not np.isnan(atr_val) and atr_val > 0:
                    bullet.trailing_stop_active = True
                    if bullet.side == "long":
                        bullet.trailing_stop = close - TRAILING_ATR_MULT * atr_val
                    else:
                        bullet.trailing_stop = close + TRAILING_ATR_MULT * atr_val

            # --- Step 6: Fire rolling adds ---
            if bullet.n_adds < 4 and floating_pnl >= bullet.next_add_threshold:
                self._fire_add(bullet, close, floating_pnl, ts)

    def _calc_unrealized(self, price: float) -> float:
        total = 0.0
        for bullet in self.active_bullets:
            total += bullet.calc_floating_pnl(price)
        return total

    def run(self) -> dict:
        print(f"\n[V21] V20 Signals + V19 Bullets Backtest")
        print(f"  Data: {len(self.df)} candles, {self.df['timestamp'].iloc[0]} → {self.df['timestamp'].iloc[-1]}")
        print(f"  Initial Capital: ${self.initial_capital:,.0f}")
        print(f"  Bullet Size: {BULLET_MARGIN_FRAC*100:.0f}% = ${self.initial_capital * BULLET_MARGIN_FRAC:,.0f}")
        print(f"  Base Leverage: {BASE_LEVERAGE}x (liq @ ±{100/BASE_LEVERAGE:.1f}%)")
        print(f"  Add Leverages: {ADD_LEVERAGES}")

        signals = generate_signals(self.df, self.swing_highs, self.swing_lows)
        signal_map = {s.idx: s for s in signals}

        long_sigs = sum(1 for s in signals if s.side == "long")
        short_sigs = sum(1 for s in signals if s.side == "short")
        print(f"  Signals: {len(signals)} ({long_sigs} long, {short_sigs} short)")

        closes = self.df["close"].values
        highs = self.df["high"].values
        lows = self.df["low"].values
        atrs = self.df["atr"].values

        for i in range(len(self.df)):
            price = closes[i]
            high = highs[i]
            low = lows[i]
            ts = self.df["timestamp"].iloc[i]
            atr_val = atrs[i] if not np.isnan(atrs[i]) else 0.0

            # Process existing bullets first
            self._process_bullets(i, high, low, price, ts, atr_val)

            # Open new bullet if signal fires
            if i in signal_map:
                sig = signal_map[i]
                self._open_bullet(sig, ts)

            # Track equity
            unrealized = self._calc_unrealized(price)
            self.equity_curve.append({
                "timestamp": ts,
                "equity": self.capital + unrealized,
                "capital": self.capital,
                "n_active_bullets": len(self.active_bullets),
            })

        # End of backtest: close all remaining bullets at last price
        last_price = closes[-1]
        last_ts = self.df["timestamp"].iloc[-1]
        for bullet in list(self.active_bullets):
            self._close_bullet(bullet, last_price, last_ts, "end_of_backtest")
            self.stats["end_of_backtest_bullets"] += 1

        return self._compute_metrics()

    def _compute_metrics(self) -> dict:
        if not self.closed_bullets:
            return {"version": "v21", "error": "No bullets fired"}

        eq_df = pd.DataFrame(self.equity_curve)
        equity = eq_df["equity"].values

        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_dd = np.max(drawdown)
        peak_equity = np.max(equity)

        returns = np.diff(equity) / equity[:-1]
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(2190)
                  if len(returns) > 1 and np.std(returns) > 0 else 0)

        bullet_pnls = [b.total_pnl for b in self.closed_bullets]
        wins = [p for p in bullet_pnls if p > 0]
        losses = [p for p in bullet_pnls if p <= 0]
        win_rate = len(wins) / len(bullet_pnls) * 100 if bullet_pnls else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        durations = []
        for b in self.closed_bullets:
            if b.entry_time and b.exit_time:
                durations.append((b.exit_time - b.entry_time).total_seconds() / 3600)
        avg_dur = np.mean(durations) if durations else 0

        long_bullets = [b for b in self.closed_bullets if b.side == "long"]
        short_bullets = [b for b in self.closed_bullets if b.side == "short"]

        # Top bullets by PnL
        top_bullets = sorted(self.closed_bullets, key=lambda b: b.total_pnl, reverse=True)[:5]
        top_bullet_info = [
            {
                "side": b.side,
                "entry_time": str(b.entry_time)[:10],
                "entry_price": round(b.entry_price, 0),
                "n_adds": b.n_adds,
                "close_reason": b.close_reason,
                "pnl": round(b.total_pnl, 2),
                "pnl_multiple": round(b.total_pnl / b.initial_margin, 1),
                "confluence": b.confluence,
            }
            for b in top_bullets
        ]

        return {
            "version": "v21",
            "strategy": "V20 Signals + V19 Bullets",
            "total_return_pct": round(total_return, 2),
            "final_equity": round(equity[-1], 2),
            "peak_equity": round(peak_equity, 2),
            "peak_multiplier": round(peak_equity / self.initial_capital, 2),
            "initial_capital": self.initial_capital,
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(pf, 2),
            "total_bullets": len(self.closed_bullets),
            "long_bullets": len(long_bullets),
            "short_bullets": len(short_bullets),
            "long_pnl": round(sum(b.total_pnl for b in long_bullets), 2),
            "short_pnl": round(sum(b.total_pnl for b in short_bullets), 2),
            "winning_bullets": len(wins),
            "losing_bullets": len(losses),
            "liquidated_bullets": self.stats["liquidated_bullets"],
            "trailing_stop_bullets": self.stats["trailing_stop_bullets"],
            "end_of_backtest_bullets": self.stats["end_of_backtest_bullets"],
            "liquidation_rate_pct": round(self.stats["liquidated_bullets"] / max(1, len(self.closed_bullets)) * 100, 1),
            "total_legs": self.stats["total_legs"],
            "liquidated_legs": self.stats["liquidated_legs"],
            "total_adds_fired": self.stats["total_adds_fired"],
            "bullets_with_adds": self.stats["bullets_with_adds"],
            "avg_win_bullet": round(np.mean(wins), 2) if wins else 0,
            "avg_loss_bullet": round(np.mean(losses), 2) if losses else 0,
            "largest_win_bullet": round(max(wins), 2) if wins else 0,
            "largest_loss_bullet": round(min(losses), 2) if losses else 0,
            "avg_duration_hours": round(avg_dur, 1),
            "avg_duration_days": round(avg_dur / 24, 1),
            "top_5_bullets": top_bullet_info,
            "data_start": str(self.df["timestamp"].iloc[0]),
            "data_end": str(self.df["timestamp"].iloc[-1]),
            "total_candles": len(self.df),
            "bullet_margin": round(self.initial_capital * BULLET_MARGIN_FRAC, 2),
            "base_leverage": BASE_LEVERAGE,
            "add_leverages": ADD_LEVERAGES,
        }

    def generate_report(self, metrics: dict):
        os.makedirs(self.results_dir, exist_ok=True)

        with open(os.path.join(self.results_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        report = self._format_report(metrics)
        with open(os.path.join(self.results_dir, "report.txt"), "w") as f:
            f.write(report)
        print(report)

        self._plot_equity_curve()
        self._save_bullet_log()

    def _format_report(self, m: dict) -> str:
        sep = "=" * 65
        lines = [
            sep,
            "  BTC/USDT V21 — V20 Signals + V19 Bullets",
            sep, "",
            f"  Data:    {m['data_start'][:10]} → {m['data_end'][:10]} ({m['total_candles']} 4H bars)",
            f"  Capital: ${m['initial_capital']:,.0f}",
            f"  Bullet:  {BULLET_MARGIN_FRAC*100:.0f}% = ${m['bullet_margin']:,.0f} @ {m['base_leverage']}x (liq ±{100/m['base_leverage']:.1f}%)",
            f"  Adds:    {m['add_leverages']} (4 max, 80% profit trigger)",
            "",
            sep, "  PERFORMANCE", sep, "",
            f"  Return:         {m['total_return_pct']:+.2f}%",
            f"  Final Equity:   ${m['final_equity']:,.2f}",
            f"  Peak Equity:    ${m['peak_equity']:,.2f} ({m['peak_multiplier']:.1f}x)",
            f"  Max Drawdown:   {m['max_drawdown_pct']:.2f}%",
            f"  Sharpe Ratio:   {m['sharpe_ratio']:.2f}",
            f"  Profit Factor:  {m['profit_factor']:.2f}",
            f"  Win Rate:       {m['win_rate_pct']:.1f}% (of bullets)",
            "",
            sep, "  BULLETS", sep, "",
            f"  Total Bullets:  {m['total_bullets']}",
            f"    Long:         {m['long_bullets']} (PnL: ${m['long_pnl']:+,.2f})",
            f"    Short:        {m['short_bullets']} (PnL: ${m['short_pnl']:+,.2f})",
            f"  Winners:        {m['winning_bullets']}",
            f"  Losers:         {m['losing_bullets']}",
            f"  Liquidated:     {m['liquidated_bullets']} ({m['liquidation_rate_pct']:.0f}%)",
            f"  Trailing Stop:  {m['trailing_stop_bullets']}",
            f"  End of BT:      {m['end_of_backtest_bullets']}",
            f"  Avg Win:        ${m['avg_win_bullet']:,.2f}",
            f"  Avg Loss:       ${m['avg_loss_bullet']:,.2f}",
            f"  Best Bullet:    ${m['largest_win_bullet']:,.2f}",
            f"  Worst Bullet:   ${m['largest_loss_bullet']:,.2f}",
            f"  Avg Duration:   {m['avg_duration_days']:.1f} days",
            "",
            sep, "  ROLLING ADDS", sep, "",
            f"  Total Legs:     {m['total_legs']}",
            f"  Adds Fired:     {m['total_adds_fired']}",
            f"  Bullets w/Adds: {m['bullets_with_adds']}",
            f"  Liquidated Legs:{m['liquidated_legs']}",
            "",
            sep, "  TOP 5 BULLETS", sep, "",
        ]

        for i, b in enumerate(m.get("top_5_bullets", []), 1):
            conf = "|".join(b["confluence"]) if b["confluence"] else "?"
            lines.append(
                f"  #{i}: {b['side'].upper()} {b['entry_time']} @ ${b['entry_price']:,.0f} "
                f"| adds={b['n_adds']} | {b['close_reason']}"
            )
            lines.append(
                f"     PnL=${b['pnl']:+,.0f} ({b['pnl_multiple']:.1f}x bullet) | {conf}"
            )

        lines += [
            "",
            sep, "  COMPARISON", sep, "",
            f"  V6  (4yr): +34%       (old champion)",
            f"  V20 (4yr): +3,954%    (Fib+TD, 3-10x)",
            f"  V19 (8.5yr):+6,851%   (Bullets, MACD signals)",
            f"  V20 (8.5yr):+153,854% (Fib+TD, 3-10x)",
            f"  V21 this run: {m['total_return_pct']:+.2f}%",
            "",
            sep,
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            sep,
        ]
        return "\n".join(lines)

    def _plot_equity_curve(self):
        eq_df = pd.DataFrame(self.equity_curve)
        equity = eq_df["equity"].values
        timestamps = pd.to_datetime(eq_df["timestamp"])

        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])

        ax1.plot(timestamps, equity, color="#FF6F00", linewidth=1.2, label="V21 (V20 Signals + V19 Bullets)")
        ax1.axhline(y=self.initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")

        for bullet in self.closed_bullets:
            eq_idx = min(bullet.entry_idx, len(equity) - 1)
            color = "#4CAF50" if bullet.total_pnl > 0 else ("#9C27B0" if bullet.close_reason == "liquidation" else "#F44336")
            marker = "^" if bullet.side == "long" else "v"
            ax1.plot(bullet.entry_time, equity[eq_idx], marker=marker,
                    color=color, markersize=4 + bullet.n_adds, alpha=0.7)

        ax1.set_title("V21 — V20 Signals + V19 Bullets Equity Curve", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Equity ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        ax2.fill_between(timestamps, -dd, 0, color="#F44336", alpha=0.3)
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "equity_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart saved → {self.results_dir}/equity_curve.png")

    def _save_bullet_log(self):
        records = []
        for b in self.closed_bullets:
            records.append({
                "bullet_id": b.bullet_id,
                "side": b.side,
                "entry_time": b.entry_time,
                "exit_time": b.exit_time,
                "entry_price": round(b.entry_price, 2),
                "initial_margin": round(b.initial_margin, 2),
                "n_adds": b.n_adds,
                "total_pnl": round(b.total_pnl, 2),
                "pnl_multiple": round(b.total_pnl / max(b.initial_margin, 1), 2),
                "close_reason": b.close_reason,
                "trailing_activated": b.trailing_stop_active,
                "peak_floating_pnl": round(b.peak_floating_pnl, 2),
                "confluence": "|".join(b.confluence),
                "fib_level": b.fib_level,
                "duration_days": round((b.exit_time - b.entry_time).total_seconds() / 86400, 1)
                    if b.exit_time and b.entry_time else 0,
                "n_legs_total": len(b.legs),
                "n_legs_liquidated": sum(1 for l in b.legs if l.close_reason == "liquidation"),
            })
        log_path = os.path.join(self.results_dir, "bullet_log.csv")
        pd.DataFrame(records).to_csv(log_path, index=False)
        print(f"  Bullet log saved → {log_path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    results = {}

    # ── Dataset 1: 4-year (2022-2026) ────────────────────────────────
    print("\n" + "=" * 65)
    print("  DATASET 1: 4-Year (2022-2026)")
    print("=" * 65)

    df_4yr = fetch_ohlcv(use_cache=True)
    results_dir_4yr = os.path.join(os.path.dirname(__file__), "results_v21")
    engine_4yr = V21Engine(df_4yr, initial_capital=10_000.0, results_dir=results_dir_4yr)
    m4 = engine_4yr.run()
    engine_4yr.generate_report(m4)
    results["4yr"] = m4

    # ── Dataset 2: 8.5-year (2017-2026) ──────────────────────────────
    data_path_8yr = os.path.join(os.path.dirname(__file__), "data", "btc_usdt_4h_2017.csv")
    if os.path.exists(data_path_8yr):
        print("\n" + "=" * 65)
        print("  DATASET 2: 8.5-Year (2017-2026)")
        print("=" * 65)

        df_8yr = pd.read_csv(data_path_8yr, parse_dates=["timestamp"])
        results_dir_8yr = os.path.join(os.path.dirname(__file__), "results_v21_8yr")
        engine_8yr = V21Engine(df_8yr, initial_capital=10_000.0, results_dir=results_dir_8yr)
        m8 = engine_8yr.run()
        engine_8yr.generate_report(m8)
        results["8yr"] = m8
    else:
        print(f"\n  [SKIP] 8.5yr dataset not found at {data_path_8yr}")

    # ── Summary comparison ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL COMPARISON")
    print("=" * 65)
    print(f"\n  {'Strategy':<28} {'Return':>12} {'MaxDD':>8} {'WR':>7} {'Trades':>8}")
    print(f"  {'-'*28} {'-'*12} {'-'*8} {'-'*7} {'-'*8}")
    print(f"  {'V6 (4yr, MACD 3x)':<28} {'~34%':>12} {'~40%':>8} {'~55%':>7} {'~50':>8}")
    print(f"  {'V20 (4yr, Fib+TD 3-10x)':<28} {'~3,954%':>12} {'~28%':>8} {'~50%':>7} {'~85':>8}")

    if "4yr" in results:
        m = results["4yr"]
        ret = f"{m['total_return_pct']:+.0f}%"
        dd = f"{m['max_drawdown_pct']:.0f}%"
        wr = f"{m['win_rate_pct']:.0f}%"
        bt = str(m["total_bullets"])
        print(f"  {'V21 (4yr, V20sig+V19bulk)':<28} {ret:>12} {dd:>8} {wr:>7} {bt:>8}")

    print(f"  {'V19 (8.5yr, Bullets+MACD)':<28} {'~6,851%':>12} {'~81%':>8} {'~9%':>7} {'~32':>8}")
    print(f"  {'V20 (8.5yr, Fib+TD 3-10x)':<28} {'~153,854%':>12} {'~?%':>8} {'~50%':>7} {'~?':>8}")

    if "8yr" in results:
        m = results["8yr"]
        ret = f"{m['total_return_pct']:+.0f}%"
        dd = f"{m['max_drawdown_pct']:.0f}%"
        wr = f"{m['win_rate_pct']:.0f}%"
        bt = str(m["total_bullets"])
        print(f"  {'V21 (8.5yr, V20sig+V19bulk)':<28} {ret:>12} {dd:>8} {wr:>7} {bt:>8}")

    print()
    return results


if __name__ == "__main__":
    main()

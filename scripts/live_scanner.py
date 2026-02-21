#!/usr/bin/env python3
"""V20 Live Signal Scanner — Fibonacci + TD Sequential + MACD Divergence

Runs on 4H BTC/USDT candles from Bybit demo. Checks the latest completed
candle for a V20 signal. If fired, places an order and sends a Telegram alert.

Usage:
    python scripts/live_scanner.py            # live mode
    python scripts/live_scanner.py --dry-run  # skip order placement
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import ccxt
import requests
from dotenv import load_dotenv

# ── Environment ──────────────────────────────────────────────────────

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY", "")
BYBIT_SECRET     = os.getenv("BYBIT_SECRET", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

SYMBOL      = "BTC/USDT:USDT"  # perp contract
SYMBOL_SPOT = "BTC/USDT"
TIMEFRAME   = "4h"
FETCH_LIMIT = 500               # candles to fetch
RISK_PCT    = 0.08              # 8% risk per trade
FIB_TOL     = 0.012             # 1.2% tolerance

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("v20_scanner")


# ══════════════════════════════════════════════════════════════════════
# INDICATOR FUNCTIONS (ported faithfully from backtest/backtest_v20.py)
# ══════════════════════════════════════════════════════════════════════

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_macd(df: pd.DataFrame, fast: int = 13, slow: int = 34, signal: int = 9) -> pd.DataFrame:
    ema_fast = compute_ema(df["close"], fast)
    ema_slow = compute_ema(df["close"], slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    df = df.copy()
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    prev  = df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_ma(df: pd.DataFrame, period: int = 200) -> pd.Series:
    return df["close"].rolling(window=period).mean()


def compute_td_sequential(df: pd.DataFrame) -> pd.DataFrame:
    """TD Sequential setup (9) and countdown (13) — exact port from backtest_v20."""
    n      = len(df)
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    buy_setup      = np.zeros(n, dtype=int)
    sell_setup     = np.zeros(n, dtype=int)
    buy_countdown  = np.zeros(n, dtype=int)
    sell_countdown = np.zeros(n, dtype=int)

    buy_cd_active  = False
    sell_cd_active = False
    buy_cd_count   = 0
    sell_cd_count  = 0

    for i in range(4, n):
        if closes[i] < closes[i - 4]:
            buy_setup[i]  = buy_setup[i - 1] + 1 if buy_setup[i - 1] > 0 else 1
            sell_setup[i] = 0
        elif closes[i] > closes[i - 4]:
            sell_setup[i] = sell_setup[i - 1] + 1 if sell_setup[i - 1] > 0 else 1
            buy_setup[i]  = 0
        else:
            buy_setup[i]  = 0
            sell_setup[i] = 0

        if buy_setup[i] == 9:
            buy_cd_active = True
            buy_cd_count  = 0
        if sell_setup[i] == 9:
            sell_cd_active = True
            sell_cd_count  = 0

        if buy_cd_active and i >= 2:
            if closes[i] <= lows[i - 2]:
                buy_cd_count += 1
                buy_countdown[i] = buy_cd_count
            if buy_cd_count >= 13:
                buy_cd_active = False
                buy_cd_count  = 0
            if sell_setup[i] >= 4:
                buy_cd_active = False
                buy_cd_count  = 0

        if sell_cd_active and i >= 2:
            if closes[i] >= highs[i - 2]:
                sell_cd_count += 1
                sell_countdown[i] = sell_cd_count
            if sell_cd_count >= 13:
                sell_cd_active = False
                sell_cd_count  = 0
            if buy_setup[i] >= 4:
                sell_cd_active = False
                sell_cd_count  = 0

    df = df.copy()
    df["td_buy_setup"]      = buy_setup
    df["td_sell_setup"]     = sell_setup
    df["td_buy_countdown"]  = buy_countdown
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
        "swing_low":  swing_low,
    }


def find_swing_points(highs: np.ndarray, lows: np.ndarray, lookback: int = 20):
    """Find significant swing highs and lows — exact port from backtest_v20."""
    swing_highs = []
    swing_lows  = []

    for i in range(lookback, len(highs) - lookback):
        start = i - lookback
        end   = i + lookback + 1

        if highs[i] >= np.max(highs[start:end]):
            if not swing_highs or i - swing_highs[-1][0] >= lookback // 2:
                swing_highs.append((i, highs[i]))

        if lows[i] <= np.min(lows[start:end]):
            if not swing_lows or i - swing_lows[-1][0] >= lookback // 2:
                swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def get_active_fib_levels(idx: int, swing_highs: list, swing_lows: list,
                           closes: np.ndarray) -> Optional[dict]:
    """Most relevant Fib levels for current price action."""
    recent_highs = [(i, v) for i, v in swing_highs if i < idx - 5]
    recent_lows  = [(i, v) for i, v in swing_lows  if i < idx - 5]

    if not recent_highs or not recent_lows:
        return None

    last_high = recent_highs[-1]
    last_low  = recent_lows[-1]

    # Both directions use same levels (high→low retracement)
    return compute_fib_levels(last_high[1], last_low[1])


def price_near_fib(price: float, fib_levels: dict, tolerance: float = FIB_TOL) -> Optional[str]:
    """Return nearest key Fib level name if within tolerance."""
    for level_name in ["0.618", "0.500", "0.382"]:
        level_price = fib_levels[level_name]
        if abs(price - level_price) / price <= tolerance:
            return level_name
    return None


def check_bottom_divergence(idx: int, hist: np.ndarray, lows: np.ndarray,
                             lookback: int = 120) -> int:
    """Bullish MACD histogram divergence — exact port from backtest_v20."""
    troughs    = []
    start      = max(0, idx - lookback)
    in_trough  = False
    trough_val = 0.0
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
            in_trough  = False
            trough_val = 0.0

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


# ══════════════════════════════════════════════════════════════════════
# SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════════════

class SignalResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return (
            f"<Signal side={self.side} leverage={self.leverage}x "
            f"entry={self.entry_price:.2f} sl={self.stop_loss:.2f} "
            f"tp1={self.take_profit_1:.2f} tp2={self.take_profit_2:.2f} "
            f"confluence={self.confluence}>"
        )


def detect_signal(df: pd.DataFrame) -> Optional[SignalResult]:
    """Check if the LAST completed candle (index -2, i.e. second-to-last) fires a V20 signal.

    We use -2 because -1 is the still-forming candle.
    """
    # Compute all indicators
    df = compute_macd(df, fast=13, slow=34, signal=9)
    df["atr"]    = compute_atr(df, period=14)
    df["sma200"] = compute_ma(df, period=200)
    df = compute_td_sequential(df)

    swing_highs, swing_lows = find_swing_points(
        df["high"].values, df["low"].values, lookback=20
    )

    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values
    volumes = df["volume"].values
    atr     = df["atr"].values
    hist    = df["macd_hist"].values
    sma200  = df["sma200"].values

    td_buy_setup     = df["td_buy_setup"].values
    td_buy_countdown = df["td_buy_countdown"].values

    # Last COMPLETED candle
    i = len(df) - 2

    log.info(f"Checking candle {df['timestamp'].iloc[i]} | close={closes[i]:.2f}")
    log.info(f"  TD buy setup={td_buy_setup[i]}  countdown={td_buy_countdown[i]}")
    log.info(f"  SMA200={sma200[i]:.2f}  ATR={atr[i]:.2f}  MACD_hist={hist[i]:.4f}")

    if np.isnan(atr[i]) or np.isnan(sma200[i]):
        log.warning("Indicators not ready (NaN) — not enough history?")
        return None

    # Volume confirmation
    avg_vol = np.mean(volumes[max(0, i - 20):i]) if i > 20 else volumes[i]
    vol_ok  = volumes[i] > avg_vol * 0.8

    if not vol_ok:
        log.info(f"  Volume too low: {volumes[i]:.2f} vs avg {avg_vol:.2f} (need 80%)")
        return None

    # Fibonacci levels
    fib       = get_active_fib_levels(i, swing_highs, swing_lows, closes)
    fib_level = price_near_fib(closes[i], fib, tolerance=FIB_TOL) if fib else None

    # TD signals
    has_td9_buy  = td_buy_setup[i] == 9
    has_td13_buy = td_buy_countdown[i] == 13

    # MACD divergence
    macd_div     = check_bottom_divergence(i, hist, lows)
    has_macd_div = macd_div >= 1

    # Build confluence list
    confluence = []
    if has_td9_buy:
        confluence.append("TD9_buy")
    if has_td13_buy:
        confluence.append("TD13_buy")
    if has_macd_div:
        confluence.append(f"MACD_div_{macd_div}")
    if fib_level:
        confluence.append(f"Fib_{fib_level}")
    if volumes[i] > avg_vol * 1.5:
        confluence.append("volume_spike")

    log.info(f"  Confluence factors: {confluence or 'none'}")

    # ── Determine signal tier ──────────────────────────────────────
    long_signal = False
    leverage    = 0

    if has_td13_buy and fib_level:                        # Tier 1 (20x)
        long_signal = True
        leverage    = 20
    elif has_td9_buy and has_macd_div:                    # Tier 2 (20x)
        long_signal = True
        leverage    = 20
    elif has_td9_buy and fib_level:                       # Tier 3 (10x)
        long_signal = True
        leverage    = 10
    elif has_td13_buy:                                    # Tier 4 (10x)
        long_signal = True
        leverage    = 10
    elif has_td9_buy and volumes[i] > avg_vol * 1.5:     # Tier 5 (5x)
        long_signal = True
        leverage    = 5
        if "volume_spike" not in confluence:
            confluence.append("volume_spike")
    elif has_macd_div and fib_level and macd_div >= 2:   # Tier 6 (5x)
        long_signal = True
        leverage    = 5

    if not long_signal:
        log.info("  → No signal")
        return None

    # ── Compute levels ────────────────────────────────────────────
    sl_atr = closes[i] - atr[i] * 1.5
    sl_fib  = fib["0.786"] if fib else sl_atr
    sl      = max(sl_atr, sl_fib)   # tighter stop

    if fib:
        tp1 = fib["ext_1.272"]
        tp2 = fib["ext_1.618"]
    else:
        risk = closes[i] - sl
        tp1  = closes[i] + risk * 1.5
        tp2  = closes[i] + risk * 2.5

    return SignalResult(
        side           = "long",
        leverage       = leverage,
        entry_price    = closes[i],
        stop_loss      = sl,
        take_profit_1  = tp1,
        take_profit_2  = tp2,
        confluence     = confluence,
        fib_level      = fib_level or "",
        fib            = fib,
        candle_time    = df["timestamp"].iloc[i],
        td_buy_setup   = int(td_buy_setup[i]),
        td_buy_countdown = int(td_buy_countdown[i]),
        atr            = float(atr[i]),
        sma200         = float(sma200[i]),
        macd_hist      = float(hist[i]),
        volume         = float(volumes[i]),
        avg_volume     = float(avg_vol),
    )


# ══════════════════════════════════════════════════════════════════════
# MARKET ANALYSIS (always returns context, even without a signal)
# ══════════════════════════════════════════════════════════════════════

def get_market_analysis(df: pd.DataFrame) -> dict:
    """Analyze current market state for dashboard reasoning.

    Always returns a rich context dict regardless of whether a signal fires.
    Uses index -2 (last completed candle), same as detect_signal().
    """
    # Compute all indicators
    df = compute_macd(df, fast=13, slow=34, signal=9)
    df["atr"]    = compute_atr(df, period=14)
    df["sma200"] = compute_ma(df, period=200)
    df = compute_td_sequential(df)

    swing_highs, swing_lows = find_swing_points(
        df["high"].values, df["low"].values, lookback=20
    )

    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values
    volumes = df["volume"].values
    atr_arr = df["atr"].values
    hist    = df["macd_hist"].values
    sma200  = df["sma200"].values

    td_buy_setup_arr     = df["td_buy_setup"].values
    td_sell_setup_arr    = df["td_sell_setup"].values
    td_buy_countdown_arr = df["td_buy_countdown"].values

    # Last COMPLETED candle
    i = len(df) - 2

    price  = float(closes[i])
    sma200_val = float(sma200[i]) if not np.isnan(sma200[i]) else None
    atr_val    = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else None

    # ── Trend ──────────────────────────────────────────────────────
    trend = "BULL" if (sma200_val and price > sma200_val) else "BEAR"

    # ── TD Sequential ──────────────────────────────────────────────
    td_buy_setup     = int(td_buy_setup_arr[i])
    td_sell_setup    = int(td_sell_setup_arr[i])
    td_buy_countdown = int(td_buy_countdown_arr[i])

    # ── MACD ────────────────────────────────────────────────────────
    macd_hist_val = float(hist[i])

    # Direction: compare last 3 bars
    if i >= 2:
        h0, h1, h2 = hist[i - 2], hist[i - 1], hist[i]
        if h2 > h1 > h0:
            macd_direction = "turning_up"
        elif h2 < h1 < h0:
            macd_direction = "turning_down"
        elif h2 > h1:
            macd_direction = "turning_up"
        elif h2 < h1:
            macd_direction = "turning_down"
        else:
            macd_direction = "flat"
    else:
        macd_direction = "flat"

    macd_div = check_bottom_divergence(i, hist, lows)

    # ── Fibonacci ───────────────────────────────────────────────────
    fib = get_active_fib_levels(i, swing_highs, swing_lows, closes)
    fib_levels_out = None
    nearest_fib    = None
    fib_distance_pct = None

    if fib:
        fib_levels_out = {
            "0.382": float(round(fib["0.382"], 2)),
            "0.500": float(round(fib["0.500"], 2)),
            "0.618": float(round(fib["0.618"], 2)),
        }
        # Find nearest key fib level
        best_level = None
        best_dist  = float("inf")
        for lvl in ["0.618", "0.500", "0.382"]:
            dist = abs(price - fib[lvl]) / price
            if dist < best_dist:
                best_dist  = dist
                best_level = lvl
        nearest_fib      = best_level
        fib_distance_pct = round(best_dist * 100, 2)

    # ── Volume ──────────────────────────────────────────────────────
    vol_window  = volumes[max(0, i - 20):i]
    avg_vol     = float(np.mean(vol_window)) if len(vol_window) > 0 else float(volumes[i])
    volume_ratio = round(float(volumes[i]) / avg_vol, 2) if avg_vol > 0 else 1.0
    vol_ok       = volumes[i] > avg_vol * 0.8

    # ── Confluence assessment ───────────────────────────────────────
    confluence_count = 0
    missing = []

    has_td9_buy      = td_buy_setup == 9
    has_td13_buy     = td_buy_countdown == 13
    has_macd_div     = macd_div >= 1
    has_fib_near     = nearest_fib is not None and fib_distance_pct is not None and fib_distance_pct <= 1.2
    has_volume_spike = volume_ratio >= 1.5

    if has_td9_buy or has_td13_buy:
        confluence_count += 1
    if has_macd_div:
        confluence_count += 1
    if has_fib_near:
        confluence_count += 1
    if has_volume_spike:
        confluence_count += 1

    # ── Signal detection (quick check) ─────────────────────────────
    signal_fired = False
    if vol_ok and (
        (has_td13_buy and has_fib_near)
        or (has_td9_buy and has_macd_div)
        or (has_td9_buy and has_fib_near)
        or has_td13_buy
        or (has_td9_buy and has_volume_spike)
        or (has_macd_div and has_fib_near and macd_div >= 2)
    ):
        signal_fired = True

    # ── Signal proximity ────────────────────────────────────────────
    if signal_fired:
        signal_proximity = "FIRED"
    elif confluence_count >= 2:
        signal_proximity = "CLOSE"
    elif confluence_count >= 1 or td_buy_setup >= 7:
        signal_proximity = "APPROACHING"
    else:
        signal_proximity = "NONE"

    # ── Build reasoning lines ───────────────────────────────────────
    reasoning = []

    # Trend line
    if sma200_val:
        trend_icon = "📈" if trend == "BULL" else "📉"
        direction  = "above" if trend == "BULL" else "below"
        reasoning.append(
            f"{trend_icon} {trend} trend — price ${price:,.0f} {direction} SMA200 ${sma200_val:,.0f}"
        )

    # TD Setup line
    if td_buy_setup > 0:
        remaining = 9 - td_buy_setup
        if td_buy_setup == 9:
            reasoning.append(f"🔢 TD buy setup COMPLETE (9/9) — watching for momentum confirmation")
        else:
            reasoning.append(f"🔢 TD buy setup at {td_buy_setup}/9 — {remaining} candles from potential TD9")
    elif td_sell_setup > 0:
        if td_sell_setup == 9:
            reasoning.append(f"🔢 TD sell setup COMPLETE (9/9) — potential sell exhaustion zone")
        else:
            remaining = 9 - td_sell_setup
            reasoning.append(f"🔢 TD sell setup at {td_sell_setup}/9 — {remaining} candles to potential sell signal")
    else:
        reasoning.append("🔢 TD setup reset — no active setup building")

    # TD Countdown line
    if td_buy_countdown > 0:
        remaining_cd = 13 - td_buy_countdown
        if td_buy_countdown == 13:
            reasoning.append(f"🎯 TD buy countdown COMPLETE (13/13) — high-probability reversal zone")
        else:
            reasoning.append(f"🎯 TD buy countdown at {td_buy_countdown}/13 — {remaining_cd} bars to completion")

    # MACD line
    if i >= 2:
        h_prev, h_curr = float(hist[i - 1]), float(hist[i])
        if macd_direction == "turning_up":
            reasoning.append(
                f"📊 MACD histogram turning up ({h_prev:+.0f} → {h_curr:+.0f}) — momentum shifting"
            )
        elif macd_direction == "turning_down":
            reasoning.append(
                f"📊 MACD histogram turning down ({h_prev:+.0f} → {h_curr:+.0f}) — momentum weakening"
            )
        else:
            reasoning.append(f"📊 MACD histogram flat ({h_curr:+.0f}) — no directional momentum")

    if has_macd_div:
        reasoning.append(f"✨ Bullish MACD divergence detected (strength {macd_div}/3) — higher lows on price, higher troughs on MACD")
    else:
        missing.append("MACD divergence")

    # Fib line
    if nearest_fib and fib_distance_pct is not None and fib_levels_out:
        fib_price = fib_levels_out.get(nearest_fib, 0)
        direction = "above" if price > fib_price else "below"
        if has_fib_near:
            reasoning.append(
                f"📐 Price near Fib {nearest_fib} (${fib_price:,.0f}) — {fib_distance_pct:.1f}% away, within tolerance"
            )
        else:
            reasoning.append(
                f"📐 Price {fib_distance_pct:.1f}% {direction} Fib {nearest_fib} (${fib_price:,.0f}) — approaching key level"
            )
        if not has_fib_near:
            missing.append(f"Fib {nearest_fib} touch")
    else:
        reasoning.append("📐 No clear Fibonacci level in range — waiting for retracement setup")
        missing.append("Fibonacci confluence")

    # Volume line
    if volume_ratio >= 1.5:
        reasoning.append(f"📈 Volume spike ({volume_ratio:.0%} of avg) — strong participation, confirmation present")
    elif vol_ok:
        reasoning.append(f"📉 Volume adequate ({volume_ratio:.0%} of avg) — normal participation")
    else:
        reasoning.append(f"📉 Volume low ({volume_ratio:.0%} of avg) — needs pickup for valid signal")
        missing.append("volume confirmation")

    # ATR line
    if atr_val:
        reasoning.append(f"📏 ATR ${atr_val:,.0f} — market volatility gauge")

    # Summary / missing line
    if signal_fired:
        reasoning.append("⚡ All confluence factors met — SIGNAL FIRED")
    elif missing:
        missing_str = " + ".join(missing[:3])
        reasoning.append(f"⏳ Missing: {missing_str}")

    candle_time = df["timestamp"].iloc[i]
    if hasattr(candle_time, "isoformat"):
        candle_time_str = candle_time.isoformat()
    else:
        candle_time_str = str(candle_time)

    return {
        "price":             round(price, 2),
        "sma200":            round(sma200_val, 2) if sma200_val else None,
        "trend":             trend,
        "td_buy_setup":      td_buy_setup,
        "td_buy_countdown":  td_buy_countdown,
        "td_sell_setup":     td_sell_setup,
        "macd_hist":         round(macd_hist_val, 2),
        "macd_direction":    macd_direction,
        "macd_divergence":   macd_div,
        "fib_levels":        fib_levels_out,
        "nearest_fib":       nearest_fib,
        "fib_distance_pct":  fib_distance_pct,
        "volume_ratio":      volume_ratio,
        "atr":               round(atr_val, 2) if atr_val else None,
        "candle_time":       candle_time_str,
        "reasoning":         reasoning,
        "signal_proximity":  signal_proximity,
    }


# ══════════════════════════════════════════════════════════════════════
# DATA FETCH (Bybit public OHLCV — no auth required)
# ══════════════════════════════════════════════════════════════════════

def fetch_candles() -> pd.DataFrame:
    """Fetch latest 500 4H BTC/USDT candles from Bybit (public API)."""
    log.info("Fetching 4H BTC/USDT candles from Bybit demo…")
    exchange = ccxt.bybit({"enableRateLimit": True})

    # Demo endpoint override (no auth needed for OHLCV)
    exchange.urls["api"] = {
        "public":  "https://api-demo.bybit.com",
        "private": "https://api-demo.bybit.com",
    }
    exchange.has["fetchCurrencies"] = False

    raw = exchange.fetch_ohlcv("BTC/USDT", "4h", limit=FETCH_LIMIT)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    log.info(f"  Got {len(df)} candles, last: {df['timestamp'].iloc[-1]}")
    return df


# ══════════════════════════════════════════════════════════════════════
# BYBIT DEMO — ACCOUNT & ORDER EXECUTION
# ══════════════════════════════════════════════════════════════════════

def get_exchange() -> ccxt.Exchange:
    """Authenticated Bybit demo exchange instance."""
    exchange = ccxt.bybit({
        "apiKey":          BYBIT_API_KEY,
        "secret":          BYBIT_SECRET,
        "enableRateLimit": True,
        "options":         {"defaultType": "linear"},
    })
    exchange.urls["api"] = {
        "public":  "https://api-demo.bybit.com",
        "private": "https://api-demo.bybit.com",
    }
    exchange.has["fetchCurrencies"] = False
    return exchange


def get_balance(exchange: ccxt.Exchange) -> float:
    """Return USDT balance from Bybit demo."""
    try:
        bal = exchange.fetch_balance()
        usdt = bal.get("USDT", {}).get("free", 0)
        log.info(f"Account balance: {usdt:.2f} USDT")
        return float(usdt)
    except Exception as e:
        log.error(f"Failed to fetch balance: {e}")
        return 0.0


def calc_position_size(balance: float, entry: float, stop_loss: float) -> float:
    """BTC quantity for 8% risk of balance."""
    risk_usdt = balance * RISK_PCT
    risk_per_btc = entry - stop_loss
    if risk_per_btc <= 0:
        return 0.0
    return risk_usdt / risk_per_btc


def place_orders(exchange: ccxt.Exchange, signal: SignalResult, qty_btc: float) -> dict:
    """Place market entry + stop-loss + TP1 limit order on Bybit demo perp.

    Returns dict with order IDs / status.
    """
    sym = SYMBOL
    results = {}

    # 1. Set leverage
    try:
        exchange.set_leverage(signal.leverage, sym)
        log.info(f"Leverage set to {signal.leverage}x")
    except Exception as e:
        log.warning(f"set_leverage: {e}")

    # Round qty to 3 decimal places (Bybit min step for BTC perp)
    qty = round(qty_btc, 3)
    if qty <= 0:
        log.error("Calculated qty is 0 — skipping order placement")
        return {"error": "qty=0"}

    # 2. Market entry
    try:
        entry_order = exchange.create_market_buy_order(sym, qty, params={
            "reduceOnly": False,
            "positionIdx": 0,
        })
        results["entry"] = entry_order
        log.info(f"  ✅ Market BUY {qty} BTC @ market | id={entry_order.get('id')}")
    except Exception as e:
        log.error(f"Market entry failed: {e}")
        results["entry_error"] = str(e)
        return results

    # 3. Stop-loss order (conditional / stop-market)
    try:
        sl_order = exchange.create_order(sym, "stop_market", "sell", qty, signal.stop_loss, params={
            "stopPrice":  signal.stop_loss,
            "reduceOnly": True,
            "positionIdx": 0,
            "triggerDirection": 2,  # below
        })
        results["stop_loss"] = sl_order
        log.info(f"  ✅ Stop-loss @ {signal.stop_loss:.2f} | id={sl_order.get('id')}")
    except Exception as e:
        log.error(f"Stop-loss order failed: {e}")
        results["sl_error"] = str(e)

    # 4. TP1 limit order (50% of position)
    tp1_qty = round(qty / 2, 3)
    try:
        tp1_order = exchange.create_limit_sell_order(sym, tp1_qty, signal.take_profit_1, params={
            "reduceOnly": True,
            "positionIdx": 0,
        })
        results["tp1"] = tp1_order
        log.info(f"  ✅ TP1 limit sell {tp1_qty} BTC @ {signal.take_profit_1:.2f} | id={tp1_order.get('id')}")
    except Exception as e:
        log.error(f"TP1 order failed: {e}")
        results["tp1_error"] = str(e)

    return results


# ══════════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════════

def send_telegram(message: str) -> bool:
    """Send message to Telegram via bot API."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured (missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "Markdown",
        }, timeout=15)
        resp.raise_for_status()
        log.info("Telegram alert sent ✅")
        return True
    except Exception as e:
        log.error(f"Telegram send failed: {e}")
        return False


def format_signal_alert(signal: SignalResult, balance: float, qty: float,
                         order_results: dict, dry_run: bool) -> str:
    """Format a Markdown alert message for Telegram."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    conf_str = " + ".join(signal.confluence)
    mode = "🔬 DRY RUN" if dry_run else "⚡ LIVE"

    risk_pct  = RISK_PCT * 100
    risk_usdt = balance * RISK_PCT
    rr_ratio  = (signal.take_profit_1 - signal.entry_price) / (signal.entry_price - signal.stop_loss)

    lines = [
        f"🚀 *V20 SIGNAL FIRED* — {mode}",
        f"⏰ {now_str}  |  Candle: {signal.candle_time}",
        f"",
        f"📈 *{signal.side.upper()}* BTC/USDT Perp  @ {signal.leverage}x leverage",
        f"",
        f"📊 *Trade Levels*",
        f"  Entry:    `${signal.entry_price:,.2f}`",
        f"  Stop:     `${signal.stop_loss:,.2f}`",
        f"  TP1 (50%):`${signal.take_profit_1:,.2f}`",
        f"  TP2 (rem):`${signal.take_profit_2:,.2f}`",
        f"  R/R:      `{rr_ratio:.2f}x`",
        f"",
        f"💡 *Confluence*",
        f"  {conf_str}",
        f"",
        f"📐 *Fib Level*: {signal.fib_level or 'n/a'}",
        f"  ATR:   `{signal.atr:.2f}`",
        f"  SMA200:`{signal.sma200:,.2f}`",
        f"",
        f"💰 *Position Size*",
        f"  Balance:  `${balance:,.2f}` USDT",
        f"  Risk:     `{risk_pct:.0f}%` = `${risk_usdt:,.2f}`",
        f"  Qty:      `{qty:.4f}` BTC",
    ]

    if dry_run:
        lines.append(f"")
        lines.append(f"⚠️ _Dry-run — no orders placed_")
    else:
        entry_id = order_results.get("entry", {}).get("id", "n/a")
        sl_id    = order_results.get("stop_loss", {}).get("id", "n/a")
        tp1_id   = order_results.get("tp1", {}).get("id", "n/a")
        lines += [
            f"",
            f"🔖 *Order IDs*",
            f"  Entry: `{entry_id}`",
            f"  SL:    `{sl_id}`",
            f"  TP1:   `{tp1_id}`",
        ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main(dry_run: bool = False):
    log.info("=" * 60)
    log.info("V20 Signal Scanner — Fibonacci + TD Sequential + MACD")
    log.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    log.info("=" * 60)

    # 1. Fetch candles
    df = fetch_candles()
    if len(df) < 250:
        log.error(f"Not enough candles ({len(df)}) to compute indicators — aborting")
        sys.exit(1)

    # 2. Detect signal
    signal = detect_signal(df)

    if signal is None:
        print("✗ No signal — scan complete, no action taken.")
        return

    # 3. Signal fired!
    print()
    print("=" * 60)
    print("  ⚡ V20 SIGNAL FIRED!")
    print("=" * 60)
    print(f"  Side:        {signal.side.upper()}")
    print(f"  Leverage:    {signal.leverage}x")
    print(f"  Confluence:  {' + '.join(signal.confluence)}")
    print(f"  Entry:       ${signal.entry_price:,.2f}")
    print(f"  Stop Loss:   ${signal.stop_loss:,.2f}")
    print(f"  TP1 (50%):   ${signal.take_profit_1:,.2f}")
    print(f"  TP2 (rest):  ${signal.take_profit_2:,.2f}")
    print(f"  Fib Level:   {signal.fib_level or 'n/a'}")
    print(f"  ATR:         {signal.atr:.2f}")
    print(f"  Candle time: {signal.candle_time}")
    print()

    # 3.5. Risk Guard pre-trade check
    try:
        from scripts.risk_guard import load_state as rg_load_state
        rg_state = rg_load_state()
        no_trade = rg_state.get("no_trade_until")
        if no_trade:
            from datetime import timezone as _tz
            until_dt = datetime.fromisoformat(no_trade).replace(tzinfo=_tz.utc)
            if datetime.now(_tz.utc) < until_dt:
                print(f"  🛡️ RISK GUARD: Trading BLOCKED until {no_trade} UTC (daily loss limit)")
                print("  Signal detected but NOT executed. Skipping.")
                return
            else:
                print("  🛡️ Risk Guard: Trade block expired, proceeding.")
    except ImportError:
        log.warning("Risk guard module not found — skipping pre-trade check")
    except Exception as e:
        log.warning(f"Risk guard pre-check failed: {e} — proceeding anyway")

    # 4. Get balance and calculate size
    exchange  = get_exchange()
    balance   = get_balance(exchange)
    qty_btc   = calc_position_size(balance, signal.entry_price, signal.stop_loss)
    risk_usdt = balance * RISK_PCT

    print(f"  Balance:     ${balance:,.2f} USDT")
    print(f"  Risk (8%):   ${risk_usdt:,.2f} USDT")
    print(f"  Position:    {qty_btc:.4f} BTC")
    print()

    # 5. Place orders (or skip in dry-run)
    order_results = {}
    if dry_run:
        print("  [DRY RUN] Skipping order placement.")
    else:
        if balance < 10:
            log.error("Balance too low to trade — aborting")
            return
        if qty_btc < 0.001:
            log.error(f"Calculated qty {qty_btc:.4f} BTC is below minimum — aborting")
            return
        order_results = place_orders(exchange, signal, qty_btc)

    # 6. Format and send Telegram alert
    alert_msg = format_signal_alert(signal, balance, qty_btc, order_results, dry_run)
    print("\n--- Telegram Alert ---")
    print(alert_msg)
    print("----------------------\n")
    send_telegram(alert_msg)

    # 7. Run Risk Guard post-trade check
    try:
        from scripts.risk_guard import main as risk_guard_main
        print("\n--- Risk Guard Post-Check ---")
        risk_guard_main()
        print("-----------------------------\n")
    except Exception as e:
        log.warning(f"Risk guard post-check failed: {e}")

    print("✓ Scanner complete.")
    return signal


def run_cli():
    parser = argparse.ArgumentParser(description="V20 Live Signal Scanner")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute signal and print details, but skip order placement."
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)


if __name__ == "__main__":
    run_cli()

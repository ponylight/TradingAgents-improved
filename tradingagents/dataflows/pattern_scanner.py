"""
Pattern Scanner — Automated detection of verified trader strategies.

Runs all pattern checks against current market data and returns structured signals
that agents can consume directly. No truncation, no prompt bloat — just data.

Patterns from LabSpeculation (投机实验室) research:
1. Buying Climax Short (Kyle Williams)
2. VCP Breakout (Minervini)
3. New High Momentum Breakout (Hong Inki)
4. BNF Contrarian Mean Reversion
5. BNF Trend Following (sector laggard)
6. MACD Triple Divergence (半木夏)
7. HH+HL Reversal Detection (比特皇)
8. Volume Exhaustion Bottom (比特皇)
9. Triangle Consolidation (比特皇)
10. Sykes Stage Identification
11. Bonde Home Run Breakout
12. Qullamaggie Breakout + EP
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("pattern_scanner")


def _fetch_ohlcv(symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data and return as DataFrame."""
    try:
        from tradingagents.dataflows.ccxt_crypto import get_crypto_ohlcv
        from io import StringIO

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        raw = get_crypto_ohlcv(symbol, start, end, timeframe)
        if not raw or "Error" in str(raw):
            return None

        lines = raw.strip().split("\n")
        header_idx = None
        for i, line in enumerate(lines):
            if "date" in line.lower() or "open" in line.lower():
                header_idx = i
                break
        if header_idx is None:
            return None

        csv_data = "\n".join(lines[header_idx:])
        df = pd.read_csv(StringIO(csv_data))

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure ascending date order — some sources return descending
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)

        return df
    except Exception as e:
        log.warning(f"Failed to fetch OHLCV: {e}")
        return None


def _fetch_fear_greed() -> Optional[int]:
    """Get current Fear & Greed index value."""
    try:
        from tradingagents.dataflows.ccxt_crypto import get_fear_greed_index
        raw = get_fear_greed_index()
        # Parse the number from the response
        for line in raw.split("\n"):
            if "value" in line.lower() or "index" in line.lower():
                import re
                nums = re.findall(r'\d+', line)
                if nums:
                    return int(nums[0])
        return None
    except Exception:
        return None


# ========== PATTERN DETECTORS ==========


def detect_buying_climax(df: pd.DataFrame, fg_value: Optional[int] = None) -> Dict[str, Any]:
    """
    Kyle Williams — Buying Climax Short (顶部高潮空)
    Detect parabolic blow-off tops ripe for mean-reversion short.
    """
    result = {"pattern": "buying_climax_short", "detected": False, "confidence": 0, "details": ""}

    if df is None or len(df) < 10:
        return result

    close = df["close"].values
    volume = df["volume"].values
    high = df["high"].values

    # Check last 5 days for parabolic move
    last5_returns = [(close[i] - close[i-1]) / close[i-1] for i in range(-4, 0)]
    consecutive_green = sum(1 for r in last5_returns if r > 0)
    total_gain = (close[-1] - close[-5]) / close[-5] if close[-5] > 0 else 0

    # Accelerating gains
    accelerating = all(last5_returns[i] < last5_returns[i+1] for i in range(len(last5_returns)-1) if last5_returns[i] > 0 and last5_returns[i+1] > 0)

    # Volume climax — last day volume vs 20-day average
    vol_avg_20 = np.mean(volume[-21:-1]) if len(volume) > 21 else np.mean(volume[:-1])
    vol_ratio = volume[-1] / vol_avg_20 if vol_avg_20 > 0 else 1

    score = 0
    details = []

    if consecutive_green >= 3:
        score += 2
        details.append(f"{consecutive_green} consecutive green days")
    if total_gain >= 0.15:  # 15%+ gain in 5 days
        score += 2
        details.append(f"5-day gain: {total_gain:.1%}")
    if accelerating:
        score += 2
        details.append("gains accelerating")
    if vol_ratio >= 2.0:
        score += 2
        details.append(f"volume climax: {vol_ratio:.1f}x avg")
    if fg_value and fg_value >= 75:
        score += 2
        details.append(f"F&G: {fg_value} (extreme greed)")

    if score >= 6:
        result["detected"] = True
        result["confidence"] = min(10, score)
        result["details"] = f"BEARISH — Buying climax detected: {', '.join(details)}. " \
                           f"Entry: short on weakness below ${close[-1]:.0f}. " \
                           f"Stop: above ${max(high[-5:]):.0f}. " \
                           f"TP: -10% of recent range."
        result["direction"] = "SHORT"
        result["stop_price"] = float(max(high[-5:]))
    return result


def detect_vcp_breakout(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Minervini — VCP Breakout (缩量整理突破)
    Detect volatility contraction pattern with decreasing pullback depth.
    """
    result = {"pattern": "vcp_breakout", "detected": False, "confidence": 0, "details": ""}

    if df is None or len(df) < 200:
        return result

    close = df["close"].values
    volume = df["volume"].values

    # Trend template check
    sma50 = pd.Series(close).rolling(50).mean().values
    sma150 = pd.Series(close).rolling(150).mean().values
    sma200 = pd.Series(close).rolling(200).mean().values

    if np.isnan(sma200[-1]):
        return result

    score = 0
    details = []

    # Bullish stack: price > 50MA > 150MA > 200MA
    if close[-1] > sma50[-1] > sma150[-1] > sma200[-1]:
        score += 2
        details.append("bullish MA stack")

    # 200MA rising for 1+ month
    if sma200[-1] > sma200[-22]:
        score += 1
        details.append("200MA rising")

    # Price >25% above 52-week low
    low_52w = min(close[-252:]) if len(close) >= 252 else min(close)
    if close[-1] > low_52w * 1.25:
        score += 1
        details.append(f">{((close[-1]/low_52w)-1)*100:.0f}% above 52w low")

    # VCP: Look for contracting pullback swings in last 30 days
    last30_high = max(close[-30:])
    pullbacks = []
    in_pullback = False
    pb_start = 0
    # Use positive indices to avoid empty-slice bug at boundaries
    base_idx = len(close) - 30
    for j in range(base_idx, len(close)):
        i_prev = j - 1
        if close[j] < close[i_prev] and not in_pullback:
            in_pullback = True
            pb_start = close[i_prev]
        elif close[j] > close[i_prev] and in_pullback:
            in_pullback = False
            seg_start = max(base_idx, j - 5)
            seg = close[seg_start:j + 1]
            pb_depth = (pb_start - min(seg)) / pb_start if pb_start > 0 and len(seg) > 0 else 0
            pullbacks.append(pb_depth)

    if len(pullbacks) >= 2 and all(pullbacks[i] > pullbacks[i+1] for i in range(len(pullbacks)-1)):
        score += 3
        details.append(f"VCP: {len(pullbacks)} contracting pullbacks")

    # Volume drying up in last 5 days
    vol_recent = np.mean(volume[-5:])
    vol_avg = np.mean(volume[-30:])
    if vol_recent < vol_avg * 0.7:
        score += 1
        details.append("volume drying up")

    if score >= 5:
        result["detected"] = True
        result["confidence"] = min(10, score)
        result["details"] = f"BULLISH — VCP setup: {', '.join(details)}. " \
                           f"Entry: on breakout above ${last30_high:.0f} with volume. " \
                           f"Stop: below last pivot low."
        result["direction"] = "BUY"
        result["entry_price"] = float(last30_high)
    return result


def detect_new_high_breakout(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Hong Inki — New High Momentum Breakout (首次新高突破)
    First breakout above 6-month high with strong candle.
    """
    result = {"pattern": "new_high_breakout", "detected": False, "confidence": 0, "details": ""}

    if df is None or len(df) < 180:
        return result

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    # 6-month high (prior to last 3 days)
    high_6m = max(high[:-3]) if len(high) > 3 else max(high[:-1])

    score = 0
    details = []

    # Check last 3 candles for breakout
    for lookback in range(1, min(4, len(close))):
        idx = -lookback
        candle_range = high[idx] - low[idx]
        avg_range = np.mean([high[i] - low[i] for i in range(-30, -1)])
        body = abs(close[idx] - (close[idx-1] if idx > -len(close) else close[0]))
        gain = (close[idx] - close[idx-1]) / close[idx-1] if close[idx-1] > 0 else 0

        if close[idx] > high_6m:
            score += 3
            details.append(f"broke 6M high ${high_6m:.0f}")

            if gain >= 0.05:
                score += 2
                details.append(f"breakout gain: {gain:.1%}")

            if candle_range > avg_range * 2:
                score += 1
                details.append("long candle (2x avg range)")

            vol_avg = np.mean(volume[-30:])
            if volume[idx] > vol_avg * 1.5:
                score += 2
                details.append(f"volume {volume[idx]/vol_avg:.1f}x avg")
            break

    if score >= 5:
        result["detected"] = True
        result["confidence"] = min(10, score)
        # Hong Inki: trade within 2 days of breakout
        result["details"] = f"BULLISH — New high breakout: {', '.join(details)}. " \
                           f"Entry: within 2 days of breakout. " \
                           f"Stop: below breakout candle low. " \
                           f"Day 3: exit if doji/-3% or +5% profit-take."
        result["direction"] = "BUY"
    return result


def detect_bnf_contrarian(df: pd.DataFrame) -> Dict[str, Any]:
    """
    BNF — 25MA Deviation Mean Reversion (逆势投资)
    Buy when price is significantly below 25MA (oversold bounce).
    """
    result = {"pattern": "bnf_contrarian", "detected": False, "confidence": 0, "details": ""}

    if df is None or len(df) < 30:
        return result

    close = df["close"].values
    sma25 = pd.Series(close).rolling(25).mean().values

    if np.isnan(sma25[-1]):
        return result

    deviation = (close[-1] - sma25[-1]) / sma25[-1]

    score = 0
    details = []

    # BNF: significant negative deviation → buy for bounce
    if deviation <= -0.10:
        score += 3
        details.append(f"25MA deviation: {deviation:.1%}")
    elif deviation <= -0.07:
        score += 2
        details.append(f"25MA deviation: {deviation:.1%}")
    elif deviation <= -0.05:
        score += 1
        details.append(f"25MA deviation: {deviation:.1%}")

    # Significant positive deviation → overbought warning
    if deviation >= 0.10:
        result["details"] = f"OVERBOUGHT — 25MA deviation +{deviation:.1%}. " \
                           f"BNF: price is overheated, caution on longs."
        result["confidence"] = min(10, int(deviation * 50))
        result["direction"] = "CAUTION_LONG"
        result["detected"] = True
        return result

    # Check if we're in a downtrend (BNF contrarian works in bear markets)
    sma50 = pd.Series(close).rolling(50).mean().values
    if not np.isnan(sma50[-1]) and close[-1] < sma50[-1]:
        score += 1
        details.append("below 50MA (bear context)")

    # Volume declining (selling exhaustion)
    if len(df["volume"].values) > 10:
        vol_recent = np.mean(df["volume"].values[-5:])
        vol_prior = np.mean(df["volume"].values[-15:-5])
        if vol_recent < vol_prior * 0.7:
            score += 1
            details.append("volume declining (exhaustion)")

    if score >= 3 and deviation < 0:
        result["detected"] = True
        result["confidence"] = min(10, score + 2)
        result["details"] = f"BULLISH (contrarian) — BNF mean reversion: {', '.join(details)}. " \
                           f"Price ${close[-1]:.0f} vs 25MA ${sma25[-1]:.0f}. " \
                           f"Entry: buy for bounce. Hold: 1 day to 1 week. " \
                           f"BNF rule: NO leverage during accumulation."
        result["direction"] = "BUY"
    return result


def detect_hh_hl_reversal(df: pd.DataFrame) -> Dict[str, Any]:
    """
    比特皇 — Higher High + Higher Low Reversal Detection
    After a downtrend, if 2nd bounce > 1st bounce AND 2nd dip > 1st dip → reversal.
    """
    result = {"pattern": "hh_hl_reversal", "detected": False, "confidence": 0, "details": ""}

    if df is None or len(df) < 60:
        return result

    close = df["close"].values[-60:]
    high = df["high"].values[-60:]
    low = df["low"].values[-60:]

    # Find swing points in last 60 candles
    swing_highs = []
    swing_lows = []
    window = 5

    for i in range(window, len(close) - window):
        if all(high[i] >= high[j] for j in range(i-window, i+window+1)):
            swing_highs.append((i, float(high[i])))
        if all(low[i] <= low[j] for j in range(i-window, i+window+1)):
            swing_lows.append((i, float(low[i])))

    # Need at least 2 swing highs and 2 swing lows
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return result

    # Check last 2 swing highs and lows (比特皇: B > A and D > C)
    sh1, sh2 = swing_highs[-2], swing_highs[-1]
    sl1, sl2 = swing_lows[-2], swing_lows[-1]

    score = 0
    details = []

    # Higher high
    if sh2[1] > sh1[1]:
        score += 3
        details.append(f"higher high: ${sh1[1]:.0f} → ${sh2[1]:.0f}")

    # Higher low
    if sl2[1] > sl1[1]:
        score += 3
        details.append(f"higher low: ${sl1[1]:.0f} → ${sl2[1]:.0f}")

    # Was there a prior downtrend? (close was lower 30 days ago)
    if len(close) > 30 and close[-30] > close[-15]:
        score += 2
        details.append("prior downtrend present")

    # If B < A (lower high) → bearish continuation flag
    if sh2[1] < sh1[1] and sl2[1] < sl1[1]:
        result["detected"] = True
        result["confidence"] = 5
        result["details"] = f"BEARISH — Lower highs + lower lows: " \
                           f"highs ${sh1[1]:.0f} → ${sh2[1]:.0f}, " \
                           f"lows ${sl1[1]:.0f} → ${sl2[1]:.0f}. " \
                           f"比特皇: likely downtrend continuation."
        result["direction"] = "SELL"
        return result

    if score >= 5:
        result["detected"] = True
        result["confidence"] = min(10, score)
        result["details"] = f"BULLISH — HH+HL reversal: {', '.join(details)}. " \
                           f"比特皇: bottom likely forming. " \
                           f"Entry: on confirmation above ${sh2[1]:.0f}."
        result["direction"] = "BUY"
    return result


def detect_volume_exhaustion(df: pd.DataFrame) -> Dict[str, Any]:
    """
    比特皇 — Volume Exhaustion + Slope Decline Bottom
    Declining slope + declining volume = selling exhaustion = bottoming.
    """
    result = {"pattern": "volume_exhaustion", "detected": False, "confidence": 0, "details": ""}

    if df is None or len(df) < 30:
        return result

    close = df["close"].values
    volume = df["volume"].values

    score = 0
    details = []

    # Slope of price declining but flattening
    if len(close) >= 20:
        slope_early = (close[-15] - close[-20]) / 5  # slope 15-20 days ago
        slope_recent = (close[-1] - close[-5]) / 5    # slope last 5 days

        if slope_early < 0 and slope_recent < 0:
            if abs(slope_recent) < abs(slope_early) * 0.6:
                score += 3
                details.append(f"decline slowing: slope {slope_early:.0f} → {slope_recent:.0f}")

    # Volume declining over last 10 days
    if len(volume) >= 20:
        vol_early = np.mean(volume[-20:-10])
        vol_recent = np.mean(volume[-10:])
        if vol_recent < vol_early * 0.65:
            score += 3
            details.append(f"volume declining: {vol_recent/vol_early:.0%} of prior")

    # Price is in a downtrend
    if len(close) >= 30 and close[-1] < close[-30]:
        score += 1
        details.append("in downtrend")

    # Hammer/doji on recent candle
    if len(df) >= 2:
        last_body = abs(close[-1] - df["open"].values[-1]) if "open" in df.columns else 0
        last_range = df["high"].values[-1] - df["low"].values[-1]
        if last_range > 0 and last_body / last_range < 0.3:
            score += 2
            details.append("hammer/doji candle")

    if score >= 5:
        result["detected"] = True
        result["confidence"] = min(10, score)
        result["details"] = f"BULLISH (exhaustion) — Volume exhaustion bottom: {', '.join(details)}. " \
                           f"比特皇: selling exhausted, bounce likely. " \
                           f"Entry: on first bullish candle confirmation."
        result["direction"] = "BUY"
    return result


def detect_sykes_stage(df: pd.DataFrame, fg_value: Optional[int] = None) -> Dict[str, Any]:
    """
    Sykes — 7-Stage Lifecycle Identification
    Identify which stage the market is in.
    """
    result = {"pattern": "sykes_stage", "detected": True, "confidence": 5, "details": ""}

    if df is None or len(df) < 90:
        result["details"] = "Insufficient data for stage identification"
        result["detected"] = False
        return result

    close = df["close"].values
    volume = df["volume"].values

    # Find peak and trough in last 90 days
    peak_90 = max(close[-90:])
    peak_idx = len(close) - 90 + np.argmax(close[-90:])
    current = close[-1]
    drawdown = (current - peak_90) / peak_90

    # Volume trends
    vol_avg_30 = np.mean(volume[-30:])
    vol_avg_60 = np.mean(volume[-60:-30]) if len(volume) >= 60 else vol_avg_30

    # Stage identification
    if drawdown > -0.05 and current > close[-30]:
        # Near highs and trending up
        if vol_avg_30 > vol_avg_60 * 1.3:
            stage = 3
            stage_name = "SUPERNOVA (Stage 3)"
            action = "LONG entry zone — momentum + media frenzy. Trail with 10d MA."
        elif vol_avg_30 < vol_avg_60 * 0.7:
            stage = 1
            stage_name = "QUIET ACCUMULATION (Stage 1)"
            action = "Wait — no action. Smart money loading."
        else:
            stage = 2
            stage_name = "PRE-BREAKOUT (Stage 2)"
            action = "Prepare — watch for volume surge to confirm Stage 3 entry."
    elif drawdown <= -0.35:
        if fg_value and fg_value <= 25:
            stage = 5
            stage_name = "DIP BUY (Stage 5)"
            action = f"LONG entry zone — panic bounce. F&G={fg_value}. TP: 5-10%, max 5 day hold."
        else:
            stage = 4
            stage_name = "CLIFF DIVE (Stage 4)"
            action = "Caution — distribution phase. Wait for Stage 5 confirmation."
    elif drawdown <= -0.15:
        if peak_idx < len(close) - 30:  # Peak was >30 days ago
            stage = 6
            stage_name = "DEAD BOUNCE (Stage 6)"
            action = "Avoid — low reward bounces."
        else:
            stage = 4
            stage_name = "CLIFF DIVE (Stage 4)"
            action = "Caution — still falling."
    else:
        stage = 2
        stage_name = "CONSOLIDATION (Stage 2)"
        action = "Wait for breakout confirmation."

    result["confidence"] = 6
    result["details"] = f"Sykes Lifecycle: {stage_name}. " \
                       f"Drawdown from 90d peak: {drawdown:.1%}. " \
                       f"Action: {action}"
    result["stage"] = stage
    result["stage_name"] = stage_name
    return result


def detect_bonde_home_run(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Bonde — Home Run Breakout (本垒打动量突破)
    Tight consolidation + breakout with specific pre-conditions.
    """
    result = {"pattern": "bonde_home_run", "detected": False, "confidence": 0, "details": ""}

    if df is None or len(df) < 30:
        return result

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    score = 0
    details = []

    # 2Lynch check: NOT 2 consecutive up days before today
    if not (close[-2] > close[-3] and close[-3] > close[-4]):
        score += 1
        details.append("2Lynch: no 2 consecutive up days prior")

    # N (Narrow/Down): yesterday was down or narrow
    yesterday_range = high[-2] - low[-2]
    avg_range = np.mean([high[i] - low[i] for i in range(-15, -1)])
    if close[-2] < close[-3] or yesterday_range < avg_range * 0.5:
        score += 2
        details.append("narrow/down day prior")

    # Consolidation: decreasing volume over last 10 days
    if len(volume) >= 15:
        vol_5 = np.mean(volume[-6:-1])
        vol_15 = np.mean(volume[-16:-6])
        if vol_5 < vol_15 * 0.7:
            score += 2
            details.append("consolidation: vol declining")

    # Breakout: today closes in top 25% of range
    today_range = high[-1] - low[-1]
    if today_range > 0:
        close_position = (close[-1] - low[-1]) / today_range
        if close_position >= 0.75:
            score += 2
            details.append(f"high close: {close_position:.0%} of range")

    # Volume surge on breakout
    vol_avg_50 = np.mean(volume[-51:-1]) if len(volume) > 51 else np.mean(volume[:-1])
    if volume[-1] > vol_avg_50 * 1.5:
        score += 2
        details.append(f"volume surge: {volume[-1]/vol_avg_50:.1f}x avg")

    if score >= 6:
        result["detected"] = True
        result["confidence"] = min(10, score)
        result["details"] = f"BULLISH — Bonde Home Run: {', '.join(details)}. " \
                           f"Entry: confirmed. TP1: +8%, TP2: +15-20%. " \
                           f"Time stop: exit if no move in 3 days."
        result["direction"] = "BUY"
    return result


def detect_qullamaggie_breakout(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Qullamaggie — Breakout from tight range after strong prior move.
    """
    result = {"pattern": "qullamaggie_breakout", "detected": False, "confidence": 0, "details": ""}

    if df is None or len(df) < 60:
        return result

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    score = 0
    details = []

    # Prior strong move: 20%+ gain in prior 30-60 days
    if len(close) >= 60:
        prior_gain = (max(close[-60:-10]) - min(close[-60:-30])) / min(close[-60:-30]) if min(close[-60:-30]) > 0 else 0
        if prior_gain >= 0.20:
            score += 2
            details.append(f"prior move: +{prior_gain:.0%}")

    # Tight consolidation: last 10 days range is narrow
    if len(close) >= 15:
        range_10d = (max(high[-10:]) - min(low[-10:])) / close[-10] if close[-10] > 0 else 0
        if range_10d < 0.08:  # Less than 8% range
            score += 3
            details.append(f"tight range: {range_10d:.1%} over 10d")

    # Volume drying up during consolidation
    if len(volume) >= 15:
        vol_base = np.mean(volume[-10:-1])
        vol_prior = np.mean(volume[-30:-10])
        if vol_base < vol_prior * 0.6:
            score += 2
            details.append("volume dried up")

    # Breakout candle
    if len(close) >= 11:
        resistance = max(high[-11:-1])
        if close[-1] > resistance:
            score += 2
            details.append(f"broke resistance ${resistance:.0f}")

    if score >= 6:
        result["detected"] = True
        result["confidence"] = min(10, score)
        result["details"] = f"BULLISH — Qullamaggie breakout: {', '.join(details)}. " \
                           f"Entry: on breakout with volume confirmation. " \
                           f"Stop: below consolidation low."
        result["direction"] = "BUY"
    return result


# ========== MASTER SCANNER ==========


_SCAN_CACHE: Dict[str, Any] = {"result": None, "ts": 0}
MAX_SCAN_OUTPUT = 3000  # Cap output size for prompt safety


def scan_all_patterns(symbol: str = "BTC/USDT") -> str:
    """
    Run all pattern detectors against current market data.
    Returns a formatted report for agent consumption.
    Results cached for 5 minutes to avoid redundant API calls.
    """
    import time as _time
    now = _time.time()
    cache_key = symbol
    if _SCAN_CACHE.get("key") == cache_key and _SCAN_CACHE["result"] and (now - _SCAN_CACHE["ts"]) < 300:
        return _SCAN_CACHE["result"]

    log.info(f"🔍 Running pattern scan for {symbol}")

    # Fetch data
    df_daily = _fetch_ohlcv(symbol, "1d", 250)
    df_4h = _fetch_ohlcv(symbol, "4h", 60)
    fg_value = _fetch_fear_greed()

    if df_daily is None or len(df_daily) < 30:
        return f"Pattern scan failed: insufficient data for {symbol}"

    current_price = float(df_daily["close"].values[-1])

    # Run all detectors — each isolated so one failure doesn't kill the scan
    results = []

    def _safe_detect(name: str, fn, *args, **kwargs) -> None:
        try:
            results.append(fn(*args, **kwargs))
        except Exception as e:
            log.warning(f"Pattern detector '{name}' failed: {e}")
            results.append({"pattern": name, "detected": False, "confidence": 0, "details": f"Error: {e}"})

    _safe_detect("buying_climax_short", detect_buying_climax, df_daily, fg_value)
    _safe_detect("vcp_breakout", detect_vcp_breakout, df_daily)
    _safe_detect("new_high_breakout", detect_new_high_breakout, df_daily)
    _safe_detect("bnf_contrarian", detect_bnf_contrarian, df_daily)

    # 5. MACD Triple Divergence
    try:
        from tradingagents.dataflows.macd_divergence import detect_triple_divergence
        macd_daily = detect_triple_divergence(df_daily, lookback=180)
        results.append({
            "pattern": "macd_triple_divergence_daily",
            "detected": macd_daily["signal"] != "NONE",
            "confidence": macd_daily["confidence"],
            "details": macd_daily["description"],
            "direction": "SELL" if "BEARISH" in macd_daily["signal"] else "BUY" if "BULLISH" in macd_daily["signal"] else "",
        })
        if df_4h is not None:
            macd_4h = detect_triple_divergence(df_4h, lookback=200)
            results.append({
                "pattern": "macd_triple_divergence_4h",
                "detected": macd_4h["signal"] != "NONE",
                "confidence": macd_4h["confidence"],
                "details": macd_4h["description"],
            })
    except Exception as e:
        log.warning(f"MACD divergence detector failed: {e}")

    _safe_detect("hh_hl_reversal", detect_hh_hl_reversal, df_daily)
    _safe_detect("volume_exhaustion", detect_volume_exhaustion, df_daily)
    _safe_detect("sykes_stage", detect_sykes_stage, df_daily, fg_value)
    _safe_detect("bonde_home_run", detect_bonde_home_run, df_daily)
    _safe_detect("qullamaggie_breakout", detect_qullamaggie_breakout, df_daily)

    # Format report
    detected = [r for r in results if r.get("detected")]
    not_detected = [r for r in results if not r.get("detected")]

    lines = [
        f"=== PATTERN SCAN: {symbol} @ ${current_price:,.0f} ===",
        f"Fear & Greed: {fg_value or 'N/A'}",
        f"Patterns checked: {len(results)} | Detected: {len(detected)}",
        "",
    ]

    if detected:
        lines.append("ACTIVE SIGNALS:")
        for r in sorted(detected, key=lambda x: x.get("confidence", 0), reverse=True):
            lines.append(f"  [{r['confidence']}/10] {r['pattern']}: {r['details']}")
            lines.append("")
    else:
        lines.append("No patterns currently active. Market is in a neutral/unclear state.")

    lines.append(f"Inactive: {', '.join(r['pattern'] for r in not_detected)}")

    report = "\n".join(lines)
    if len(report) > MAX_SCAN_OUTPUT:
        report = report[:MAX_SCAN_OUTPUT] + "\n... (truncated)"

    log.info(f"Pattern scan complete: {len(detected)} signals detected")

    # Cache the result
    _SCAN_CACHE["result"] = report
    _SCAN_CACHE["ts"] = _time.time()
    _SCAN_CACHE["key"] = symbol

    return report

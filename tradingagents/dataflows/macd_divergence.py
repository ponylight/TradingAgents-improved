"""
MACD Triple Divergence Detector

Implements 半木夏's triple divergence strategy:
- Price makes 3 successive peaks/troughs
- MACD histogram shrinks twice consecutively
- Aligns with Elliott Wave 5-wave completion
- Best on Daily and 4H timeframes
- Frequency: ~1-2 per year on daily

References: LabSpeculation article Feb 11, 2026
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("macd_divergence")

# Standard MACD parameters (半木夏 default)
FAST = 12
SLOW = 26
SIGNAL = 9

# Minimum histogram segment length (candles) to count as a "wave"
MIN_SEGMENT_LEN = 3

# Minimum shrinkage ratio to count (peak must be at least 10% smaller)
MIN_SHRINKAGE_PCT = 0.10


def compute_macd(close: pd.Series, fast: int = FAST, slow: int = SLOW,
                 signal: int = SIGNAL) -> pd.DataFrame:
    """Compute MACD, Signal, and Histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }, index=close.index)


def _segment_histogram(histogram: pd.Series) -> List[Dict[str, Any]]:
    """
    Split histogram into contiguous positive/negative segments.
    
    Returns list of segments, each with:
      - sign: "positive" or "negative"
      - start_idx: integer position of segment start
      - end_idx: integer position of segment end (inclusive)
      - peak_value: max absolute value in segment
      - peak_pos: integer position of peak within segment
    """
    segments = []
    if len(histogram) == 0:
        return segments

    values = histogram.values
    current_sign = "positive" if values[0] >= 0 else "negative"
    start = 0

    for i in range(1, len(values)):
        new_sign = "positive" if values[i] >= 0 else "negative"
        if new_sign != current_sign:
            # Close previous segment
            seg_values = values[start:i]
            if len(seg_values) >= MIN_SEGMENT_LEN:
                if current_sign == "positive":
                    peak_rel = int(np.argmax(seg_values))
                    peak_val = float(seg_values[peak_rel])
                else:
                    peak_rel = int(np.argmin(seg_values))
                    peak_val = float(seg_values[peak_rel])
                segments.append({
                    "sign": current_sign,
                    "start_idx": start,
                    "end_idx": i - 1,
                    "peak_value": peak_val,
                    "peak_pos": start + peak_rel,
                    "length": len(seg_values),
                })
            current_sign = new_sign
            start = i

    # Close final segment
    seg_values = values[start:]
    if len(seg_values) >= MIN_SEGMENT_LEN:
        if current_sign == "positive":
            peak_rel = int(np.argmax(seg_values))
            peak_val = float(seg_values[peak_rel])
        else:
            peak_rel = int(np.argmin(seg_values))
            peak_val = float(seg_values[peak_rel])
        segments.append({
            "sign": current_sign,
            "start_idx": start,
            "end_idx": len(values) - 1,
            "peak_value": peak_val,
            "peak_pos": start + peak_rel,
            "length": len(seg_values),
        })

    return segments



def detect_triple_divergence(
    df: pd.DataFrame,
    fast: int = FAST,
    slow: int = SLOW,
    signal: int = SIGNAL,
    lookback: int = 200,
) -> Dict[str, Any]:
    """
    Detect MACD triple divergence on OHLCV data.
    
    Args:
        df: DataFrame with at least a 'close' column (or 'Close')
        fast, slow, signal: MACD parameters
        lookback: How many candles to analyze (from the end)
    
    Returns:
        {
            "signal": "BEARISH_DIVERGENCE" | "BULLISH_DIVERGENCE" | "NONE",
            "confidence": 0-10,
            "segments_used": [...],  # The 3 histogram segments
            "price_points": [(idx, price), ...],  # The 3 peaks/troughs
            "histogram_shrinkage": [pct1, pct2],  # How much histogram shrank
            "description": str,
        }
    """
    result = {
        "signal": "NONE",
        "confidence": 0,
        "segments_used": [],
        "price_points": [],
        "histogram_shrinkage": [],
        "description": "No triple divergence detected",
    }

    # Normalize column names
    close_col = "close" if "close" in df.columns else "Close"
    if close_col not in df.columns:
        log.warning("No 'close' column found in DataFrame")
        return result

    close = df[close_col].iloc[-lookback:].reset_index(drop=True).astype(float)
    if len(close) < slow + signal + 20:
        log.warning(f"Not enough data: {len(close)} candles (need {slow + signal + 20}+)")
        return result

    # Compute MACD
    macd_df = compute_macd(close, fast, slow, signal)
    histogram = macd_df["histogram"]

    # Segment the histogram
    segments = _segment_histogram(histogram)

    # Check for bearish triple divergence (3 positive segments with shrinking peaks + rising price)
    bearish = _check_divergence(close, segments, sign="positive", divergence_type="bearish")
    if bearish:
        return bearish

    # Check for bullish triple divergence (3 negative segments with shrinking troughs + falling price)
    bullish = _check_divergence(close, segments, sign="negative", divergence_type="bullish")
    if bullish:
        return bullish

    return result


def _check_divergence(
    close: pd.Series,
    segments: List[Dict],
    sign: str,
    divergence_type: str,  # "bearish" or "bullish"
) -> Optional[Dict[str, Any]]:
    """
    Check for triple divergence among segments of the given sign.
    
    For bearish: 3 positive segments where histogram peaks decrease
                 while price peaks increase
    For bullish: 3 negative segments where histogram troughs get less negative
                 while price troughs decrease
    """
    # Filter segments of the right sign
    target_segments = [s for s in segments if s["sign"] == sign]

    if len(target_segments) < 3:
        return None

    # Check the last 3 segments of this sign (most recent pattern)
    for i in range(len(target_segments) - 2):
        s1, s2, s3 = target_segments[i], target_segments[i + 1], target_segments[i + 2]

        # Ensure segments have alternating segments between them (opposite color)
        # This is guaranteed by our segmentation since we skip segments of the other sign

        h1 = abs(s1["peak_value"])
        h2 = abs(s2["peak_value"])
        h3 = abs(s3["peak_value"])

        # Histogram must shrink twice: h1 > h2 > h3
        if not (h1 > h2 > h3):
            continue

        shrinkage1 = (h1 - h2) / h1 if h1 != 0 else 0
        shrinkage2 = (h2 - h3) / h2 if h2 != 0 else 0

        if shrinkage1 < MIN_SHRINKAGE_PCT or shrinkage2 < MIN_SHRINKAGE_PCT:
            continue

        # Find price peaks/troughs for each segment
        p1_idx, p1_price = _get_price_extreme(close, s1, sign)
        p2_idx, p2_price = _get_price_extreme(close, s2, sign)
        p3_idx, p3_price = _get_price_extreme(close, s3, sign)

        if divergence_type == "bearish":
            # Price should be making higher highs while histogram shrinks
            if not (p1_price < p2_price < p3_price):
                # Also accept roughly equal (within 1%) — flat top divergence
                if not (p2_price >= p1_price * 0.99 and p3_price >= p2_price * 0.99):
                    continue
        else:
            # Price should be making lower lows while histogram shrinks
            if not (p1_price > p2_price > p3_price):
                if not (p2_price <= p1_price * 1.01 and p3_price <= p2_price * 1.01):
                    continue

        # Calculate confidence (0-10)
        confidence = _calc_confidence(shrinkage1, shrinkage2, s1, s2, s3, close)

        # Check recency — pattern should be recent (s3 should end near current candle)
        recency = len(close) - s3["end_idx"]
        if recency > 20:
            confidence = max(0, confidence - 2)  # Penalize old patterns

        signal_name = "BEARISH_DIVERGENCE" if divergence_type == "bearish" else "BULLISH_DIVERGENCE"
        
        desc_dir = "higher highs" if divergence_type == "bearish" else "lower lows"
        desc_hist = "shrinking green bars" if divergence_type == "bearish" else "shrinking red bars"

        # === Price Range Proximity Filter ===
        # If divergence price points span too wide a range relative to current price,
        # the pattern is too stale/broad to be actionable.
        current_price = float(close.iloc[-1])
        range_base = min(p1_price, p3_price) if min(p1_price, p3_price) > 0 else 1
        price_range_pct = abs(p3_price - p1_price) / range_base * 100
        max_distance_pct = abs(p1_price - current_price) / current_price * 100 if current_price > 0 else 0
        
        if price_range_pct > 30:
            log.info(
                f"ℹ️ MACD divergence discarded: price range {p1_price:.0f}→{p3_price:.0f} "
                f"spans {price_range_pct:.1f}% (>30%) — too broad to be actionable"
            )
            continue
        if max_distance_pct > 50:
            log.info(
                f"ℹ️ MACD divergence discarded: 1st point ${p1_price:.0f} is "
                f"{max_distance_pct:.1f}% from current ${current_price:.0f} — too distant"
            )
            continue
        if price_range_pct > 20:
            confidence = max(0, confidence - 2)
            log.info(f"ℹ️ MACD divergence confidence penalized: wide range ({price_range_pct:.1f}%)")

        log.info(
            f"🔺 MACD Triple {divergence_type.upper()} Divergence detected! "
            f"Confidence: {confidence}/10, "
            f"Histogram shrinkage: {shrinkage1:.1%}, {shrinkage2:.1%}, "
            f"Prices: {p1_price:.0f} → {p2_price:.0f} → {p3_price:.0f}"
        )

        return {
            "signal": signal_name,
            "confidence": confidence,
            "segments_used": [s1, s2, s3],
            "price_points": [
                (p1_idx, p1_price),
                (p2_idx, p2_price),
                (p3_idx, p3_price),
            ],
            "histogram_shrinkage": [round(shrinkage1, 3), round(shrinkage2, 3)],
            "description": (
                f"Triple {divergence_type} divergence: price making {desc_dir} "
                f"({p1_price:.0f} → {p2_price:.0f} → {p3_price:.0f}) "
                f"while MACD histogram shows {desc_hist} "
                f"(shrinkage: {shrinkage1:.0%}, {shrinkage2:.0%}). "
                f"This suggests momentum exhaustion and potential reversal. "
                f"Confidence: {confidence}/10."
            ),
        }

    return None


def _get_price_extreme(close: pd.Series, segment: Dict, sign: str) -> Tuple[int, float]:
    """Get the price peak (positive segment) or trough (negative segment)."""
    start = segment["start_idx"]
    end = segment["end_idx"] + 1
    seg_close = close.iloc[start:end]
    if sign == "positive":
        pos = int(np.argmax(seg_close.values))
    else:
        pos = int(np.argmin(seg_close.values))
    actual_idx = start + pos
    return actual_idx, float(close.iloc[actual_idx])


def _calc_confidence(shrinkage1: float, shrinkage2: float,
                     s1: Dict, s2: Dict, s3: Dict,
                     close: pd.Series) -> int:
    """Score confidence 0-10 based on pattern quality."""
    score = 5  # Base score for a valid triple divergence

    # Bigger shrinkage = stronger signal
    avg_shrinkage = (shrinkage1 + shrinkage2) / 2
    if avg_shrinkage > 0.40:
        score += 2
    elif avg_shrinkage > 0.25:
        score += 1

    # Consistent shrinkage (both similar magnitude)
    shrinkage_ratio = min(shrinkage1, shrinkage2) / max(shrinkage1, shrinkage2) if max(shrinkage1, shrinkage2) > 0 else 0
    if shrinkage_ratio > 0.6:
        score += 1

    # Pattern completeness — is the 3rd segment near the end of data?
    recency = len(close) - s3["end_idx"]
    if recency <= 5:
        score += 1  # Very recent

    # Longer segments = more reliable
    avg_len = (s1["length"] + s2["length"] + s3["length"]) / 3
    if avg_len >= 10:
        score += 1

    return min(10, max(0, score))


def check_macd_divergence_for_symbol(symbol: str = "BTC/USDT",
                                     timeframe: str = "1d") -> str:
    """Alias for agent tool compatibility."""
    days = 180 if timeframe == "1d" else 60
    return get_divergence_report(symbol, timeframe, days)


def get_divergence_report(symbol: str = "BTC/USDT", timeframe: str = "1d",
                          days_back: int = 180) -> str:
    """
    High-level function to check for MACD triple divergence.
    Designed to be called as a tool by trading agents.
    
    Returns a formatted string report.
    """
    from datetime import datetime, timedelta

    try:
        from tradingagents.dataflows.ccxt_crypto import get_crypto_ohlcv
    except ImportError:
        return "Error: ccxt_crypto module not available"

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    raw = get_crypto_ohlcv(symbol, start, end, timeframe)
    if not raw or "Error" in str(raw):
        return f"Error fetching OHLCV data for {symbol}: {raw}"

    # Parse the CSV-like string into DataFrame
    try:
        from io import StringIO
        lines = raw.strip().split("\n")
        header_idx = None
        for i, line in enumerate(lines):
            if "date" in line.lower() or "open" in line.lower():
                header_idx = i
                break
        
        if header_idx is None:
            df = pd.read_csv(StringIO(raw))
        else:
            csv_data = "\n".join(lines[header_idx:])
            df = pd.read_csv(StringIO(csv_data))
    except Exception as e:
        return f"Error parsing OHLCV data: {e}"

    if df.empty or len(df) < 50:
        return f"Insufficient data: {len(df)} candles (need 50+)"

    result = detect_triple_divergence(df, lookback=min(len(df), 200))

    if result["signal"] == "NONE":
        return (
            f"MACD Triple Divergence Check ({symbol} {timeframe}, last {days_back}d):\n"
            f"No triple divergence detected.\n"
            f"This is a rare signal (~1-2 per year on daily). "
            f"Absence of divergence is normal."
        )

    report = [
        f"⚠️ MACD TRIPLE DIVERGENCE DETECTED ({symbol} {timeframe})",
        f"Signal: {result['signal']}",
        f"Confidence: {result['confidence']}/10",
        f"",
        f"Price points: {' → '.join(f'${p:.0f}' for _, p in result['price_points'])}",
        f"Histogram shrinkage: {result['histogram_shrinkage'][0]:.0%}, {result['histogram_shrinkage'][1]:.0%}",
        f"",
        result["description"],
        f"",
        f"Strategy notes (半木夏):",
        f"- Entry: At daily close when 3rd divergence confirmed",
        f"- Stop: Above/below the 3rd peak/trough",
        f"- If next candle histogram doesn't continue shrinking → immediate stop",
        f"- Effective for 15-50 candles at this timeframe",
    ]
    return "\n".join(report)

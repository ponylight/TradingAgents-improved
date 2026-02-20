"""Technical indicators for the MACD Reversal strategy."""

import numpy as np
import pandas as pd


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_macd(df: pd.DataFrame, fast: int = 13, slow: int = 34, signal: int = 9) -> pd.DataFrame:
    """Compute MACD with custom Fibonacci-based parameters. Focus on histogram."""
    ema_fast = compute_ema(df["close"], fast)
    ema_slow = compute_ema(df["close"], slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line

    df = df.copy()
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range for stop-loss calculation."""
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_ma(df: pd.DataFrame, period: int = 30) -> pd.Series:
    """Simple moving average for rolling position pullback detection."""
    return df["close"].rolling(window=period).mean()


def detect_histogram_peaks_troughs(hist: pd.Series, lookback: int = 5) -> tuple[list, list]:
    """Find local peaks (>0) and troughs (<0) in histogram for divergence detection.

    Returns lists of (index, value) tuples for peaks and troughs.
    """
    peaks = []
    troughs = []
    values = hist.values

    for i in range(lookback, len(values) - 1):
        window = values[i - lookback : i + 1]
        if values[i] > 0:
            if values[i] == max(window) and values[i] > values[i - 1] and values[i] >= values[i + 1] if i + 1 < len(values) else True:
                # Local peak in positive territory
                if not peaks or i - peaks[-1][0] >= lookback:
                    peaks.append((i, values[i]))
        elif values[i] < 0:
            if values[i] == min(window) and values[i] < values[i - 1] and values[i] <= values[i + 1] if i + 1 < len(values) else True:
                # Local trough in negative territory
                if not troughs or i - troughs[-1][0] >= lookback:
                    troughs.append((i, values[i]))

    return peaks, troughs


def find_price_swing_lows(df: pd.DataFrame, lookback: int = 10) -> list[tuple[int, float]]:
    """Find local price lows for divergence comparison."""
    lows = []
    for i in range(lookback, len(df) - 1):
        window_low = df["low"].iloc[i - lookback : i + 1].min()
        if df["low"].iloc[i] == window_low:
            if not lows or i - lows[-1][0] >= lookback:
                lows.append((i, df["low"].iloc[i]))
    return lows


def find_price_swing_highs(df: pd.DataFrame, lookback: int = 10) -> list[tuple[int, float]]:
    """Find local price highs for divergence comparison."""
    highs = []
    for i in range(lookback, len(df) - 1):
        window_high = df["high"].iloc[i - lookback : i + 1].max()
        if df["high"].iloc[i] == window_high:
            if not highs or i - highs[-1][0] >= lookback:
                highs.append((i, df["high"].iloc[i]))
    return highs

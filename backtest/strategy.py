"""MACD Reversal Strategy with Rolling Positions.

Implements the 半木夏 MACD divergence strategy from STRATEGY.md:
- MACD(13, 34, 9) histogram-only divergence
- Long: price new lows + histogram bottom divergence + Key K-line
- Short: price new highs + histogram top divergence
- Filter: histogram peak height diff >30%
- Stop-loss: Key K-line low minus ATR(14)
- Exit: half at 1:1.5 R/R, trail rest
- Rolling positions on breakouts / MA30 pullbacks
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Trade:
    side: str  # "long" or "short"
    entry_price: float
    entry_idx: int
    entry_time: pd.Timestamp
    size: float  # position size in BTC
    stop_loss: float
    take_profit_1: float  # 1:1.5 R/R
    initial_risk: float  # distance from entry to SL
    is_rolling: bool = False  # rolling/add-on position

    # tracking
    exit_price: float = 0.0
    exit_idx: int = 0
    exit_time: pd.Timestamp = None
    pnl: float = 0.0  # total PnL including TP1 partial
    pnl_tp1: float = 0.0  # PnL from the TP1 half close
    original_size: float = 0.0  # original size before TP1 halving
    closed: bool = False
    half_closed: bool = False  # first half taken at TP1
    trailing_stop: float = 0.0


@dataclass
class Signal:
    idx: int
    side: str  # "long" or "short"
    entry_price: float
    stop_loss: float
    divergence_strength: float  # how many divergences (1=regular, 2=double, 3=triple)
    signal_type: str = "primary"  # "primary" or "rolling_add"


def compute_signals(df: pd.DataFrame) -> list[Signal]:
    """Scan the DataFrame for MACD divergence entry signals."""
    signals = []
    hist = df["macd_hist"].values
    closes = df["close"].values
    lows = df["low"].values
    highs = df["high"].values
    atr = df["atr"].values

    # We need enough data for indicators to stabilize
    start_idx = 50
    # Minimum cooldown between signals (30 bars = 5 days on 4h chart)
    min_cooldown = 30
    last_signal_idx = -min_cooldown

    for i in range(start_idx, len(df)):
        if i - last_signal_idx < min_cooldown:
            continue

        # --- LONG SIGNAL: Bottom Divergence ---
        # Key K-line: histogram negative and turning (dark red -> light red)
        # Require at least 3 bars of declining histogram before the turn
        if hist[i] < 0 and i >= 4:
            is_key_kline = (
                hist[i] > hist[i - 1]  # current bar less negative
                and hist[i - 1] < 0    # previous bar negative
                and hist[i - 1] < hist[i - 2]  # was declining
                and hist[i - 2] < hist[i - 3]  # sustained decline (3 bars)
            )

            # Require minimum histogram magnitude (filter out noise)
            min_hist_size = closes[i] * 0.001  # 0.1% of price
            if is_key_kline and abs(hist[i - 1]) > min_hist_size:
                long_div = _check_bottom_divergence(i, hist, lows, lookback=120)
                if long_div is not None:
                    strength, peak_diff = long_div
                    if peak_diff > 0.30:
                        sl = lows[i] - atr[i] * 1.5 if not np.isnan(atr[i]) else lows[i] * 0.97
                        signals.append(Signal(
                            idx=i,
                            side="long",
                            entry_price=closes[i],
                            stop_loss=sl,
                            divergence_strength=strength,
                        ))
                        last_signal_idx = i

        # --- SHORT SIGNAL: Top Divergence ---
        if hist[i] > 0 and i >= 4:
            is_key_kline = (
                hist[i] < hist[i - 1]  # current bar less positive
                and hist[i - 1] > 0    # previous bar positive
                and hist[i - 1] > hist[i - 2]  # was rising
                and hist[i - 2] > hist[i - 3]  # sustained rise (3 bars)
            )

            min_hist_size = closes[i] * 0.001
            if is_key_kline and abs(hist[i - 1]) > min_hist_size:
                short_div = _check_top_divergence(i, hist, highs, lookback=120)
                if short_div is not None:
                    strength, peak_diff = short_div
                    if peak_diff > 0.30:
                        sl = highs[i] + atr[i] * 1.5 if not np.isnan(atr[i]) else highs[i] * 1.03
                        signals.append(Signal(
                            idx=i,
                            side="short",
                            entry_price=closes[i],
                            stop_loss=sl,
                            divergence_strength=strength,
                        ))
                        last_signal_idx = i

    return signals


def _check_bottom_divergence(
    current_idx: int, hist: np.ndarray, lows: np.ndarray, lookback: int = 80
) -> tuple[float, float] | None:
    """Check for bullish (bottom) divergence.

    Returns (divergence_count, peak_height_diff_ratio) or None.
    Price makes lower lows but histogram troughs are higher (less negative).
    """
    # Find histogram troughs in negative territory
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

    # Include current trough zone
    if in_trough and trough_val < 0:
        troughs.append((trough_idx, trough_val))

    if len(troughs) < 2:
        return None

    # Check last two troughs for divergence
    divergence_count = 0
    for k in range(len(troughs) - 1):
        t1_idx, t1_val = troughs[k]
        t2_idx, t2_val = troughs[k + 1]

        # Price lower low but histogram higher trough (less negative)
        price_lower = lows[t2_idx] < lows[t1_idx]
        hist_higher = t2_val > t1_val  # less negative = higher

        if price_lower and hist_higher:
            divergence_count += 1

    if divergence_count == 0:
        return None

    # Calculate peak height difference ratio
    last_two = troughs[-2:]
    peak_diff = abs(abs(last_two[1][1]) - abs(last_two[0][1])) / abs(last_two[0][1])
    return (min(divergence_count, 3), peak_diff)


def _check_top_divergence(
    current_idx: int, hist: np.ndarray, highs: np.ndarray, lookback: int = 80
) -> tuple[float, float] | None:
    """Check for bearish (top) divergence.

    Price makes higher highs but histogram peaks are lower.
    """
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

        price_higher = highs[p2_idx] > highs[p1_idx]
        hist_lower = p2_val < p1_val

        if price_higher and hist_lower:
            divergence_count += 1

    if divergence_count == 0:
        return None

    last_two = peaks[-2:]
    peak_diff = abs(last_two[0][1] - last_two[1][1]) / last_two[0][1]
    return (min(divergence_count, 3), peak_diff)


def compute_rolling_signals(
    df: pd.DataFrame, active_trades: list[Trade], current_idx: int
) -> list[Signal]:
    """Detect rolling position add-on opportunities.

    Add to winners on:
    1. Consolidation breakouts (price breaks above recent high with volume)
    2. Pullbacks to MA30
    """
    signals = []
    if not active_trades:
        return signals

    close = df["close"].iloc[current_idx]
    ma30 = df["ma30"].iloc[current_idx]
    atr = df["atr"].iloc[current_idx]

    if np.isnan(ma30) or np.isnan(atr):
        return signals

    for trade in active_trades:
        if trade.closed:
            continue

        unrealized_pnl_pct = 0
        if trade.side == "long":
            unrealized_pnl_pct = (close - trade.entry_price) / trade.entry_price
        else:
            unrealized_pnl_pct = (trade.entry_price - close) / trade.entry_price

        # Only add to winners (at least 3% in profit)
        if unrealized_pnl_pct < 0.03:
            continue

        if trade.side == "long":
            # Pullback to MA30: price touches MA30 from above and bounces
            near_ma30 = abs(close - ma30) / close < 0.005
            above_ma30 = close > ma30

            # Breakout: price breaks above recent 20-bar high
            recent_high = df["high"].iloc[max(0, current_idx - 20) : current_idx].max()
            breakout = close > recent_high

            # Volume confirmation: current volume > 1.5x average
            avg_vol = df["volume"].iloc[max(0, current_idx - 20) : current_idx].mean()
            vol_confirm = df["volume"].iloc[current_idx] > avg_vol * 1.5

            if (near_ma30 and above_ma30) or (breakout and vol_confirm):
                sl = close - atr * 1.5
                signals.append(Signal(
                    idx=current_idx,
                    side="long",
                    entry_price=close,
                    stop_loss=sl,
                    divergence_strength=1,
                    signal_type="rolling_add",
                ))

        elif trade.side == "short":
            near_ma30 = abs(close - ma30) / close < 0.005
            below_ma30 = close < ma30

            recent_low = df["low"].iloc[max(0, current_idx - 20) : current_idx].min()
            breakdown = close < recent_low

            avg_vol = df["volume"].iloc[max(0, current_idx - 20) : current_idx].mean()
            vol_confirm = df["volume"].iloc[current_idx] > avg_vol * 1.5

            if (near_ma30 and below_ma30) or (breakdown and vol_confirm):
                sl = close + atr * 1.5
                signals.append(Signal(
                    idx=current_idx,
                    side="short",
                    entry_price=close,
                    stop_loss=sl,
                    divergence_strength=1,
                    signal_type="rolling_add",
                ))

    return signals

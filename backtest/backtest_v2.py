"""Backtest V2 — MACD Reversal with improved filters.

Changes from V1:
1. 50 SMA trend filter: long-only when price > SMA50, short-only when price < SMA50
2. Tighter TP1: 1:1 R/R (was 1:1.5)
3. Double divergence minimum: require divergence_strength >= 2 before entry
4. Skip any trade that goes against the 50-period SMA direction
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

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.fetch_data import fetch_ohlcv
from backtest.indicators import compute_macd, compute_atr, compute_ma

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v2")


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class Trade:
    side: str
    entry_price: float
    entry_idx: int
    entry_time: pd.Timestamp
    size: float
    stop_loss: float
    take_profit_1: float
    initial_risk: float
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


@dataclass
class Signal:
    idx: int
    side: str
    entry_price: float
    stop_loss: float
    divergence_strength: float
    signal_type: str = "primary"


# ── Strategy: signal generation with V2 filters ──────────────────────

def compute_signals_v2(df: pd.DataFrame) -> list[Signal]:
    """Scan for MACD divergence signals with V2 filters applied.

    V2 changes:
    - Require divergence_strength >= 2 (double divergence minimum)
    - 50 SMA trend filter: long only when close > SMA50, short only when close < SMA50
    """
    signals = []
    hist = df["macd_hist"].values
    closes = df["close"].values
    lows = df["low"].values
    highs = df["high"].values
    atr = df["atr"].values
    sma50 = df["sma50"].values

    start_idx = 55  # enough for SMA50 + a few extra bars
    min_cooldown = 30
    last_signal_idx = -min_cooldown

    for i in range(start_idx, len(df)):
        if i - last_signal_idx < min_cooldown:
            continue

        # Skip if SMA50 is not yet computed
        if np.isnan(sma50[i]):
            continue

        # ── LONG SIGNAL ──
        if hist[i] < 0 and i >= 4:
            # V2 TREND FILTER: only go long when price > SMA50
            if closes[i] <= sma50[i]:
                pass  # skip — against trend
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
                        # V2: require double divergence (strength >= 2)
                        if strength >= 2 and peak_diff > 0.30:
                            sl = lows[i] - atr[i] * 1.5 if not np.isnan(atr[i]) else lows[i] * 0.97
                            signals.append(Signal(
                                idx=i,
                                side="long",
                                entry_price=closes[i],
                                stop_loss=sl,
                                divergence_strength=strength,
                            ))
                            last_signal_idx = i

        # ── SHORT SIGNAL ──
        if hist[i] > 0 and i >= 4:
            # V2 TREND FILTER: only go short when price < SMA50
            if closes[i] >= sma50[i]:
                pass  # skip — against trend
            else:
                is_key_kline = (
                    hist[i] < hist[i - 1]
                    and hist[i - 1] > 0
                    and hist[i - 1] > hist[i - 2]
                    and hist[i - 2] > hist[i - 3]
                )

                min_hist_size = closes[i] * 0.001
                if is_key_kline and abs(hist[i - 1]) > min_hist_size:
                    short_div = _check_top_divergence(i, hist, highs, lookback=120)
                    if short_div is not None:
                        strength, peak_diff = short_div
                        # V2: require double divergence (strength >= 2)
                        if strength >= 2 and peak_diff > 0.30:
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
    """Check for bullish (bottom) divergence. Same logic as V1."""
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
        price_lower = lows[t2_idx] < lows[t1_idx]
        hist_higher = t2_val > t1_val

        if price_lower and hist_higher:
            divergence_count += 1

    if divergence_count == 0:
        return None

    last_two = troughs[-2:]
    peak_diff = abs(abs(last_two[1][1]) - abs(last_two[0][1])) / abs(last_two[0][1])
    return (min(divergence_count, 3), peak_diff)


def _check_top_divergence(
    current_idx: int, hist: np.ndarray, highs: np.ndarray, lookback: int = 80
) -> tuple[float, float] | None:
    """Check for bearish (top) divergence. Same logic as V1."""
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


def compute_rolling_signals_v2(
    df: pd.DataFrame, active_trades: list[Trade], current_idx: int
) -> list[Signal]:
    """Rolling add-on signals with V2 trend filter."""
    signals = []
    if not active_trades:
        return signals

    close = df["close"].iloc[current_idx]
    ma30 = df["ma30"].iloc[current_idx]
    atr = df["atr"].iloc[current_idx]
    sma50 = df["sma50"].iloc[current_idx]

    if np.isnan(ma30) or np.isnan(atr) or np.isnan(sma50):
        return signals

    for trade in active_trades:
        if trade.closed:
            continue

        unrealized_pnl_pct = 0
        if trade.side == "long":
            unrealized_pnl_pct = (close - trade.entry_price) / trade.entry_price
        else:
            unrealized_pnl_pct = (trade.entry_price - close) / trade.entry_price

        if unrealized_pnl_pct < 0.03:
            continue

        if trade.side == "long":
            # V2: trend filter — only add if price still above SMA50
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
                    idx=current_idx,
                    side="long",
                    entry_price=close,
                    stop_loss=sl,
                    divergence_strength=1,
                    signal_type="rolling_add",
                ))

        elif trade.side == "short":
            # V2: trend filter — only add if price still below SMA50
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
                    idx=current_idx,
                    side="short",
                    entry_price=close,
                    stop_loss=sl,
                    divergence_strength=1,
                    signal_type="rolling_add",
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
    rr_ratio: float = 1.0  # V2: tighter TP1 at 1:1 R/R (was 1.5)


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

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_macd(df, fast=13, slow=34, signal=9)
        df["atr"] = compute_atr(df, period=14)
        df["ma30"] = compute_ma(df, period=30)
        df["sma50"] = compute_ma(df, period=50)  # V2: 50-period SMA for trend filter
        return df

    def run(self) -> dict:
        print(f"[V2] Running backtest on {len(self.df)} candles...")
        print(f"  Date range: {self.df['timestamp'].iloc[0]} to {self.df['timestamp'].iloc[-1]}")
        print(f"  V2 changes: SMA50 trend filter | TP1 at 1:{self.config.rr_ratio} R/R | double divergence minimum")

        primary_signals = compute_signals_v2(self.df)
        signal_map: dict[int, Signal] = {s.idx: s for s in primary_signals}
        print(f"  Found {len(primary_signals)} primary divergence signals (after V2 filters)")

        for i in range(len(self.df)):
            price = self.df["close"].iloc[i]
            high = self.df["high"].iloc[i]
            low = self.df["low"].iloc[i]
            ts = self.df["timestamp"].iloc[i]

            self._manage_positions(i, high, low, price, ts)

            if self.active_trades:
                rolling_sigs = compute_rolling_signals_v2(self.df, self.active_trades, i)
                for sig in rolling_sigs:
                    self._execute_rolling_add(sig, i, ts)

            if i in signal_map:
                sig = signal_map[i]
                self._execute_entry(sig, ts)

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

    def _execute_entry(self, signal: Signal, ts: pd.Timestamp):
        same_dir = [t for t in self.active_trades if t.side == signal.side and not t.is_rolling]
        if same_dir:
            return

        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            return

        risk_amount = self.capital * self.config.risk_per_trade_pct
        size_btc = risk_amount / risk
        max_size_btc = (self.capital * self.config.max_position_pct) / signal.entry_price
        size_btc = min(size_btc, max_size_btc)

        if size_btc * signal.entry_price < 10:
            return

        commission = size_btc * signal.entry_price * self.config.commission_rate
        self.capital -= commission

        if signal.side == "long":
            tp1 = signal.entry_price + risk * self.config.rr_ratio
        else:
            tp1 = signal.entry_price - risk * self.config.rr_ratio

        trade = Trade(
            side=signal.side,
            entry_price=signal.entry_price,
            entry_idx=signal.idx,
            entry_time=ts,
            size=size_btc,
            stop_loss=signal.stop_loss,
            take_profit_1=tp1,
            initial_risk=risk,
            original_size=size_btc,
            trailing_stop=signal.stop_loss,
        )
        self.active_trades.append(trade)
        self.trades.append(trade)

    def _execute_rolling_add(self, signal: Signal, idx: int, ts: pd.Timestamp):
        base_trades = [t for t in self.active_trades if t.side == signal.side and not t.is_rolling and not t.closed]
        if not base_trades:
            return

        base = base_trades[0]
        base_key = base.entry_idx
        add_count = self.rolling_add_counts.get(base_key, 0)
        if add_count >= self.config.max_rolling_adds:
            return

        add_size = base.size * (self.config.pyramid_decay ** (add_count + 1))
        cost = add_size * signal.entry_price
        if cost > self.capital * 0.2:
            return

        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            return

        commission = cost * self.config.commission_rate
        self.capital -= commission

        if signal.side == "long":
            tp1 = signal.entry_price + risk * self.config.rr_ratio
        else:
            tp1 = signal.entry_price - risk * self.config.rr_ratio

        trade = Trade(
            side=signal.side,
            entry_price=signal.entry_price,
            entry_idx=idx,
            entry_time=ts,
            size=add_size,
            stop_loss=signal.stop_loss,
            take_profit_1=tp1,
            initial_risk=risk,
            original_size=add_size,
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
            return {"error": "No trades executed"}

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

        # Long/short breakdown
        long_trades = [t for t in self.closed_trades if t.side == "long"]
        short_trades = [t for t in self.closed_trades if t.side == "short"]
        long_pnl = sum(t.pnl for t in long_trades)
        short_pnl = sum(t.pnl for t in short_trades)

        return {
            "version": "v2",
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
            "data_start": str(self.df["timestamp"].iloc[0]),
            "data_end": str(self.df["timestamp"].iloc[-1]),
            "total_candles": len(self.df),
        }

    def generate_report(self, metrics: dict):
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Metrics JSON
        with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Text report
        report = self._format_report(metrics)
        with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
            f.write(report)
        print(report)

        # Equity curve PNG
        self._plot_equity_curve()

        # Trade log CSV
        self._save_trade_log()

        # Comparison with V1 if available
        self._generate_comparison(metrics)

    def _format_report(self, metrics: dict) -> str:
        sep = "=" * 60
        lines = [
            sep,
            "  BTC/USDT MACD REVERSAL BACKTEST — V2",
            "  Improvements: SMA50 trend filter, 1:1 TP, double divergence",
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
            "  V2 FILTER DETAILS",
            sep,
            "",
            "  1. 50 SMA trend filter: long only above SMA50, short only below",
            "  2. TP1 at 1:1 R/R (V1 was 1:1.5)",
            "  3. Require double divergence (strength >= 2) for entry",
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

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[3, 1, 1])

        # Equity curve
        ax1.plot(timestamps, equity, color="#2196F3", linewidth=1.2, label="Equity (V2)")
        ax1.axhline(y=self.config.initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")

        # Plot SMA50 trade direction zones as background
        sma50 = self.df["sma50"].values
        prices = self.df["close"].values
        ts_all = pd.to_datetime(self.df["timestamp"])

        ax1.set_title("BTC/USDT MACD Reversal V2 — Equity Curve (SMA50 filter + double divergence + 1:1 TP)", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Equity ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        for trade in self.closed_trades[:50]:
            marker_color = "#4CAF50" if trade.pnl > 0 else "#F44336"
            if trade.entry_time in timestamps.values:
                entry_equity_idx = timestamps[timestamps == trade.entry_time].index
                if len(entry_equity_idx) > 0:
                    ax1.plot(trade.entry_time, equity[entry_equity_idx[0]],
                            marker="^" if trade.side == "long" else "v",
                            color=marker_color, markersize=5, alpha=0.7)

        # Drawdown
        ax2.fill_between(timestamps, -drawdown_pct, 0, color="#F44336", alpha=0.3)
        ax2.plot(timestamps, -drawdown_pct, color="#F44336", linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Drawdown", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # BTC price with SMA50 overlay
        ax3.plot(ts_all, prices, color="#FF9800", linewidth=0.8, label="BTC Price")
        valid_sma = ~np.isnan(sma50)
        ax3.plot(ts_all[valid_sma], sma50[valid_sma], color="#9C27B0", linewidth=0.8, alpha=0.7, label="SMA50")
        ax3.set_ylabel("BTC Price ($)")
        ax3.set_title("BTC/USDT Price with SMA50", fontsize=11)
        ax3.legend(loc="upper left", fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        fig.subplots_adjust(hspace=0.35)
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
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "size_btc": round(t.size, 6),
                "stop_loss": round(t.stop_loss, 2),
                "pnl": round(t.pnl, 2),
                "half_closed_at_tp1": t.half_closed,
                "duration_hours": round((t.exit_time - t.entry_time).total_seconds() / 3600, 1) if t.exit_time and t.entry_time else 0,
            })
        trade_df = pd.DataFrame(records)
        trade_file = os.path.join(RESULTS_DIR, "trade_log.csv")
        trade_df.to_csv(trade_file, index=False)
        print(f"  Trade log saved to {trade_file} ({len(records)} trades)")

    def _generate_comparison(self, v2_metrics: dict):
        """Generate V1 vs V2 side-by-side comparison."""
        v1_path = os.path.join(os.path.dirname(__file__), "results", "metrics.json")
        if not os.path.exists(v1_path):
            print("  V1 results not found — skipping comparison")
            return

        with open(v1_path) as f:
            v1 = json.load(f)

        v2 = v2_metrics

        def delta(v2_val, v1_val, higher_is_better=True):
            diff = v2_val - v1_val
            if diff == 0:
                return "  (same)"
            arrow = "+" if diff > 0 else ""
            quality = ""
            if higher_is_better:
                quality = " BETTER" if diff > 0 else " worse"
            else:
                quality = " BETTER" if diff < 0 else " worse"
            return f"  ({arrow}{diff:.2f}{quality})"

        sep = "=" * 72
        lines = [
            sep,
            "  BACKTEST COMPARISON: V1 vs V2",
            sep,
            "",
            f"  {'Metric':<30} {'V1':>15} {'V2':>15}   {'Change':>15}",
            f"  {'-'*30} {'-'*15} {'-'*15}   {'-'*15}",
            "",
            f"  {'Total Return %':<30} {v1['total_return_pct']:>14.2f}% {v2['total_return_pct']:>14.2f}%{delta(v2['total_return_pct'], v1['total_return_pct'])}",
            f"  {'Final Equity':<30} ${v1['final_equity']:>13,.2f} ${v2['final_equity']:>13,.2f}{delta(v2['final_equity'], v1['final_equity'])}",
            f"  {'Max Drawdown %':<30} {v1['max_drawdown_pct']:>14.2f}% {v2['max_drawdown_pct']:>14.2f}%{delta(v2['max_drawdown_pct'], v1['max_drawdown_pct'], higher_is_better=False)}",
            f"  {'Sharpe Ratio':<30} {v1['sharpe_ratio']:>15.2f} {v2['sharpe_ratio']:>15.2f}{delta(v2['sharpe_ratio'], v1['sharpe_ratio'])}",
            f"  {'Win Rate %':<30} {v1['win_rate_pct']:>14.2f}% {v2['win_rate_pct']:>14.2f}%{delta(v2['win_rate_pct'], v1['win_rate_pct'])}",
            f"  {'Profit Factor':<30} {v1['profit_factor']:>15.2f} {v2['profit_factor']:>15.2f}{delta(v2['profit_factor'], v1['profit_factor'])}",
            "",
            f"  {'Total Trades':<30} {v1['total_trades']:>15} {v2['total_trades']:>15}",
            f"  {'  Primary':<30} {v1['primary_trades']:>15} {v2['primary_trades']:>15}",
            f"  {'  Rolling Adds':<30} {v1['rolling_adds']:>15} {v2['rolling_adds']:>15}",
            f"  {'Avg Win':<30} ${v1['avg_win']:>13,.2f} ${v2['avg_win']:>13,.2f}",
            f"  {'Avg Loss':<30} ${v1['avg_loss']:>13,.2f} ${v2['avg_loss']:>13,.2f}",
            f"  {'Avg Duration (days)':<30} {v1['avg_trade_duration_days']:>15.1f} {v2['avg_trade_duration_days']:>15.1f}",
            "",
            sep,
            "  V2 CHANGES APPLIED",
            sep,
            "",
            "  1. SMA50 trend filter: long only above SMA50, short only below",
            "  2. TP1 tightened to 1:1 R/R (V1 used 1:1.5)",
            "  3. Double divergence minimum (strength >= 2 required)",
            "",
            sep,
        ]

        comparison = "\n".join(lines)
        comp_path = os.path.join(RESULTS_DIR, "comparison_v1_vs_v2.txt")
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
        rr_ratio=1.0,  # V2: tighter TP
    )

    engine = BacktestEngine(df, config)
    metrics = engine.run()
    engine.generate_report(metrics)

    print("\nBacktest V2 complete!")
    return metrics


if __name__ == "__main__":
    main()

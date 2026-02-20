"""Backtest engine for MACD Reversal + Rolling Positions strategy.

Handles position management, risk management, and performance metrics.
Starting capital: $10,000, commission: 0.1%.
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest.indicators import compute_macd, compute_atr, compute_ma
from backtest.strategy import compute_signals, compute_rolling_signals, Trade, Signal

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    commission_rate: float = 0.001  # 0.1%
    max_position_pct: float = 0.40  # max 40% of account per position
    risk_per_trade_pct: float = 0.02  # risk 2% of capital per trade
    pyramid_decay: float = 0.6  # each rolling add is 60% of previous size
    max_rolling_adds: int = 3
    trailing_stop_atr_mult: float = 2.5
    rr_ratio: float = 1.5  # take half profit at 1:1.5 risk/reward


class BacktestEngine:
    def __init__(self, df: pd.DataFrame, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.df = self._prepare_data(df)

        # State
        self.capital = self.config.initial_capital
        self.equity_curve = []
        self.trades: list[Trade] = []
        self.active_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []

        # Rolling position tracking
        self.rolling_add_counts: dict[int, int] = {}  # base_trade_idx -> add count

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators to the DataFrame."""
        df = compute_macd(df, fast=13, slow=34, signal=9)
        df["atr"] = compute_atr(df, period=14)
        df["ma30"] = compute_ma(df, period=30)
        return df

    def run(self) -> dict:
        """Execute the backtest. Returns performance metrics."""
        print(f"Running backtest on {len(self.df)} candles...")
        print(f"Date range: {self.df['timestamp'].iloc[0]} to {self.df['timestamp'].iloc[-1]}")

        # Pre-compute all primary signals
        primary_signals = compute_signals(self.df)
        signal_map: dict[int, Signal] = {s.idx: s for s in primary_signals}
        print(f"Found {len(primary_signals)} primary divergence signals")

        for i in range(len(self.df)):
            price = self.df["close"].iloc[i]
            high = self.df["high"].iloc[i]
            low = self.df["low"].iloc[i]
            ts = self.df["timestamp"].iloc[i]

            # 1. Check stop-losses and take-profits on active trades
            self._manage_positions(i, high, low, price, ts)

            # 2. Check for rolling add signals on active trades
            if self.active_trades:
                rolling_sigs = compute_rolling_signals(self.df, self.active_trades, i)
                for sig in rolling_sigs:
                    self._execute_rolling_add(sig, i, ts)

            # 3. Check for new primary entry signal
            if i in signal_map:
                sig = signal_map[i]
                self._execute_entry(sig, ts)

            # 4. Record equity
            unrealized = self._calc_unrealized(price)
            self.equity_curve.append({
                "timestamp": ts,
                "equity": self.capital + unrealized,
                "capital": self.capital,
                "unrealized": unrealized,
                "num_positions": len(self.active_trades),
            })

        # Close any remaining positions at last price
        last_price = self.df["close"].iloc[-1]
        last_ts = self.df["timestamp"].iloc[-1]
        for trade in list(self.active_trades):
            self._close_trade(trade, last_price, len(self.df) - 1, last_ts, reason="end_of_backtest")

        metrics = self._compute_metrics()
        return metrics

    def _execute_entry(self, signal: Signal, ts: pd.Timestamp):
        """Open a new position based on a primary signal."""
        # Don't open if we already have a position in the same direction
        same_dir = [t for t in self.active_trades if t.side == signal.side and not t.is_rolling]
        if same_dir:
            return

        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            return

        # Position sizing: risk 2% of capital
        risk_amount = self.capital * self.config.risk_per_trade_pct
        size_btc = risk_amount / risk

        # Cap at max position size
        max_size_btc = (self.capital * self.config.max_position_pct) / signal.entry_price
        size_btc = min(size_btc, max_size_btc)

        if size_btc * signal.entry_price < 10:  # minimum $10 position
            return

        # Commission
        commission = size_btc * signal.entry_price * self.config.commission_rate
        self.capital -= commission

        # Calculate TP1 at 1:1.5 R/R
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
        """Add to a winning position (pyramid sizing)."""
        # Find the base trade we're adding to
        base_trades = [t for t in self.active_trades if t.side == signal.side and not t.is_rolling and not t.closed]
        if not base_trades:
            return

        base = base_trades[0]
        base_key = base.entry_idx

        add_count = self.rolling_add_counts.get(base_key, 0)
        if add_count >= self.config.max_rolling_adds:
            return

        # Pyramid sizing: each add is decay^n of base size
        add_size = base.size * (self.config.pyramid_decay ** (add_count + 1))

        # Check we have enough capital
        cost = add_size * signal.entry_price
        if cost > self.capital * 0.2:  # don't use more than 20% for rolling adds
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
        """Check SL, TP, and update trailing stops for all active positions."""
        for trade in list(self.active_trades):
            if trade.closed:
                continue

            if trade.side == "long":
                # Check stop-loss hit
                if low <= trade.trailing_stop:
                    exit_price = trade.trailing_stop  # assume fill at SL
                    self._close_trade(trade, exit_price, idx, ts, reason="stop_loss")
                    continue

                # Check TP1 (half position)
                if not trade.half_closed and high >= trade.take_profit_1:
                    half_size = trade.size / 2
                    pnl = (trade.take_profit_1 - trade.entry_price) * half_size
                    commission = half_size * trade.take_profit_1 * self.config.commission_rate
                    tp1_pnl = pnl - commission
                    self.capital += tp1_pnl
                    trade.pnl_tp1 = tp1_pnl
                    trade.size -= half_size
                    trade.half_closed = True

                    # Move stop to breakeven + small buffer
                    trade.trailing_stop = max(trade.trailing_stop, trade.entry_price + trade.initial_risk * 0.1)

                # Update trailing stop (trail by ATR * mult)
                atr = self.df["atr"].iloc[idx]
                if not np.isnan(atr) and trade.half_closed:
                    new_trail = close - atr * self.config.trailing_stop_atr_mult
                    trade.trailing_stop = max(trade.trailing_stop, new_trail)

            elif trade.side == "short":
                # Check stop-loss hit
                if high >= trade.trailing_stop:
                    exit_price = trade.trailing_stop
                    self._close_trade(trade, exit_price, idx, ts, reason="stop_loss")
                    continue

                # Check TP1
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
        """Close a trade and record PnL."""
        if trade.side == "long":
            pnl = (exit_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - exit_price) * trade.size

        commission = trade.size * exit_price * self.config.commission_rate
        pnl -= commission

        trade.exit_price = exit_price
        trade.exit_idx = idx
        trade.exit_time = ts
        # Total PnL includes the TP1 partial close profit
        trade.pnl = pnl + trade.pnl_tp1
        trade.closed = True

        # Add realized PnL for remaining position to capital
        self.capital += pnl

        if trade in self.active_trades:
            self.active_trades.remove(trade)
        self.closed_trades.append(trade)

    def _calc_unrealized(self, current_price: float) -> float:
        """Calculate unrealized PnL across all active trades."""
        total = 0.0
        for trade in self.active_trades:
            if trade.side == "long":
                total += (current_price - trade.entry_price) * trade.size
            else:
                total += (trade.entry_price - current_price) * trade.size
        return total

    def _compute_metrics(self) -> dict:
        """Compute comprehensive backtest performance metrics."""
        if not self.closed_trades:
            return {"error": "No trades executed"}

        equity_df = pd.DataFrame(self.equity_curve)
        equity = equity_df["equity"].values

        # Total return
        total_return = (equity[-1] - self.config.initial_capital) / self.config.initial_capital * 100

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = np.max(drawdown)

        # Sharpe ratio (annualized, assuming 4h bars = 6 bars/day = 2190 bars/year)
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(2190)
        else:
            sharpe = 0.0

        # Trade stats
        pnls = [t.pnl for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average trade duration (in hours, since 4h candles)
        durations = []
        for t in self.closed_trades:
            if t.entry_time is not None and t.exit_time is not None:
                dur = (t.exit_time - t.entry_time).total_seconds() / 3600
                durations.append(dur)
        avg_duration_hours = np.mean(durations) if durations else 0
        avg_duration_days = avg_duration_hours / 24

        # Rolling position stats
        primary_trades = [t for t in self.closed_trades if not t.is_rolling]
        rolling_trades = [t for t in self.closed_trades if t.is_rolling]

        metrics = {
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
            "data_start": str(self.df["timestamp"].iloc[0]),
            "data_end": str(self.df["timestamp"].iloc[-1]),
            "total_candles": len(self.df),
        }
        return metrics

    def generate_report(self, metrics: dict):
        """Save results report and equity curve chart."""
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Save metrics JSON
        metrics_file = os.path.join(RESULTS_DIR, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_file}")

        # Generate text report
        report = self._format_report(metrics)
        report_file = os.path.join(RESULTS_DIR, "report.txt")
        with open(report_file, "w") as f:
            f.write(report)
        print(f"Report saved to {report_file}")
        print(report)

        # Generate equity curve
        self._plot_equity_curve()

        # Save trade log
        self._save_trade_log()

    def _format_report(self, metrics: dict) -> str:
        """Format metrics into a readable report."""
        sep = "=" * 60
        lines = [
            sep,
            "  BTC/USDT MACD REVERSAL BACKTEST REPORT",
            "  Strategy: 半木夏 MACD Divergence + Rolling Positions",
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
            f"  Win Rate:        {metrics['win_rate_pct']:.1f}%",
            f"  Profit Factor:   {metrics['profit_factor']:.2f}",
            "",
            f"  Avg Win:         ${metrics['avg_win']:,.2f}",
            f"  Avg Loss:        ${metrics['avg_loss']:,.2f}",
            f"  Largest Win:     ${metrics['largest_win']:,.2f}",
            f"  Largest Loss:    ${metrics['largest_loss']:,.2f}",
            "",
            f"  Avg Duration:    {metrics['avg_trade_duration_days']:.1f} days ({metrics['avg_trade_duration_hours']:.0f} hours)",
            "",
            sep,
            f"  Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            sep,
        ]
        return "\n".join(lines)

    def _plot_equity_curve(self):
        """Generate equity curve chart with drawdown overlay."""
        equity_df = pd.DataFrame(self.equity_curve)
        equity = equity_df["equity"].values
        timestamps = pd.to_datetime(equity_df["timestamp"])

        peak = np.maximum.accumulate(equity)
        drawdown_pct = (peak - equity) / peak * 100

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[3, 1, 1])

        # Equity curve
        ax1.plot(timestamps, equity, color="#2196F3", linewidth=1.2, label="Equity")
        ax1.axhline(y=self.config.initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
        ax1.set_title("BTC/USDT MACD Reversal Strategy — Equity Curve", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Equity ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # Mark trades on equity curve
        for trade in self.closed_trades[:50]:  # limit to avoid clutter
            if trade.pnl > 0:
                marker_color = "#4CAF50"
            else:
                marker_color = "#F44336"
            if trade.entry_time in timestamps.values:
                entry_equity_idx = timestamps[timestamps == trade.entry_time].index
                if len(entry_equity_idx) > 0:
                    ax1.plot(trade.entry_time, equity[entry_equity_idx[0]],
                            marker="^" if trade.side == "long" else "v",
                            color=marker_color, markersize=4, alpha=0.6)

        # Drawdown
        ax2.fill_between(timestamps, -drawdown_pct, 0, color="#F44336", alpha=0.3)
        ax2.plot(timestamps, -drawdown_pct, color="#F44336", linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Drawdown", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # BTC price
        ax3.plot(self.df["timestamp"], self.df["close"], color="#FF9800", linewidth=0.8)
        ax3.set_ylabel("BTC Price ($)")
        ax3.set_title("BTC/USDT Price", fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        fig.subplots_adjust(hspace=0.35)
        chart_path = os.path.join(RESULTS_DIR, "equity_curve.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Equity curve saved to {chart_path}")

    def _save_trade_log(self):
        """Save detailed trade log as CSV."""
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
        print(f"Trade log saved to {trade_file} ({len(records)} trades)")

"""Backtest V17b — SMART AGGRESSIVE ROLLING STRATEGY.

Lesson from V17: 10-15x leverage on weekly = liquidation from normal volatility.
The articles' insight isn't HIGH LEVERAGE, it's BIG POSITION SIZE with SURVIVABLE LEVERAGE.

KEY DESIGN:
  1. POSITION SIZE: 30% of capital as margin (aggressive!)
  2. LEVERAGE: 3x base, 5x on adds — survivable through weekly swings
     - 3x = 33% to liquidation. Weekly ATR ~8% = 4x ATR buffer
  3. ADD MARGIN: From capital + floating profits (50% of available)
  4. NO FAKEOUT AUTO-CLOSE — let ATR stops handle it
  5. SIMPLIFIED LIQ CHECK — individual only
  6. LOOSENED ADD SIGNALS — pullback within 5% of SMA20, wider fib, 3-week breakout
  7. MAX ADDS: 5 with flatter pyramid (80/60/50/40/30%)
  8. ADD STOPS: trailing at 2x ATR (wider breathing room)
  9. BASE TRAILING: 2x ATR (let winners run longer with lower leverage)
  10. SCALE LEVERAGE WITH PROFIT: When floating PnL > 50% of margin,
      allow higher leverage on next add (up to 10x on add 3+)

ALSO: Larger starting capital ($10K) and aggressive re-entry (1 week cooldown).
"""

import sys
import os
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backtest.fetch_data import fetch_ohlcv
from backtest.indicators import compute_macd, compute_atr, compute_ma

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v17b")

CAPITAL = 10_000.0
COMMISSION_RATE = 0.001
FUNDING_RATE_PER_8H = 0.0001

SWING_LOOKBACK = 10
MVRV_LOOKBACK = 52
MVRV_UNDERVALUED = 0.8
FEAR_PROXIMITY_PCT = 0.30
SOPR_DECLINE_WEEKS = 3
FIFTY_TWO_WEEK_PROXIMITY = 0.20

# SMART AGGRESSIVE: low leverage, big position
TRAILING_ATR_MULT = 2.0          # 2x ATR — lock profits on base
ADD_ATR_MULT = 2.0               # 2x ATR — adds trail too
BASE_LEVERAGE = 3                # 3x base — can survive 33% drop
ADD_LEVERAGE_BASE = 5            # 5x on early adds — can survive 20% drop
ADD_LEVERAGE_PROFIT = 10         # 10x on later adds when deep in profit
ENTRY_MARGIN_PCT = 0.30          # 30% of capital

MVRV_SCALEOUT = 2.5
MVRV_AGGRESSIVE_EXIT = 3.5

PYRAMID_SIZES = [0.80, 0.60, 0.50, 0.40, 0.30]
MAX_ADDS = 5

ADD_CAPITAL_SHARE = 0.50
ADD_MIN_SIZE_BTC = 0.001
PROFIT_THRESHOLD_FOR_HIGH_LEV = 0.50  # 50% profit on base → allow 10x adds

CONSOLIDATION_LOOKBACK = 3
FIB_RETRACE_LOW = 0.382
FIB_RETRACE_HIGH = 0.618
SMA_PROXIMITY_PCT = 0.05        # within 5% of SMA20

MIN_COOLDOWN = 1


def resample_to_weekly(df_4h):
    df = df_4h.set_index("timestamp")
    weekly = df.resample("W").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open"]).reset_index()
    weekly = compute_macd(weekly, fast=13, slow=34, signal=9)
    weekly["atr"] = compute_atr(weekly, period=14)
    weekly["sma20"] = compute_ma(weekly, period=20)
    weekly["sma50"] = compute_ma(weekly, period=50)
    return weekly


def check_bottom_divergence(current_idx, hist, lows, lookback=80):
    troughs = []
    start = max(0, current_idx - lookback)
    in_trough = False
    trough_val = 0.0
    trough_idx = 0
    for j in range(start, current_idx):
        if hist[j] < 0:
            if not in_trough or hist[j] < trough_val:
                trough_val = hist[j]; trough_idx = j
            in_trough = True
        else:
            if in_trough and trough_val < 0:
                troughs.append((trough_idx, trough_val))
            in_trough = False; trough_val = 0.0
    if in_trough and trough_val < 0:
        troughs.append((trough_idx, trough_val))
    if len(troughs) < 2:
        return None
    for k in range(len(troughs) - 1):
        if lows[troughs[k+1][0]] < lows[troughs[k][0]] and troughs[k+1][1] > troughs[k][1]:
            return (1, 0)
    return None


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


class SmartAggressiveBacktest:

    def __init__(self, weekly_df, capital=CAPITAL):
        self.initial_capital = capital
        self.capital = capital
        self.w = weekly_df
        self.active_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.equity_curve = []
        self.trade_log = []
        self.base_entry_idx = -1
        self.base_margin = 0.0
        self.add_count = 0

        self.stats = {
            "signals_generated": 0, "liquidations": 0, "liquidation_loss": 0.0,
            "total_funding_paid": 0.0, "pyramid_adds": 0, "adds_attempted": 0,
            "adds_skipped_no_profit": 0, "adds_skipped_no_signal": 0,
            "adds_skipped_max_reached": 0, "adds_skipped_too_small": 0,
            "adds_skipped_insufficient_margin": 0,
            "adds_via_pullback_sma20": 0, "adds_via_fib_retracement": 0,
            "adds_via_consolidation_breakout": 0,
            "total_add_margin": 0.0, "scale_outs": 0, "aggressive_exits": 0,
            "high_leverage_adds": 0,
        }

    def _mvrv_proxy(self, idx):
        start = max(0, idx - MVRV_LOOKBACK)
        window = self.w["close"].values[start:idx + 1]
        if len(window) < 10: return 1.0
        avg = np.mean(window)
        return self.w["close"].values[idx] / avg if avg > 0 else 1.0

    def _evaluate_entry(self, idx):
        conditions = []; flags = []
        if self._mvrv_proxy(idx) < MVRV_UNDERVALUED:
            conditions.append(1); flags.append("MVRV<0.8")
        # Fear + proximity
        start = max(0, idx - MVRV_LOOKBACK)
        window_low = self.w["low"].values[start:idx+1]
        closes = self.w["close"].values
        sma50 = self.w["sma50"].values
        if len(window_low) >= 10:
            low_365 = np.min(window_low)
            price = closes[idx]
            if (price <= low_365 * (1 + FEAR_PROXIMITY_PCT) and
                idx >= SOPR_DECLINE_WEEKS and not np.isnan(sma50[idx]) and
                price < sma50[idx]):
                decline = True
                for k in range(1, SOPR_DECLINE_WEEKS + 1):
                    if idx - k < 0 or closes[idx-k+1] >= closes[idx-k]:
                        decline = False; break
                if decline:
                    conditions.append(2); flags.append("fear_30%+decline")
        # MACD divergence
        hist = self.w["macd_hist"].values
        lows = self.w["low"].values
        if idx >= 3 and not (hist[idx] >= 0 and hist[idx-1] >= 0):
            if check_bottom_divergence(idx, hist, lows):
                conditions.append(3); flags.append("MACD_div")
        # SMA20 crossover
        sma20 = self.w["sma20"].values
        if idx >= 1 and not np.isnan(sma20[idx]) and not np.isnan(sma20[idx-1]):
            if closes[idx-1] < sma20[idx-1] and closes[idx] > sma20[idx]:
                conditions.append(4); flags.append("SMA20_cross")
        # 52-week low proximity
        if len(window_low) >= 10:
            low_52w = np.min(window_low)
            if closes[idx] <= low_52w * (1 + FIFTY_TWO_WEEK_PROXIMITY):
                conditions.append(5); flags.append("52w_low_20%")
        return len(conditions), "|".join(flags)

    def _check_add_signal(self, idx, price):
        sma20 = self.w["sma20"].values
        lows = self.w["low"].values
        highs = self.w["high"].values
        volumes = self.w["volume"].values

        # Signal A: Near SMA20 (within 5%)
        if not np.isnan(sma20[idx]):
            dist = abs(price - sma20[idx]) / sma20[idx]
            if dist < SMA_PROXIMITY_PCT:
                return True, "pullback_sma20"
            if lows[idx] <= sma20[idx] <= price:
                return True, "pullback_sma20"

        # Signal B: Fib retracement
        if self.base_entry_idx >= 0:
            sw_start = max(0, self.base_entry_idx - 20)
            sw_low = np.min(lows[sw_start:self.base_entry_idx + 1])
            sw_high = np.max(highs[self.base_entry_idx:idx + 1])
            if sw_high > sw_low:
                move = sw_high - sw_low
                f382 = sw_high - move * FIB_RETRACE_LOW
                f618 = sw_high - move * FIB_RETRACE_HIGH
                if f618 <= price <= f382:
                    return True, "fib_retracement"

        # Signal C: Breakout > 3-week high
        if idx >= CONSOLIDATION_LOOKBACK:
            prev_high = np.max(highs[idx - CONSOLIDATION_LOOKBACK:idx])
            if price > prev_high:
                avg_vol = np.mean(volumes[idx - CONSOLIDATION_LOOKBACK:idx])
                if avg_vol > 0 and volumes[idx] > avg_vol * 0.8:
                    return True, "consolidation_breakout"

        return False, ""

    def run(self):
        print(f"\n[V17b SMART AGGRESSIVE] Running on {len(self.w)} weekly candles...")
        closes = self.w["close"].values
        highs = self.w["high"].values
        lows = self.w["low"].values
        atr = self.w["atr"].values

        start_idx = 52
        for i in range(start_idx, len(self.w)):
            if not np.isnan(self.w["sma50"].values[i]):
                start_idx = i; break

        for i in range(len(self.w)):
            price = closes[i]; high = highs[i]; low = lows[i]
            ts = self.w["timestamp"].iloc[i]
            w_atr = atr[i] if not np.isnan(atr[i]) else 0

            self._deduct_funding(price)
            self._manage_positions(i, high, low, price, ts, w_atr)

            if self.active_trades and i >= start_idx:
                self._check_add(i, price, ts, w_atr)

            if i >= start_idx and not self.active_trades:
                cond_count, cond_flags = self._evaluate_entry(i)
                if cond_count >= 1:
                    last_exit = max((t.exit_idx for t in self.closed_trades), default=-999)
                    if i - last_exit >= MIN_COOLDOWN:
                        self._execute_entry(i, price, ts, w_atr, cond_count, cond_flags)

            unrealized = sum((price - t.entry_price) * t.size_btc
                             for t in self.active_trades if not t.closed)
            self.equity_curve.append({
                "timestamp": ts, "equity": self.capital + unrealized,
                "capital": self.capital, "price": price,
            })

        if self.active_trades:
            for t in list(self.active_trades):
                self._close_trade(t, closes[-1], len(self.w)-1,
                                  self.w["timestamp"].iloc[-1], "end_of_backtest")

        return self._compute_metrics()

    def _execute_entry(self, idx, price, ts, w_atr, cond_count, cond_flags):
        leverage = BASE_LEVERAGE
        self.stats["signals_generated"] += 1
        margin = self.capital * ENTRY_MARGIN_PCT
        if margin < 50: return

        notional = margin * leverage
        size_btc = notional / price
        stop_dist = w_atr * TRAILING_ATR_MULT if w_atr > 0 else price * 0.10
        stop_loss = price - stop_dist
        commission = notional * COMMISSION_RATE
        self.capital -= commission
        liq_price = price * (1 - 1 / leverage)
        min_stop = liq_price + (price - liq_price) * 0.10
        stop_loss = max(stop_loss, min_stop)

        trade = Trade(entry_price=price, entry_idx=idx, entry_time=ts,
                      size_btc=size_btc, margin=margin, leverage=leverage,
                      stop_loss=stop_loss, liquidation_price=liq_price,
                      conditions_met=cond_count, condition_flags=cond_flags,
                      highest_price=price)
        self.active_trades.append(trade)
        self.base_entry_idx = idx
        self.base_margin = margin
        self.add_count = 0

        self.trade_log.append(
            f"[{ts.strftime('%Y-%m-%d')}] ENTRY: ${price:,.0f} | "
            f"{leverage}x | margin=${margin:,.0f} | size={size_btc:.4f} BTC | "
            f"conds={cond_count} ({cond_flags}) | SL=${stop_loss:,.0f} | "
            f"liq=${liq_price:,.0f}")

    def _check_add(self, idx, price, ts, w_atr):
        if self.base_entry_idx < 0: return
        self.stats["adds_attempted"] += 1

        if self.add_count >= MAX_ADDS:
            self.stats["adds_skipped_max_reached"] += 1; return

        agg_pnl = sum((price - t.entry_price) * t.size_btc
                       for t in self.active_trades if not t.closed)
        if agg_pnl < 0:
            self.stats["adds_skipped_no_profit"] += 1; return

        signal_fired, signal_type = self._check_add_signal(idx, price)
        if not signal_fired:
            self.stats["adds_skipped_no_signal"] += 1; return

        # Available margin: capital + profits
        used_margin = sum(t.margin for t in self.active_trades if not t.closed)
        remaining = self.capital - used_margin
        available = max(0, remaining * ADD_CAPITAL_SHARE + max(0, agg_pnl) * 0.5)

        pyramid_frac = PYRAMID_SIZES[self.add_count]
        add_margin = min(available, self.base_margin * pyramid_frac)
        if add_margin < 50:
            self.stats["adds_skipped_insufficient_margin"] += 1; return

        # Determine leverage: higher if deep in profit
        base_trades = [t for t in self.active_trades if not t.is_add and not t.closed]
        if base_trades:
            base = base_trades[0]
            base_profit_pct = (price - base.entry_price) / base.entry_price
            if base_profit_pct > PROFIT_THRESHOLD_FOR_HIGH_LEV and self.add_count >= 2:
                leverage = ADD_LEVERAGE_PROFIT  # 10x when deep in profit
                self.stats["high_leverage_adds"] += 1
            else:
                leverage = ADD_LEVERAGE_BASE  # 5x normally
        else:
            leverage = ADD_LEVERAGE_BASE

        add_notional = add_margin * leverage
        add_btc = add_notional / price
        liq_price = price * (1 - 1 / leverage)

        if add_btc < ADD_MIN_SIZE_BTC:
            self.stats["adds_skipped_too_small"] += 1; return

        commission = add_notional * COMMISSION_RATE
        self.capital -= commission

        add_stop = price - w_atr * ADD_ATR_MULT if w_atr > 0 else price * 0.90
        min_stop = liq_price + (price - liq_price) * 0.10
        add_stop = max(add_stop, min_stop)

        self.stats["total_add_margin"] += add_margin
        if signal_type == "pullback_sma20": self.stats["adds_via_pullback_sma20"] += 1
        elif signal_type == "fib_retracement": self.stats["adds_via_fib_retracement"] += 1
        elif signal_type == "consolidation_breakout": self.stats["adds_via_consolidation_breakout"] += 1

        self.add_count += 1
        base = base_trades[0] if base_trades else self.active_trades[0]

        add_trade = Trade(entry_price=price, entry_idx=idx, entry_time=ts,
                          size_btc=add_btc, margin=add_margin, leverage=leverage,
                          stop_loss=add_stop, liquidation_price=liq_price,
                          conditions_met=base.conditions_met,
                          condition_flags=base.condition_flags,
                          is_add=True, add_number=self.add_count, highest_price=price)
        self.active_trades.append(add_trade)
        self.stats["pyramid_adds"] += 1

        total_btc = sum(t.size_btc for t in self.active_trades if not t.closed)
        total_margin = sum(t.margin for t in self.active_trades if not t.closed)
        total_not = total_btc * price
        eff_lev = total_not / total_margin if total_margin > 0 else leverage

        self.trade_log.append(
            f"[{ts.strftime('%Y-%m-%d')}] ADD #{self.add_count} ({signal_type}): "
            f"${price:,.0f} | {leverage}x | margin=${add_margin:,.0f} | "
            f"size={add_btc:.4f} BTC | pnl=${agg_pnl:,.0f} | "
            f"total={total_btc:.4f} BTC | eff_lev={eff_lev:.1f}x | "
            f"stop=${add_stop:,.0f}")

    def _deduct_funding(self, price):
        for t in self.active_trades:
            if t.closed: continue
            cost = t.size_btc * price * FUNDING_RATE_PER_8H * 21
            self.capital -= cost
            t.funding_paid += cost
            self.stats["total_funding_paid"] += cost

    def _manage_positions(self, idx, high, low, price, ts, w_atr):
        mvrv = self._mvrv_proxy(idx)
        for trade in list(self.active_trades):
            if trade.closed: continue
            trade.highest_price = max(trade.highest_price, high)

            # Liquidation
            if low <= trade.liquidation_price:
                self.stats["liquidations"] += 1
                self.stats["liquidation_loss"] += trade.margin
                self._close_trade(trade, trade.liquidation_price, idx, ts, "liquidation")
                continue

            # Trail stops (both base and adds)
            if w_atr > 0:
                mult = TRAILING_ATR_MULT if not trade.is_add else ADD_ATR_MULT
                new_trail = trade.highest_price - w_atr * mult
                trade.stop_loss = max(trade.stop_loss, new_trail)

            if low <= trade.stop_loss:
                reason = "trailing_stop" if not trade.is_add else "add_stop"
                self._close_trade(trade, trade.stop_loss, idx, ts, reason)
                continue

            # Scale out
            if mvrv > MVRV_SCALEOUT and not trade.scale_out_done:
                sz = trade.size_btc * 0.20
                pnl = (price - trade.entry_price) * sz - sz * price * COMMISSION_RATE
                self.capital += pnl
                trade.size_btc -= sz
                trade.scale_out_done = True
                self.stats["scale_outs"] += 1

            if mvrv > MVRV_AGGRESSIVE_EXIT:
                self.stats["aggressive_exits"] += 1
                self._close_trade(trade, price, idx, ts, "mvrv_exit")

    def _close_trade(self, trade, exit_price, idx, ts, reason):
        pnl = (exit_price - trade.entry_price) * trade.size_btc
        pnl -= trade.size_btc * exit_price * COMMISSION_RATE
        if reason == "liquidation": pnl = -trade.margin

        trade.exit_price = exit_price; trade.exit_idx = idx
        trade.exit_time = ts; trade.pnl = pnl
        trade.closed = True; trade.close_reason = reason
        self.capital += pnl

        if trade in self.active_trades:
            self.active_trades.remove(trade)
        self.closed_trades.append(trade)

        self.trade_log.append(
            f"[{ts.strftime('%Y-%m-%d')}] EXIT ({reason}): ${exit_price:,.0f} | "
            f"pnl=${pnl:,.0f} | funding=${trade.funding_paid:,.0f} | "
            f"{trade.leverage}x | add={trade.add_number}")

        if not trade.is_add:
            for at in list(self.active_trades):
                if at.is_add and not at.closed:
                    self._close_trade(at, exit_price, idx, ts, "base_closed")
            self.base_entry_idx = -1; self.base_margin = 0; self.add_count = 0

    def _compute_metrics(self):
        eq = pd.DataFrame(self.equity_curve)
        equity = eq["equity"].values
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak * 100
        max_dd = np.max(dd)
        returns = np.diff(equity) / equity[:-1]
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(52)
                  if len(returns) > 1 and np.std(returns) > 0 else 0)

        pnls = [t.pnl for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

        base_trades = [t for t in self.closed_trades if not t.is_add]
        add_trades = [t for t in self.closed_trades if t.is_add]

        return {
            "system": "v17b_smart_aggressive",
            "total_return_pct": round(total_return, 2),
            "final_equity": round(equity[-1], 2),
            "initial_capital": self.initial_capital,
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(pf, 2),
            "total_trades": len(self.closed_trades),
            "base_trades": len(base_trades),
            "pyramid_adds": len(add_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "largest_win": round(max(pnls), 2) if pnls else 0,
            "largest_loss": round(min(pnls), 2) if pnls else 0,
            "gross_pnl": round(sum(pnls), 2),
            "total_funding_cost": round(self.stats["total_funding_paid"], 2),
            "net_pnl_after_funding": round(sum(pnls) - self.stats["total_funding_paid"], 2),
            **self.stats,
        }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_4h = fetch_ohlcv(use_cache=True)
    weekly = resample_to_weekly(df_4h)
    print(f"Loaded {len(df_4h)} 4H → {len(weekly)} weekly candles")

    bt = SmartAggressiveBacktest(weekly)
    metrics = bt.run()

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Report
    lines = ["=" * 70, "  V17b — SMART AGGRESSIVE ROLLING", "=" * 70, ""]
    lines.append(f"CAPITAL: ${CAPITAL:,.0f} | BASE: {BASE_LEVERAGE}x | ADD: {ADD_LEVERAGE_BASE}x/{ADD_LEVERAGE_PROFIT}x")
    lines.append(f"MARGIN: {ENTRY_MARGIN_PCT:.0%} | TRAIL: {TRAILING_ATR_MULT}x ATR | ADD STOP: {ADD_ATR_MULT}x ATR")
    lines.append(f"PYRAMID: {PYRAMID_SIZES} (max {MAX_ADDS})\n")
    for k, v in metrics.items():
        if k in ("system",): continue
        if isinstance(v, float):
            lines.append(f"  {k:35s}: {v:>12,.2f}")
        else:
            lines.append(f"  {k:35s}: {v:>12}")
    lines.append("\n── TRADE LOG ──")
    for entry in bt.trade_log:
        lines.append(f"  {entry}")
    lines.append("\n── TRADE DETAILS ──")
    for i, t in enumerate(bt.closed_trades, 1):
        add_str = f" (Add #{t.add_number})" if t.is_add else ""
        lines.append(
            f"  #{i}{add_str}: {t.entry_time.strftime('%Y-%m-%d')} → "
            f"{t.exit_time.strftime('%Y-%m-%d') if t.exit_time else '?'} | "
            f"${t.entry_price:,.0f} → ${t.exit_price:,.0f} | "
            f"{t.leverage}x | PnL=${t.pnl:,.2f} | {t.close_reason}")

    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
        f.write(report)

    # Equity plot
    eq = pd.DataFrame(bt.equity_curve)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[2, 1, 1])
    fig.suptitle(f"V17b Smart Aggressive | Return: {metrics['total_return_pct']:+.1f}% | "
                 f"DD: {metrics['max_drawdown_pct']:.1f}%", fontsize=13)
    axes[0].plot(eq["timestamp"], eq["equity"], color="royalblue", linewidth=1.5)
    axes[0].axhline(y=CAPITAL, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Equity ($)"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(eq["timestamp"], eq["price"], color="orange", linewidth=1)
    for t in bt.closed_trades:
        c = "green" if not t.is_add else "lime"
        axes[1].scatter(t.entry_time, t.entry_price, marker="^", color=c, s=60, zorder=5)
        if t.exit_time:
            mc = "red" if t.pnl < 0 else "blue"
            axes[1].scatter(t.exit_time, t.exit_price, marker="v", color=mc, s=40, zorder=5)
    axes[1].set_ylabel("BTC ($)"); axes[1].grid(True, alpha=0.3)
    equity = eq["equity"].values
    peak = np.maximum.accumulate(equity)
    dd_pct = (peak - equity) / peak * 100
    axes[2].fill_between(eq["timestamp"], 0, dd_pct, color="red", alpha=0.3)
    axes[2].set_ylabel("DD (%)"); axes[2].invert_yaxis(); axes[2].grid(True, alpha=0.3)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "equity_curve.png"), dpi=150)
    plt.close()

    print(f"\n{'='*50}")
    print(f"  V17b SMART AGGRESSIVE")
    print(f"{'='*50}")
    print(f"  Return:       {metrics['total_return_pct']:+.2f}%")
    print(f"  Final Equity: ${metrics['final_equity']:,.2f}")
    print(f"  Max DD:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe:       {metrics['sharpe_ratio']:.2f}")
    print(f"  Trades:       {metrics['total_trades']} ({metrics['pyramid_adds']} adds)")
    print(f"  Win Rate:     {metrics['win_rate_pct']:.1f}%")
    print(f"  Liquidations: {metrics['liquidations']}")
    print(f"  Funding:      ${metrics['total_funding_cost']:,.2f}")
    print(f"  Hi-Lev Adds:  {metrics['high_leverage_adds']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

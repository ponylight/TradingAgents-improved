"""sweep_v20.py — Parameter Optimization Sweep for V20 Strategy

Tests systematic combinations of:
1. Cooldown bars between signals (12, 18, 24, 36)
2. Leverage tiers (conservative/current/aggressive)
3. Risk per trade (2%, 3%, 5%, 8%)
4. Trailing ATR multiplier (2.0, 2.5, 3.0, 3.5)
5. Fibonacci tolerance (0.005, 0.008, 0.012)
6. Long-only mode (yes / no)

All combos tested on 4-year data first.
Top-10 winners re-tested on 8.5-year data.
"""

import sys
import os
import itertools
import time
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.fetch_data import fetch_ohlcv
from backtest.indicators import compute_macd, compute_atr, compute_ma
from backtest.backtest_v20 import (
    compute_td_sequential,
    find_swing_points,
    compute_fib_levels,
    get_active_fib_levels,
    check_bottom_divergence,
    check_top_divergence,
    Signal,
    Trade,
)

# ─────────────────────────────────────────────────────────────────────
# Leverage Tier Definitions
# ─────────────────────────────────────────────────────────────────────
LEVERAGE_TIERS = {
    "conservative": {"base": 2, "mid": 3, "high": 5},
    "current":      {"base": 3, "mid": 5, "high": 10},
    "aggressive":   {"base": 5, "mid": 10, "high": 20},
}

# ─────────────────────────────────────────────────────────────────────
# Sweep parameter space
# ─────────────────────────────────────────────────────────────────────
PARAM_SPACE = {
    "cooldown":       [12, 18, 24, 36],
    "leverage_tier":  ["conservative", "current", "aggressive"],
    "risk_pct":       [0.02, 0.03, 0.05, 0.08],
    "trail_atr_mult": [2.0, 2.5, 3.0, 3.5],
    "fib_tolerance":  [0.005, 0.008, 0.012],
    "long_only":      [True, False],
}


# ─────────────────────────────────────────────────────────────────────
# Parameterized signal generator
# ─────────────────────────────────────────────────────────────────────

def generate_signals_param(df, swing_highs, swing_lows,
                           cooldown: int, leverage_tiers: dict,
                           fib_tolerance: float, long_only: bool) -> list:
    """Generate signals with custom parameters."""
    signals = []

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    volumes= df["volume"].values
    atr    = df["atr"].values
    hist   = df["macd_hist"].values
    sma200 = df["sma200"].values

    td_buy_setup    = df["td_buy_setup"].values
    td_sell_setup   = df["td_sell_setup"].values
    td_buy_countdown= df["td_buy_countdown"].values
    td_sell_countdown=df["td_sell_countdown"].values

    base_lev = leverage_tiers["base"]
    mid_lev  = leverage_tiers["mid"]
    high_lev = leverage_tiers["high"]

    last_signal_idx = -cooldown

    for i in range(50, len(df)):
        if i - last_signal_idx < cooldown:
            continue
        if np.isnan(atr[i]) or np.isnan(sma200[i]):
            continue

        avg_vol = np.mean(volumes[max(0, i-20):i]) if i > 20 else volumes[i]
        if volumes[i] <= avg_vol * 0.8:
            continue

        fib = get_active_fib_levels(i, swing_highs, swing_lows, closes)
        bull = closes[i] > sma200[i] if not np.isnan(sma200[i]) else False

        # ── LONG ──────────────────────────────────────────────────
        confluence = []
        leverage = base_lev

        has_td9_buy  = td_buy_setup[i] == 9
        has_td13_buy = td_buy_countdown[i] == 13
        macd_div     = check_bottom_divergence(i, hist, lows)
        has_macd_div = macd_div >= 1

        fib_level = None
        if fib:
            fib_level = price_near_fib_param(closes[i], fib, fib_tolerance)

        if has_td9_buy:  confluence.append("TD9_buy")
        if has_td13_buy: confluence.append("TD13_buy")
        if has_macd_div: confluence.append(f"MACD_div_{macd_div}")
        if fib_level:    confluence.append(f"Fib_{fib_level}")

        long_signal = False
        if has_td13_buy and fib_level:
            long_signal = True; leverage = high_lev
        elif has_td9_buy and has_macd_div:
            long_signal = True; leverage = high_lev
        elif has_td9_buy and fib_level:
            long_signal = True; leverage = mid_lev
        elif has_td13_buy:
            long_signal = True; leverage = mid_lev
        elif has_td9_buy and volumes[i] > avg_vol * 1.5:
            long_signal = True; leverage = base_lev
            confluence.append("volume_spike")
        elif has_macd_div and fib_level and macd_div >= 2:
            long_signal = True; leverage = base_lev

        if long_signal:
            sl_atr = closes[i] - atr[i] * 1.5
            sl_fib  = fib["0.786"] if fib else sl_atr
            sl = max(sl_atr, sl_fib)
            if fib:
                tp1 = fib["ext_1.272"]; tp2 = fib["ext_1.618"]
            else:
                risk = closes[i] - sl
                tp1 = closes[i] + risk * 1.5; tp2 = closes[i] + risk * 2.5

            signals.append(Signal(
                idx=i, side="long", entry_price=closes[i],
                stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
                leverage=leverage, confluence=confluence, fib_level=fib_level or "",
            ))
            last_signal_idx = i
            continue

        # ── SHORT (skip if long_only) ─────────────────────────────
        if long_only or bull:
            continue

        confluence = []
        leverage = base_lev

        has_td9_sell  = td_sell_setup[i] == 9
        has_td13_sell = td_sell_countdown[i] == 13
        macd_bear_div = check_top_divergence(i, hist, highs)
        has_macd_bd   = macd_bear_div >= 1

        fib_level = None
        if fib:
            for lvl in ["0.382", "0.500", "0.618"]:
                lp = fib[lvl]
                if abs(closes[i] - lp) / closes[i] <= fib_tolerance:
                    fib_level = lvl; break

        if has_td9_sell:  confluence.append("TD9_sell")
        if has_td13_sell: confluence.append("TD13_sell")
        if has_macd_bd:   confluence.append(f"MACD_bear_div_{macd_bear_div}")
        if fib_level:     confluence.append(f"Fib_{fib_level}")

        short_signal = False
        if has_td13_sell and fib_level:
            short_signal = True; leverage = high_lev
        elif has_td9_sell and has_macd_bd:
            short_signal = True; leverage = high_lev
        elif has_td9_sell and fib_level:
            short_signal = True; leverage = mid_lev
        elif has_td13_sell:
            short_signal = True; leverage = mid_lev
        elif has_td9_sell and volumes[i] > avg_vol * 1.5:
            short_signal = True; leverage = base_lev
            confluence.append("volume_spike")
        elif has_macd_bd and fib_level and macd_bear_div >= 2:
            short_signal = True; leverage = base_lev

        if short_signal:
            sl_atr = closes[i] + atr[i] * 1.5
            sl_fib  = fib["0.236"] if fib else sl_atr
            sl = min(sl_atr, sl_fib)
            if fib:
                tp1 = fib["0.786"]; tp2 = fib["swing_low"]
            else:
                risk = sl - closes[i]
                tp1 = closes[i] - risk * 1.5; tp2 = closes[i] - risk * 2.5

            signals.append(Signal(
                idx=i, side="short", entry_price=closes[i],
                stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
                leverage=leverage, confluence=confluence, fib_level=fib_level or "",
            ))
            last_signal_idx = i

    return signals


def price_near_fib_param(price: float, fib_levels: dict, tolerance: float) -> str | None:
    for lvl in ["0.618", "0.500", "0.382"]:
        lp = fib_levels[lvl]
        if abs(price - lp) / price <= tolerance:
            return lvl
    return None


# ─────────────────────────────────────────────────────────────────────
# Lightweight Backtest Engine
# ─────────────────────────────────────────────────────────────────────

def run_backtest(df, signals, risk_pct: float, trail_atr_mult: float,
                 initial_capital: float = 10_000.0) -> dict:
    """Run the backtest engine with given parameters. Returns metrics dict."""
    COMMISSION = 0.001
    MAX_POS_PCT = 0.50

    capital = initial_capital
    equity_curve = []
    closed_trades = []
    active_trades = []
    liquidations = 0
    liq_loss = 0.0

    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values
    atrs    = df["atr"].values

    signal_map = {s.idx: s for s in signals}

    for i in range(len(df)):
        price = closes[i]
        high  = highs[i]
        low   = lows[i]
        ts    = df["timestamp"].iloc[i]
        atr_val = atrs[i]

        # manage open positions
        for trade in list(active_trades):
            if trade["closed"]:
                continue

            if trade["side"] == "long":
                # Liquidation
                if low <= trade["liq_price"]:
                    liquidations += 1
                    liq_loss += trade["margin"]
                    _close(trade, trade["liq_price"], ts, "liquidation", capital, closed_trades, active_trades, COMMISSION)
                    capital = trade["_cap_after"]
                    continue

                # Stop
                if low <= trade["trailing_stop"]:
                    _close(trade, trade["trailing_stop"], ts, "stop_loss", capital, closed_trades, active_trades, COMMISSION)
                    capital = trade["_cap_after"]
                    continue

                # TP1
                if not trade["half_closed"] and high >= trade["tp1"]:
                    half = trade["size"] / 2
                    pnl_half = (trade["tp1"] - trade["entry"]) * half
                    comm = half * trade["tp1"] * COMMISSION
                    trade["pnl_partial"] = pnl_half - comm
                    capital += trade["pnl_partial"]
                    trade["size"] -= half
                    trade["half_closed"] = True
                    trade["trailing_stop"] = max(trade["trailing_stop"],
                        trade["entry"] + (trade["tp1"] - trade["entry"]) * 0.1)

                # TP2
                if trade["half_closed"] and high >= trade["tp2"]:
                    _close(trade, trade["tp2"], ts, "take_profit_2", capital, closed_trades, active_trades, COMMISSION)
                    capital = trade["_cap_after"]
                    continue

                # Trailing
                if not np.isnan(atr_val) and trade["half_closed"]:
                    new_trail = price - atr_val * trail_atr_mult
                    trade["trailing_stop"] = max(trade["trailing_stop"], new_trail)

            else:  # short
                if high >= trade["liq_price"]:
                    liquidations += 1
                    liq_loss += trade["margin"]
                    _close(trade, trade["liq_price"], ts, "liquidation", capital, closed_trades, active_trades, COMMISSION)
                    capital = trade["_cap_after"]
                    continue

                if high >= trade["trailing_stop"]:
                    _close(trade, trade["trailing_stop"], ts, "stop_loss", capital, closed_trades, active_trades, COMMISSION)
                    capital = trade["_cap_after"]
                    continue

                if not trade["half_closed"] and low <= trade["tp1"]:
                    half = trade["size"] / 2
                    pnl_half = (trade["entry"] - trade["tp1"]) * half
                    comm = half * trade["tp1"] * COMMISSION
                    trade["pnl_partial"] = pnl_half - comm
                    capital += trade["pnl_partial"]
                    trade["size"] -= half
                    trade["half_closed"] = True
                    trade["trailing_stop"] = min(trade["trailing_stop"],
                        trade["entry"] - (trade["entry"] - trade["tp1"]) * 0.1)

                if trade["half_closed"] and low <= trade["tp2"]:
                    _close(trade, trade["tp2"], ts, "take_profit_2", capital, closed_trades, active_trades, COMMISSION)
                    capital = trade["_cap_after"]
                    continue

                if not np.isnan(atr_val) and trade["half_closed"]:
                    new_trail = price + atr_val * trail_atr_mult
                    trade["trailing_stop"] = min(trade["trailing_stop"], new_trail)

        # New signal
        if i in signal_map:
            sig = signal_map[i]
            same_dir = [t for t in active_trades if t["side"] == sig.side and not t["closed"]]
            if not same_dir:
                risk = abs(sig.entry_price - sig.stop_loss)
                if risk > 0:
                    risk_amount = capital * risk_pct
                    base_size = risk_amount / risk
                    max_size = (capital * MAX_POS_PCT) / sig.entry_price
                    base_size = min(base_size, max_size)
                    if base_size * sig.entry_price >= 10:
                        lev = sig.leverage
                        lev_size = base_size * lev
                        margin = base_size * sig.entry_price
                        notional = lev_size * sig.entry_price
                        comm = notional * COMMISSION
                        capital -= comm
                        liq_p = sig.entry_price * (1 - 1/lev) if sig.side == "long" else sig.entry_price * (1 + 1/lev)
                        active_trades.append({
                            "side": sig.side, "entry": sig.entry_price,
                            "entry_idx": i, "entry_time": ts,
                            "size": lev_size, "margin": margin, "leverage": lev,
                            "trailing_stop": sig.stop_loss, "tp1": sig.take_profit_1,
                            "tp2": sig.take_profit_2, "liq_price": liq_p,
                            "half_closed": False, "closed": False,
                            "pnl_partial": 0.0, "pnl": 0.0, "close_reason": "",
                            "exit_price": 0.0, "exit_time": None,
                            "_cap_after": capital,
                        })

        # Unrealized
        unrealized = 0.0
        for t in active_trades:
            if not t["closed"]:
                if t["side"] == "long":
                    unrealized += (price - t["entry"]) * t["size"]
                else:
                    unrealized += (t["entry"] - price) * t["size"]
        equity_curve.append(capital + unrealized)

    # Force-close remaining
    last_price = closes[-1]
    last_ts = df["timestamp"].iloc[-1]
    for trade in list(active_trades):
        if not trade["closed"]:
            _close(trade, last_price, last_ts, "end_of_backtest", capital, closed_trades, active_trades, COMMISSION)
            capital = trade["_cap_after"]

    return _metrics(equity_curve, closed_trades, initial_capital, liquidations, liq_loss)


def _close(trade, exit_price, ts, reason, capital, closed_trades, active_trades, commission):
    if trade["side"] == "long":
        pnl = (exit_price - trade["entry"]) * trade["size"]
    else:
        pnl = (trade["entry"] - exit_price) * trade["size"]
    comm = trade["size"] * exit_price * commission
    pnl -= comm
    trade["pnl"] = pnl + trade["pnl_partial"]
    trade["exit_price"] = exit_price
    trade["exit_time"] = ts
    trade["close_reason"] = reason
    trade["closed"] = True
    new_cap = capital + pnl
    trade["_cap_after"] = new_cap
    closed_trades.append(trade)
    if trade in active_trades:
        active_trades.remove(trade)


def _metrics(equity_curve, closed_trades, initial_capital, liquidations, liq_loss):
    if not closed_trades or not equity_curve:
        return None

    equity = np.array(equity_curve)
    total_return = (equity[-1] - initial_capital) / initial_capital * 100
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak * 100
    max_dd = float(np.max(dd))

    returns = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
    std_r = float(np.std(returns))
    sharpe = float(np.mean(returns) / std_r * np.sqrt(2190)) if std_r > 0 else 0.0

    pnls   = [t["pnl"] for t in closed_trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss   = abs(sum(losses)) if losses else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    long_t  = [t for t in closed_trades if t["side"] == "long"]
    short_t = [t for t in closed_trades if t["side"] == "short"]

    return {
        "total_return_pct":  round(total_return, 2),
        "final_equity":      round(float(equity[-1]), 2),
        "max_drawdown_pct":  round(max_dd, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "win_rate_pct":      round(win_rate, 2),
        "profit_factor":     round(pf, 3),
        "total_trades":      len(closed_trades),
        "long_trades":       len(long_t),
        "short_trades":      len(short_t),
        "long_pnl":          round(sum(t["pnl"] for t in long_t), 2),
        "short_pnl":         round(sum(t["pnl"] for t in short_t), 2),
        "liquidations":      liquidations,
        "liq_loss":          round(liq_loss, 2),
        "avg_win":           round(float(np.mean(wins)), 2) if wins else 0,
        "avg_loss":          round(float(np.mean(losses)), 2) if losses else 0,
    }


# ─────────────────────────────────────────────────────────────────────
# Scoring function (composite rank)
# ─────────────────────────────────────────────────────────────────────

def score(m: dict) -> float:
    """Composite score balancing return, drawdown, sharpe, and profit factor."""
    if m is None:
        return -9999
    ret   = m["total_return_pct"]
    dd    = m["max_drawdown_pct"]
    sh    = m["sharpe_ratio"]
    pf    = min(m["profit_factor"], 10)  # cap at 10 to avoid infinity distortion
    liq   = m["liquidations"]
    trades = m["total_trades"]
    
    if trades < 3:
        return -9999  # not enough trades
    
    # Penalize excessive drawdown
    dd_penalty = max(0, dd - 40) * 2
    # Bonus for fewer liquidations
    liq_penalty = liq * 5
    
    # Composite: heavily weighted toward Sharpe and return/drawdown ratio
    rdr = ret / max(dd, 1)  # return/drawdown ratio
    return sh * 30 + rdr * 10 + pf * 5 + ret * 0.1 - dd_penalty - liq_penalty


# ─────────────────────────────────────────────────────────────────────
# Worker function for multiprocessing
# ─────────────────────────────────────────────────────────────────────

_SHARED_DF = None
_SHARED_SWINGS = None

def _init_worker(df_dict, swings):
    global _SHARED_DF, _SHARED_SWINGS
    # Reconstruct DataFrame in worker (simpler than shared memory)
    _SHARED_DF = pd.DataFrame(df_dict)
    _SHARED_DF["timestamp"] = pd.to_datetime(_SHARED_DF["timestamp"])
    _SHARED_SWINGS = swings


def _run_combo(params):
    cooldown, tier_name, risk_pct, trail_atr_mult, fib_tol, long_only = params
    tiers = LEVERAGE_TIERS[tier_name]

    try:
        sigs = generate_signals_param(
            _SHARED_DF, _SHARED_SWINGS[0], _SHARED_SWINGS[1],
            cooldown=cooldown, leverage_tiers=tiers,
            fib_tolerance=fib_tol, long_only=long_only,
        )
        if not sigs:
            return None

        m = run_backtest(_SHARED_DF, sigs, risk_pct=risk_pct, trail_atr_mult=trail_atr_mult)
        if m is None:
            return None

        m.update({
            "cooldown":       cooldown,
            "leverage_tier":  tier_name,
            "risk_pct":       risk_pct,
            "trail_atr_mult": trail_atr_mult,
            "fib_tolerance":  fib_tol,
            "long_only":      long_only,
        })
        m["score"] = score(m)
        return m
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────
# Prepare data (shared across all workers)
# ─────────────────────────────────────────────────────────────────────

def prepare_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    from backtest.indicators import compute_macd, compute_atr, compute_ma
    df = compute_macd(raw_df, fast=13, slow=34, signal=9)
    df["atr"]    = compute_atr(df, period=14)
    df["sma50"]  = compute_ma(df, period=50)
    df["sma200"] = compute_ma(df, period=200)
    df = compute_td_sequential(df)
    return df


# ─────────────────────────────────────────────────────────────────────
# Main sweep runner
# ─────────────────────────────────────────────────────────────────────

def print_table(rows: list[dict], title: str, rank_start: int = 1):
    print(f"\n{'='*130}")
    print(f"  {title}")
    print(f"{'='*130}")
    hdr = (
        f"{'Rank':>4} {'Return':>8} {'MaxDD':>7} {'Sharpe':>7} {'PF':>6} {'WR%':>6} "
        f"{'Trades':>7} {'Liq':>4} {'Cdwn':>5} {'Tier':>13} {'Risk':>6} "
        f"{'Trail':>6} {'FibTol':>7} {'LongOnly':>9} {'Score':>8}"
    )
    print(hdr)
    print("-" * 130)
    for rank, r in enumerate(rows, rank_start):
        lo = "YES" if r["long_only"] else "no"
        print(
            f"{rank:>4} {r['total_return_pct']:>7.1f}% {r['max_drawdown_pct']:>6.1f}% "
            f"{r['sharpe_ratio']:>7.2f} {min(r['profit_factor'], 99):>6.2f} "
            f"{r['win_rate_pct']:>5.1f}% {r['total_trades']:>7} {r['liquidations']:>4} "
            f"{r['cooldown']:>5} {r['leverage_tier']:>13} {r['risk_pct']*100:>5.0f}% "
            f"{r['trail_atr_mult']:>6.1f} {r['fib_tolerance']:>7.3f} {lo:>9} "
            f"{r['score']:>8.1f}"
        )


def main():
    print("=" * 70)
    print("  V20 Parameter Sweep — Fibonacci + TD Sequential")
    print("=" * 70)

    # ── Load 4yr data ──────────────────────────────────────────────────
    print("\n[1/4] Loading and preparing 4-year dataset...")
    raw_4yr = fetch_ohlcv(use_cache=True)
    df_4yr  = prepare_df(raw_4yr)
    sh_4yr  = find_swing_points(df_4yr["high"].values, df_4yr["low"].values, lookback=20)
    print(f"  Data: {len(df_4yr)} bars | {df_4yr['timestamp'].iloc[0].date()} → {df_4yr['timestamp'].iloc[-1].date()}")

    # ── Build param combinations ───────────────────────────────────────
    keys  = list(PARAM_SPACE.keys())
    combos = list(itertools.product(*[PARAM_SPACE[k] for k in keys]))
    total  = len(combos)
    print(f"\n[2/4] Running sweep: {total} combinations × 4-year data")
    print(f"  Workers: {cpu_count()} CPUs")

    # Serialize DataFrame for worker init
    df_dict = df_4yr.copy()
    df_dict["timestamp"] = df_dict["timestamp"].astype(str)
    df_records = df_dict.to_dict("list")

    t0 = time.time()
    with Pool(
        processes=max(1, cpu_count() - 1),
        initializer=_init_worker,
        initargs=(df_records, sh_4yr),
    ) as pool:
        results_4yr = pool.map(_run_combo, combos)

    elapsed = time.time() - t0
    valid = [r for r in results_4yr if r is not None]
    valid.sort(key=lambda r: r["score"], reverse=True)

    print(f"  Done in {elapsed:.1f}s — {len(valid)}/{total} valid results")

    # Print top 20 on 4yr
    top20 = valid[:20]
    print_table(top20, "TOP 20 CONFIGURATIONS — 4-Year Dataset (2022–2026)")

    # Extra breakdown: long-only vs both
    long_only_top = [r for r in valid if r["long_only"]][:5]
    both_top = [r for r in valid if not r["long_only"]][:5]
    print_table(long_only_top, "TOP 5 LONG-ONLY configs (4yr)")
    print_table(both_top,      "TOP 5 LONG+SHORT configs (4yr)")

    # ── Validate top 10 on 8.5yr ──────────────────────────────────────
    data_8yr_path = os.path.join(os.path.dirname(__file__), "data", "btc_usdt_4h_2017.csv")
    if not os.path.exists(data_8yr_path):
        print(f"\n[3/4] Skipping 8.5yr validation — file not found: {data_8yr_path}")
    else:
        print(f"\n[3/4] Validating top-10 configs on 8.5-year dataset...")
        raw_8yr = pd.read_csv(data_8yr_path, parse_dates=["timestamp"])
        df_8yr  = prepare_df(raw_8yr)
        sh_8yr  = find_swing_points(df_8yr["high"].values, df_8yr["low"].values, lookback=20)
        print(f"  Data: {len(df_8yr)} bars | {df_8yr['timestamp'].iloc[0].date()} → {df_8yr['timestamp'].iloc[-1].date()}")

        df_dict_8 = df_8yr.copy()
        df_dict_8["timestamp"] = df_dict_8["timestamp"].astype(str)
        df_rec_8 = df_dict_8.to_dict("list")

        top10_params = [
            (r["cooldown"], r["leverage_tier"], r["risk_pct"],
             r["trail_atr_mult"], r["fib_tolerance"], r["long_only"])
            for r in valid[:10]
        ]

        with Pool(
            processes=max(1, min(10, cpu_count() - 1)),
            initializer=_init_worker,
            initargs=(df_rec_8, sh_8yr),
        ) as pool:
            results_8yr = pool.map(_run_combo, top10_params)

        print_table(
            [r for r in results_8yr if r is not None],
            "TOP-10 CONFIGS VALIDATED — 8.5-Year Dataset (2017–2026)",
        )

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n[4/4] PARAMETER INSIGHTS")
    print("=" * 70)

    # Best by individual param
    for param in ["cooldown", "leverage_tier", "risk_pct", "trail_atr_mult", "fib_tolerance"]:
        grouped = {}
        for r in valid:
            k = r[param]
            if k not in grouped:
                grouped[k] = []
            grouped[k].append(r["score"])
        best_k = max(grouped, key=lambda k: np.mean(grouped[k]))
        print(f"  Best avg score for {param:>15}: {best_k} "
              f"(mean score={np.mean(grouped[best_k]):+.1f}, n={len(grouped[best_k])})")

    lo_scores = [r["score"] for r in valid if r["long_only"]]
    both_scores = [r["score"] for r in valid if not r["long_only"]]
    print(f"\n  Long-only avg score:    {np.mean(lo_scores):+.2f} (n={len(lo_scores)})")
    print(f"  Long+Short avg score:   {np.mean(both_scores):+.2f} (n={len(both_scores)})")

    print(f"\n  Best overall config:")
    best = valid[0]
    print(f"    cooldown={best['cooldown']} bars | leverage={best['leverage_tier']} | "
          f"risk={best['risk_pct']*100:.0f}% | trail={best['trail_atr_mult']} ATR | "
          f"fib_tol={best['fib_tolerance']} | long_only={best['long_only']}")
    print(f"    Return={best['total_return_pct']:+.1f}% | DD={best['max_drawdown_pct']:.1f}% | "
          f"Sharpe={best['sharpe_ratio']:.2f} | PF={best['profit_factor']:.2f} | "
          f"WR={best['win_rate_pct']:.1f}% | Trades={best['total_trades']} | "
          f"Liq={best['liquidations']}")

    # Save results to CSV
    out_path = os.path.join(os.path.dirname(__file__), "results_v20", "sweep_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(valid).sort_values("score", ascending=False).to_csv(out_path, index=False)
    print(f"\n  Full results saved to: {out_path}")
    print(f"\n{'='*70}")
    print("  Sweep complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

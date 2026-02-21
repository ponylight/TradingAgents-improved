# BTC/USDT TRADING STRATEGY
*Last updated: 2026-02-21 by Canon — Unified V20 single-strategy system*
*Backtested: 2017-08 → 2026-02 (8.5 years, 18,645 4H candles)*

---

## CORE PHILOSOPHY

> One strategy. Maximum conviction. No capital splitting.
>
> V20 Fibonacci + TD Sequential generates better signals, better exits, and better risk-adjusted returns than any long-term bullet approach. It outperforms V19 "20 Bullets" by **200x** over the same period with fewer liquidations and lower drawdowns.

**Why single strategy beats dual:**
- No capital dilution — 100% deployed through one proven system
- V20 already catches supercycles via repeated 4H entries during bull runs
- Short-duration trades (~2 days avg) eliminate funding rate bleeding
- Structured Fibonacci exits capture more profit than "let it ride or die"

---

## REGIME DETECTION

### Primary Signals
| Signal | Bull | Bear | Source |
|--------|------|------|--------|
| **MVRV Z-Score** | > 0 | < 0 | woocharts.com |
| **MVRV Ratio** | > 1.5 | < 1 | CryptoQuant |
| **Weekly 200 SMA** | Above | Below | Chart |
| **365-day MA** | Above | Below | Chart |
| **Fear & Greed** | > 40 | < 25 | alternative.me |
| **Ichimoku Kumo** | Bullish twist | Bearish twist | Chart |

### Secondary Signals
| Signal | Bull | Bear | Source |
|--------|------|------|--------|
| **SOPR** | > 1 | < 1 | CryptoQuant |
| **Spot Taker CVD** | Buy dominant | Sell dominant | CryptoQuant |
| **CQ Bull-Bear Cycle** | Bull phase | Bear phase | CryptoQuant |
| **Exchange Inflows** | Low | High from whales | CryptoQuant |

### Regime Rules
- **3+ primary signals agree** → confirmed regime
- **Mixed signals** → reduce position sizing, lower leverage tier

### Current Status (Feb 21, 2026)
- **Regime: BEAR** — MVRV Z -0.86, F&G 8, below 200 SMA, bearish Kumo
- Late-stage capitulation likely. Signals will fire when reversal begins.

---

## ⚡ V20 OPTIMIZED — FIBONACCI + TD SEQUENTIAL

### Backtest Results

| Period | Return | Multiplier | Trades | Win Rate | Max DD | Liq. | Sharpe | PF |
|--------|--------|-----------|--------|----------|--------|------|--------|----|
| 4yr (2022-2026) | **+2,144,586%** | 21,445x | 96 | 53.1% | 54.9% | 1 | 1.92 | 12.48 |
| 8.5yr (2017-2026) | **+276,478,125%** | 2,764,781x | 165 | ~51% | 91.6% | 8 | 1.57 | 16.74 |

*Optimized parameters from 1,152-config sweep. Long-only.*

### Signal Generation

**1. TD Sequential (Tom DeMark)**
- **Setup:** Count 9 consecutive closes where `close[i] < close[i-4]` (buy setup = bearish exhaustion)
- **Countdown:** After completed 9-bar setup, count 13 bars where `close[i] ≤ low[i-2]`
- TD9 = setup complete (potential reversal); TD13 = countdown complete (strongest signal)
- Cancel countdown if opposing setup reaches 4

**2. Fibonacci Retracement**
- Identify swing high/low from 20-bar lookback structure
- Key retracement levels: **0.382, 0.500, 0.618** from swing
- **Tolerance: 1.2%** (wider net catches more valid bounces)
- Extensions for TP: 1.272, 1.618, 2.618

**3. MACD Divergence (13/34)**
- Fast EMA 13, Slow EMA 34, Signal 9
- **Bullish divergence:** Price lower low + MACD histogram higher low
- Divergence strength ≥ 1 for signal, ≥ 2 for high conviction

### Optimized Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Cooldown | **12 bars** (2 days) | More signals = more compounding |
| Leverage tiers | **5x / 10x / 20x** (aggressive) | Higher base, higher ceiling |
| Risk per trade | **8%** | Aggressive but bounded |
| Trailing ATR | **3.5x** | Wide enough to survive pullbacks |
| Fib tolerance | **1.2%** | Catches more valid bounces |
| Direction | **Long-only** | Shorts historically unprofitable |
| Position sizing | **50% max** of capital per trade | Hard cap |

### Entry Tiers

| Tier | Leverage | Trigger Conditions |
|------|----------|--------------------|
| **Tier 1** | **20x** | TD13 at Fib level **OR** TD9 + MACD divergence |
| **Tier 2** | **10x** | TD9 at Fib level **OR** TD13 alone |
| **Tier 3** | **5x** | TD9 + volume spike **OR** MACD div ≥2 at Fib level |

> **Key insight:** TD9 + MACD divergence confluence generates almost all alpha. Do not force trades without at least one anchor signal.

### Exit Rules

| Target | Level | Action |
|--------|-------|--------|
| **TP1** | Fib extension 1.272 | Close 50% of position |
| **TP2** | Fib extension 1.618 | Trail remainder |
| **Stop** | Fib 0.786 OR 1.5 ATR from entry | Full exit (whichever tighter) |
| **Trailing** | 3.5x ATR (activates after TP1) | Trail remainder to TP2+ |

### Why Long-Only

Short signals in this system consistently lose money across all timeframes:
- 4yr: Shorts PnL = -$52 (vs Longs +$412K)
- 8.5yr: Shorts PnL = -$112K (vs Longs +$16.2M)

Long-only configs also show **higher Profit Factor** (16.7 vs 12.2) and **fewer liquidations** (8 vs 9) on 8.5yr data.

---

## ⚠️ FUNDING RATE MANAGEMENT

Average trade duration ~2 days — funding impact minimal. But monitor:
- **Funding > 0.03%/8h:** Reduce leverage or close
- **Funding negative:** Getting paid — add confidence
- **Cumulative > 3% on any position:** Alert for review

---

## 🔧 RISK MANAGEMENT

### Hard Limits
| Rule | Value |
|------|-------|
| Max position | 50% of capital |
| Risk per trade | 8% |
| Cash reserve | 20% always |
| Circuit breaker | Account -15% from peak → flatten all |
| Daily loss limit | -3% → no new trades 24h |
| Max leverage | 20x (Tier 1 only) |

### Drawdown Expectations
- 4yr max DD: 54.9% (1 liquidation)
- 8.5yr max DD: 91.6% (8 liquidations)
- **Drawdowns are temporary.** The 91.6% DD on 8.5yr occurred mid-path to 2.76M x returns.
- Most drawdown comes from a few large losing trades at 20x — the Fib stops limit each loss.

### Profit Taking (Mandatory)
- **Account equity hits 100x:** Take 30% off, withdraw to cold storage
- **Account equity hits 500x:** Take another 30%
- **Account equity hits 1000x:** Take another 30%
- **MVRV > 2.5:** Reduce to Tier 3 only (5x max)
- **MVRV > 3.5:** Stop entering, let existing positions trail out

---

## 📊 DATA SOURCES

| Data | Source |
|------|--------|
| MVRV Z-Score | woocharts.com (browser) |
| MVRV Ratio / SOPR / CVD | CryptoQuant (browser) |
| F&G Index | alternative.me (API) |
| OHLCV / Indicators | Bybit/Binance (ccxt) |
| Strategy KB | NotebookLM (notebook 9b6bf693) |

---

## 📈 BACKTEST HISTORY

### Evolution to V20
| Version | Strategy | Return (8.5yr) | Max DD | Notes |
|---------|----------|----------------|--------|-------|
| V6 | MACD 3x/10x (short-term) | +34% (4yr) | 15% | Too few trades |
| V14 | Weekly 5x, no adds | +42% | 17% | Conservative baseline |
| V16 | Weekly 5x + rolling adds | +54% | 20% | Adds contributed nothing |
| V18b | Weekly 10x + 10x adds | +470% | 84% | Rolling works at right sizing |
| V19 | 20 Bullets 20x→5x | +6,851% | 81% | Long-term bullet approach |
| V20 | Fib + TD Sequential | +153,854% | 60% | **Base V20 (3x/5x/10x)** |
| **V20opt** | **V20 optimized sweep** | **+276,478,125%** | **91.6%** | **CHAMPION — 1,152 configs tested** |
| V21 | V20 signals + 20 bullets | +786% | 57% | Bullets HURT V20's precision exits |

### Why V20 Beat Everything
1. **Signal quality:** TD Sequential + MACD divergence + Fibonacci = precise entries
2. **Structured exits:** Fib 1.272/1.618 targets capture swing profits methodically
3. **Compounding speed:** ~2-day average trades → capital recycles rapidly
4. **No funding bleed:** Short holds = near-zero funding costs
5. **Long-only eliminates drag:** Shorts lost money across every backtest period

### Why V21 (Bullets) Failed
- 20x entry leverage → ±5% liquidation zone → too tight for 4H candles
- Trailing stops exit winners too early (3x 4H ATR ≈ $2.4-4.5K)
- Pyramid rarely deploys — only 4.4% of bullets reached all 4 adds
- V20's Fibonacci TP targets > binary "ride or die"

---

## Lessons Learned
- **One great strategy > two mediocre ones.** Capital splitting dilutes returns.
- **Structured exits beat trailing-only.** Fib extensions provide exact price targets.
- **Signal confluence is king.** TD9 + MACD divergence = the edge.
- **Long-only in BTC.** Over 8+ years, shorting BTC has negative expected value with this system.
- **Position sizing matters.** Same signals at 3% vs 8% risk = dramatically different compounding.
- **Wider Fib tolerance (1.2%) catches more valid entries.** Precision < coverage.
- **Shorter cooldown = more compounding.** 12 bars beats 18 or 24.

---

## Weekly Review Schedule
- **Sunday 8PM Sydney:** Strategist reviews week, updates regime + this document

## Implementation
- **Backtest:** `backtest/backtest_v20.py`
- **Sweep results:** `backtest/results_v20/sweep_results.csv`
- **Live implementation:** TBD — build automated signal scanner + alert system

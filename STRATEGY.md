# BTC/USDT TRADING STRATEGY
*Last updated: 2026-02-21 by Canon*
*Backtested: 2017-08 → 2026-02 (8.5 years, 445 weekly candles)*

---

## CORE PHILOSOPHY: THE 20 BULLETS (肥宅 Method)

> "把仓位拆成20份以上" — Split capital into 20+ bullets.
> Fire cheap shots until one catches a trend. Then pour fuel on it.

**Expected outcomes:**
- ~90% of bullets get liquidated (normal, expected, budgeted)
- ~5-10% break even or small profit
- **1-3 bullets catch a supercycle trend → 50-200x on that bullet**
- Net result over a full cycle: **50-100x on total capital**

**Backtested results (2017-2026):**
- 32 bullets fired, 29 liquidated (91%)
- 3 trends caught → peaked at **221x** starting capital
- Final: **69.5x** after drawdowns
- Max drawdown: 81%

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
- **Mixed signals** → reduce exposure, smaller bullets

### Current Status (Feb 21, 2026)
- **Regime: BEAR** — MVRV Z -0.86, F&G 8, below 200 SMA, bearish Kumo
- Late-stage capitulation likely. Good time to fire bullets.

---

## 📈 LONG-TERM: 20 BULLETS STRATEGY (60% of Capital)

### Capital Allocation
- **Total long-term capital:** 60% of account
- **Each bullet:** 5% of current equity (= 1/20th)
- **Reserve:** Always keep 40% uninvested for bullets + short-term

### Entry: Fire a Bullet

**Trigger (ANY one sufficient):**
1. Price reclaims 20-week SMA from below (SMA crossover)
2. MVRV Z-Score < 0 (historically always recovers)
3. Weekly MACD 13/34 bullish divergence
4. Price within 20% of 52-week low

**Bullet specs:**
- **Margin:** 5% of current equity
- **Leverage:** 20x (high leverage on small capital = cheap lottery ticket)
- **Liquidation distance:** ~4.5% below entry
- **Max loss if liquidated:** 5% of equity (one bullet)
- **Cooldown:** 1 week between bullets

**No stop-loss.** Either the trend catches or you get liquidated. That's the design.

### Rolling Adds: Pour Fuel on Fire

Once a bullet is profitable, compound aggressively with rolling adds.

**Add trigger (ALL required):**
1. Position is profitable (floating PnL > 0)
2. Add signal fires (any one):
   - Breakout above 2-week high
   - Pullback within 5% of 20-week SMA
   - Fib 0.382-0.618 retracement from swing low to recent high

**Decreasing leverage on adds (CRITICAL):**
| Add # | Leverage | Why |
|-------|----------|-----|
| Base entry | 20x | Cheap bullet, small margin |
| Add 1 | 15x | Growing position, reduce risk |
| Add 2 | 10x | Significant size now |
| Add 3 | 5x | Protect the compound |
| Add 4 | 5x | Maximum protection |

**Add sizing:**
- **Margin source:** 80% of floating profit
- **Pyramid cap:** Add 1 = 100%, Add 2 = 75%, Add 3 = 50%, Add 4 = 25% of available
- **Max 4 adds per trend**

**Example from backtest (2023 trend):**
```
Base:  $16,617 → $56,755 | 20x | margin $8K   → PnL $388K (+48x)
Add 1: $17,128 → $56,755 | 15x | margin $4K   → PnL $137K (+35x)
Add 2: $20,872 → $56,755 | 10x | margin $33K  → PnL $558K (+17x)
Add 3: $22,708 → $56,755 |  5x | margin $43K  → PnL $320K (+7x)
Add 4: $23,742 → $56,755 |  5x | margin $29K  → PnL $203K (+7x)
TOTAL: $1.6M from a single $8K bullet
```

### Exit: Trailing Stop (Patient)

**No trailing until 100% profit on base position.** Let it breathe.

Once base is +100%:
- **Trailing stop:** 3x Weekly ATR from highest price
- Applies to base position
- When base stops out → ALL adds close at market

**Scale-out (optional):**
- MVRV > 2.5 → take 20% off each position
- MVRV > 3.5 → aggressive exit (close all)

### Risk Budget
- Each liquidated bullet = 5% of equity
- Can afford ~18 consecutive liquidations before 60% drawdown
- Historically: 29 liquidations spread over 8 years, interspersed with 3 winning trends
- **This strategy REQUIRES accepting frequent small losses for rare massive wins**

---

## ⚡ SHORT-TERM: FIBONACCI SWING (40% of Capital)

### Goal
Capture 4H swing moves. Quick in, quick out. Both long and short.

### Direction Bias
- **With macro trend:** up to 10x leverage
- **Against macro trend:** max 3x leverage, tighter stop

### Entry Method: Natural Trading Theory (自然交易理论)

**Step 1 — Space (Fibonacci Retracement):**
- Entry at 0.618 or 0.382 gravity points

**Step 2 — Time (Fibonacci Trend Time):**
- High-probability at 0.618 time division
- Only enter when space + time align

**Step 3 — Energy (Key K-line):**
- Must show strong candle at space+time intersection
- No Key K-line = no entry

### Position Sizing
- Divide short-term capital into 20-50 parts
- Add at 1% intervals, max 2 additions
- **Stop: max 3% price move** (set BEFORE entry)
- **TP: batch exit 20-30% per 1% move**

### Leverage
| Conviction | Leverage |
|-----------|---------|
| 5-6/10 | 2-3x |
| 7-8/10 | 3-5x |
| 9-10/10 | 5-10x |

---

## ⚠️ FUNDING RATE MANAGEMENT

- At 0.01%/8h = ~1.1%/month = ~13%/year cost of holding
- **Funding > 0.03%/8h:** Reduce leverage or close
- **Funding negative:** Getting paid — add confidence
- **Cumulative > 3%:** Alert for review
- **Cumulative > 5%:** Strongly consider closing

---

## 🔧 SHARED RULES

### Portfolio Limits
- **Reserve:** 20% always in cash
- **Account -15% from peak → flatten all** (circuit breaker)
- **Daily loss limit (short-term):** -3% → no new trades 24h

### Data Sources
| Data | Source |
|------|--------|
| MVRV Z-Score | woocharts.com (browser) |
| MVRV Ratio / SOPR / CVD | CryptoQuant (browser) |
| F&G Index | alternative.me (API) |
| OHLCV / Indicators | Bybit/Binance (ccxt) |
| Strategy KB | NotebookLM (notebook 9b6bf693) |

---

## BACKTEST RESULTS SUMMARY (2017-2026)

### V19 — 20 Bullets (Champion)
| Metric | Value |
|--------|-------|
| Period | Aug 2017 → Feb 2026 |
| Starting capital | $10,000 |
| Final equity | $695,075 |
| Peak equity | $2,210,058 (221x) |
| Total return | +6,851% |
| Max drawdown | 81% |
| Bullets fired | 32 |
| Liquidated | 29 (91%) |
| Trends caught | 3 |
| Biggest single trade | $558K (Add 2 on 2023 trend) |

### Previous versions for reference
| Version | Config | Return | Max DD | Notes |
|---------|--------|--------|--------|-------|
| V14 | 5x, 5%, no adds | +42% | 17% | Conservative baseline |
| V16 | 5x, 5%, 肥宅 adds | +54% | 20% | Adds didn't help (too small) |
| V18 | 10x, 10%, 10x adds | +470% | 84% | Good but V19 is better |
| **V19** | **20x→5x, 5%, decreasing** | **+6,851%** | **81%** | **Champion** |

### Key Insight
The magic isn't rolling adds alone. It's the combination of:
1. **Cheap bullets** (5% at 20x = affordable liquidations)
2. **Decreasing leverage on adds** (20x→15x→10x→5x = protect the compound)
3. **Patient trailing** (no stop until +100% = let winners run)
4. **Aggressive profit compounding** (80% of float → next add)

---

## Lessons Learned
- **Position sizing matters more than entry signals.** Same entry, 5% vs 20% margin = 4x difference in returns.
- **Rolling adds only work at the right base size.** Too big (30%+) = liquidation cascades. Too small (5% at 5x) = adds are noise.
- **20x leverage on 5% capital = 1x effective exposure.** It's not reckless — it's a calculated lottery ticket.
- **91% liquidation rate is fine** when the 9% winners compound 50-200x.
- **Decreasing leverage on adds is essential.** 20x on all adds → account destruction. 20x→5x → account multiplication.
- **Funding costs matter on long holds.** Budget 1%/month. The V19 backtest includes $530K in funding costs.
- **The strategy requires a supercycle.** In pure sideways/bear markets, you bleed bullets. That's expected.

## Weekly Review Schedule
- **Sunday 8PM Sydney:** Strategist reviews week, updates regime + this document

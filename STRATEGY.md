# BTC/USDT TRADING STRATEGY
*Last updated: 2026-02-21 by Canon*

## Current Regime: ACCUMULATION
Post-capitulation base-building. MVRV Z-Score: -0.86 (extreme undervalue).

---

## DUAL-TIMEFRAME SYSTEM

Two independent strategies running on separate capital allocations:

| Category | Timeframe | Capital | Max Leverage | Duration |
|----------|-----------|---------|-------------|----------|
| **Long-Term Trend** | Weekly | 60% | 50x | Weeks to years |
| **Short-Term Trend** | 4H | 40% | 10x | Hours to days |

**Rules:** Each category manages its own P&L, stops, and positions independently. Never mix capital between categories.

---

## 📈 LONG-TERM TREND (Weekly Charts)

### Goal
Catch major market cycle moves. Hold for weeks, months, or even years. Compound via rolling positions.

### Entry Rules
- **MACD Divergence on WEEKLY:** Fast 13, Slow 34, histogram only. Double divergence minimum.
- **Trend Filter:** Long only above Weekly 50 SMA. Short only below Weekly 50 SMA.
- **Bull Market Filter (200 SMA):** When weekly price > 200 SMA → **NO SHORTS**.
- **MVRV Z-Score confirmation:** Z < 0 = strong long signal. Z > 5 = distribution/exit.
- **123 Rule as leverage booster:** If confirmed → increase leverage tier.

### Leverage Tiers
| Conviction | Base Leverage | With 123 Rule |
|-----------|-------------|--------------|
| Normal setup | 5x | 10x |
| Double divergence + bottom zone | 10x | 25x |
| Triple divergence + MVRV < 0 + 123 confirmed | 25x | **50x** |

**Hard cap: 50x.** Only at extreme conviction with multiple confirmations.

### Rolling Positions (滚仓) — The Core Strategy
- **When:** MVRV Z < 0 OR within 30% of 365-day low OR confirmed cycle bottom
- **Inverted pyramid:** Each add LARGER as trend proves itself (150% → 200% → 250% of base)
- **Add on:** Weekly pullbacks to MA30, Fib 0.5-0.618, consolidation breakouts
- **Confirm:** Big weekly candles + volume + price >2-3% beyond resistance. Fails to hold 3 weekly candles → exit add
- **Funding:** Use floating profits only for adds. Never add new capital at risk
- **Trailing stop:** 2x Weekly ATR from highest point reached
- **Each add:** Protected at breakeven (entry price)

### ⚠️ Funding Rate & Fee Management
- **Monitor weekly funding rate cost.** At 0.01% per 8h = ~1.1% per month = ~13% per year
- **If funding > 0.03% per 8h (bullish sentiment extreme):** Consider reducing leverage or hedging
- **If funding negative (bearish sentiment):** You get PAID to hold longs — increase position confidence
- **Rule:** If cumulative funding fees exceed 5% of position value, reassess hold vs. close
- **Prefer low-fee exchanges** for long-duration holds

### Exit Rules
- **No fixed TP** — ride the trend. Let trailing stop close the position
- **Regime shift to Distribution (MVRV > 5, RSI weekly > 70):** Begin scaling out 20% at a time
- **Circuit breaker:** If drawdown from peak > 25%, reassess entire position

### Regime Rules (Long-Term)
| Regime | Action | Max Leverage | MVRV Zone |
|--------|--------|-------------|-----------|
| Accumulation | Scale in + rolling | 25-50x | Z < 0 |
| Markup | Hold + add on pullbacks | 10-25x | Z 0-3 |
| Distribution | Scale out gradually | 1-5x | Z 3-5 |
| Decline | Cash or hedge | 1x short max | Z > 5 then falling |

---

## ⚡ SHORT-TERM TREND (4H Charts)

### Goal
Capture swing moves within the broader trend. Quick entries, disciplined exits.

### Entry Rules — V6 MACD Strategy (Backtested: +34.26% over 4 years)
- **MACD Settings:** Fast 13, Slow 34 (Fibonacci), histogram only
- **Long Entry:** Price new lows + MACD histogram double bottom divergence → Key K-line (dark red → light red)
- **Short Entry:** Price new highs + double top divergence → dark green → light green
- **Trend Filter (50 SMA on 4H):** Long only above, short only below
- **Bull Market Filter (200 SMA on 4H):** Price > 200 SMA → NO SHORTS
- **Filter:** Histogram peak height diff >30%. Double divergence required.

### Entry Rules — Natural Trading Theory (Fibonacci)
- **Space:** Enter at Fibonacci gravity points (0.382, 0.618)
- **Time:** Fibonacci Trend Time alignment (long-short-long rhythm)
- **Energy:** K-line volume confirmation (absolute/relative strength)
- **Entry:** All three (space + time + energy) must align
- **Timeframes:** 1H/2H for precision, 4H for confirmation

### Leverage (Short-Term)
| Conviction | Leverage |
|-----------|---------|
| 5-6/10 | 2-3x |
| 7-8/10 | 3-5x |
| 9-10/10 | 5-10x |

**Hard cap: 10x** for short-term trades.

### Stop-Loss & Exit (Short-Term)
- **Stop-loss:** ATR-based. Key K-line low minus ATR(14). **Never exceed 3% price movement.**
- **Exit option A:** Half at 1:1 R/R, trail rest
- **Exit option B (Natural Trading Theory):** Batch profit-taking — 20-30% off for every 1% move, repeated 3-5 times
- **Max 2 additions** per short-term trade (strict limit)
- **Position adds:** Only with floating profits, each smaller than previous

### Regime Rules (Short-Term)
- **Bull market (price > 200 SMA weekly):** LONG ONLY. No shorts.
- **Bear market (price < 200 SMA weekly):** SHORT ONLY. No longs.
- Trade WITH the macro trend, not against it. Short-term captures pullbacks/bounces within the larger trend.
- Short-term trades use separate capital — do NOT conflict with long-term positions.

---

## 🔧 SHARED RULES

### MVRV Z-Score (On-Chain Valuation)
- **Source:** https://woocharts.com/bitcoin-mvrv-z/
- **Current (Feb 21, 2026):** -0.86 (extreme undervalue)
- **Z < 0:** 🟢 Strong accumulation. Long-term: max aggression. Short-term: long bias.
- **Z 0-2:** ⚪ Fair value. Normal trading both timeframes.
- **Z 3-5:** 🟡 Expensive. Long-term: start scaling out. Short-term: reduce leverage.
- **Z > 7:** 🔴 Extreme. Long-term: distribution mode. Short-term: short bias.

### Trend Reversal Detection (123 Rule + 2B Rule)
- **123 Rule:** (1) Trendline break → (2) Failed retest → (3) Pivot break
- **2B Rule:** Fakeout reversal entry with tight stop
- **Usage:** Confidence booster for leverage decisions in BOTH timeframes

### Portfolio Rules
- **Reserve:** Always keep 20% in cash (USDT) — combined across both categories
- **Long-term allocation:** 60% of trading capital
- **Short-term allocation:** 40% of trading capital
- **Drawdown circuit breaker:** Account -15% from peak → flatten all, review
- **Daily loss limit (short-term only):** -3% of short-term capital → no new short-term trades for 24h

### NotebookLM Knowledge Base
- **Notebook ID:** 9b6bf693-4196-4266-ad42-6a3e21ffa33b
- **Query command:** `source ~/trading-agents-env/bin/activate && notebooklm use 9b6bf693-4196-4266-ad42-6a3e21ffa33b && notebooklm ask "<question>"`
- Agents should query for strategy clarification before trading decisions

---

## Current Positions
| Category | Pair | Side | Entry | Size | SL | TP1 | TP2 | TP3 | Status |
|----------|------|------|-------|------|-----|-----|-----|-----|--------|
| Long-term | BTC/USDT | Long | $68,164 | 0.037 BTC | $63,500 | $72,000 | $75,500 | $82,000 | Open |

## Key Levels
- **Support:** $60,000 (must hold), $63,500 (current stop)
- **Resistance:** $69,000 (23.6% Fib), $74,500 (38.2% Fib)
- **Invalidation:** Weekly close below $58,000 → regime shifts to Decline

---

## Lessons Learned
- **Backtest evolution (V1→V10):** V6 is the champion for short-term (+34.26%, 4yr). Rolling adds never triggered on 4H — needs weekly timeframe or active monitoring.
- **Shorts are dangerous** — only short in confirmed bear (price < 200 SMA).
- **Rolling works at bottoms** — best done on weekly charts with patient holds.
- **MMCrypto lesson:** 50x bottom-fishing works conceptually but needs graduated entry (V8) not flat 50x (V7).
- **Funding rates matter** for long-duration positions. Monitor weekly.
- **Dual-timeframe** prevents mixing long-term conviction with short-term noise.

## Weekly Review Schedule
- **Sunday 8PM Sydney:** Strategist agent reviews week, updates this document

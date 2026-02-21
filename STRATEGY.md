# BTC/USDT TRADING STRATEGY
*Last updated: 2026-02-21 by Canon*

---

## REGIME DETECTION (Bull vs Bear)

Use multiple on-chain and technical signals to determine market regime:

### Primary Signals
| Signal | Bull | Bear | Source |
|--------|------|------|--------|
| **MVRV Z-Score** | > 0 | < 0 | https://woocharts.com/bitcoin-mvrv-z/ |
| **MVRV Ratio** | > 1.5 | < 1 | https://cryptoquant.com (free) |
| **Weekly 200 SMA** | Price above | Price below | Chart |
| **365-day MA** | Price above | Price below | Chart (key bear market resistance) |
| **Fear & Greed** | > 40 | < 25 | alternative.me |
| **Ichimoku Kumo (Weekly)** | Bullish cloud twist | Bearish cloud twist | Chart (Senkou Span A/B cross) |

### Secondary Signals
| Signal | Bull | Bear | Source |
|--------|------|------|--------|
| **SOPR** | > 1 (selling at profit) | < 1 (selling at loss) | CryptoQuant |
| **Spot Taker CVD** | Buy Dominant | Sell Dominant | CryptoQuant |
| **Stablecoin Supply Ratio** | Low (buying power) | High | CryptoQuant |
| **Whale Order Size** | Big Whale Orders | Retail Orders | CryptoQuant |
| **CQ Bull-Bear Cycle** | Bull phase | Bear phase | https://cryptoquant.com/community/dashboard/66934cb3840887109d27d4a3 |
| **Exchange Inflows** | Low (accumulation) | High from large holders (distribution) | CryptoQuant |

### Regime Rules
- **Bull confirmed:** 3+ primary signals agree = BULL
- **Bear confirmed:** 3+ primary signals agree = BEAR
- **Transitional:** Mixed signals = reduce exposure, tighten stops

### Current Status (Feb 21, 2026)
- MVRV Z-Score: **-0.86** (BEAR)
- MVRV Ratio: **1.24** (neutral — above 1 but below 1.5)
- Weekly 200 SMA: **Below** (BEAR)
- 365-day MA (~$101K): **Far below** (BEAR)
- F&G: ~7 Extreme Fear (BEAR)
- Ichimoku Kumo (Weekly): **Bearish twist** since late 2025 (BEAR)
- CQ Bull-Bear Cycle: **Bear phase** since Oct 2025 (BEAR)
- Spot Taker CVD: Buy Dominant (BULL signal)
- Exchange Inflows: Large holders distributing (BEAR)
- **Regime: BEAR — but drawdown diminishing each cycle (75%→81%→74%→30% so far). Late-stage capitulation likely.**

---

## DUAL-TIMEFRAME SYSTEM

| Category | Timeframe | Capital | Max Leverage | Duration |
|----------|-----------|---------|-------------|----------|
| **Long-Term Trend** | Weekly | 60% | 50x | Weeks to years |
| **Short-Term Trend** | 4H | 40% | 10x | Hours to days |

---

## 📈 LONG-TERM TREND (Weekly Charts)

### Goal
Catch major cycle moves. Hold weeks to years. Compound via rolling.

### Entry Conditions (ANY one is sufficient to enter)
1. **MVRV Z-Score < 0** — alone is enough (historically always a good buy)
2. **Price within 30% of 365-day low + F&G < 25** — fear + discount combo
3. **Weekly MACD 13/34 divergence** — single divergence sufficient (histogram bottoming)
4. **Price reclaims 20-week SMA from below** — trend reversal confirmation

**Entry:** When ANY condition is met, open long position.
**More signals aligning = higher leverage tier.**

### Leverage Tiers (Time-Based Scaling)
| Time Profitable | Leverage | Action |
|----------------|---------|--------|
| Entry | 5x | Test position (5% of capital) |
| 2 weeks profitable | 10x | Add 150% of base |
| 1 month profitable | 25x | Add 200% of base |
| 3+ months profitable | **50x** | Add 250% of base |

**Only increase leverage when position is profitable.** If position goes negative, hold current level — never add to a loser.

### Rolling Positions (滚仓)
- **Trigger:** Position is +5% profitable AND regime still bearish/accumulation
- **Inverted pyramid:** 150% → 200% → 250% of base size on each add
- **Add on:** Weekly pullbacks to MA30, Fib 0.5-0.618, consolidation breakouts
- **Confirm:** Big weekly candles + volume + price >2-3% beyond resistance
- **Funding:** Floating profits AND/OR additional capital. Can add new funds to increase position.
- **Safety:** Run `scripts/rolling_calculator.py` before every add. Liquidation price must stay below previous swing low.
- **Each add:** Protected at breakeven (entry price)
- **Master trailing stop:** 2x Weekly ATR from highest point

### Exit Rules
- **No fixed TP** — ride the trend via trailing stop
- **Scale out when regime shifts to Bull:** MVRV > 3, F&G > 75 → take 20% off at a time
- **MVRV > 3.7:** Begin aggressive distribution
- **Circuit breaker:** Drawdown > 25% from peak → reassess

### ⚠️ Funding Rate & Fee Management (CRITICAL for long-term holds)
- **Monitor EVERY 8h funding settlement.** Log cumulative cost.
- At 0.01%/8h = ~1.1%/month = ~13%/year — this is the cost of holding
- **Funding > 0.03%/8h:** Reduce leverage or consider closing/hedging
- **Funding negative:** Getting PAID to hold — increase confidence, add to position
- **Cumulative funding > 3% of position value:** Alert Master for review
- **Cumulative funding > 5% of position value:** Strongly consider closing or reducing
- **Track all fees:** Entry/exit commissions + funding = total cost of trade. Must be factored into P&L.
- **Long holds (months+):** Funding cost can exceed the trade profit. Always compare unrealised gain vs cumulative funding paid.

---

## ⚡ SHORT-TERM TREND (4H Charts)

### Goal
Capture swing moves within the macro trend. Quick in, quick out.

### Direction Rule
- **No hard direction restriction.** Trade both long and short based on MACD/Fibonacci signals.
- **Leverage bias:** Trading WITH macro trend → up to 10x. Trading AGAINST → max 3x, tighter stop.
- Regime determined by the signals table above (3+ primary must agree).

### Entry Methods

**Method A: MACD Reversal (V6 — backtested +34.26%)**
- MACD 13/34 histogram, double divergence minimum on 4H
- Trend filter: 4H 50 SMA direction
- Stop-loss: ATR(14) based. Never exceed 3%.
- Exit: Half at 1:1 R/R, trail rest

**Method B: Natural Trading Theory (Fibonacci)**
- Space: Fibonacci gravity points (0.382, 0.618)
- Time: Fibonacci Trend Time alignment
- Energy: K-line volume confirmation
- Entry: All three align
- Timeframe: 1H/2H for precision, 4H confirmation
- Stop-loss: Never exceed 3%
- Exit: Batch profit-taking (20-30% per 1% move, 3-5 times)

### Leverage (Short-Term)
| Conviction | Leverage |
|-----------|---------|
| 5-6/10 | 2-3x |
| 7-8/10 | 3-5x |
| 9-10/10 | 5-10x |

**Hard cap: 10x.** Max 2 additions per trade.

---

## 🔧 SHARED RULES

### Portfolio
- **Reserve:** Always keep 20% in cash (USDT)
- **Long-term:** 60% of trading capital
- **Short-term:** 40% of trading capital
- **Drawdown breaker:** Account -15% from peak → flatten all
- **Daily loss limit (short-term):** -3% → no new trades 24h

### Risk Management (McDowell Rules)
- **2% max risk per trade** (margin at risk)
- **Stop-loss set BEFORE entry** — mandatory
- **Divide account into 20-50 parts** for position sizing
- **Track win rate + R/R** to ensure Risk of Ruin = 0
- **Trading journal:** agent_memory.json logs all decisions

### Data Sources (Free)
| Data | Source | How |
|------|--------|-----|
| MVRV Z-Score | woocharts.com | Browser scrape |
| MVRV Ratio | CryptoQuant | Browser scrape (1.2399 visible) |
| SOPR / CVD / Leverage | CryptoQuant | Browser scrape |
| F&G Index | alternative.me | API (free) |
| OHLCV / Indicators | Bybit/Binance | ccxt (free) |
| Funding Rate | Bybit | ccxt (free) |
| MVRV Proxy | CoinGecko | API (free, in pipeline) |
| Strategy KB | NotebookLM | notebooklm CLI |

### NotebookLM Knowledge Base
- **Notebook ID:** 9b6bf693-4196-4266-ad42-6a3e21ffa33b
- **Query:** `source ~/trading-agents-env/bin/activate && notebooklm use 9b6bf693-4196-4266-ad42-6a3e21ffa33b && notebooklm ask "<question>"`

---

## Current Positions
| Category | Pair | Side | Entry | Size | SL | TP | Status |
|----------|------|------|-------|------|-----|-----|--------|
| Long-term | BTC/USDT | Long | $67,980 | 0.7 BTC | Cross margin | Trail 2x wATR | Test position (5%) |

**Capital:** ~$170K USDT (demo). Long-term: $102K (60%). Short-term: $68K (40%).

---

## Lessons Learned
- **Backtest V1→V11:** V6 MACD on 4H = best short-term (+34.26%). Weekly signals too rare for frequent trading but perfect for long-term holds.
- **Rolling positions** need weekly timeframe + patience. Never triggered on 4H backtests.
- **Shorts are dangerous** in bull markets. Always trade WITH the macro trend.
- **MMCrypto 50x bottom-fishing** only works with graduated leverage (V8), not flat 50x.
- **Funding rates** are a hidden tax on long-duration leveraged positions. Monitor weekly.
- **Regime detection** is more important than entry timing. Right direction > perfect entry.

## Weekly Review Schedule
- **Sunday 8PM Sydney:** Strategist agent reviews week, updates regime + this document

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
| **Fear & Greed** | > 40 | < 25 | alternative.me |

### Secondary Signals
| Signal | Bull | Bear | Source |
|--------|------|------|--------|
| **SOPR** | > 1 (selling at profit) | < 1 (selling at loss) | CryptoQuant |
| **Spot Taker CVD** | Buy Dominant | Sell Dominant | CryptoQuant |
| **Stablecoin Supply Ratio** | Low (buying power) | High | CryptoQuant |
| **Whale Order Size** | Big Whale Orders | Retail Orders | CryptoQuant |

### Regime Rules
- **Bull confirmed:** 3+ primary signals agree = BULL
- **Bear confirmed:** 3+ primary signals agree = BEAR
- **Transitional:** Mixed signals = reduce exposure, tighten stops

### Current Status (Feb 21, 2026)
- MVRV Z-Score: **-0.86** (BEAR)
- MVRV Ratio: **1.24** (neutral — above 1 but below 1.5)
- Weekly 200 SMA: **Below** (BEAR)
- F&G: ~7 Extreme Fear (BEAR)
- Spot Taker CVD: Buy Dominant (BULL signal)
- **Regime: BEAR / Late Capitulation → potential Accumulation**

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

### Entry Conditions (need 2+ to enter)
1. **MVRV Z-Score < 0** — generational accumulation zone
2. **MVRV Ratio < 1** — market cap below realized cap (holders underwater)
3. **Weekly MACD 13/34** divergence (single or double) on histogram
4. **123 Rule** trendline break confirmed
5. **F&G < 15** — extreme fear / capitulation
6. **SOPR < 1** — coins moving at a loss (capitulation selling)

**Entry:** When 2+ conditions met, open long position.
**More conditions = higher leverage tier.**

### Leverage Tiers
| Conditions Met | Leverage |
|---------------|---------|
| 2 conditions | 5x |
| 3 conditions | 10x |
| 4 conditions | 25x |
| 5-6 conditions | **50x** |

### Rolling Positions (滚仓)
- **Trigger:** Position is +5% profitable AND regime still bearish/accumulation
- **Inverted pyramid:** 150% → 200% → 250% of base size on each add
- **Add on:** Weekly pullbacks to MA30, Fib 0.5-0.618, consolidation breakouts
- **Confirm:** Big weekly candles + volume + price >2-3% beyond resistance
- **Funding:** Floating profits only. Never add new capital.
- **Each add:** Protected at breakeven (entry price)
- **Master trailing stop:** 2x Weekly ATR from highest point

### Exit Rules
- **No fixed TP** — ride the trend via trailing stop
- **Scale out when regime shifts to Bull:** MVRV > 3, F&G > 75 → take 20% off at a time
- **MVRV > 3.7:** Begin aggressive distribution
- **Circuit breaker:** Drawdown > 25% from peak → reassess

### ⚠️ Funding Rate Management
- Monitor funding rate weekly. At 0.01%/8h ≈ 13%/year
- **Funding > 0.03%/8h:** Reduce leverage or hedge
- **Funding negative:** Getting PAID to hold — increase confidence
- **Cumulative funding > 5% of position:** Reassess hold vs. close

---

## ⚡ SHORT-TERM TREND (4H Charts)

### Goal
Capture swing moves within the macro trend. Quick in, quick out.

### Direction Rule
- **Bull regime:** LONG ONLY. No shorts.
- **Bear regime:** SHORT ONLY. No longs.
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
| — | — | — | — | — | — | — | No open positions |

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

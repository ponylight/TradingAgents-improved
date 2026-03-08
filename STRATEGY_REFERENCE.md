# LabSpeculation Trading Strategies (投机实验室)
Source: https://x.com/LabSpeculation
Scraped: 2026-03-08

## 🇯🇵 BNF (小手川隆) — Japan's Trading God
**Source:** Feb 28 article (128K views)
**Track Record:** ¥1.6M → ¥19B (~$130M), bought multiple buildings in Tokyo

### Contrarian Strategy (逆势交易法) — Bear Markets
- Focus: 25-day MA deviation rate (乖离率)
- When deviation is significantly negative (price far below 25MA), stock is undervalued → buy for bounce
- Example: Stock 25MA = ¥100, current = ¥80 → deviation = -20% → buy
- Different stocks/sectors have different deviation thresholds
- Large caps vs small caps have different baseline deviations
- **Key rule: Never use leverage during capital accumulation phase**
- Hold period: 1 day to 1 week (short-term swing)

### Trend Following Strategy (顺势交易法) — Bull Markets  
- "Two days, one night" strategy: buy today, sell tomorrow morning
- Hold 20-50 stocks simultaneously for diversification
- **Sector rotation/laggard play:** If 1 of 4 steel companies rallies, buy the other 3 laggards
- Look for sector momentum spillover
- Quick profit-taking or stop-loss at market open next day

### Applicability to BTC:
- 25MA deviation for mean reversion entries ✅ can implement
- Sector laggard concept → could apply to BTC vs ETH/SOL relative strength
- No leverage rule during accumulation → aligns with Master's Phase 1

---

## 🇰🇷 Hong Inki — Korea's Hottest Trader
**Source:** Mar 6 article (17K views)
**Track Record:** ₩150K (~$100) → ₩1B (~$700K) in crypto, then stocks

### Theme-Leading Stock Strategy (主导主题龙头策略)
1. **Find the dominant theme:** Look at top 15 stocks by volume with >5% gain. If multiple belong to same theme → that's the day's dominant theme
2. **Only trade the #1 stock** (highest gain) in the theme — never the #2 or #3
   - #1 has more momentum, more attention, harder to dump
   - When #1 hits limit-up, people sell #2/#3 to chase #1 → dangerous
3. **Entry: First breakout above 6-month (or all-time) high** with massive volume candle >10%
4. **Timing:** Concentrate trades within 2 days of the initial breakout candle
5. **Exit rules (Day 3+):**
   - Doji or inverted hammer → prepare to exit
   - Drops below opening price by 1% → consider exit
   - Drops 3% → mandatory exit
   - Rises 5%+ → mandatory profit-take

### Applicability to BTC:
- Volume breakout above key highs ✅ partially in Green Lane
- The "only trade #1" concept → reinforces BTC-only focus vs altcoins
- Strict day-3 exit discipline → useful for short-term trades
- Mostly equity-focused but the breakout mechanics apply

---

## 🇨🇳 半木夏 — Cycle + Liquidity + MACD Triple Divergence
**Source:** Feb 11 article (148K views) — Crypto-specific, retired 2023, briefly resurfaced

### MACD Triple Divergence Strategy
**Parameters:** Fast=12, Slow=26, Signal=9 (standard)

**Triple Top Divergence (Bearish):**
- Price makes 3 successive highs
- MACD histogram (green bars) shrinks twice consecutively
- Each histogram segment must have opposite-color bars between them
- Aligns with Elliott Wave 5-wave completion

**Triple Bottom Divergence (Bullish):**
- Price makes 3 successive lows
- MACD histogram (red bars) shrinks twice consecutively

**Entry:** At 8 AM (daily close) when 3rd divergence confirmed + histogram shrinkage confirmed
**Stop Loss (Short):** Above the 3rd peak high
**Stop Loss (Long):** Below the 3rd trough low
**Critical Rule:** If next day's close doesn't show continued histogram shrinkage → immediate stop loss
**Take Profit:** Use wave theory or other TA; effective for 15-50 candles at the given timeframe
**Best Timeframes:** Daily, 4H (1H too noisy, only 15-50 hours effective)
**Frequency:** ~1-2 times per year on daily timeframe

### Macro Framework
1. **Cycle Analysis:** BTC has exited traditional 4-year cycle → currently in bear phase
   - Old altcoins pumping = late-cycle signal
2. **Liquidity Analysis:** Watch US liquidity (SOFR spread, Fed policy)
   - Tight liquidity → bearish BTC/stocks
3. **Technical (Elliott Wave):** BTC in large Wave 4 correction
   - If bottoms ~$80K → Wave 5 target could be $240K
   - Wave 4 typically sideways (especially when Wave 2 was sharp)

### Bubble Top Indicators (for future use):
1. Mega-premium M&A deals appearing
2. Inflation rising + Fed tightening expectations growing
3. Any AI-related stock pumps regardless of fundamentals

### Applicability to BTC:
- MACD Triple Divergence ✅ **HIGHLY implementable** — clear rules, works on daily/4H
- Already have 半神MACD in our indicator toolkit
- Macro framework aligns with our CII/macro_radar approach
- Bubble indicators useful for long-term position management

---

## 🇨🇳 比特皇 (Bitcoin King) — $1,500 → $24M
**Source:** Mar 3 article (193K views, 540 bookmarks!) — Pure crypto, BTC-focused

### Seven Key Trading Points:

**1. Define Your Trading Timeframe**
- Small timeframe = short trades, don't hold long (profits evaporate)
- Large timeframe = wider SL/TP, prepare for longer holds
- **NEVER mix:** short-term trade held long, or long-term trade cut short

**2. Three High-Probability Patterns:**
- **Ascending/Descending Triangles:** Consolidation pattern showing which side (bull/bear) is retreating
- **Double-Bottom Reversal (HH + HL):** After big drop, if 2nd bounce (B) exceeds 1st bounce (A), and 2nd dip (D) stays above 1st dip (C) → higher high + higher low = reversal likely
  - If B < A → likely just a bear flag continuation
- **Slope + Volume exhaustion:** Declining slope + declining volume = selling exhaustion = bottoming

**3. Position Sizing & Pyramiding:**
- Uncertain → split entries across small range (no adding)
- High conviction → full position at once
- **Only add in profit, never in loss**
- Additions ≤ 1x original position size
- Set stop loss on additions too

**4. Stop Loss First, Then Profit:**
- Before entering: calculate distance to SL, ensure it's within tolerance
- After 3 consecutive stop-outs → **mandatory 1-day rest**
- Progressive sizing after stops: 30% → 32% → 35% (recover losses if next trade wins)
- Three TP methods: full exit, partial, or hedge

**5. Risk/Reward:**
- Small accounts need minimum **10:1 R:R** for compounding
- Wait patiently for high-R:R setups

**6. Position Size & Leverage:**
- Start small with high leverage (20x), gradually reduce as account grows
- At ¥100K-300K: max 10x
- At millions: max 5x
- At tens of millions: ~3x, using only 10-30% of capital
- **Compounding requires reducing leverage as capital grows** (one mistake resets everything)

**7. Psychology:**
- Major drawdowns caused by: overconfidence after wins (心浮) or tilt after consecutive losses (气躁)
- 3 consecutive stops = mandatory 1-day break
- Goal: trade like a machine, no emotions

**8. Backtesting/Review (复盘):**
- He memorized every BTC chart segment from 2019 onwards
- "Read charts like reading books — the more you review, the better your instinct"
- Pattern recognition through massive repetition → develops 盘感 (market sense)

### Applicability to BTC:
- HH+HL reversal detection ✅ already partially in trend reconciliation
- Triangle pattern detection → could add to Green Lane
- Volume exhaustion + slope decline → implementable
- Progressive sizing after stops → could add to executor
- Mandatory rest after 3 stops → **should implement as a circuit breaker**
- 10:1 R:R minimum for small accounts → relevant for Phase 1
- Leverage scaling with account size → Phase 2

---

## 🇯🇵 BNF + CIS Combined Trend Strategies  
**Source:** Feb 11 article (203K views)
- CIS = "Strongest retail trader in Japan"
- Both friends, similar backgrounds
- CIS strategy details in separate article (need to read)

---

## 🇨🇳 Steven Dux (杜修贤) — Short Selling Master
**Source:** Feb 18 article (24K views)
- Born 1994, Chongqing → moved to US
- Specializes in shorting junk/pump-and-dump stocks
- Process for identifying "garbage" stocks to short
- More applicable to US penny stocks than BTC

---

## 🇨🇳 退神 — Small Account Growth
**Source:** Feb 8 article (216K views, 429 bookmarks)
- ¥10,000 → ¥10M in 2 years
- Dropped out of school to trade
- Strategy details in the article (likely momentum/breakout)

---

## Priority for Implementation (BTC System)

### Phase 1 — Quick Wins:
1. **MACD Triple Divergence (半木夏)** — Clear rules, works on daily/4H, can add to analyst toolkit
2. **25MA Deviation Mean Reversion (BNF)** — Simple contrarian signal for extreme conditions  
3. **3-Stop Circuit Breaker (比特皇)** — Mandatory rest after 3 consecutive losses
4. **HH+HL Reversal Detection (比特皇)** — Already partial, enhance it

### Phase 2 — Structural:
5. **Leverage Scaling (比特皇)** — Auto-reduce leverage as account grows
6. **Volume Breakout + Theme Momentum (Hong Inki)** — Enhance Green Lane with volume confirmation
7. **Macro Cycle Framework (半木夏)** — SOFR spread, liquidity monitoring

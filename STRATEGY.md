# BTC/USDT TRADING STRATEGY
*Last updated: 2026-02-20 by Strategist Agent*

## Current Regime: ACCUMULATION
Post-capitulation base-building. Scale in cautiously.

## Regime Rules
| Regime | Action | Max Leverage | Max Exposure |
|--------|--------|-------------|-------------|
| Accumulation | Scale in on dips | 1-2x | 40% of account |
| Markup | Add to winners | 2-3x | 60% of account |
| Distribution | Take profits | 1x | 20% of account |
| Decline | Cash or short | 1x short | 20% of account |

## Regime Transition Signals
- Accumulation → Markup: Price above 50 SMA + F&G above 30 + whale accumulation confirmed
- Markup → Distribution: RSI >70 + F&G >75 + divergences forming
- Distribution → Decline: Price below 50 SMA + death cross + rising volume on red days
- Decline → Accumulation: Capitulation candle + F&G <15 + funding neutral

## Portfolio Rules
- **Reserve:** Always keep 30% in cash (USDT)
- **Max single position:** 40% of account
- **Max leverage:** Per regime table above
- **Drawdown circuit breaker:** Account -10% → flatten all, review strategy
- **Daily loss limit:** -3% → no new trades for 24h

## Conviction Scaling
| Confidence | Allocation | Leverage |
|-----------|-----------|---------|
| 5-6/10 | 10% | 1x |
| 7-8/10 | 25% | 1-2x |
| 9-10/10 | 40% | 2-3x |

## Time Horizons
- **Swing trade (default):** 1-4 weeks, 4h chart decisions
- **Position trade:** 1-6 months, daily chart, macro-driven
- **Day trade:** Only on extreme events (>5% intraday move)

## Current Positions
| Pair | Side | Entry | Size | SL | TP1 | Status |
|------|------|-------|------|-----|-----|--------|
| BTC/USDT | Long | $68,164 | 0.037 BTC | $63,500 | $72,000 | Open |

## Key Levels
- **Support:** $60,000 (must hold), $63,500 (stop level)
- **Resistance:** $69,000 (23.6% Fib), $74,500 (38.2% Fib)
- **Invalidation:** Weekly close below $58,000 → regime shifts to Decline

## Technical Entry Rules — V3 (Backtested: +4.67% return, 2.1% max DD, 3.26 PF)

### MACD Reversal Strategy (半木夏)
- **MACD Settings:** Fast 13, Slow 34 (Fibonacci), focus on histogram only
- **Long Entry:** Price makes new lows + MACD histogram shows **double divergence minimum** (strength ≥ 2) → enter on "Key K-line" (dark red → light red)
- **Short Entry:** Price makes new highs + MACD histogram shows double top divergence → enter on dark green → light green
- **Trend Filter (50 SMA):** Long only above 50 SMA, short only below 50 SMA
- **Bull Market Filter (200 SMA):** When price > 200 SMA → **NO SHORTS**. Long only.
- **Filter:** Histogram peak height diff must be >30%. **Double divergence required** (single = skip)
- **Stop-loss:** Key K-line low minus ATR(14) for longs, high plus ATR(14) for shorts
- **Exit:** Half position at **1:1 R/R**, hold rest with trailing stop

### Trend Reversal Detection (123 Rule + 2B Rule)
- **123 Rule:** (1) Break the trendline → (2) Failed retest of previous high/low → (3) Break of previous pivot point
- **2B Rule:** Price breaks a level but immediately reverses back = fakeout entry with tight stop
- **High-probability trends** start after long horizontal consolidation or when volatility hits new lows

### Rolling Positions (滚仓) — Only Near Market Bottoms
- **When:** Only when price is within 30% of 365-day low (near bottom accumulation)
- **Never roll** in ranging or mid-trend markets
- **Add on:** Consolidation breakouts (triangles) or pullbacks to MA30 / Fib 0.5-0.618
- **Confirm breakout:** Big candles + volume + price >2-3% beyond resistance. If fails to hold 3 candles → exit immediately
- **Sizing:** Start with 5-10% test position at 2-3x max. Pyramid adds using **floating profits only**. Each add smaller than previous
- **Must use:** Trailing stop-losses (ATR-based) on all rolling positions

### NotebookLM Knowledge Base
- **Notebook ID:** 9b6bf693-4196-4266-ad42-6a3e21ffa33b
- **Query command:** `source ~/trading-agents-env/bin/activate && notebooklm use 9b6bf693-4196-4266-ad42-6a3e21ffa33b && notebooklm ask "<question>"`
- Agents should query this notebook for strategy clarification before making trading decisions

## Lessons Learned
*(Updated weekly by Strategist)*
- First trade: entered during Extreme Fear (7/100). Historical edge is strong but timing uncertain.
- **Backtest V1→V3 evolution:** Raw MACD divergence lost 24.2% over 4 years. Adding trend filters (50/200 SMA), double divergence requirement, 1:1 TP, no bull market shorts, and bottom-only rolling turned it profitable (+4.67%, 2.1% DD, 3.26 profit factor). Quality over quantity — 7 trades beat 220.
- **Shorts are dangerous** — V1 shorts lost $1,299. Only short in confirmed bear (price < 200 SMA).
- **Rolling works at bottoms** — rolling adds were the only profitable part of V1. Restricting to near 365-day lows makes it even better.

## Weekly Review Schedule
- **Sunday 8PM Sydney:** Strategist agent reviews week, updates this document

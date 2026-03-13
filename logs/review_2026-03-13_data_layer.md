# Data Layer & Analyst Accuracy Review — 2026-03-13

## Executive Summary

Full audit of all data fetching modules, analyst prompts, tool bindings, and pipeline efficiency. Found **8 CRITICAL**, **13 HIGH**, **14 MEDIUM**, and **7 LOW** severity issues across the stack.

**Key themes:**
1. Analyst prompts contain contradictory instructions and reference non-existent data fields
2. Several tools are imported but never bound to analysts (dead code / silent failures)
3. ~50+ API calls per cycle with significant redundancy (Fear & Greed fetched 3x, hash rate 2x)
4. Previous calculation fixes (VWAP z-score, Stochastic K/D, volume/MA ratio) are confirmed correct

---

## 1. Data Source Audit

### 1.1 Previously Fixed Calculations — All Verified Correct

| Fix | File | Lines | Status |
|-----|------|-------|--------|
| VWAP z-score (was -10σ) | crypto_technical_brief.py | 367-376 | ✅ Correct — uses residuals with [-5,+5] clamp |
| Stochastic K/D (K=0.59, D=7.93) | crypto_technical_brief.py | 44-51 | ✅ Correct — K is smoothed %K, D is SMA of K |
| Volume/MA ratio (was 0.03) | crypto_technical_brief.py | 403-430 | ✅ Correct — fallback to median, guard against <0.05 |

### 1.2 Issues Found in Data Modules

#### MEDIUM — OI Time-Series Period Calculation Off for Short Spans
**File:** `ccxt_crypto.py:354-358`

```python
hours_span = (oi_points[0][2] - oi_points[-1][2]) / 1000 / 3600
periods_4h = max(hours_span / 4, 1)
rate_per_4h = total_change_pct / periods_4h
```

If data spans <4 hours, `periods_4h` floors to 1 regardless of actual interval count. Rate-per-4H becomes unreliable for intra-session data. Should use `max(len(oi_points) - 1, 1)` for data-point-based periods.

#### MEDIUM — Dual Caching Systems Don't Share State
**File:** `candle_cache.py:23-24` vs `ccxt_crypto.py:78`

- `candle_cache.py` keys on `(symbol, timeframe, since_ms, limit)`
- `ccxt_crypto.py` keys on `(symbol, exchange, timeframe, start_date, end_date)`
- Overlapping date ranges cause re-fetches instead of cache hits

#### MEDIUM — Completeness Check Missing Anomaly Detection
**File:** `completeness.py:9-62`

Checks staleness, gaps, zero volume, frozen feed — but misses backwards timestamps, extreme price moves (>50% single bar), and volume spikes (>100x normal).

#### MEDIUM — Geopolitical News Atom Namespace Dead Code
**File:** `geopolitical_news.py:99-143`

Atom XML parsing code exists but GDELT returns JSON only. The Atom fallback path is unreachable. Confusing but non-breaking.

#### LOW — Pattern Scanner Cache Key Missing Date Range
**File:** `pattern_scanner.py:683-782`

Cache key is just `symbol` with 300s TTL. Same symbol at different times returns stale 250-day data within TTL window.

#### LOW — Data Quality Scoring Missing Technical Bounds Checks
**File:** `data_quality.py:25-119`

No validation for NaN/inf in VWAP z-score, divide-by-zero in RSI, or Stochastic K/D bounds (0-100). Catches high-level failures but not calculation anomalies.

#### LOW — Social Sentiment Keyword-Based (By Design)
**File:** `social_sentiment.py:73-95`

Cannot detect sarcasm or coded language. Acknowledged in code (lines 230-238) with LLM overlay as mitigation.

### 1.3 Macro Data Fallbacks — Working Correctly

| Source | Primary | Fallback | Status |
|--------|---------|----------|--------|
| DXY | yfinance DX-Y.NYB | FRED DTWEXBGS | ✅ Both paths tested |
| 10Y Yield | yfinance ^TNX | FRED DGS10 | ✅ Independent tracking |
| 13W Yield | yfinance ^IRX | FRED DGS2 | ✅ Independent tracking |
| S&P 500 | yfinance ^GSPC | (none) | ✅ Single source OK |

---

## 2. Analyst Prompt Accuracy

### 2.1 crypto_market_analyst.py

#### HIGH — Data Format Not Clearly Described
**Lines 57-114:** Prompt says "Your input is a pre-computed Technical Brief (JSON)" but doesn't explain the nested structure (`timeframes[].trend`, `timeframes[].momentum.macd_cross`, etc.). LLM may reference fields that don't exist.

#### HIGH — Contradictory AVWAP Handling
**Lines 64-66:** Says AVWAP convergence makes S/R levels "unreliable" but doesn't instruct analyst to skip analysis or cap confidence score explicitly.

#### MEDIUM — Vague Momentum Divergence Instruction
**Lines 74-81:** "RSI divergence on higher TF is a strong signal" but data only provides `momentum.rsi_divergence` (boolean) without per-timeframe breakdown.

#### MEDIUM — Token-Wasting ATR Framework
**Lines 100-102:** Asks for "ATR-based stop distance" but ATR is not in the output schema. ~50 tokens wasted.

#### MEDIUM — Redundant String Operation
**Line 48:** `company_name.replace("/USDT", "/USDT")` replaces with identical string. Dead code from incomplete refactoring.

### 2.2 crypto_sentiment_analyst.py

#### CRITICAL — Tool Count Mismatch
**Line 109 vs 76-80:** Prompt says "Call all three tools" but four tools are available (get_funding_rate, get_oi_timeseries, get_open_interest, get_crypto_fear_greed). Analyst may skip one.

#### CRITICAL — Contradictory Tool Precedence
Three conflicting instructions:
- Line 78: get_oi_timeseries is "PREFERRED over get_open_interest"
- Line 81: get_open_interest is "BACKUP: Use only if get_oi_timeseries fails"
- Line 109: "Call all three tools" (implies both simultaneously)

#### HIGH — Conflicting Weighting Framework
**Lines 96-101:** "Verdict driven primarily by POSITIONING DATA" but also "weight Fear & Greed LOW" and pre-fetched social sentiment injected at lines 72-73 with no weight guidance. Ambiguous when social contradicts positioning.

#### HIGH — Pre-Fetched Data Bypasses Tool Validation
**Line 39:** Social sentiment pre-fetched via `get_social_sentiment_enhanced(llm=llm)` and injected directly. LLM doesn't call a tool for it, creating implicit data injection without sourcing transparency.

#### MEDIUM — Missing Field Name Reference for Fear & Greed
**Line 80:** Says "Do NOT anchor on Fear & Greed" without specifying which field contains the value.

### 2.3 crypto_fundamentals_analyst.py

#### CRITICAL — Macro Data Injected Without Instructions
**Lines 76-79, 183:** Code injects `macro_summary` and `stablecoin_summary` into the user message, but the system prompt (lines 112-172) never mentions macro data, "7-signal composite," or stablecoin health. Analyst receives data it wasn't instructed to analyze.

#### CRITICAL — No Interpretation Framework for Injected Data
**Lines 180-187:** User message includes `MACRO SIGNAL RADAR (7-signal composite — weight heavily)` but the prompt never explains what the 7 signals are, how to interpret `stablecoin['overall']`, or what format to expect.

#### CRITICAL — References Disabled Feature (CII)
**Lines 100-107, 189:** `get_crisis_impact_index()` is commented out with fallback `"Geopolitical risk (CII disabled)"`. This string is injected into the prompt but no instructions explain what CII is or what to do when it's disabled.

#### HIGH — Missing Weight Guidance for Macro vs Fundamentals
**Lines 163-171:** Asks for macro radar verdict but doesn't define how it should influence the fundamentals verdict. Only hint: "if macro says CASH, bias bearish."

#### HIGH — Tool Pre-Fetch Without Deduplication
**Line 51:** Calls `get_onchain_fundamentals.invoke({})` to pre-fetch, but the LLM may invoke the same tool again during its tool-calling loop. No deduplication guard.

#### MEDIUM — Staleness Warning Not Integrated into Framework
**Lines 93-98:** Warning injected when data >12h old, but the analyst's instruction framework doesn't mention staleness or how to interpret the warning.

### 2.4 news_analyst.py

#### CRITICAL — String Interpolation Logic Bug
**Line 85:** `'geo_data' in dir()` always evaluates True since geo_data is defined in the local scope (line 47). The fallback `'?'` is unreachable. If fetch fails, geo_data is still defined (empty), returning 0 instead of indicating failure.

#### CRITICAL — Template Variable Interpolation Error
**Lines 82-86:** f-string in system message tries `.get('sources_fetched', 0)` which depends on geo_data being a dict. If geo_data is malformed, this crashes at prompt construction time.

#### HIGH — Available Tools Not Mentioned in Instructions
**Lines 90-95:** System message lists 5 macro tools but omits `get_news` and `get_global_news` from the instruction text, despite both being in the tools list (line 33). Analyst may not know these tools exist.

#### HIGH — Framework Doesn't Match Role Title
**Lines 80-109:** Role is "news and macro analyst" but framework only asks for "Macro Environment" as a sub-item. No explicit macro regime classification (Risk-On / Risk-Off) requested.

#### HIGH — User Message Assumes Tool Calls Without Guarantee
**Lines 115-118:** Says "Call macro tools (DXY, yields, economic_calendar)" but nothing enforces the LLM will actually make these calls.

#### MEDIUM — No Fallback for DEGRADED Quality
**Lines 71-78:** Staleness warning only injected for STALE/FAILED quality. DEGRADED quality passes with no guidance on confidence capping.

### 2.5 macro_analyst.py

#### CRITICAL — FINAL TRANSACTION PROPOSAL Instruction Violates Analyst Role
**Lines 92-93:** System prompt tells the macro analyst to output `FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`. This is a **macro analyst**, not a trader. Instruction copied from a different role template, violating the analyst-only responsibility.

#### CRITICAL — Generic Template Overrides Role-Specific Instructions
**Lines 88-95:** Generic "helpful AI assistant" template overshadows the detailed macro analysis instructions at lines 29-82 (454 tokens of excellent macro theory, largely wasted).

#### HIGH — Tool Calls Not Executed (No Agentic Loop)
**Lines 106-107:** Uses `chain = prompt | llm.bind_tools(tools)` then `chain.invoke()` — but unlike other analysts, there is NO tool-calling loop. If LLM requests tool calls, they are returned as-is without execution. Analyst cannot actually fetch macro data.

#### HIGH — Report Extraction Assumes No Tool Calls
**Lines 109-115:** `if len(result.tool_calls) == 0: report = result.content` — only extracts content if no tools were called. If LLM tries to call tools (which won't execute), report is empty string.

#### HIGH — State Field Collision
**Line 115:** Returns `"news_report": report` with comment `# Reusing news_report slot for macro data`. This overwrites the news analyst's output if both run. Comment acknowledges the problem but doesn't fix it.

#### MEDIUM — Instructions Don't Map to Tools
**Lines 71-82:** Steps say "Check DXY trend", "Review Fed Funds Rate" but don't explicitly say "call get_dollar_index" or "call get_economic_data".

---

## 3. Tool Bindings

### 3.1 Tool Binding Matrix

| Tool | Defined In | Bound To | Status |
|------|-----------|----------|--------|
| get_crypto_price_data | crypto_tools.py | Market analyst | ✅ OK |
| get_orderbook_depth | crypto_tools.py | Market analyst | ✅ OK |
| get_crypto_fear_greed | crypto_tools.py | Sentiment analyst | ✅ OK |
| get_funding_rate | crypto_tools.py | Sentiment analyst | ✅ OK |
| get_open_interest | crypto_tools.py | Sentiment analyst | ✅ OK |
| get_oi_timeseries | crypto_tools.py | Sentiment analyst | ✅ OK |
| get_onchain_fundamentals | crypto_tools.py | Fundamentals analyst | ✅ OK |
| get_macro_signal_radar | macro_tools.py | Fundamentals analyst | ✅ OK |
| get_stablecoin_peg_health | macro_tools.py | Fundamentals analyst | ✅ OK |
| get_news | news_data_tools.py | News analyst | ⚠️ Expects ticker, gets symbol |
| get_global_news | news_data_tools.py | News analyst | ✅ OK |
| get_dollar_index | macro_tools.py | News analyst | ✅ OK |
| get_yields | macro_tools.py | News analyst | ✅ OK |
| get_sp500 | macro_tools.py | News analyst | ✅ OK |
| get_economic_data | macro_tools.py | News analyst | ✅ OK |
| get_economic_calendar | macro_tools.py | News analyst | ✅ OK |
| check_macd_divergence | crypto_tools.py | Market (internal) | ✅ OK |
| run_pattern_scan | crypto_tools.py | Market (internal) | ✅ OK |

### 3.2 Unbound / Dead Tools

#### CRITICAL — get_crisis_impact_index Missing @tool Decorator
**File:** `crypto_tools.py:180`

Function defined but NOT decorated with `@tool`. Cannot be called by any LLM tool-calling mechanism. Will produce "Unknown tool" errors. The fundamentals analyst already disables it (line 107), but the root cause is missing decorator.

#### HIGH — get_crypto_technical_indicators Imported but Never Bound
**File:** `crypto_trading_graph.py:35`

Imported but never added to any analyst's ToolNode. Market analyst uses `build_crypto_technical_brief()` (deterministic) instead. Dead import.

#### HIGH — get_liquidation_info Imported but Never Bound
**File:** `crypto_trading_graph.py:41`

Imported but not included in any ToolNode binding. Sentiment analyst has positioning tools but cannot access liquidation data.

#### HIGH — create_macro_analyst Imported but Never Wired
**File:** `crypto_trading_graph.py:25`

Imported but not added to `analyst_creators` dict (lines 238-243). Dead code — macro analysis is scattered across fundamentals pre-fetch and news analyst tools.

#### MEDIUM — get_news Expects Ticker, Receives Crypto Symbol
**File:** `news_data_tools.py:6-10`

`get_news(ticker: str, ...)` expects stock ticker (e.g., "AAPL") but crypto context passes "BTC/USDT". The vendor router may fail silently or return no results.

#### MEDIUM — get_insider_transactions Semantically Wrong for Crypto
**File:** `news_data_tools.py:42-53`

Stock insider transactions tool available to news analyst in crypto context. BTC has no "insiders" — calls waste tokens and return empty data.

#### LOW — Backtest Tools Depend on clawquant CLI
**File:** `backtest_tools.py:15-35`

Tools expect clawquant in `.venv/bin/`. If not installed, return JSON error silently. No pre-binding validation.

---

## 4. Data Pipeline Efficiency

### 4.1 API Calls Per Execution Cycle

| Source | Calls | Cached? | Notes |
|--------|-------|---------|-------|
| BTC OHLCV (1h/4h/1d) | 3 | ✅ candle_cache | Shared across modules |
| Fear & Greed Index | 3 | ⚠️ Partial | macro_radar cached (300s), ccxt_crypto and pattern_scanner not |
| Hash rate | 2 | ❌ | onchain_fundamentals + macro_radar fetch independently |
| DXY | 1 | ❌ | No caching |
| Treasury yields | 1-2 | ❌ | macro_data + macro_radar may double-fetch ^TNX |
| S&P 500 | 1 | ❌ | No caching |
| News RSS feeds | 30+ | ❌ | Rate-limited 0.3s/req |
| Social (Reddit/Twitter) | 4 | ❌ | Rate-limited 0.5s/req |
| On-chain data | 6 | ✅ 300s TTL | In-memory cache |
| Orderbook | 1 | ❌ | Real-time, no cache needed |
| **TOTAL** | **~52+** | | |

### 4.2 Redundant Fetches

| Data | Fetched By | Times/Cycle | Savings if Deduplicated |
|------|-----------|-------------|------------------------|
| Fear & Greed | ccxt_crypto, macro_radar, pattern_scanner | 3 | 2 API calls |
| Hash rate | onchain_fundamentals, macro_radar | 2 | 1 API call |
| 10Y Yield (^TNX) | macro_data, macro_radar | 2 | 1 API call |
| BTC OHLCV 1d | technical_brief, pattern_scanner | 2 | Partially cached |

### 4.3 Token Cost Estimate

Assuming GPT-4-class LLM at ~$10/1M input tokens:

| Analyst | Prompt Tokens (est.) | Tool Response Tokens (est.) | Total |
|---------|---------------------|---------------------------|-------|
| Market | ~3,000 | ~2,000 | ~5,000 |
| Sentiment | ~2,500 | ~1,500 | ~4,000 |
| Fundamentals | ~3,500 | ~2,000 | ~5,500 |
| News | ~3,000 | ~3,000 | ~6,000 |
| Research debate (2 rounds) | ~4,000 | — | ~4,000 |
| Risk debate (2 rounds) | ~3,000 | — | ~3,000 |
| Trader | ~2,000 | — | ~2,000 |
| **TOTAL per cycle** | | | **~29,500** |

At $10/1M tokens: **~$0.30/cycle** for input. With output tokens (~15,000 at $30/1M): **~$0.75/cycle total**.

### 4.4 Efficiency Recommendations

1. **Add global Fear & Greed cache** — save 2 API calls/cycle
2. **Add global hash rate cache** — save 1 API call/cycle
3. **Unify OHLCV caching** — candle_cache.py and ccxt_crypto.py should share cache keys
4. **Cache treasury yields globally** — prevent double-fetch of ^TNX
5. **Remove dead tool imports** — reduce code complexity
6. **Fix macro_analyst tool loop** — currently cannot execute any tool calls

---

## 5. Issue Summary by Severity

### CRITICAL (8)
1. `crypto_sentiment_analyst.py:109` — Tool count mismatch ("three" vs four)
2. `crypto_sentiment_analyst.py:78,81,109` — Contradictory tool precedence
3. `crypto_fundamentals_analyst.py:76-79,183` — Macro data injected without instructions
4. `crypto_fundamentals_analyst.py:180-187` — No interpretation framework for injected data
5. `crypto_fundamentals_analyst.py:107,189` — References disabled CII feature
6. `news_analyst.py:85` — String interpolation logic bug (`dir()` check always True)
7. `macro_analyst.py:92-93` — FINAL TRANSACTION PROPOSAL violates analyst role
8. `macro_analyst.py:88-95` — Generic template overrides role-specific instructions

### HIGH (13)
1. `crypto_market_analyst.py:57-114` — Data format not clearly described
2. `crypto_market_analyst.py:64-66` — Contradictory AVWAP handling
3. `crypto_sentiment_analyst.py:96-101` — Conflicting weighting framework
4. `crypto_sentiment_analyst.py:39` — Pre-fetched data bypasses tool validation
5. `crypto_fundamentals_analyst.py:163-171` — Missing weight guidance
6. `crypto_fundamentals_analyst.py:51` — Tool pre-fetch without deduplication
7. `news_analyst.py:90-95` — get_news/get_global_news not mentioned in instructions
8. `news_analyst.py:80-109` — Framework doesn't match role title
9. `news_analyst.py:115-118` — Assumes tool calls without guarantee
10. `macro_analyst.py:106-107` — Tool calls not executed (no agentic loop)
11. `macro_analyst.py:109-115` — Report extraction assumes tool calls fail
12. `macro_analyst.py:115` — State field collision (news_report reused)
13. `crypto_trading_graph.py:35,41,25` — 3 dead imports (technical_indicators, liquidation_info, macro_analyst)

### MEDIUM (14)
1. `ccxt_crypto.py:354-358` — OI time-series period calculation
2. `candle_cache.py:23` vs `ccxt_crypto.py:78` — Dual caching systems
3. `completeness.py:9-62` — Missing anomaly detection
4. `geopolitical_news.py:99-143` — Atom namespace dead code
5. `crypto_market_analyst.py:74-81` — Vague momentum divergence instruction
6. `crypto_market_analyst.py:100-102` — Token-wasting ATR framework
7. `crypto_market_analyst.py:48` — Redundant string operation
8. `crypto_sentiment_analyst.py:80` — Missing field name reference
9. `crypto_fundamentals_analyst.py:93-98` — Staleness warning not in framework
10. `crypto_fundamentals_analyst.py:138-142` — Vague "7-signal composite"
11. `news_analyst.py:71-78` — No DEGRADED quality fallback
12. `macro_analyst.py:71-82` — Instructions don't map to tools
13. `news_data_tools.py:6-10` — get_news expects ticker, gets symbol
14. `news_data_tools.py:42-53` — Insider transactions irrelevant for crypto

### LOW (7)
1. `pattern_scanner.py:683-782` — Cache key missing date range
2. `data_quality.py:25-119` — Missing technical bounds checks
3. `social_sentiment.py:73-95` — Keyword-based (by design)
4. `crypto_tools.py:180` — get_crisis_impact_index missing @tool (disabled anyway)
5. `crypto_market_analyst.py:25,40` — Pattern tools bound inside analyst
6. `backtest_tools.py:15-35` — clawquant dependency not validated
7. `macro_analyst.py:29-82` — 454 tokens of prompt bloat (shadowed by template)

---

## 6. Priority Fix Recommendations

### Immediate (next session)
1. **Fix macro_analyst.py** — add agentic tool loop, remove FINAL TRANSACTION PROPOSAL, fix state field collision
2. **Fix sentiment analyst prompt** — correct tool count, resolve tool precedence contradiction
3. **Fix fundamentals analyst prompt** — add macro radar interpretation framework, remove CII reference
4. **Fix news analyst** — fix dir() logic bug, mention get_news/get_global_news in instructions

### Short-term
5. Add global caching for Fear & Greed, hash rate, treasury yields
6. Unify OHLCV caching between candle_cache.py and ccxt_crypto.py
7. Remove dead imports from crypto_trading_graph.py
8. Fix get_news ticker/symbol mismatch for crypto context

### Medium-term
9. Improve completeness.py with anomaly detection
10. Add technical bounds checking to data_quality.py
11. Clean up geopolitical_news.py Atom dead code
12. Remove get_insider_transactions from crypto news analyst tools

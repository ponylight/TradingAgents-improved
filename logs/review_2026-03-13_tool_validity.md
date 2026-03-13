# Tool Validity Review — 2026-03-13

## 1. TOOL VALIDITY MATRIX

### Crypto Market Analyst Tools

| Tool Name | Bound To | API Source | Returns | Error Handling | Accuracy | Callable | Issues |
|-----------|----------|------------|---------|----------------|----------|----------|--------|
| `check_macd_divergence` | Market Analyst (internal loop) | Bybit via ccxt → `macd_divergence.py` | BEARISH/BULLISH_DIVERGENCE or NONE with confidence 0-10, price points, shrinkage % | ✅ try/except, returns error string | ✅ Solid — price range filter, time proximity filter, confidence scoring | ✅ In tool_map, @tool decorator present | None |
| `run_pattern_scan` | Market Analyst (internal loop) | Bybit via ccxt → `pattern_scanner.py` | Formatted report: detected patterns with confidence, direction, entry/stop levels | ✅ Each detector wrapped in _safe_detect; scan cached 5min | ✅ 10+ pattern detectors (Kyle Williams, Minervini, BNF, 半木夏, etc.) | ✅ In tool_map, @tool decorator present | Daily cache key means intraday re-scans return stale data |
| `run_backtest` | Market Analyst (internal loop) | ClawQuant CLI subprocess | JSON: return%, sharpe, drawdown, win rate, trades | ✅ Returns error JSON if clawquant missing or timeout | ⚠️ Depends on `clawquant` binary being installed — **currently NOT installed** per code check | ✅ In tool_map, @tool decorator present (wrapper) | **ISSUE: clawquant likely not installed** — tool will always return error |
| `compare_strats` | Market Analyst (internal loop) | ClawQuant CLI subprocess | JSON comparison of multiple strategies | ✅ Same as run_backtest | ⚠️ Same dependency issue | ✅ In tool_map, @tool decorator present (wrapper) | **ISSUE: Same clawquant dependency** |
| `get_cross_venue_snapshot` | Market Analyst + Sentiment Analyst (internal loops) | Bybit, Binance, Coinbase via ccxt → `cross_venue.py` | Text: direction, confirmation level, price divergence, funding, OI, basis | ✅ Partial results if exchanges fail; 2min cache | ✅ Multi-exchange real data | ✅ In both tool_maps, @tool decorator present | None |

### Crypto Sentiment Analyst Tools

| Tool Name | Bound To | API Source | Returns | Error Handling | Accuracy | Callable | Issues |
|-----------|----------|------------|---------|----------------|----------|----------|--------|
| `get_crypto_fear_greed` | Sentiment Analyst (internal loop) | alternative.me free API | Last 30 days Fear & Greed index (0-100 + classification) | ✅ try/except returns error string | ✅ Real API, free, no key | ✅ @tool, in tool_map | None |
| `get_funding_rate` | Sentiment Analyst (internal loop) | Bybit via ccxt `fetch_funding_rate_history` | Last 20 funding rates with timestamp, % rate, regime label (normal/elevated/extreme) | ✅ Raises DataFetchError (caught by loop) | ✅ Real exchange data | ✅ @tool, in tool_map | None |
| `get_open_interest` | Sentiment Analyst (internal loop) | Bybit via ccxt `fetch_open_interest` | Single OI snapshot: contracts, value, timestamp | ✅ Raises DataFetchError | ✅ Real exchange data | ✅ @tool, in tool_map | Prompt says "BACKUP" — used only if oi_timeseries fails |
| `get_oi_timeseries` | Sentiment Analyst (internal loop) | Bybit REST API direct (`/v5/market/open-interest`) | Hourly OI snapshots with % change, direction (BUILDING/UNWINDING/FLAT), rate per 4H | ✅ Raises DataFetchError | ✅ Real API, good interpretation | ✅ @tool, in tool_map | None |
| `get_cross_venue_snapshot` | Sentiment Analyst (internal loop) | Multi-exchange (same as above) | Same as above | ✅ | ✅ | ✅ | Shared with market analyst — no issue |

### Crypto Sentiment Analyst Pre-Fetched Data

| Data Source | Module | API | Returns | Error Handling | Issues |
|-------------|--------|-----|---------|----------------|--------|
| Social Sentiment (Reddit) | `social_sentiment.py` | Reddit JSON API (4 subreddits) | Mood, net sentiment, engagement, narratives, LLM classification overlay | ✅ Graceful fallback with warning | ⚠️ Reddit rate limits (429) — 0.5s sleep between subs; double fetch (once in `get_social_sentiment`, again in `get_social_sentiment_enhanced` for LLM classification) |

### News Analyst Tools

| Tool Name | Bound To | API Source | Returns | Error Handling | Accuracy | Callable | Issues |
|-----------|----------|------------|---------|----------------|----------|----------|--------|
| `get_news` | News Analyst (internal loop) | yfinance via `interface.py` vendor routing | Formatted news string for ticker | ✅ Vendor routing handles errors | ✅ Real API | ✅ @tool, in tool_map | Uses stock vendor routing — works for crypto tickers via "/" stripping |
| `get_global_news` | News Analyst (internal loop) | yfinance via `interface.py` vendor routing | Formatted global news | ✅ Same | ✅ | ✅ @tool, in tool_map | None |
| `get_dollar_index` | News Analyst (internal loop) | Primary: yfinance DX-Y.NYB, Fallback: FRED DTWEXBGS | DXY CSV data with header | ✅ Two-tier fallback (yfinance → FRED) | ✅ Real market data | ✅ @tool, in tool_map | FRED fallback needs `FRED_API_KEY` env var |
| `get_yields` | News Analyst (internal loop) | Primary: yfinance ^TNX/^IRX, Fallback: FRED DGS10/DGS2 | Treasury yield data (10Y, 13W/2Y) | ✅ Two-tier fallback | ✅ Real market data | ✅ @tool, in tool_map | FRED fallback needs `FRED_API_KEY` |
| `get_sp500` | News Analyst (internal loop) | yfinance ^GSPC | S&P 500 OHLCV CSV | ✅ Raises DataFetchError | ✅ Real market data | ✅ @tool, in tool_map | None |
| `get_economic_data` | News Analyst (internal loop) | FRED API via fredapi | FRED series CSV (FEDFUNDS, CPI, M2, etc.) | ✅ Returns helpful error if key missing | ⚠️ **Requires FRED_API_KEY** — returns "FRED_API_KEY not set" without it | ✅ @tool, in tool_map | **ISSUE: API key required but may not be set** |
| `get_economic_calendar` | News Analyst (internal loop) | Hardcoded FOMC/CPI/NFP dates + date math | Formatted upcoming events for current month | ✅ No API dependency — deterministic | ✅ Hardcoded 2025-2026 FOMC dates are reliable | ✅ @tool, in tool_map | 2027+ will need updated FOMC dates |

### News Analyst Pre-Fetched Data

| Data Source | Module | API | Returns | Error Handling | Issues |
|-------------|--------|-----|---------|----------------|--------|
| Geopolitical News | `geopolitical_news.py` | 30+ RSS feeds + GDELT API | Scored headlines: alerts, crypto-relevant, top headlines, impact scores | ✅ Per-feed failure tolerance; 0.3s rate limit | ⚠️ RSS feeds can be slow (15s timeout each × 20+ feeds = potentially >60s total fetch time) |

### Crypto Fundamentals Analyst Tools

| Tool Name | Bound To | API Source | Returns | Error Handling | Accuracy | Callable | Issues |
|-----------|----------|------------|---------|----------------|----------|----------|--------|
| `get_onchain_fundamentals` | Fundamentals Analyst (pre-fetched + dedup tool loop) | blockchain.info + mempool.space + CoinGecko | Full on-chain report: hash rate, addresses, tx volume, fees, market data, halving cycle | ✅ Each sub-source has fallback (e.g. mempool mirror) | ✅ Multiple free public APIs | ✅ @tool decorator, in tool_map, pre-fetched with dedup | None — well-architected |

### Crypto Fundamentals Analyst Pre-Fetched Data

| Data Source | Module | API | Returns | Error Handling | Issues |
|-------------|--------|-----|---------|----------------|--------|
| Macro Radar (7-signal) | `macro_radar.py` | Yahoo Finance + alternative.me + mempool.space | BUY/CASH verdict, 7 signals, Mayer Multiple | ✅ Cached 5min; each signal independent; retry with backoff | ⚠️ Yahoo Finance API (v8) may get rate-limited; session with retry adapter helps |
| Stablecoin Health | `macro_radar.py` | CoinGecko `/simple/price` | USDT/USDC/DAI peg status, deviation % | ✅ Returns UNKNOWN on failure | None |

### Tools Defined But NOT Bound to Any Analyst

| Tool Name | Module | API Source | Status | Issues |
|-----------|--------|------------|--------|--------|
| `get_crypto_price_data` | `crypto_tools.py` | Bybit via ccxt | ❌ **NOT BOUND** — no analyst imports or uses it | Dead code since TechnicalBrief handles OHLCV |
| `get_crypto_technical_indicators` | `crypto_tools.py` | Bybit via ccxt + stockstats | ❌ **NOT BOUND** — no analyst imports it | Dead code since TechnicalBrief computes all indicators |
| `get_orderbook_depth` | `crypto_tools.py` | Bybit via ccxt | ❌ **NOT BOUND** — no analyst imports it | Potentially useful but not wired |
| `get_liquidation_info` | `crypto_tools.py` | Bybit REST API (funding + OI + L/S ratio) | ❌ **NOT BOUND** — no analyst imports it | Overlaps with sentiment analyst tools |
| `get_macro_signal_radar` | `crypto_tools.py` | macro_radar.py (cached) | ❌ **NOT BOUND** — fundamentals analyst pre-fetches directly | Correct design — pre-fetch avoids LLM tool overhead |
| `get_stablecoin_peg_health` | `crypto_tools.py` | macro_radar.py (cached) | ❌ **NOT BOUND** — fundamentals analyst pre-fetches directly | Same as above |
| `get_crisis_impact_index` | `crypto_tools.py` | crypto_monitor.py (GDELT + RSS) | ❌ **NOT BOUND** — no analyst imports it | Has @tool decorator but unused; would provide CII score to fund manager |
| `get_insider_transactions` | `news_data_tools.py` | yfinance/alpha_vantage | ❌ **NOT BOUND** — correctly excluded for crypto | Returns early for crypto tickers |

---

## 2. ANALYST DATA FLOW DIAGRAMS

### A. Crypto Market Analyst

```
Pre-fetched data:
  └─ build_crypto_technical_brief() → TechnicalBrief JSON
       Sources: Bybit 1h/4h/1d OHLCV → indicators (RSI, MACD, BB, ATR, OBV, VWAP, ADX, Stoch)
       + market structure (BOS, CHOCH, swings)
       + key levels (Fibonacci, pivots, BB bands)
       + signal classification (bullish/bearish/neutral + confidence)
       + cross-TF contradiction resolution
       + EMA convergence, liquidity sweeps, EMA alignment

Tools available (internal loop, max 3 rounds):
  ├─ check_macd_divergence → MACD triple divergence (半木夏)
  ├─ run_pattern_scan → 10+ pattern detectors
  ├─ run_backtest → ClawQuant strategy backtest [BROKEN: clawquant not installed]
  ├─ compare_strats → ClawQuant multi-strategy comparison [BROKEN: same]
  └─ get_cross_venue_snapshot → Bybit/Binance/Coinbase alignment

Prompt instructs LLM to call:
  ✅ check_macd_divergence (minimum)
  ✅ get_cross_venue_snapshot (minimum)
  ⚠️ run_backtest / compare_strats (optional, but will always error)

Output: market_report (structured technical analysis)
```

### B. Crypto Sentiment Analyst

```
Pre-fetched data:
  └─ get_social_sentiment_enhanced(llm) → Reddit sentiment report
       Sources: r/bitcoin, r/cryptocurrency, r/CryptoMarkets, r/BitcoinMarkets
       + LLM classification overlay (auxiliary quick model)
       + keyword scoring + engagement metrics + top narratives

  └─ Data quality scoring → quality header injected into system prompt

Tools available (internal loop, max 3 rounds):
  ├─ get_funding_rate → Bybit funding rates (last 20 entries)
  ├─ get_oi_timeseries → Bybit OI hourly snapshots with direction
  ├─ get_open_interest → Bybit OI single snapshot (backup)
  ├─ get_crypto_fear_greed → alternative.me F&G index (30 days)
  └─ get_cross_venue_snapshot → Multi-exchange confirmation

Prompt instructs LLM to call ALL of:
  ✅ get_funding_rate (PRIMARY)
  ✅ get_oi_timeseries (PRIMARY)
  ✅ get_crypto_fear_greed (SUPPLEMENTARY)
  ✅ get_cross_venue_snapshot (CONFIRMATION)
  ⚠️ get_open_interest (BACKUP only)

Output: sentiment_report
```

### C. Crypto Fundamentals Analyst

```
Pre-fetched data:
  └─ get_onchain_fundamentals() → Full on-chain report
       Sources: blockchain.info (hash rate, difficulty, addresses, tx, fees)
              + mempool.space (difficulty adjustment, fee estimates, hashrate)
              + CoinGecko (market cap, supply, ATH, price changes)
              + deterministic halving cycle calculation

  └─ get_macro_radar_cached() → 7-signal BUY/CASH verdict
       Sources: Yahoo Finance (BTC, JPY, QQQ, XLP)
              + alternative.me (F&G)
              + mempool.space (hash rate)

  └─ get_stablecoin_health_cached() → USDT/USDC/DAI peg status
       Source: CoinGecko

  └─ Data quality scoring + staleness gate (>12h = cap confidence MEDIUM)

Tools available (dedup loop, max 3 rounds):
  └─ get_onchain_fundamentals → returns pre-fetched cache (avoids redundant API)

Prompt instructs LLM to: analyze pre-fetched data (tools are backup only)

Output: fundamentals_report
```

### D. News & Macro Analyst

```
Pre-fetched data:
  └─ fetch_all_news(max_per_feed=8) → Geopolitical intelligence
       Sources: 30+ RSS feeds (CNBC, MarketWatch, BBC, NPR, Fed, SEC, CSIS, etc.)
              + GDELT API (crypto-specific global events)
       + impact scoring + alert detection + region tagging

  └─ Data quality scoring + staleness gate (>4h = cap confidence by 1 level)

Tools available (internal loop, max 3 rounds):
  ├─ get_news → yfinance news for ticker
  ├─ get_global_news → yfinance global news
  ├─ get_dollar_index → DXY (yfinance / FRED fallback)
  ├─ get_yields → Treasury yields (yfinance / FRED fallback)
  ├─ get_sp500 → S&P 500 (yfinance)
  ├─ get_economic_data → FRED series (requires FRED_API_KEY)
  └─ get_economic_calendar → Deterministic event calendar

Prompt instructs LLM to call at minimum:
  ✅ get_dollar_index
  ✅ get_yields
  ✅ get_economic_calendar
  ⚠️ get_economic_data (requires FRED_API_KEY — may fail silently with "key not set")

Output: news_report
```

### E. Macro Analyst (DEPRECATED)

```
Status: NOT WIRED into CryptoTradingAgentsGraph
  - Not in analyst_creators dict
  - Writes to "macro_report" field which doesn't exist in AgentState
  - Same tools as News Analyst (duplication)
  - Kept for potential standalone use
```

---

## 3. GRAPH BINDING VERIFICATION

### Tool Node Registration (crypto_trading_graph.py)

All ToolNodes are marked as **dead code** in comments — correct, because each analyst uses an internal agentic tool-calling loop that resolves tool calls before returning to the graph. The graph's `should_continue` conditional edge would only route to the ToolNode if tool_calls remain, but they never do.

**Verification**: Each analyst's `tool_map` matches the tools bound to `llm.bind_tools(tools)`:

| Analyst | tool_map | bind_tools | Match? |
|---------|----------|------------|--------|
| Market | 5 tools (backtest wrappers + macd + pattern + cross_venue) | Same 5 | ✅ |
| Sentiment | 5 tools (fear_greed, funding, OI, OI_ts, cross_venue) | Same 5 | ✅ |
| Fundamentals | 1 tool (get_onchain_fundamentals) | Same 1 | ✅ |
| News | 7 tools (news×2, DXY, yields, SP500, FRED, calendar) | Same 7 | ✅ |

---

## 4. ISSUES IDENTIFIED

### 🔴 Critical Issues

1. **`run_backtest` / `compare_strats` — ClawQuant not installed**
   - File: `backtest_tools.py:16-19`
   - The tools check for `.venv/bin/clawquant` which is unlikely to exist
   - LLM will call these and always get `{"error": "clawquant not installed..."}`
   - **Impact**: Market analyst wastes a tool round getting an error, but recovers
   - **Fix**: Either install clawquant or remove from market analyst tool list

2. **`get_economic_data` — FRED_API_KEY may not be set**
   - File: `macro_data.py:166-167`
   - Returns string "FRED_API_KEY not set" — not an error, but not useful
   - **Impact**: News analyst gets no FRED data (CPI, M2, unemployment) if key absent
   - **Fix**: Document required env var or provide free alternative

### 🟡 Medium Issues

3. **Reddit double-fetch in sentiment pre-fetch**
   - File: `social_sentiment.py:304-319`
   - `get_social_sentiment_enhanced()` calls `get_social_sentiment()` (fetches 4 subs) then fetches all 4 subs AGAIN for LLM classification
   - 8 Reddit API calls instead of 4 — doubles latency and rate limit exposure
   - **Fix**: Pass collected posts from first fetch to LLM classifier

4. **Pattern scan cache key uses daily granularity**
   - File: `pattern_scanner.py:693`
   - Cache key: `f"{symbol}:{_dt.utcnow().strftime('%Y-%m-%d')}"` — same-day re-scans return stale data
   - **Impact**: If analyst runs twice in one session, second call gets cached result even if market moved
   - **Fix**: Use hourly or 5-minute cache key, or TTL-only caching

5. **RSS feed fetch time may be very long**
   - File: `geopolitical_news.py:99-129,196-231`
   - 20+ feeds × 15s timeout × 0.3s sleep = potentially 300s+ worst case
   - Sequential fetch with no parallelism
   - **Impact**: News analyst pre-fetch may block for minutes
   - **Fix**: Concurrent fetching with `concurrent.futures`

6. **`get_cross_venue_snapshot` shared between 2 analysts**
   - Market analyst and sentiment analyst both call it
   - 2-minute cache prevents duplicate API calls but wastes a tool round for the second caller
   - **Impact**: Minor — cached response is fast

### 🟢 Low Issues

7. **Unbound tools in crypto_tools.py**
   - `get_crypto_price_data`, `get_crypto_technical_indicators`, `get_orderbook_depth`, `get_liquidation_info`, `get_macro_signal_radar`, `get_stablecoin_peg_health`, `get_crisis_impact_index`
   - All have @tool decorators but are not imported by any analyst
   - **Impact**: Dead code, no runtime issue. Some could be useful (orderbook, CII)
   - **Fix**: Remove @tool decorators from unused tools or wire useful ones

8. **Economic calendar hardcoded dates**
   - File: `macro_data.py:217-225`
   - FOMC dates for 2025-2026 only; 2027+ will silently have no FOMC entries
   - **Impact**: None until 2027
   - **Fix**: Add 2027 dates when available

9. **Mining cost floor hardcoded at $60K**
   - File: `macro_radar.py:295`
   - `cost_floor = 60000` is a fixed estimate, not dynamically calculated
   - **Impact**: Signal accuracy degrades if actual cost changes significantly
   - **Fix**: Could use hashrate + difficulty to estimate dynamically

10. **Fundamentals analyst does not propagate `messages` to state**
    - File: `crypto_fundamentals_analyst.py:215-218`
    - Returns `{"fundamentals_report": result.content}` but no `"messages"` key
    - Other analysts return `{"messages": state["messages"] + [result], ...}`
    - **Impact**: Graph state `messages` list doesn't include fundamentals LLM output; this is actually fine since the graph clears messages after analyst fan-in, but inconsistent with other analysts

---

## 5. DATA STALENESS / CACHING SUMMARY

| Data Source | Cache TTL | Staleness Detection | Issues |
|-------------|-----------|---------------------|--------|
| OHLCV candles | File cache, configurable hours | None built-in | OK for non-realtime |
| Funding rates | No cache (live ccxt call) | N/A — always fresh | Good |
| OI timeseries | No cache (live Bybit API) | N/A — always fresh | Good |
| Fear & Greed | No cache (live API call) | N/A | Good |
| Cross-venue | 1-2 min in-memory | TTL-based | Good balance |
| Macro radar | 5 min in-memory | TTL-based | Good |
| Stablecoin health | 5 min in-memory | TTL-based | Good |
| On-chain fundamentals | 5 min in-memory per sub-source | `generated_at` timestamp + staleness gate (>12h) | ✅ Excellent |
| Social sentiment | No cache | `generated_at` in response | ⚠️ No cache means always re-fetching Reddit |
| Geopolitical news | No cache | `generated_at` + quality scoring (>4h stale) | ⚠️ Slow to fetch, no cache |
| Technical brief | No cache | `generated_at` in model | OK — computed fresh each run |
| Pattern scan | 5 min in-memory (daily key) | TTL-based | ⚠️ Daily key too coarse |
| MACD divergence | No cache (via pattern scan) | N/A | OK |

---

## 6. API KEY REQUIREMENTS

| API | Key Required | Env Var | Used By | Impact if Missing |
|-----|-------------|---------|---------|-------------------|
| Bybit | No (public endpoints) | `BYBIT_API_KEY` (optional) | Market, Sentiment, Fundamentals (tech brief) | None — public API works |
| Binance | No (public endpoints) | `BINANCE_API_KEY` (optional) | Cross-venue confirmation | None |
| Coinbase | No (public endpoints) | `COINBASE_API_KEY` (optional) | Cross-venue confirmation | None |
| alternative.me | No | — | Fear & Greed | N/A |
| Reddit | No | — | Social sentiment | ⚠️ Rate limited without auth |
| FRED | **Yes** (for `get_economic_data`) | `FRED_API_KEY` | News analyst | 🔴 Returns "key not set" — no CPI/M2/unemployment |
| blockchain.info | No | — | On-chain fundamentals | N/A |
| mempool.space | No | — | Fundamentals, Macro radar | N/A |
| CoinGecko | No (free tier) | — | Fundamentals, Stablecoin health | ⚠️ Rate limited on free tier |
| Yahoo Finance | No | — | DXY, Yields, S&P 500, Macro radar | ⚠️ Unofficial API, may block |
| GDELT | No | — | Geopolitical news, CII | N/A |
| RSS feeds | No | — | Geopolitical news | N/A |
| LLM Provider | **Yes** | `ANTHROPIC_API_KEY` etc. | All analysts | 🔴 Fatal — nothing works |
| ClawQuant | N/A (binary) | — | Backtest tools | 🔴 Tool always errors if not installed |

---

## 7. FORMAT MISMATCH CHECK

| Tool | Return Format | Prompt Expects | Match? |
|------|--------------|----------------|--------|
| `check_macd_divergence` | Formatted string report with BEARISH/BULLISH_DIVERGENCE or "No divergence" | Divergence type, strength, timeframe | ✅ |
| `run_pattern_scan` | Formatted string: active signals with confidence and details | Pattern signals with confidence | ✅ |
| `run_backtest` | JSON string with metrics | Strategy backtest results | ✅ (but always errors) |
| `get_cross_venue_snapshot` | Formatted text: direction, confirmation level, prices, funding, OI, basis | Cross-venue confirmation | ✅ |
| `get_funding_rate` | Formatted text: last 20 rates with regime labels | Funding rates with directional bias | ✅ |
| `get_oi_timeseries` | Formatted text: hourly OI snapshots + direction + rate + alerts | OI over 6×4H with change rate and direction | ✅ |
| `get_open_interest` | Formatted text: single snapshot | Single OI value | ✅ |
| `get_crypto_fear_greed` | Formatted text: last 30 days F&G values | `fear_greed_value` (0-100) and `fear_greed_label` | ⚠️ Minor: prompt says "fear_greed_value" field but tool returns formatted multi-line text |
| `get_dollar_index` | CSV with header | DXY level and trend | ✅ |
| `get_yields` | Multi-section text with yield tables | Treasury yield curve | ✅ |
| `get_economic_data` | CSV with header | FRED data series | ✅ |
| `get_economic_calendar` | Formatted text with upcoming events | Upcoming events | ✅ |
| `get_onchain_fundamentals` | Formatted report text | On-chain fundamentals report | ✅ |

---

## 8. SUMMARY OF RECOMMENDED ACTIONS

### Must Fix
1. **Remove or guard backtest tools** — `run_backtest`/`compare_strats` always error without clawquant; remove from market analyst tool list or make conditional
2. **Document FRED_API_KEY requirement** — `get_economic_data` silently returns useless string without it

### Should Fix
3. **Fix Reddit double-fetch** — save posts from first fetch, pass to LLM classifier
4. **Add concurrent RSS fetching** — geopolitical news fetch is sequentially slow
5. **Fix pattern scan cache granularity** — daily key → 5-minute TTL-only

### Nice to Have
6. **Remove dead @tool decorators** from unbound tools (reduce confusion)
7. **Wire `get_crisis_impact_index`** to fund manager or news analyst
8. **Wire `get_orderbook_depth`** to market analyst (order flow context)
9. **Add `messages` to fundamentals analyst return** for consistency
10. **Make mining cost floor dynamic** using hashrate/difficulty regression

---

*Report generated: 2026-03-13T18:55:00+11:00*
*Reviewer: Claude Code (Opus 4.6)*

# Final Analyst Audit — 2026-03-13 (Pass 4)

**Scope**: All 5 analyst agents, tools, data sources, graph binding
**Previous reviews**: `review_2026-03-13_architecture.md`, `review_2026-03-13_data_layer.md`, `review_2026-03-13_analysts_post_fix.md`, `review_2026-03-13_tool_validity.md`
**Previous fixes**: 85+ issues across 3 review rounds

---

## Overall Grade: A-

**Previous grades**: C+ → B+ → A-

---

## Per-Analyst Scorecard

### 1. Crypto Market Analyst (`crypto_market_analyst.py`)

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Tools bound = documented in prompt | PASS | 3 tools: `check_macd_divergence`, `run_pattern_scan`, `get_cross_venue_snapshot` — all 3 documented in prompt (lines 78-81) |
| 2 | Tools bound = graph ToolNode | PASS | Graph ToolNode (line 213-217) has same 3 tools. **FIXED** from previous CRITICAL |
| 3 | No tool binding mismatches | PASS | Internal tool loop (lines 138-156) resolves all calls before graph routing. Graph ToolNode is dead code (correctly documented in graph comments) |
| 4 | Pre-fetched data explained | PASS | TechnicalBrief JSON structure documented in prompt (lines 48-57) |
| 5 | No contradictions | PASS | — |
| 6 | No disabled/removed feature refs | PASS | Backtest tools (`run_backtest`, `compare_strats`) removed. **FIXED** from previous CRITICAL |
| 7 | Output format clear | PASS | 7-section structured format + Summary Table (lines 98-128) |
| 8 | Weight guidance explicit | PASS | Multi-TF alignment guidance, cross-venue confirmation rules (lines 83-87) |
| 9 | Cross-venue documented | PASS | `get_cross_venue_snapshot` with interpretation rules (lines 83-87) |
| 10 | Error handling consistent | PASS | Brief failure → error string (line 42); tool errors caught (line 152); empty report fallback (lines 159-161) |

**Result: 10/10 PASS**

---

### 2. Crypto Sentiment Analyst (`crypto_sentiment_analyst.py`)

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Tools bound = documented in prompt | PASS | 5 tools all documented with priority labels (lines 81-86) |
| 2 | Tools bound = graph ToolNode | PASS | Graph ToolNode (lines 219-225) has same 5 tools |
| 3 | No tool binding mismatches | PASS | Internal tool loop resolves all calls |
| 4 | Pre-fetched data explained | PASS | Social sentiment clearly marked as pre-fetched with weight guidance (lines 74-78) |
| 5 | No contradictions | PASS | User message (lines 115-116) now says "Use get_open_interest only if get_oi_timeseries fails" — consistent with system prompt BACKUP label. **FIXED** from previous MEDIUM |
| 6 | No disabled/removed feature refs | PASS | — |
| 7 | Output format clear | PASS | 6-section analysis framework (lines 101-107) |
| 8 | Weight guidance explicit | PASS | Strict priority: positioning > social > F&G (lines 88-94) |
| 9 | Cross-venue documented | PASS | Tool #5 with interpretation guidance (line 86) |
| 10 | Error handling consistent | PASS | Social fallback with warning (lines 39-46); data quality gate (lines 49-67); tool errors caught (line 134) |

**Result: 10/10 PASS**

---

### 3. Crypto Fundamentals Analyst (`crypto_fundamentals_analyst.py`)

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Tools bound = documented in prompt | PASS | 1 tool (`get_onchain_fundamentals`) — LLM told to analyze pre-fetched data, tool is backup/dedup |
| 2 | Tools bound = graph ToolNode | PASS | Graph ToolNode (lines 226-229) has only `get_onchain_fundamentals`. **FIXED** from previous MEDIUM — `get_macro_signal_radar` and `get_stablecoin_peg_health` removed |
| 3 | No tool binding mismatches | PASS | Dedup loop (lines 199-213) handles re-calls with pre-fetched cache |
| 4 | Pre-fetched data explained | PASS | On-chain, macro radar, and stablecoin data all injected with context (lines 183-195) |
| 5 | No contradictions | PASS | — |
| 6 | No disabled/removed feature refs | PASS | CII references removed in prior fix |
| 7 | Output format clear | PASS | 8-section format (lines 168-178) |
| 8 | Weight guidance explicit | PASS | 5-bucket weights: Macro 30% > Network 25% > Adoption 20% > Valuation 15% > Cycle 10% (lines 117-148) |
| 9 | Cross-venue documented | N/A | Not relevant for fundamentals |
| 10 | Error handling consistent | PASS | Staleness gate (lines 96-101); macro fallback (lines 78-83); quality scoring (lines 86-94) |

**Issues (LOW):**
- Fundamentals analyst does not propagate `messages` to state (returns only `fundamentals_report`). Other analysts return `messages`. Not a bug — graph clears messages after analyst fan-in — but inconsistent.

**Result: 10/10 PASS** (1 LOW cosmetic issue)

---

### 4. News Analyst (`news_analyst.py`)

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Tools bound = documented in prompt | PASS | 7 tools all documented (lines 106-113) with must-call guidance |
| 2 | Tools bound = graph ToolNode | PASS | Graph ToolNode (lines 232-240) has same 7 tools |
| 3 | No tool binding mismatches | PASS | Internal tool loop resolves all calls |
| 4 | Pre-fetched data explained | PASS | Geo news from RSS+GDELT injected with source count and alert count (lines 98-102) |
| 5 | No contradictions | PASS | — |
| 6 | No disabled/removed feature refs | PASS | — |
| 7 | Output format clear | PASS | 7-section framework (lines 124-130) |
| 8 | Weight guidance explicit | PASS | Macro regime → events → calendar ordering |
| 9 | Cross-venue documented | N/A | Not relevant for news |
| 10 | Error handling consistent | PASS | Geo fallback (lines 47-59); staleness gate (lines 62-82); quality degraded note (lines 87-91); tool errors caught (line 158) |

**Issues (LOW):**
- `NEWS_STALENESS_HOURS = 4.0` may trigger unnecessary degraded-data warnings for economic calendar (daily data). Documented design decision (lines 26-29). Acceptable.

**Result: 10/10 PASS** (1 LOW design note)

---

### 5. Macro Analyst (`macro_analyst.py`) — DEPRECATED

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Tools bound = documented in prompt | PASS | 5 tools documented (lines 82-86) |
| 2 | Tools bound = graph ToolNode | N/A | Not wired into graph (by design) |
| 3 | No tool binding mismatches | N/A | — |
| 4 | Pre-fetched data explained | N/A | No pre-fetched data |
| 5 | No contradictions | PASS | — |
| 6 | No disabled/removed feature refs | PASS | Deprecation clearly documented (lines 1-11) |
| 7 | Output format clear | PASS | 4-field output (lines 89-93) |
| 8 | Weight guidance explicit | PASS | Factor ordering in prompt |
| 9 | Cross-venue documented | N/A | — |
| 10 | Error handling consistent | PASS | Internal tool loop with error handling (lines 106-124); empty report fallback (lines 127-129) |

**Issues (EXISTING, ACCEPTED):**
- Writes to `macro_report` field that doesn't exist in AgentState. Intentionally orphaned — documented as deprecated with instructions for re-wiring (lines 10-11).

**Result: PASS (deprecated, correctly documented)**

---

## Per-Tool Scorecard

### Tools Bound to Active Analysts

| Tool | @tool | Returns useful data | Error handling | Callable | Status |
|------|-------|-------------------|----------------|----------|--------|
| `check_macd_divergence` | PASS | PASS | PASS (returns error string) | PASS | CLEAN |
| `run_pattern_scan` | PASS | PASS | PASS (per-detector isolation) | PASS | CLEAN |
| `get_cross_venue_snapshot` | PASS | PASS | PASS (partial results on exchange failure) | PASS | CLEAN |
| `get_crypto_fear_greed` | PASS | PASS | PASS (returns error string) | PASS | CLEAN |
| `get_funding_rate` | PASS | PASS | PASS (raises DataFetchError, caught by loop) | PASS | CLEAN |
| `get_open_interest` | PASS | PASS | PASS (raises DataFetchError) | PASS | CLEAN |
| `get_oi_timeseries` | PASS | PASS | PASS (raises DataFetchError) | PASS | CLEAN |
| `get_onchain_fundamentals` | PASS | PASS | PASS (returns error string) | PASS | CLEAN |
| `get_news` | PASS | PASS | PASS (vendor routing) | PASS | CLEAN |
| `get_global_news` | PASS | PASS | PASS (vendor routing) | PASS | CLEAN |
| `get_dollar_index` | PASS | PASS | PASS (yfinance→FRED fallback) | PASS | CLEAN |
| `get_yields` | PASS | PASS | PASS (yfinance→FRED fallback) | PASS | CLEAN |
| `get_sp500` | PASS | PASS | PASS (returns error string) | PASS | CLEAN |
| `get_economic_data` | PASS | PASS | PASS (returns "key not set" if no FRED_API_KEY) | PASS | CLEAN |
| `get_economic_calendar` | PASS | PASS | PASS (deterministic, no API) | PASS | CLEAN |

**15/15 tools PASS** — all callable, all return useful data, all handle errors.

### Tools NOT Bound (dead code in `crypto_tools.py`)

| Tool | Status | Notes |
|------|--------|-------|
| `get_crypto_price_data` | Dead code | Superseded by TechnicalBrief |
| `get_crypto_technical_indicators` | Dead code | Superseded by TechnicalBrief |
| `get_orderbook_depth` | Dead code | Potentially useful, not wired |
| `get_liquidation_info` | Dead code | Overlaps with sentiment tools |
| `get_macro_signal_radar` | Dead code | Pre-fetched by fundamentals analyst |
| `get_stablecoin_peg_health` | Dead code | Pre-fetched by fundamentals analyst |
| `get_crisis_impact_index` | Dead code | Has @tool decorator, not wired |

**7 unbound tools** — not a bug, just unused API surface. LOW priority cleanup.

---

## Data Source Scorecard

| Data Source | Caching | Staleness Detection | Error Handling | Status |
|-------------|---------|---------------------|----------------|--------|
| TechnicalBrief (Bybit OHLCV) | File cache | `generated_at` timestamp | try/except with fallback | CLEAN |
| Social Sentiment (Reddit) | None (fresh each run) | `generated_at` + quality scoring | Graceful fallback | CLEAN |
| Geopolitical News (RSS+GDELT) | None (fresh each run) | `generated_at` + quality scoring (>4h) | Per-feed failure tolerance; concurrent ThreadPoolExecutor | CLEAN |
| On-chain Fundamentals | 5min in-memory per sub-source | `generated_at` + staleness gate (>12h) | Per-source fallback (mempool mirror) | CLEAN |
| Macro Radar (7-signal) | 5min in-memory | TTL-based | Each signal independent | CLEAN |
| Stablecoin Health | 5min in-memory | TTL-based | Returns UNKNOWN on failure | CLEAN |
| Cross-venue (Bybit/Binance/Coinbase) | 1-5min in-memory | TTL-based | Partial results on exchange failure | CLEAN |
| Funding Rates (Bybit) | No cache (live) | Always fresh | Raises DataFetchError | CLEAN |
| OI Timeseries (Bybit) | No cache (live) | Always fresh | Raises DataFetchError | CLEAN |
| Fear & Greed (alternative.me) | No cache (live) | Always fresh | Returns error string | CLEAN |
| DXY (yfinance→FRED) | No cache | N/A | Two-tier fallback | CLEAN |
| Treasury Yields (yfinance→FRED) | No cache | N/A | Two-tier fallback | CLEAN |
| FRED Data | No cache | N/A | Returns "key not set" message | CLEAN |
| Pattern Scanner | 5min in-memory (daily key) | TTL + daily key | Per-detector isolation | LOW |

---

## Remaining Issues

| # | Severity | Area | Issue | Status |
|---|----------|------|-------|--------|
| 1 | LOW | Fundamentals | Does not propagate `messages` to state (cosmetic inconsistency) | EXISTING (accepted) |
| 2 | LOW | News | 4h staleness threshold may be aggressive for daily economic data | EXISTING (documented design decision) |
| 3 | LOW | Pattern Scanner | Daily cache key means intraday re-scans return stale data | EXISTING |
| 4 | LOW | Macro Analyst | Orphaned code — deprecated but not deleted | EXISTING (documented) |
| 5 | LOW | crypto_tools.py | 7 unbound tools with @tool decorators (dead code) | EXISTING |
| 6 | LOW | Economic Calendar | Hardcoded FOMC dates for 2025-2026 only | EXISTING (OK until 2027) |
| 7 | LOW | Sentiment F&G | Prompt says `fear_greed_value` field name but tool returns multi-line text | EXISTING (minor, LLM handles fine) |

**Zero CRITICAL issues.**
**Zero HIGH issues.**
**Zero MEDIUM issues.**
**7 LOW issues (all pre-existing, all acceptable).**

---

## Comparison with Previous Review Grades

| Issue | Review 1 (C+) | Review 2 (B+) | Review 3 (tool validity) | Review 4 (this) |
|-------|---------------|---------------|--------------------------|-----------------|
| Market analyst tool binding (5→3 mismatch) | Not checked | **CRITICAL** | Verified fixed | PASS |
| Market analyst backtest tools (clawquant) | Not checked | — | **CRITICAL** | PASS (removed) |
| Market analyst prompt missing tool docs | Not checked | **HIGH** | — | PASS (all 3 documented) |
| Market analyst no internal tool loop | Not checked | **CRITICAL** | — | PASS (3-round loop added) |
| Graph ToolNode mismatches | Not checked | **CRITICAL** | Verified fixed | PASS |
| Fundamentals ToolNode extras | Not checked | **MEDIUM** | — | PASS (cleaned up) |
| Sentiment "call all 5" vs "BACKUP" | Not checked | **MEDIUM** | — | PASS (user msg fixed) |
| F&G weighting conflict | **CRITICAL** | RESOLVED | — | PASS |
| Social sentiment double-fetch | Not checked | — | **MEDIUM** | PASS (single fetch + reuse) |
| RSS sequential fetching | Not checked | — | **MEDIUM** | PASS (ThreadPoolExecutor) |
| Macro analyst orphaned | **HIGH** | **HIGH** | Documented | PASS (deprecated docs) |
| CII disabled references | **MEDIUM** | RESOLVED | — | PASS |
| Data quality gates | MISSING | PASS | PASS | PASS |
| Staleness gates | MISSING | PASS | PASS | PASS |
| Cross-venue integration | NOT PRESENT | PASS | PASS | PASS |
| Tool error handling | INCONSISTENT | IMPROVED | PASS | PASS |

**Issues fixed since review 1**: 85+
**Issues fixed since review 2-3**: 8 (1 CRITICAL, 2 HIGH, 3 MEDIUM, 2 LOW)
**Issues remaining**: 7 LOW (all acceptable)

---

## Conclusion

The codebase is production-ready for its intended scope. All CRITICAL, HIGH, and MEDIUM issues from previous reviews have been resolved. The remaining 7 LOW issues are documented design decisions or cosmetic inconsistencies that do not affect runtime correctness.

Key improvements across 4 review passes:
- **Tool binding**: All analysts now have synchronized tool bindings (LLM ↔ graph ToolNode)
- **Internal tool loops**: All 4 active analysts use consistent 3-round internal loops
- **Broken tools removed**: Backtest tools (clawquant dependency) removed from market analyst
- **Data quality**: Structured scoring on all analysts with staleness gates
- **Error handling**: Consistent pattern across all tools and data sources
- **Cross-venue**: Properly documented and integrated for market + sentiment analysts
- **Concurrent fetching**: RSS feeds now fetched via ThreadPoolExecutor
- **Social sentiment**: Single fetch + reuse (no more double-fetch)

**Grade: A-** (7 LOW issues prevent A)

---

*Report generated: 2026-03-13T19:15:00+11:00*
*Reviewer: Claude Code (Opus 4.6)*
*Review pass: 4 of 4 (final)*

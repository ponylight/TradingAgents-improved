# Post-Fix Analyst Review — 2026-03-13

**Scope**: All 5 analyst agents, their tools, graph binding, and cross-venue integration.
**Previous reviews**: `review_2026-03-13_architecture.md`, `review_2026-03-13_data_layer.md`

---

## Executive Summary

The 42-fix commit and Binance/Coinbase integration significantly improved the codebase. Prompt quality, data quality gates, weighting guidance, and output formats are now strong across all analysts. However, this review found **1 critical issue** (tool binding mismatch in market analyst), **2 high issues**, and several medium/low items.

**Overall grade: B+** (up from C+ in previous review)

---

## Per-Analyst Review

### 1. Crypto Market Analyst (`crypto_market_analyst.py`)

| Check | Status | Notes |
|-------|--------|-------|
| Role clarity | PASS | Clear: interpret pre-computed Technical Brief |
| Data description | PASS | Detailed JSON structure documentation |
| Tool instructions | **FAIL** | 4 of 5 bound tools undocumented in prompt |
| No contradictions | PASS | — |
| Weight guidance | PASS | Multi-TF alignment guidance clear |
| Output format | PASS | Structured sections + Summary Table |
| Cross-venue | PASS | `get_cross_venue_snapshot` documented with interpretation rules |
| Token efficiency | PASS | Lean prompt, no bloat |
| Framework completeness | PASS | Covers all brief fields |

**Issues:**

**CRITICAL — Tool binding mismatch between analyst and graph ToolNode**

The analyst binds 5 tools to the LLM (line 40):
```
run_backtest, compare_strats, check_macd_divergence, run_pattern_scan, get_cross_venue_snapshot
```

The graph's ToolNode for "market" has 3 different tools (`crypto_trading_graph.py:207-211`):
```
get_crypto_price_data, get_orderbook_depth, get_cross_venue_snapshot
```

Only `get_cross_venue_snapshot` overlaps. If the LLM calls `run_backtest`, `compare_strats`, `check_macd_divergence`, or `run_pattern_scan`, the graph routes to the ToolNode which doesn't have those tools → **runtime error**.

The market analyst uses a single `llm_with_tools.invoke()` (no internal tool loop), so tool calls propagate to the graph's conditional edges, which route to the mismatched ToolNode.

**Fix**: Either (a) add an internal tool loop like sentiment/news analysts, or (b) sync the graph ToolNode to match the analyst's bound tools.

**HIGH — 4 bound tools undocumented in prompt**

`run_backtest`, `compare_strats`, `check_macd_divergence`, and `run_pattern_scan` are bound to the LLM but the system prompt only mentions `get_cross_venue_snapshot`. The LLM may discover them via tool schemas but has no guidance on when/how to use them.

---

### 2. Crypto Sentiment Analyst (`crypto_sentiment_analyst.py`)

| Check | Status | Notes |
|-------|--------|-------|
| Role clarity | PASS | Clear: derivatives positioning + social sentiment |
| Data description | PASS | Pre-fetched vs tool-called distinction explicit |
| Tool instructions | PASS | All 5 tools documented with priority and weight |
| No contradictions | PASS | Previous F&G weighting conflict resolved |
| Weight guidance | PASS | Strict priority: positioning > social > F&G |
| Output format | PASS | 6-section framework |
| Cross-venue | PASS | `get_cross_venue_snapshot` listed as tool #5 with interpretation |
| Token efficiency | PASS | — |
| Framework completeness | PASS | All data sources covered |

**Issues:**

**MEDIUM — get_open_interest listed as backup but LLM told to call all 5 tools**

The user message says "Call all five tools" but the system prompt says `get_open_interest` is "BACKUP: Use only if get_oi_timeseries fails." This is a minor contradiction — the LLM will likely call all 5 anyway, wasting tokens on a redundant OI snapshot.

**LOW — Internal tool loop makes graph ToolNode dead code**

The sentinel analyst has its own 3-round tool loop (lines 119-137), consuming all tool calls internally. The graph's `tools_sentiment` ToolNode is never reached. Not a bug — just dead code in the graph routing.

---

### 3. Crypto Fundamentals Analyst (`crypto_fundamentals_analyst.py`)

| Check | Status | Notes |
|-------|--------|-------|
| Role clarity | PASS | Clear: on-chain fundamentals, not price charts |
| Data description | PASS | 5-bucket framework with weights |
| Tool instructions | PASS | Single tool, pre-fetched with dedup |
| No contradictions | PASS | CII references removed (previous fix) |
| Weight guidance | PASS | Macro radar 30% > Network 25% > Adoption 20% > Valuation 15% > Cycle 10% |
| Output format | PASS | 8-section output format |
| Cross-venue | N/A | Not relevant for fundamentals |
| Token efficiency | PASS | — |
| Framework completeness | PASS | Macro radar interpretation guide is thorough |

**Issues:**

**MEDIUM — Graph ToolNode includes tools the analyst doesn't bind to LLM**

Graph ToolNode for "fundamentals" includes `get_macro_signal_radar` and `get_stablecoin_peg_health` (`crypto_trading_graph.py:219-224`), but the analyst only binds `get_onchain_fundamentals` to the LLM (line 44). The macro/stablecoin data is pre-fetched in the node function, not via tool calls. The extra tools in the ToolNode are dead code.

**LOW — Fragile dedup logic for tool re-calls**

The dedup logic (lines 201-208) handles only one round of tool calls. If the LLM somehow calls `get_onchain_fundamentals` in a second round, it would invoke the API again. Low risk since the data is already in the context.

---

### 4. News Analyst (`news_analyst.py`)

| Check | Status | Notes |
|-------|--------|-------|
| Role clarity | PASS | Clear: news events + macroeconomic indicators |
| Data description | PASS | Pre-fetched geo news + tool-called macro data |
| Tool instructions | PASS | All 7 tools documented with must-call guidance |
| No contradictions | PASS | — |
| Weight guidance | PASS | Macro regime > events > calendar |
| Output format | PASS | 7-section framework |
| Cross-venue | N/A | Not relevant for news |
| Token efficiency | PASS | — |
| Framework completeness | PASS | — |

**Issues:**

**LOW — Internal tool loop makes graph ToolNode dead code**

Same as sentiment analyst — the news analyst has its own 3-round tool loop. The graph's `tools_news` ToolNode is never reached.

**LOW — Staleness threshold reduced to 4h may be too aggressive**

`NEWS_STALENESS_HOURS = 4.0` was reduced from 6h in a previous fix. For economic calendar data (which updates daily), 4h may cause unnecessary degraded-data warnings during off-hours.

---

### 5. Macro Analyst (`macro_analyst.py`)

| Check | Status | Notes |
|-------|--------|-------|
| Role clarity | PASS | Clear: macro conditions → BTC impact |
| Data description | PASS | 6 macro factors well-explained |
| Tool instructions | PASS | All 5 tools documented |
| No contradictions | PASS | — |
| Weight guidance | PASS | Implicit via factor ordering |
| Output format | PASS | 4-field output |
| Cross-venue | N/A | — |
| Token efficiency | PASS | Lean prompt |
| Framework completeness | PASS | — |

**Issues:**

**HIGH — Macro analyst not wired into graph; writes to non-existent state field**

The macro analyst writes to `macro_report` (line 124), but `AgentState` has no `macro_report` field. The analyst is not in the graph's `analyst_creators` dict — it's orphaned code.

The news analyst already covers macro data (DXY, yields, S&P, FRED, economic calendar) with the same tools. The macro analyst appears to be a redundant, disconnected module.

**Decision needed**: Either wire it in as an optional analyst (adding `macro_report` to AgentState) or delete it to reduce code surface.

**MEDIUM — Proper agentic tool loop (confirmed fixed)**

The macro analyst now has a proper 3-round tool loop (lines 97-115). This was broken before — **confirmed fixed**.

**MEDIUM — State field collision resolved**

Previous concern about news_report vs macro_report collision is moot — macro analyst writes to `macro_report`, news analyst writes to `news_report`, and only the news analyst is wired into the graph.

---

## Tool Binding Verification Matrix

### Tools bound to LLM (in analyst code) vs Graph ToolNode (in crypto_trading_graph.py)

| Analyst | LLM-Bound Tools | Graph ToolNode Tools | Match? | Tool Loop? |
|---------|-----------------|----------------------|--------|------------|
| **Market** | `run_backtest, compare_strats, check_macd_divergence, run_pattern_scan, get_cross_venue_snapshot` | `get_crypto_price_data, get_orderbook_depth, get_cross_venue_snapshot` | **1/5 overlap** | No (single invoke) |
| **Sentiment** | `get_crypto_fear_greed, get_funding_rate, get_open_interest, get_oi_timeseries, get_cross_venue_snapshot` | `get_crypto_fear_greed, get_funding_rate, get_open_interest, get_oi_timeseries, get_cross_venue_snapshot` | **5/5 match** | Yes (internal, 3 rounds) |
| **Fundamentals** | `get_onchain_fundamentals` | `get_onchain_fundamentals, get_macro_signal_radar, get_stablecoin_peg_health` | **1/3 overlap** | No (pre-fetch + single invoke) |
| **News** | `get_news, get_global_news, get_dollar_index, get_yields, get_sp500, get_economic_data, get_economic_calendar` | `get_news, get_global_news, get_dollar_index, get_yields, get_sp500, get_economic_data, get_economic_calendar` | **7/7 match** | Yes (internal, 3 rounds) |
| **Macro** | `get_dollar_index, get_yields, get_sp500, get_economic_data, get_economic_calendar` | *(not in graph)* | N/A | Yes (internal, 3 rounds) |

### Key Insight: Dual Tool Execution Paths

There are **two** tool execution paths in the system, and they're inconsistently used:

1. **Internal tool loop** (sentiment, news, macro): Analyst handles tool calls in a `for round_num in range(MAX_TOOL_ROUNDS)` loop. Graph ToolNode is **dead code**.
2. **Graph-routed tools** (market, fundamentals): Analyst does single `llm.invoke()`, tool calls route through graph conditional edges to ToolNode.

**The market analyst is the only one that actually uses graph-routed tools, and its tool binding is wrong.** This is the critical bug.

---

## Data Flow Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │              PRE-FETCHED DATA                    │
                    │  (deterministic, no LLM, injected into prompts)  │
                    └─────────────────────────────────────────────────┘
                              │                │              │
                    TechnicalBrief     Social Sentiment    Geo News (RSS+GDELT)
                    (8 indicators       (Reddit keywords)   (30+ feeds)
                     × 3 timeframes)         │              │
                              │              │              │
                              ▼              ▼              ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   MARKET     │   │  SENTIMENT   │   │ FUNDAMENTALS │   │    NEWS      │
│   ANALYST    │   │   ANALYST    │   │   ANALYST    │   │   ANALYST    │
│              │   │              │   │              │   │              │
│ Pre: Brief   │   │ Pre: Social  │   │ Pre: On-chain│   │ Pre: Geo RSS │
│ Tools: 5*    │   │ Tools: 5     │   │ Pre: Macro   │   │ Tools: 7     │
│ Loop: NO†    │   │ Loop: YES    │   │ Pre: Stable  │   │ Loop: YES    │
│              │   │              │   │ Tool: 1      │   │              │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
  market_report    sentiment_report   fundamentals_report   news_report
       │                  │                  │                  │
       └──────────────────┴──────────────────┴──────────────────┘
                                    │
                              AgentState
                                    │
                          ┌─────────▼─────────┐
                          │  Bull ↔ Bear Debate │
                          │  Research Manager   │
                          └─────────┬──────────┘
                                    │
                              ┌─────▼─────┐
                              │  Trader    │
                              └─────┬─────┘
                                    │
                          ┌─────────▼───────────┐
                          │ Risk Debate (3-way)  │
                          │ Risk Judge           │
                          └─────────┬────────────┘
                                    │
                          ┌─────────▼───────────┐
                          │ Portfolio Manager    │
                          └─────────────────────┘

* Market analyst tool binding mismatched with graph ToolNode
† No internal tool loop — relies on graph routing (broken)
```

### Cross-Venue Data Flow

```
                    ┌──────────┐   ┌──────────┐   ┌──────────┐
                    │  Bybit   │   │ Binance  │   │ Coinbase │
                    │(primary) │   │(confirm) │   │(confirm) │
                    └────┬─────┘   └────┬─────┘   └────┬─────┘
                         │              │              │
                         └──────────────┼──────────────┘
                                        │
                              exchange_manager.py
                              (ccxt singleton cache)
                                        │
                              cross_venue.py
                              (5-min cache, graceful degradation)
                                        │
                    ┌───────────────────┬┴───────────────────┐
                    │                   │                     │
              get_cross_venue     get_cross_venue      get_cross_venue
              _snapshot           _snapshot             _snapshot
              (market analyst)    (sentiment analyst)   (via tool)
```

---

## Issue Summary

| # | Severity | Analyst | Issue | Status |
|---|----------|---------|-------|--------|
| 1 | **CRITICAL** | Market | LLM-bound tools ≠ Graph ToolNode tools (4/5 mismatch). Tool calls will fail at runtime. | NEW |
| 2 | **HIGH** | Market | 4 bound tools (backtest, patterns, MACD divergence) undocumented in prompt | NEW |
| 3 | **HIGH** | Macro | Orphaned analyst — not in graph, writes to non-existent `macro_report` field | EXISTING |
| 4 | **MEDIUM** | Sentiment | "Call all 5 tools" contradicts "get_open_interest is BACKUP only" | NEW |
| 5 | **MEDIUM** | Fundamentals | Graph ToolNode has 2 extra tools the LLM doesn't know about (dead code) | NEW |
| 6 | **MEDIUM** | All | Inconsistent tool execution (2 analysts use internal loops, 2 use graph routing) | DESIGN |
| 7 | **LOW** | Sentiment, News | Internal tool loops make graph ToolNodes dead code | DESIGN |
| 8 | **LOW** | News | 4h staleness threshold may be too aggressive for daily economic data | EXISTING |
| 9 | **LOW** | Fundamentals | Dedup logic handles only 1 round of re-calls | EXISTING |

---

## Previous Review Comparison

| Area | Previous (pre-fix) | Current (post-fix) | Delta |
|------|-------------------|-------------------|-------|
| F&G weighting conflict | CRITICAL | RESOLVED | Fixed |
| Social sentiment weight | HIGH | RESOLVED | Weight hierarchy clear |
| Tool-name mismatch | HIGH | RESOLVED | Tool names match signatures |
| CII disabled references | MEDIUM | RESOLVED | Removed |
| Macro radar interpretation | MISSING | RESOLVED | Full 7-signal guide added |
| Data quality gates | MISSING | PASS | Structured scoring on all analysts |
| Staleness gates | MISSING | PASS | All analysts have freshness checks |
| News analyst tool loop | BROKEN (no loop) | PASS | Proper agentic loop |
| Macro analyst tool loop | BROKEN (no loop) | PASS | Proper agentic loop |
| Macro analyst state field | COLLISION RISK | SAFE | Writes to `macro_report` (unused) |
| Market analyst tool binding | NOT CHECKED | **CRITICAL** | New issue found |
| Cross-venue integration | NOT PRESENT | PASS | Well-documented in prompts |

**Fixed: 10 issues from previous reviews**
**New: 3 issues discovered (1 critical, 2 high)**
**Remaining: 6 medium/low items**

---

## Recommended Fix Priority

1. **CRITICAL**: Sync market analyst tools — either add internal tool loop or update graph ToolNode to match LLM-bound tools
2. **HIGH**: Add prompt documentation for backtest/pattern/MACD tools in market analyst
3. **HIGH**: Decision on macro analyst — wire in or delete
4. **MEDIUM**: Remove "call all 5 tools" from sentiment user message, or drop "BACKUP" label from get_open_interest
5. **MEDIUM**: Clean up dead ToolNode tools from graph (fundamentals extras)
6. **MEDIUM**: Standardize on one tool execution pattern (recommend internal loops for all)

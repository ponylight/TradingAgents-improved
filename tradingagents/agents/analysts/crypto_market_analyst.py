"""
Crypto Technical Analyst

Receives a pre-computed TechnicalBrief (deterministic, zero LLM) and interprets it.
The brief covers 1h/4h/1d: trend, momentum, VWAP, volatility, volume,
market structure, key levels, Fibonacci, and signal classification.

The LLM's job: interpret the numbers, identify the setup, produce a structured report.
NOT the LLM's job: fetch data or calculate indicators (that's Tier 1).
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

log = logging.getLogger("crypto_market_analyst")

MAX_TOOL_ROUNDS = 3


def create_crypto_market_analyst(llm):
    """Create the crypto technical analyst agent node."""

    # Pre-compute the technical brief (no LLM needed)
    from tradingagents.dataflows.crypto_technical_brief import build_crypto_technical_brief

    # MACD Triple Divergence detector (半木夏 strategy)
    from tradingagents.agents.utils.crypto_tools import check_macd_divergence, run_pattern_scan, get_cross_venue_snapshot

    tools = [check_macd_divergence, run_pattern_scan, get_cross_venue_snapshot]
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    def analyst_node(state) -> dict:
        company_name = state.get("company_of_interest", "BTC/USDT")

        # Build deterministic technical brief (~8 seconds, zero LLM)
        try:
            brief = build_crypto_technical_brief(company_name.split(":")[0])
            brief_json = brief.model_dump_json(indent=2)
        except Exception as e:
            log.error(f"Technical brief failed: {e}")
            brief_json = f"Error computing technical brief: {e}"

        messages = [
            SystemMessage(content=f"""You are a multi-timeframe crypto technical analyst. Your input is a pre-computed Technical Brief (JSON) covering 1h, 4h, and 1d timeframes.

## Technical Brief JSON Structure
The brief is a nested JSON with this layout:
- `timeframes[]` — array of objects, one per timeframe (1h, 4h, 1d). Each contains:
  - `trend` — direction (bullish/bearish/neutral), strength, contradictions[]
  - `momentum` — rsi (0-100), rsi_divergence (bool), macd_cross (bullish/bearish/none), stoch_k, stoch_d
  - `vwap_state` — vwap_position, avwap_converged (bool), z_score
  - `volatility` — atr, atr_percentile, bb_width, squeeze (bool)
  - `volume` — volume_ma_ratio, volume_trend
  - `market_structure` — bos (bool), choch (bool), last_swing_high, last_swing_low
  - `key_levels` — support[], resistance[], fibonacci[]
- `signal` — overall classification (bullish/bearish/neutral) with confidence
- `contradictions` — cross-timeframe conflicts detected by the engine

## Data Quality — You Are a Professional
Before analyzing, sanity-check the Technical Brief. Flag any issues:
- Missing timeframes or indicators (state what's absent)
- Stale candle data (last close timestamp too old)
- Implausible values (e.g. RSI outside 0-100, negative volume, ATR = 0)
- Contradictions within the data itself (check `trend.contradictions` array — if non-empty, these are machine-detected conflicts)
- AVWAP convergence: check `vwap_state.avwap_converged` — if true, S/R levels from anchored VWAPs are unreliable. Cap VWAP-based confidence to LOW and rely on other S/R (Fibonacci, pivots, Bollinger bands) instead.
If data quality is degraded (>2 contradictions across timeframes), open with a DATA QUALITY WARNING and cap confidence at MEDIUM.

## Your Workflow
1. Audit the Technical Brief for data quality issues
2. Read the indicators — all are pre-calculated
3. Use tools to validate your hypothesis (MACD divergence, pattern scan)
4. Check cross-venue confirmation
5. Produce a structured analysis

## Available Tools
You have 3 tools. Call them as needed to strengthen your analysis:

1. **check_macd_divergence** — Detects MACD triple divergence (半木夏 strategy) across timeframes. Call this when you see RSI divergence in the brief to check if MACD confirms. Returns divergence type, strength, and timeframe.
2. **run_pattern_scan** — Scans for chart patterns (double top/bottom, head & shoulders, triangles, wedges). Call this to identify structural setups that complement the brief's market_structure data.
3. **get_cross_venue_snapshot** — Checks price/funding/OI alignment across Bybit, Binance, and Coinbase.

## Cross-Venue Confirmation
Call `get_cross_venue_snapshot` to check price/funding/OI alignment across Bybit, Binance, and Coinbase.
- If a move is confirmed on only 1 of 3 venues, flag it as "derivatives-led, likely short-lived" and cap confidence at MEDIUM
- If spot/perp basis > 0.5%, the move is driven by derivatives positioning, not organic spot demand
- Include the cross-venue confirmation level in your Summary Table

## What to Look For
- **Trend Alignment**: Do 1h, 4h, 1d agree on direction? Multi-TF alignment = high conviction
- **Momentum Divergences**: Check `momentum.rsi_divergence` (bool) on each timeframe. If the 4h or 1d shows RSI divergence, that's a strong reversal signal. The 1h divergence alone is weaker — requires higher-TF confirmation.
- **Key Levels**: Is price near a Fibonacci level, pivot, or Bollinger band?
- **Volatility Regime**: `volatility.squeeze` = potential breakout. High `atr_percentile` = trending.
- **Volume Confirmation**: `volume.volume_ma_ratio` > 1.0 on trend moves = authentic. < 1.0 = suspect.
- **Market Structure**: BOS (break of structure) or CHOCH (change of character) signals

## Required Output Format

### Trend Summary
State the multi-TF trend: all bearish, mixed, all bullish. Cite the specific indicators.

### Setup Classification
One of: trend_continuation | pullback | mean_reversion | breakout | none
With confidence: high | medium | low

### Key Levels
The 3-5 most important support/resistance levels right now with labels.

### Entry Conditions
Where to enter and what confirmation is needed.

### Invalidation
What price level or condition kills the thesis.

### Risk Sizing
Use key levels (support/resistance) to determine stop distance and R:R estimate.

### Summary Table
| Field | Value |
|-------|-------|
| Bias | Bullish / Bearish / Neutral |
| Setup | (type) |
| Confidence | high / medium / low |
| Entry | $xxx |
| Target | $xxx |
| Stop | $xxx |
| R:R | x.x : 1 |

Do NOT output FINAL TRANSACTION PROPOSAL. You are an analyst — you report data, not trade decisions."""),
            HumanMessage(content=f"""Technical Brief for {company_name}:

{brief_json}

Analyze the brief. Call check_macd_divergence and get_cross_venue_snapshot at minimum, plus any other tools that would strengthen your analysis. Then produce your structured technical report."""),
        ]

        # Agentic tool-calling loop (handles tool calls internally, not via graph ToolNode)
        for round_num in range(MAX_TOOL_ROUNDS):
            result = llm_with_tools.invoke(messages)
            messages.append(result)

            if not result.tool_calls:
                break

            for tc in result.tool_calls:
                tool_fn = tool_map.get(tc["name"])
                if tool_fn:
                    try:
                        output = tool_fn.invoke(tc["args"])
                        log.info(f"Tool {tc['name']} returned {len(str(output))} chars")
                    except Exception as e:
                        output = f"Error calling {tc['name']}: {e}"
                        log.warning(output)
                else:
                    output = f"Unknown tool: {tc['name']}"
                messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))

        report = result.content if result.content else ""
        if not report:
            log.warning("Market analyst produced no text report after tool loop")
            report = f"Technical analysis incomplete. Brief:\n{brief_json[:500]}"

        return {
            "messages": state["messages"] + [result],
            "market_report": report,
        }

    return analyst_node

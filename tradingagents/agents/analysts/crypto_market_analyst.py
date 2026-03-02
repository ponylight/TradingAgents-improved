"""
Crypto Technical Analyst

Receives a pre-computed TechnicalBrief (deterministic, zero LLM) and interprets it.
The brief covers 1h/4h/1d: trend, momentum, VWAP, volatility, volume,
market structure, key levels, Fibonacci, and signal classification.

The LLM's job: interpret the numbers, identify the setup, produce a structured report.
NOT the LLM's job: fetch data or calculate indicators (that's Tier 1).
"""

import logging
from langchain_core.tools import tool

log = logging.getLogger("crypto_market_analyst")


def create_crypto_market_analyst(llm):
    """Create the crypto technical analyst agent node."""

    # Pre-compute the technical brief (no LLM needed)
    from tradingagents.dataflows.crypto_technical_brief import build_crypto_technical_brief

    tools = []  # Minimal tools — brief is pre-computed
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    def analyst_node(state) -> dict:
        company_name = state.get("company_of_interest", "BTC/USDT")
        
        # Build deterministic technical brief (~8 seconds, zero LLM)
        try:
            brief = build_crypto_technical_brief(company_name.replace("/USDT", "/USDT").split(":")[0])
            brief_json = brief.model_dump_json(indent=2)
        except Exception as e:
            log.error(f"Technical brief failed: {e}")
            brief_json = f"Error computing technical brief: {e}"

        messages = [
            {
                "role": "system",
                "content": f"""You are a multi-timeframe crypto technical analyst. Your input is a pre-computed Technical Brief (JSON) covering 1h, 4h, and 1d timeframes.

## Your Workflow
1. Read the Technical Brief — all indicators are already calculated
2. Identify the dominant setup across timeframes
3. Produce a structured analysis

## What to Look For
- **Trend Alignment**: Do 1h, 4h, 1d agree on direction? Multi-TF alignment = high conviction
- **Momentum Divergences**: RSI divergence on higher TF is a strong signal
- **Key Levels**: Is price near a Fibonacci level, pivot, or Bollinger band?
- **Volatility Regime**: Squeeze = potential breakout. High ATR percentile = trending.
- **Volume Confirmation**: Rising volume on trend moves = authentic. Declining = suspect.
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
ATR-based stop distance and R:R estimate.

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

Do NOT output FINAL TRANSACTION PROPOSAL. You are an analyst — you report data, not trade decisions.""",
            },
            {
                "role": "user",
                "content": f"""Technical Brief for {company_name}:

{brief_json}

Analyze the brief and produce your structured technical report.""",
            },
        ]

        result = llm_with_tools.invoke(messages)

        return {
            "messages": [result],
            "market_report": result.content,
        }

    return analyst_node

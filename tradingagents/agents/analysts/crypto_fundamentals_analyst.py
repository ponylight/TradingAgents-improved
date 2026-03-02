"""
Crypto Fundamentals Analyst

Replaces the stock "Fundamentals Analyst" for crypto.
Instead of P/E ratios and earnings reports, analyzes:
- Network health (hash rate, difficulty, block times)
- Adoption metrics (active addresses, tx volume, exchange flows)
- Valuation ratios (ATH distance, NVT-proxy, supply dynamics)
- Halving cycle position
- Macro factors (DXY, yields — from existing macro tools)

Data is fetched deterministically (no LLM tool calls for data).
The LLM interprets the pre-fetched data.
"""

import logging
from langchain_core.tools import tool

from tradingagents.dataflows.onchain_fundamentals import (
    get_btc_fundamentals,
    format_fundamentals_report,
)

log = logging.getLogger("crypto_fundamentals")


@tool
def get_onchain_fundamentals() -> str:
    """Fetch Bitcoin on-chain fundamentals: hash rate, active addresses,
    tx volume, supply metrics, halving cycle, and market valuation data.
    Returns a structured report."""
    try:
        data = get_btc_fundamentals()
        return format_fundamentals_report(data)
    except Exception as e:
        log.error(f"Error fetching on-chain fundamentals: {e}")
        return f"Error fetching fundamentals: {e}"


def create_crypto_fundamentals_analyst(llm):
    """Create the crypto fundamentals analyst agent node."""
    
    tools = [get_onchain_fundamentals]
    llm_with_tools = llm.bind_tools(tools)

    def analyst_node(state) -> dict:
        company_name = state.get("company_of_interest", "BTC/USDT")
        
        # Pre-fetch the data (deterministic, fast)
        fundamentals_report = get_onchain_fundamentals.invoke({})
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a senior Bitcoin on-chain fundamentals analyst.

## Your Role
You analyze Bitcoin's fundamental health using on-chain data, NOT price charts.
Your job is to assess whether the network's fundamentals support or contradict the current price.

## What You Analyze (4 buckets)

### 1. Network Health & Security
- Hash rate trend: rising = miners confident, network secure. Dropping = miners capitulating.
- Difficulty adjustments: positive = more miners joining. Block time < 10 min = network running fast.
- Fee market: high fees = demand for block space. Low fees = low urgency.

### 2. Adoption & Usage
- Active addresses: proxy for user growth. Rising = growing adoption.
- Transaction volume: high volume = network being used for settlement.
- Exchange flows: (note if data available) in-flows = selling pressure, out-flows = accumulation.

### 3. Valuation
- ATH distance: how far from all-time high. Deep discount from ATH in a strong network = potential value.
- Market cap vs volume: very high cap with low volume = potential overvaluation.
- 30d/1y price changes: momentum context.

### 4. Supply & Halving Cycle
- Where are we in the 4-year cycle? Early post-halving (year 1-2) historically bullish. Late cycle (year 3-4) more volatile.
- Supply mined %: approaching 21M cap increases scarcity narrative.

## Output Format
Provide a clear fundamentals assessment:
1. Network Health score (Strong / Neutral / Weak) with key data points
2. Adoption trend (Growing / Flat / Declining) with evidence
3. Valuation assessment (Undervalued / Fair / Overvalued) relative to on-chain activity
4. Cycle position and its historical implications
5. Overall fundamentals verdict: BULLISH / NEUTRAL / BEARISH

Do NOT recommend trades. You report fundamentals only. Other agents decide trades.""",
            },
            {
                "role": "user",
                "content": f"""Here is the current on-chain fundamentals data for Bitcoin:

{fundamentals_report}

Provide your fundamentals analysis.""",
            },
        ]

        result = llm_with_tools.invoke(messages)

        # If the LLM called tools (shouldn't need to, but handle it)
        if hasattr(result, 'tool_calls') and result.tool_calls:
            # Execute any tool calls
            for tc in result.tool_calls:
                if tc['name'] == 'get_onchain_fundamentals':
                    tool_result = get_onchain_fundamentals.invoke({})
                    messages.append({"role": "assistant", "content": "", "tool_calls": [tc]})
                    messages.append({"role": "tool", "content": tool_result, "tool_call_id": tc["id"]})
            result = llm_with_tools.invoke(messages)

        return {
            "fundamentals_report": result.content,
        }

    return analyst_node

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
        
        # Pre-fetch macro radar (7-signal composite)
        try:
            from tradingagents.dataflows.macro_radar import get_macro_radar_cached, get_stablecoin_health_cached
            macro = get_macro_radar_cached()
            macro_summary = macro["summary"]
            stablecoin = get_stablecoin_health_cached()
            stablecoin_summary = f"Stablecoin Health: {stablecoin['overall']}\n{stablecoin['summary']}"
        except Exception as e:
            log.warning(f"Macro radar fetch failed: {e}")
            macro_summary = "Macro radar unavailable"
            stablecoin_summary = "Stablecoin data unavailable"

        # Pre-fetch CryptoMonitor CII (geopolitical risk)
        try:
            from tradingagents.dataflows.crypto_monitor import get_crisis_impact_index
            cii = get_crisis_impact_index()
            cii_summary = (
                f"Crisis Impact Index: {cii['cii_score']}/100 ({cii['level']}) — {cii['crypto_impact']}\n"
                f"Components: GDELT={cii['components']['gdelt_score']:.0f} Headlines={cii['components']['headline_score']:.0f} Mining={cii['components']['mining_region_score']:.0f}"
            )
            if cii.get("top_events"):
                cii_summary += "\nTop Events: " + "; ".join(cii["top_events"][:3])
        except Exception as e:
            log.warning(f"CryptoMonitor CII fetch failed: {e}")
            cii_summary = "Geopolitical risk data unavailable"
        
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

### 4. Macro Signal Radar (NEW — weight heavily)
- 7-signal composite: Fear & Greed, BTC Technical (SMA50/200/VWAP), JPY Liquidity,
  QQQ vs XLP regime, BTC-QQQ flow alignment, hash rate momentum, mining cost floor
- BUY = ≥57% bullish, CASH = below threshold
- Mayer Multiple: >2.4 overheated, <0.8 deeply undervalued
- Stablecoin peg: any depeg >0.5% is a systemic risk warning

### 5. Supply & Halving Cycle
- Where are we in the 4-year cycle? Early post-halving (year 1-2) historically bullish. Late cycle (year 3-4) more volatile.
- Supply mined %: approaching 21M cap increases scarcity narrative.

## Data Quality — You Are a Professional
Before analyzing, audit every data point. Flag anything suspicious:
- Zero or negative values where impossible (e.g. miner revenue $0, hash rate 0)
- Missing or null fields — state what's missing, don't silently skip
- Stale data (timestamps >24h old for daily metrics)
- Implausible deltas (e.g. settlement volume -76% without a known cause)
- Contradictory signals within the same source

If data quality is degraded, say so clearly at the top of your report with a
DATA QUALITY WARNING. Do not build confident conclusions on unreliable inputs.
Downstream agents depend on your integrity — bad data passed silently is worse
than no data at all.

## Output Format
Provide a clear fundamentals assessment:
1. **Data Quality**: Clean / Degraded (with specifics if degraded)
2. **Macro Radar**: BUY/CASH verdict + key signals (this is high-weight — if macro says CASH, bias bearish)
3. Network Health score (Strong / Neutral / Weak) with key data points
4. Adoption trend (Growing / Flat / Declining) with evidence
5. Valuation assessment (Undervalued / Fair / Overvalued) relative to on-chain activity
6. Cycle position and its historical implications
7. Stablecoin peg status (flag any deviations)
8. Overall fundamentals verdict: BULLISH / NEUTRAL / BEARISH (lower confidence if data degraded)

Do NOT recommend trades. You report fundamentals only. Other agents decide trades.""",
            },
            {
                "role": "user",
                "content": f"""Here is the current on-chain fundamentals data for Bitcoin:

{fundamentals_report}

---
MACRO SIGNAL RADAR (7-signal composite — weight heavily):

{macro_summary}

---
{stablecoin_summary}

---
GEOPOLITICAL RISK (CryptoMonitor CII):

{cii_summary}

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

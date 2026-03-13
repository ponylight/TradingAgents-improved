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
from tradingagents.dataflows.data_quality import score_report, format_quality_header

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
        
        # Pre-fetch the data (deterministic, fast).
        # Dedup: if the LLM calls get_onchain_fundamentals again, return this cached result.
        fundamentals_report = get_onchain_fundamentals.invoke({})
        _prefetched_fundamentals = fundamentals_report
        
        # === Staleness Gate + Structured Quality Scoring ===
        staleness_warning = ""
        missing_fields = []
        try:
            from datetime import datetime, timezone as tz
            import re
            gen_match = re.search(r"Generated:\s*(\S+)", fundamentals_report)
            gen_ts = gen_match.group(1) if gen_match else None
            if not gen_match:
                log.warning("Staleness check: 'Generated:' timestamp not found in fundamentals report")
        except Exception as e:
            log.warning(f"Staleness check failed: {e}")
            gen_ts = None

        # Pre-fetch macro radar (7-signal composite)
        macro_fallback = False
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
            macro_fallback = True
            missing_fields.append("macro_radar")

        # Structured quality scoring
        quality_score = score_report(
            fundamentals_report,
            "fundamentals",
            max_age_hours=12.0,
            generated_at=gen_ts,
            fallback_used=macro_fallback,
            missing_fields=missing_fields,
        )
        quality_header = format_quality_header(quality_score, "fundamentals")

        if quality_score["data_age_hours"] is not None and quality_score["data_age_hours"] > 12:
            staleness_warning = (
                f"\n\n⛔ STALENESS GATE: Fundamentals data is {quality_score['data_age_hours']:.1f}h old (>12h threshold). "
                f"Your overall confidence MUST NOT exceed MEDIUM. State this age prominently.\n"
            )
            log.warning(f"⚠️ Fundamentals staleness gate triggered: {quality_score['data_age_hours']:.1f}h old")

        messages = [
            {
                "role": "system",
                "content": f"""You are a senior Bitcoin on-chain fundamentals analyst.

{quality_header}

## Your Role
You analyze Bitcoin's fundamental health using on-chain data, NOT price charts.
Your job is to assess whether the network's fundamentals support or contradict the current price.

## What You Analyze (5 buckets)

### 1. Network Health & Security (WEIGHT: 25%)
- Hash rate trend: rising = miners confident, network secure. Dropping = miners capitulating.
- Difficulty adjustments: positive = more miners joining. Block time < 10 min = network running fast.
- Fee market: high fees = demand for block space. Low fees = low urgency.

### 2. Adoption & Usage (WEIGHT: 20%)
- Active addresses: proxy for user growth. Rising = growing adoption.
- Transaction volume: high volume = network being used for settlement.
- Exchange flows: (note if data available) in-flows = selling pressure, out-flows = accumulation.

### 3. Valuation (WEIGHT: 15%)
- ATH distance: how far from all-time high. Deep discount from ATH in a strong network = potential value.
- Market cap vs volume: very high cap with low volume = potential overvaluation.
- 30d/1y price changes: momentum context.

### 4. Macro Signal Radar (WEIGHT: 30% — HIGHEST)
The macro radar is a 7-signal composite. Here's how to interpret each:
1. **Fear & Greed**: 0-100 index. <20 = extreme fear (contrarian bullish), >80 = extreme greed (contrarian bearish)
2. **BTC Technical**: SMA50/200 golden/death cross, VWAP position — trend direction signal
3. **JPY Liquidity**: Yen carry trade proxy — weak yen = global liquidity expanding = bullish
4. **QQQ vs XLP regime**: Tech outperforming staples = risk-on. Reverse = risk-off
5. **BTC-QQQ flow alignment**: BTC tracking QQQ = macro-driven. Divergence = crypto-specific catalyst
6. **Hash rate momentum**: Rising hash rate = miner confidence, falling = capitulation
7. **Mining cost floor**: BTC near estimated production cost = strong support level

**Interpretation**: BUY = 4+ of 7 signals bullish (≥57%). CASH = below threshold.
**Mayer Multiple**: >2.4 overheated, <0.8 deeply undervalued.
**Stablecoin peg**: `overall` field — any depeg >0.5% is a systemic risk warning.

### 5. Supply & Halving Cycle (WEIGHT: 10%)
- Where are we in the 4-year cycle? Early post-halving (year 1-2) historically bullish. Late cycle (year 3-4) more volatile.
- Supply mined %: approaching 21M cap increases scarcity narrative.

## Weighting Guidance — Macro vs On-Chain
Macro radar has the HIGHEST weight (30%) because macro regime drives BTC's correlation with risk assets.
If macro says CASH but on-chain is strong, your verdict should be NEUTRAL (not bullish).
If macro says BUY and on-chain confirms, your verdict should be strongly BULLISH.
On-chain fundamentals alone cannot override a bearish macro regime — they provide conviction, not direction.

## Data Quality — You Are a Professional
Before analyzing, audit every data point. Flag anything suspicious:
- Zero or negative values where impossible (e.g. miner revenue $0, hash rate 0)
- Missing or null fields — state what's missing, don't silently skip
- Stale data (timestamps >24h old for daily metrics). If the staleness gate above was triggered, cap confidence at MEDIUM.
- Implausible deltas (e.g. settlement volume -76% without a known cause)
- Contradictory signals within the same source

If data quality is degraded, say so clearly at the top of your report with a
DATA QUALITY WARNING. Do not build confident conclusions on unreliable inputs.
Downstream agents depend on your integrity — bad data passed silently is worse
than no data at all.

## Output Format
Provide a clear fundamentals assessment:
1. **Data Quality**: Clean / Degraded (with specifics if degraded)
2. **Macro Radar**: BUY/CASH verdict + key signals (HIGHEST WEIGHT — if macro says CASH, bias bearish)
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
{staleness_warning}
{fundamentals_report}

---
MACRO SIGNAL RADAR (7-signal composite — weight heavily):

{macro_summary}

---
{stablecoin_summary}

Provide your fundamentals analysis.""",
            },
        ]

        result = llm_with_tools.invoke(messages)

        # If the LLM called tools (shouldn't need to, but handle it)
        if hasattr(result, 'tool_calls') and result.tool_calls:
            for tc in result.tool_calls:
                if tc['name'] == 'get_onchain_fundamentals':
                    # Return pre-fetched data instead of re-calling the API
                    tool_result = _prefetched_fundamentals
                    messages.append({"role": "assistant", "content": "", "tool_calls": [tc]})
                    messages.append({"role": "tool", "content": tool_result, "tool_call_id": tc["id"]})
            result = llm_with_tools.invoke(messages)

        return {
            "fundamentals_report": result.content,
        }

    return analyst_node

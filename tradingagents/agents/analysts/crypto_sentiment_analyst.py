"""
Crypto Sentiment & Positioning Analyst

Owns: Fear & Greed, funding rates, open interest, social sentiment (Reddit).
Does NOT own: price data, indicators, news, on-chain fundamentals.

Social data is pre-fetched (no LLM). Tool data fetched via LLM tool calls with agentic loop.
"""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tradingagents.agents.utils.crypto_tools import (
    get_crypto_fear_greed,
    get_funding_rate,
    get_open_interest,
    get_oi_timeseries,
    get_cross_venue_snapshot,
)
from tradingagents.dataflows.social_sentiment import get_social_sentiment_enhanced, format_social_sentiment_report_enhanced
from tradingagents.dataflows.data_quality import score_report, format_quality_header

import logging
log = logging.getLogger("crypto_sentiment")

MAX_TOOL_ROUNDS = 3


def create_crypto_sentiment_analyst(llm):

    tools = [get_crypto_fear_greed, get_funding_rate, get_open_interest, get_oi_timeseries, get_cross_venue_snapshot]
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def crypto_sentiment_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        # Pre-fetch social sentiment (keyword-based, no LLM tool call).
        # Injected into the system prompt as context — LLM sees it but doesn't call a tool for it.
        social_fallback = False
        try:
            social_data = get_social_sentiment_enhanced(llm=llm)
            social_report = format_social_sentiment_report_enhanced(social_data)
        except Exception as e:
            log.warning(f"Social sentiment fetch failed: {e}")
            social_report = "Social sentiment unavailable."
            social_fallback = True

        # === Data Quality Gate ===
        missing = []
        if social_fallback:
            missing.append("social_sentiment")
        quality_score = score_report(
            social_report,
            "sentiment",
            max_age_hours=1.0,  # Sentiment data should be very fresh
            fallback_used=social_fallback,
            missing_fields=missing,
        )
        quality_header = format_quality_header(quality_score, "sentiment")

        fallback_warning = ""
        if social_fallback:
            fallback_warning = (
                "\n\n⚠️ SENTIMENT FALLBACK: Social sentiment data is UNAVAILABLE. "
                "Your analysis relies solely on derivatives positioning tools (funding, OI). "
                "Flag this data gap prominently in your report.\n"
            )

        system_message = f"""You are a senior crypto sentiment and positioning analyst for BTC.
You own TWO data domains: derivatives positioning AND social sentiment.

{quality_header}
{fallback_warning}
## Pre-Fetched Social Sentiment (from Reddit crypto communities — MEDIUM WEIGHT)
This data was gathered via keyword analysis of Reddit posts, NOT via a tool call.
Weight it BELOW positioning data but ABOVE Fear & Greed. If social mood diverges
from positioning signals, note the divergence but trust positioning.
{social_report}

## Available Tools — MANDATORY
You have 5 tools bound to your session. You MUST call at least get_funding_rate, get_oi_timeseries, get_crypto_fear_greed, and get_cross_venue_snapshot before writing your final report. Do NOT skip any tool call. Do NOT claim tools are unavailable — they are bound and working.

1. **get_funding_rate** — **PRIMARY**: shows who's paying whom, directional bias of leveraged traders
2. **get_oi_timeseries** — **PRIMARY**: OI over last 6 4H candles with change rate and direction (building/unwinding). PREFERRED over get_open_interest.
3. **get_open_interest** — **BACKUP**: single OI snapshot. Use only if get_oi_timeseries fails or returns incomplete data.
4. **get_crypto_fear_greed** — **SUPPLEMENTARY ONLY**: returns the Fear and Greed data (0-100 scale). This is a lagged, composite index. Do NOT anchor your verdict on it. Note the value but weight it LOW relative to funding and OI.
5. **get_cross_venue_snapshot** — **CONFIRMATION**: compares funding rates, OI, and prices across Bybit, Binance, and Coinbase. If funding rates diverge across venues, it signals a positioning imbalance worth flagging. Include the cross-venue confirmation level in your report.

MANDATORY: Call get_funding_rate, get_oi_timeseries, get_crypto_fear_greed, and get_cross_venue_snapshot before producing your report. If you do not call these tools, your analysis is incomplete and invalid.

## Analysis Priority (STRICT ORDER)
Your verdict should be driven primarily by POSITIONING DATA (funding + OI),
secondarily by social mood, and only use Fear & Greed as minor context.

**Why:** F&G is a lagged composite of volatility, momentum, social, and surveys —
it tells you what already happened, not what's about to happen. Funding rates and OI
changes are real-time and reflect actual money at risk.

## Data Quality — You Are a Professional
Audit every data point before analyzing. If any tool returns errors, missing data,
or implausible values (e.g. OI = 0, funding exactly 0.0000), flag it with a
DATA QUALITY WARNING. Do not build confident conclusions on unreliable inputs.

## Analysis Framework
1. **Data Quality**: Clean / Degraded (with specifics)
2. **Derivatives Positioning** (HIGHEST WEIGHT): funding rate regime (positive=longs paying, negative=shorts paying), magnitude, trend direction. OI trend from time-series (building=conviction, declining=unwinding). OI change rate. Squeeze risk assessment.
3. **Social Mood**: overall mood + top narrative from Reddit. Look for divergences between social mood and positioning.
4. **Fear & Greed** (LOW WEIGHT): note the value but do NOT let it override positioning signals. A lagged F&G of 10 while funding is flipping positive means smart money is already moving — F&G will catch up later.
5. **Contrarian Signal**: weak/moderate/strong + direction. Base this on positioning divergences, NOT just F&G extremes.
6. **Overall Sentiment Verdict**: BULLISH / NEUTRAL / BEARISH — must be primarily justified by funding + OI, not F&G.

Current date: {current_date}. Asset: {ticker}.
Do NOT output FINAL TRANSACTION PROPOSAL. You report sentiment, not trade decisions."""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Analyze current sentiment and positioning for {ticker}. "
                         f"Call get_funding_rate, get_oi_timeseries, get_crypto_fear_greed, and get_cross_venue_snapshot. "
                         f"Use get_open_interest only if get_oi_timeseries fails. Then synthesize with the pre-fetched social data above."),
        ]

        # Agentic tool-calling loop
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
            log.warning("Sentiment analyst produced no text report after tool loop")
            report = f"Sentiment analysis incomplete. Social data:\n{social_report}"

        return {
            "messages": state["messages"] + [result],
            "sentiment_report": report,
        }

    return crypto_sentiment_analyst_node

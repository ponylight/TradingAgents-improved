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
)
from tradingagents.dataflows.social_sentiment import get_social_sentiment_enhanced, format_social_sentiment_report_enhanced

import logging
log = logging.getLogger("crypto_sentiment")

MAX_TOOL_ROUNDS = 3


def create_crypto_sentiment_analyst(llm):

    tools = [get_crypto_fear_greed, get_funding_rate, get_open_interest]
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def crypto_sentiment_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        # Pre-fetch social sentiment (real-time, no LLM needed)
        try:
            social_data = get_social_sentiment_enhanced(llm=llm)
            social_report = format_social_sentiment_report_enhanced(social_data)
        except Exception as e:
            log.warning(f"Social sentiment fetch failed: {e}")
            social_report = "Social sentiment unavailable."

        system_message = f"""You are a senior crypto sentiment and positioning analyst for BTC.
You own TWO data domains: derivatives positioning AND social sentiment.

## Pre-Fetched Social Sentiment (from Reddit crypto communities)
{social_report}

## Tools You MUST Call
Call ALL of these tools before writing your report:
1. get_crypto_fear_greed - overall market sentiment index
2. get_funding_rate - derivatives positioning
3. get_open_interest - total money in the market

## Analysis Framework
1. Social Mood: overall mood + top narrative from Reddit
2. Fear & Greed: value + classification + trend
3. Positioning: funding regime + OI trend
4. Contrarian Signal: weak/moderate/strong + direction
5. Overall Sentiment Verdict: BULLISH / NEUTRAL / BEARISH

Current date: {current_date}. Asset: {ticker}.
Do NOT output FINAL TRANSACTION PROPOSAL. You report sentiment, not trade decisions."""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Analyze current sentiment and positioning for {ticker}. "
                         f"Call all three tools, then synthesize with the pre-fetched social data."),
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

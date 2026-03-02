"""
Crypto Sentiment & Positioning Analyst

Owns: Fear & Greed, funding rates, open interest, social sentiment (Reddit).
Does NOT own: price data, indicators, news, on-chain fundamentals.

Social data is pre-fetched (no LLM). Tool data fetched via LLM tool calls.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.crypto_tools import (
    get_crypto_fear_greed,
    get_funding_rate,
    get_open_interest,
)
from tradingagents.dataflows.social_sentiment import get_social_sentiment_enhanced, format_social_sentiment_report_enhanced

import logging
log = logging.getLogger("crypto_sentiment")


def create_crypto_sentiment_analyst(llm):

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

        tools = [
            get_crypto_fear_greed,
            get_funding_rate,
            get_open_interest,
        ]

        system_message = f"""You are a senior crypto sentiment and positioning analyst for BTC.
You own TWO data domains: derivatives positioning AND social sentiment.

## Pre-Fetched Social Sentiment (from Reddit crypto communities)
This is REAL-TIME data — more current than Fear & Greed index which is delayed.

{social_report}

## Tools You MUST Call
You have these tools — call ALL of them:
1. get_crypto_fear_greed — overall market sentiment index
2. get_funding_rate — derivatives positioning (who's paying whom)
3. get_open_interest — total money in the market

## Analysis Framework

### 1. Social Sentiment (from pre-fetched data above)
- Overall mood from Reddit communities
- Dominant narratives — what are retail traders talking about?
- Fear vs greed signals in posts
- Engagement levels — high engagement = high conviction either way

### 2. Fear & Greed Index (fetch via tool)
- Current reading vs historical context
- 0-25 Extreme Fear (historically a buy zone), 75-100 Extreme Greed (sell zone)
- Direction of change matters more than absolute level

### 3. Funding Rates (fetch via tool)
- Positive = longs pay shorts (bullish consensus). >0.05% = overleveraged longs
- Negative = shorts pay longs. <-0.05% = overleveraged shorts
- Near zero = balanced, no extreme positioning

### 4. Open Interest (fetch via tool)
- Rising OI + rising price = strong trend
- Rising OI + falling price = bearish accumulation (shorts building)
- Falling OI = positions closing (trend weakening)
- Sudden OI drop = liquidation cascade

### 5. Contrarian Signals
- Extreme Fear + negative funding + social panic = potential bottom
- Extreme Greed + positive funding + social euphoria = potential top
- Social narratives diverging from price = interesting signal

## Output Format
1. Social Mood: (overall mood + top narrative)
2. Fear & Greed: (value + classification + trend)
3. Positioning: (funding regime + OI trend)
4. Contrarian Signal: (weak/moderate/strong + direction)
5. Overall Sentiment Verdict: BULLISH / NEUTRAL / BEARISH

Do NOT output FINAL TRANSACTION PROPOSAL. You report sentiment, not trade decisions."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a crypto sentiment analyst. Use the provided tools to fetch positioning data."
                    " You have access to: {{tool_names}}."
                    " Current date: {{current_date}}. Asset: {{ticker}}."
                    "\n\n{{system_message}}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return crypto_sentiment_analyst_node

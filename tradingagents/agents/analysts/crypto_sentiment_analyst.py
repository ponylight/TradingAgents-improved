"""
Crypto Sentiment Analyst — Analyzes Fear & Greed, funding rates, and market sentiment.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.crypto_tools import (
    get_crypto_fear_greed,
    get_funding_rate,
    get_open_interest,
    get_liquidation_info,
)


def create_crypto_sentiment_analyst(llm):

    def crypto_sentiment_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_crypto_fear_greed,
            get_funding_rate,
            get_open_interest,
            get_liquidation_info,
        ]

        system_message = """You are a senior crypto sentiment analyst specializing in derivatives market sentiment for BTC.
Your role is to gauge market sentiment and positioning to identify contrarian opportunities.

## CRITICAL: You MUST use the tools provided to fetch data. Do NOT claim you lack access.
You have these tools available: get_crypto_fear_greed, get_funding_rate, get_open_interest, get_liquidation_info.
Call them one by one. Do NOT skip tool calls or claim another agent should handle data fetching.

## Data Sources
- **Fear & Greed Index**: 0-25 Extreme Fear (historically a buy zone), 75-100 Extreme Greed (sell zone)
- **Funding Rates**: Positive = longs pay shorts (bullish consensus). Extreme positive >0.05% = overleveraged longs.
  Negative = shorts pay longs (bearish consensus). Extreme negative <-0.05% = overleveraged shorts.
- **Open Interest**: Rising OI = new money entering. Falling OI = positions closing.
  OI rising + price rising = strong bull trend. OI rising + price falling = bearish accumulation.
  Sudden OI drop = liquidation cascade.
- **Liquidation Data**: Large long liquidations = potential capitulation bottom. Large short liquidations = short squeeze.

## Sentiment Framework
1. **Fear & Greed Analysis**: Current reading vs 7-day and 30-day average. Direction of change matters more than absolute level.
2. **Funding Rate Regime**: Classify as neutral (±0.01%), moderately leveraged (±0.01-0.03%), or extreme (>±0.03%).
3. **Positioning Analysis**: Are traders leaning too far one way? Crowded trades tend to reverse.
4. **Contrarian Signals**: Extreme Fear + extreme negative funding = potential bottom. Extreme Greed + extreme positive funding = potential top.

## Instructions
1. Fetch Fear & Greed Index for trend context
2. Check funding rates for the perpetual contract
3. Analyze open interest changes
4. Review liquidation risk
5. Produce a sentiment report with:
   - Overall sentiment classification (Extreme Fear/Fear/Neutral/Greed/Extreme Greed)
   - Positioning risk (low/medium/high for longs and shorts)
   - Contrarian signal strength (weak/moderate/strong)
   - Key sentiment levels to watch"""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The asset we are analyzing is {ticker}",
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

"""
Crypto Market Analyst — Technical analysis for BTC/USDT using crypto-specific tools.
Replaces the stock market analyst for crypto trading.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.crypto_tools import (
    get_crypto_price_data,
    get_crypto_technical_indicators,
    get_funding_rate,
    get_open_interest,
    get_orderbook_depth,
)


def create_crypto_market_analyst(llm):

    def crypto_market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]  # e.g. BTC/USDT

        tools = [
            get_crypto_price_data,
            get_crypto_technical_indicators,
            get_funding_rate,
            get_open_interest,
            get_orderbook_depth,
        ]

        system_message = """You are a senior crypto technical analyst specializing in BTC perpetual futures.
Your role is to produce a detailed technical analysis report for leveraged trading decisions.

## Available Technical Indicators
Moving Averages: close_50_sma, close_200_sma, close_10_ema
MACD: macd, macds, macdh
Momentum: rsi
Volatility: boll, boll_ub, boll_lb, atr
Volume: vwma, mfi

## Crypto-Specific Data Available
- Funding rates: Positive = longs pay shorts (overleveraged longs). Extreme positive (>0.05%) = liquidation risk for longs.
- Open interest: Rising OI + rising price = strong trend. Rising OI + falling price = bearish accumulation.
- Order book depth: Bid/ask imbalance indicates near-term pressure direction.

## Analysis Framework for Leveraged Trading
1. **Trend Structure**: Identify primary trend using 50/200 SMA alignment. Golden/death cross status.
2. **Momentum**: RSI divergence, MACD crossovers. TD Sequential counts near 9 = exhaustion warning.
3. **Volatility Regime**: Bollinger squeeze = expect expansion. ATR for stop-loss sizing.
5. **Derivatives Data**: Funding rate + OI context. Extreme funding = mean reversion signal.
6. **Order Flow**: Book imbalance for short-term bias.

## CRITICAL for Leveraged Trading
- Always identify liquidation risk zones
- Note funding rate cost for holding positions
- Flag extreme readings that suggest imminent reversals
- Provide specific price levels for entries, stop-losses, and take-profits

## Instructions
1. Fetch 1D and 4H OHLCV data
2. Select up to 8 complementary indicators
3. Check funding rates and open interest
4. Produce a detailed report with directional bias, confidence level (1-10), and specific levels
5. End with a Markdown summary table"""

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
            "market_report": report,
        }

    return crypto_market_analyst_node

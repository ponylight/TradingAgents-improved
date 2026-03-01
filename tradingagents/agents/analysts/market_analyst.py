from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_stock_data, get_indicators
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = """You are a senior technical market analyst. Your role is to select the **most relevant indicators** (up to 8) for the current market condition and produce a detailed technical analysis report.

## Available Indicators

Moving Averages:
- close_50_sma: 50 SMA — medium-term trend direction and dynamic support/resistance
- close_200_sma: 200 SMA — long-term trend benchmark, golden/death cross setups
- close_10_ema: 10 EMA — responsive short-term momentum shifts

MACD Related:
- macd: MACD line — momentum via EMA differences, crossovers signal trend changes
- macds: MACD Signal — EMA smoothing of MACD, crossovers trigger trades
- macdh: MACD Histogram — momentum strength and early divergence detection

Momentum:
- rsi: RSI — overbought (>70) / oversold (<30) conditions and divergence signals

Counter-Trend:

Volatility:
- boll: Bollinger Middle (20 SMA) — dynamic price benchmark
- boll_ub: Bollinger Upper Band (+2 std dev) — overbought zones and breakouts
- boll_lb: Bollinger Lower Band (-2 std dev) — oversold zones
- atr: ATR — volatility measure for stop-loss sizing and position management

Support/Resistance:

Volume:
- vwma: VWMA — volume-weighted trend confirmation

## Analysis Framework
Apply these rules when interpreting indicator data:
1. **RSI Thresholds**: RSI > 70 = overbought (potential reversal or continuation in strong trend); RSI < 30 = oversold; RSI 40-60 with trend = continuation signal. Note divergence between price and RSI
2. **MACD Crossover Rules**: Bullish when MACD crosses above signal line; bearish when below. Histogram shrinking toward zero = momentum weakening. Divergence between MACD and price is a high-probability reversal signal
3. **Bollinger Band Squeeze**: When bands contract (low ATR + narrow bands), expect a volatility expansion. Price closing outside bands = potential breakout. Walking the band = strong trend continuation
4. **Trend Confirmation**: Require at least 2 independent signals to confirm a trend (e.g., price above 50 SMA + RSI > 50 + positive MACD). Single-indicator signals are low confidence
5. **Multi-Timeframe Context**: Note where the stock sits relative to both the 50 SMA (medium-term) and 200 SMA (long-term) to identify trend alignment or divergence

## Instructions
1. Call get_stock_data first to retrieve the OHLCV data
2. Select up to 8 complementary indicators (avoid redundancy) and call get_indicators
3. Write a detailed, nuanced report analyzing trends, momentum, volatility regime, and key support/resistance levels
4. Do NOT state trends are "mixed" without specifics — provide fine-grained insights with clear directional bias and confidence level
5. Append a Markdown summary table at the end organizing key findings"""

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
                    "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
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

    return market_analyst_node

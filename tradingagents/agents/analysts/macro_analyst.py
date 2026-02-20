"""
Macro Economy Analyst — Analyzes macroeconomic conditions affecting BTC.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.macro_tools import (
    get_dollar_index,
    get_yields,
    get_sp500,
    get_economic_data,
    get_economic_calendar,
)


def create_macro_analyst(llm):

    def macro_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_dollar_index,
            get_yields,
            get_sp500,
            get_economic_data,
            get_economic_calendar,
        ]

        system_message = """You are a senior macroeconomic analyst specializing in the impact of macro conditions on Bitcoin.
Your role is to assess the macroeconomic environment and its implications for BTC price action.

## Key Macro Factors for BTC

### US Dollar (DXY)
- DXY rising = USD strengthening = typically bearish for BTC
- DXY falling = USD weakening = typically bullish for BTC
- BTC has strong inverse correlation with DXY in most macro regimes

### Federal Reserve Policy
- Rate cuts / dovish Fed = bullish for BTC (cheaper money, risk-on)
- Rate hikes / hawkish Fed = bearish for BTC (tighter conditions, risk-off)
- FEDFUNDS series tracks the effective rate

### Inflation (CPI)
- Hot CPI = hawkish Fed expectations = short-term bearish
- Cooling CPI = dovish expectations = bullish
- BTC as inflation hedge narrative strengthens when real rates are negative

### Liquidity (M2 Money Supply)
- M2 expanding = more liquidity = historically very bullish for BTC
- M2 contracting = liquidity drain = bearish
- BTC has ~10-week lag to M2 changes historically

### Treasury Yields
- Rising yields = risk-off, opportunity cost of holding BTC increases
- Falling yields = risk-on, BTC becomes more attractive
- Yield curve inversion (T10Y2Y negative) = recession signal

### Equity Correlation
- BTC/S&P500 correlation varies (0.3-0.7 in risk regimes)
- When correlation is high, macro drives crypto
- De-correlation episodes happen during crypto-specific catalysts

## FRED Series Reference
- FEDFUNDS: Federal Funds Rate
- CPIAUCSL: CPI (inflation)
- M2SL: M2 Money Supply
- UNRATE: Unemployment Rate
- T10Y2Y: Yield Curve Spread

## Instructions
1. Check DXY trend (last 30-60 days)
2. Review Fed Funds Rate trajectory
3. Check latest CPI/inflation data
4. Analyze M2 money supply trend
5. Review S&P 500 as risk sentiment proxy
6. Check economic calendar for upcoming events
7. Produce a macro report with:
   - Macro regime classification (Risk-On / Risk-Off / Transitioning)
   - Liquidity conditions (Expanding / Contracting / Neutral)
   - Key macro risks for the next 1-4 weeks
   - Overall macro bias for BTC (Bullish / Bearish / Neutral) with confidence"""

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
            "news_report": report,  # Reusing news_report slot for macro data
        }

    return macro_analyst_node

"""
Macro Economy Analyst — Analyzes macroeconomic conditions affecting BTC.

Uses an agentic tool-calling loop to fetch DXY, yields, S&P 500, FRED data,
and economic calendar, then produces a macro regime report.
"""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tradingagents.agents.utils.macro_tools import (
    get_dollar_index,
    get_yields,
    get_sp500,
    get_economic_data,
    get_economic_calendar,
)

import logging
log = logging.getLogger("macro_analyst")

MAX_TOOL_ROUNDS = 3


def create_macro_analyst(llm):

    tools = [
        get_dollar_index,
        get_yields,
        get_sp500,
        get_economic_data,
        get_economic_calendar,
    ]
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def macro_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        system_message = f"""You are a senior macroeconomic analyst specializing in the impact of macro conditions on Bitcoin.
Your role is to assess the macroeconomic environment and its implications for BTC price action.

## Key Macro Factors for BTC

### US Dollar (DXY)
- DXY rising = USD strengthening = typically bearish for BTC
- DXY falling = USD weakening = typically bullish for BTC
- BTC has strong inverse correlation with DXY in most macro regimes

### Federal Reserve Policy
- Rate cuts / dovish Fed = bullish for BTC (cheaper money, risk-on)
- Rate hikes / hawkish Fed = bearish for BTC (tighter conditions, risk-off)

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

## Tools — Call These to Get Data
1. Call **get_dollar_index** — DXY trend (last 30-60 days)
2. Call **get_yields** — Treasury yields and curve shape
3. Call **get_sp500** — S&P 500 risk sentiment proxy
4. Call **get_economic_data** — FRED series: FEDFUNDS, CPIAUCSL, M2SL, UNRATE, T10Y2Y
5. Call **get_economic_calendar** — upcoming FOMC, CPI, jobs reports

## Output Format
Produce a macro report with:
- Macro regime classification (Risk-On / Risk-Off / Transitioning)
- Liquidity conditions (Expanding / Contracting / Neutral)
- Key macro risks for the next 1-4 weeks
- Overall macro bias for BTC (Bullish / Bearish / Neutral) with confidence

Current date: {current_date}. Asset: {ticker}.
Do NOT output FINAL TRANSACTION PROPOSAL. You are a macro analyst — you report macro conditions, not trade decisions."""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Analyze the current macroeconomic environment and its impact on {ticker}. "
                         f"Call get_dollar_index, get_yields, get_sp500, get_economic_data, and get_economic_calendar, "
                         f"then produce your macro regime report."),
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
            log.warning("Macro analyst produced no text report after tool loop")
            report = "Macro analysis incomplete — tool calls may have failed."

        return {
            "messages": [result],
            "macro_report": report,
        }

    return macro_analyst_node

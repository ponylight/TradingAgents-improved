"""
Crypto News & Macro Analyst

Owns: News events, macroeconomic indicators, geopolitical catalysts.
Data sources: 30+ RSS feeds + GDELT (pre-fetched), macro tools (DXY, yields, calendar via FRED).
"""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tradingagents.agents.utils.agent_utils import get_news, get_global_news
from tradingagents.agents.utils.macro_tools import (
    get_dollar_index,
    get_yields,
    get_sp500,
    get_economic_data,
    get_economic_calendar,
)
from tradingagents.dataflows.geopolitical_news import fetch_all_news, format_geopolitical_report

import logging
log = logging.getLogger("news_analyst")

MAX_TOOL_ROUNDS = 3


def create_news_analyst(llm):

    tools = [
        get_news, get_global_news, get_dollar_index, get_yields,
        get_sp500, get_economic_data, get_economic_calendar,
    ]
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        # Pre-fetch geopolitical news from 30+ RSS feeds + GDELT
        try:
            geo_data = fetch_all_news(max_per_feed=8)
            geo_report = format_geopolitical_report(geo_data)
            alert_count = geo_data.get("alert_count", 0)
        except Exception as e:
            log.warning(f"Geopolitical news fetch failed: {e}")
            geo_report = "Geopolitical news unavailable."
            alert_count = 0

        system_message = f"""You are a senior crypto news and macro analyst for BTC.

## Pre-Fetched Geopolitical & Market Intelligence
REAL-TIME data from {geo_data.get('sources_fetched', 0) if 'geo_data' in dir() else '?'} news sources + GDELT.
{"⚠️ " + str(alert_count) + " GEOPOLITICAL ALERTS — prioritize these." if alert_count > 0 else "No active geopolitical alerts."}

{geo_report}

## Tools Available (call the macro ones to supplement news)
- get_dollar_index — DXY level and trend
- get_yields — US Treasury yield curve
- get_sp500 — S&P 500 risk appetite
- get_economic_data — FRED data (CPI, M2, UNRATE)
- get_economic_calendar — upcoming FOMC, CPI, jobs reports

## Analysis Framework
1. Macro Environment: risk-on or risk-off? Liquidity expanding or contracting?
2. Top Events: ranked by BTC impact (from pre-fetched news above)
3. Primary Catalyst: the ONE thing that matters most right now
4. Upcoming Events: next 48h calendar items (from tools)
5. Overall News/Macro Verdict: BULLISH / NEUTRAL / BEARISH

Current date: {current_date}. Asset: {ticker}.
Do NOT output FINAL TRANSACTION PROPOSAL. You report events and macro, not trade decisions."""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Analyze the current news and macro environment for {ticker}. "
                         f"The pre-fetched headlines are above. Call macro tools (DXY, yields, economic_calendar) "
                         f"to get quantitative data, then synthesize your report."),
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
            log.warning("News analyst produced no text report after tool loop")
            report = f"News analysis incomplete. Headlines:\n{geo_report}"

        return {
            "messages": state["messages"] + [result],
            "news_report": report,
        }

    return news_analyst_node

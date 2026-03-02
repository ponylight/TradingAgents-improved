"""
Crypto News & Macro Analyst

Owns: News events, macroeconomic indicators, geopolitical catalysts.
Per the paper: "analyze news articles, government announcements, and other
macroeconomic indicators to assess the market's macroeconomic state."

Data sources:
- News: get_news, get_global_news (Alpha Vantage / yfinance)
- Macro: DXY, yields, S&P500, economic data, economic calendar (FRED)
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import get_news, get_global_news
from tradingagents.agents.utils.macro_tools import (
    get_dollar_index,
    get_yields,
    get_sp500,
    get_economic_data,
    get_economic_calendar,
)

import logging
from tradingagents.dataflows.geopolitical_news import fetch_all_news, format_geopolitical_report
log = logging.getLogger("news_analyst")


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_news,
            get_global_news,
            get_dollar_index,
            get_yields,
            get_sp500,
            get_economic_data,
            get_economic_calendar,
        ]

        # Pre-fetch geopolitical news from 30+ RSS feeds + GDELT
        try:
            geo_data = fetch_all_news(max_per_feed=8)
            geo_report = format_geopolitical_report(geo_data)
            alert_count = geo_data.get("alert_count", 0)
        except Exception as e:
            log.warning(f"Geopolitical news fetch failed: {e}")
            geo_report = "Geopolitical news unavailable."
            alert_count = 0

        system_message = f"""You are a senior crypto news and macro analyst.

## Pre-Fetched Geopolitical & Market Intelligence
This is REAL-TIME data from {geo_data.get('sources_fetched', 0)} news sources + GDELT.
{"⚠️ " + str(alert_count) + " GEOPOLITICAL ALERTS DETECTED — prioritize these." if alert_count > 0 else "No active geopolitical alerts."}

{geo_report}

## Additional Context from Tools
Use your tools to supplement the above with macro data (DXY, yields, economic calendar).
The pre-fetched news covers headlines; tools provide quantitative macro data. You analyze news events AND macroeconomic indicators that impact Bitcoin's price.

## Your Two Domains

### 1. News & Events
- Geopolitical events (wars, sanctions, elections)
- Regulatory actions (SEC, CFTC, global crypto regulation)
- Institutional moves (ETF flows, corporate treasury buys, exchange news)
- Market events (liquidation cascades, exchange hacks, stablecoin depegs)

### 2. Macroeconomic Indicators
- **DXY (Dollar Index)**: Strong dollar = bearish for BTC. Weakening = bullish.
- **Treasury Yields**: Rising yields = tighter conditions = risk-off. Falling = risk-on.
- **S&P 500**: BTC correlation with equities. S&P selling = BTC likely follows.
- **Economic Data (FRED)**: CPI, unemployment, M2 money supply. Inflation up = mixed (hedge narrative vs rate hike fear).
- **Economic Calendar**: Upcoming FOMC, CPI releases, jobs reports. These cause volatility spikes.

## Tools Available
- get_news(query, start_date, end_date) — targeted news search
- get_global_news(curr_date, look_back_days, limit) — broad macro news
- get_dollar_index(curr_date) — DXY current level and trend
- get_yields(curr_date) — US Treasury yield curve
- get_sp500(curr_date) — S&P 500 for risk appetite
- get_economic_data(indicator, start_date, end_date) — FRED data (CPI, M2, UNRATE)
- get_economic_calendar(curr_date) — upcoming economic events

## Analysis Framework
1. **Macro Environment**: Risk-on or risk-off? Liquidity expanding or contracting?
2. **Event Impact**: For each major event, assess timeframe (hours/days/weeks), magnitude (1-5%), direction
3. **Catalyst Identification**: What is the SINGLE most important catalyst right now?
4. **Upcoming Risk Events**: What's on the calendar that could move markets?

## Output Format
1. Macro Environment: (risk-on/risk-off with key data)
2. Top Events: (ranked by impact on BTC)
3. Primary Catalyst: (the one thing that matters most right now)
4. Upcoming Events: (next 48h calendar items)
5. Overall News/Macro Verdict: BULLISH / NEUTRAL / BEARISH

Do NOT output FINAL TRANSACTION PROPOSAL. You report events and macro, not trade decisions."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a crypto news and macro analyst. Use the provided tools to fetch news and economic data."
                    " Tools available: {tool_names}."
                    " Current date: {current_date}. Asset: {ticker}."
                    "\n\n{system_message}",
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
            "news_report": report,
        }

    return news_analyst_node

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
from tradingagents.dataflows.data_quality import score_report, format_quality_header, DataQuality

import logging
from datetime import datetime, timezone
log = logging.getLogger("news_analyst")

MAX_TOOL_ROUNDS = 3

# News/macro data older than 4h is considered stale
NEWS_STALENESS_HOURS = 4.0


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
        geo_fallback = False
        geo_generated_at = None
        try:
            geo_data = fetch_all_news(max_per_feed=8)
            geo_report = format_geopolitical_report(geo_data)
            alert_count = geo_data.get("alert_count", 0)
            geo_generated_at = geo_data.get("generated_at")
        except Exception as e:
            log.warning(f"Geopolitical news fetch failed: {e}")
            geo_data = {}
            geo_report = "Geopolitical news unavailable."
            alert_count = 0
            geo_fallback = True

        # === Data Freshness Gate ===
        staleness_warning = ""
        missing = []
        if geo_fallback:
            missing.append("geopolitical_news")
        quality_score = score_report(
            geo_report,
            "news",
            max_age_hours=NEWS_STALENESS_HOURS,
            generated_at=geo_generated_at,
            fallback_used=geo_fallback,
            missing_fields=missing,
        )
        quality_header = format_quality_header(quality_score, "news")

        if quality_score["quality"] in (DataQuality.STALE, DataQuality.FAILED):
            age_str = f"{quality_score['data_age_hours']:.1f}h" if quality_score["data_age_hours"] else "unknown"
            staleness_warning = (
                f"\n\n⛔ STALENESS GATE: News/macro data is {age_str} old (>{NEWS_STALENESS_HOURS}h threshold). "
                f"Cap your overall confidence by ONE level (HIGH→MEDIUM, MEDIUM→LOW). State this prominently.\n"
            )
            log.warning(f"⚠️ News staleness gate triggered: {age_str} old")

        # Safe source count extraction (geo_data always defined — use isinstance guard)
        sources_fetched = geo_data.get('sources_fetched', 0) if isinstance(geo_data, dict) else 0

        quality_degraded_note = ""
        if quality_score["quality"] == DataQuality.DEGRADED:
            quality_degraded_note = (
                "\n⚠️ DATA DEGRADED: Some news sources returned errors or partial data. "
                "Cap confidence by one level if key sources are missing.\n"
            )

        system_message = f"""You are a senior crypto news and macro analyst for BTC.

{quality_header}
{staleness_warning}{quality_degraded_note}
## Pre-Fetched Geopolitical & Market Intelligence
Data from {sources_fetched} news sources + GDELT.
{"⚠️ " + str(alert_count) + " GEOPOLITICAL ALERTS — prioritize these." if alert_count > 0 else "No active geopolitical alerts."}

{geo_report}

## Tools You MUST Call
Call the macro tools to get quantitative data. Also use get_news/get_global_news for additional headlines:
- **get_news** — crypto-specific news for the asset (pass ticker and date range)
- **get_global_news** — broader market headlines (pass current date)
- **get_dollar_index** — DXY level and trend (CALL THIS)
- **get_yields** — US Treasury yield curve (CALL THIS)
- **get_sp500** — S&P 500 risk appetite
- **get_economic_data** — FRED data (CPI, M2, UNRATE, FEDFUNDS)
- **get_economic_calendar** — upcoming FOMC, CPI, jobs reports (CALL THIS)

You SHOULD call at least get_dollar_index, get_yields, and get_economic_calendar.
If tool calls fail, proceed with the pre-fetched data above and flag the gap.

## Data Quality — You Are a Professional
Before analyzing, audit every data point. If news data shows quality issues
(see header above), flag it prominently. Do NOT build confident conclusions
on stale or missing data. If the staleness gate is triggered, your confidence
MUST be capped as instructed.

## Analysis Framework
1. **Data Quality**: Clean / Degraded (with specifics if degraded)
2. **Macro Regime**: classify as Risk-On / Risk-Off / Transitioning based on DXY, yields, S&P 500
3. **Liquidity**: Expanding / Contracting / Neutral (M2, Fed policy, yield curve)
4. Top Events: ranked by BTC impact (from pre-fetched news above)
5. Primary Catalyst: the ONE thing that matters most right now
6. Upcoming Events: next 48h calendar items (from tools)
7. Overall News/Macro Verdict: BULLISH / NEUTRAL / BEARISH

Current date: {current_date}. Asset: {ticker}.
Do NOT output FINAL TRANSACTION PROPOSAL. You report events and macro, not trade decisions."""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Analyze the current news and macro environment for {ticker}. "
                         f"The pre-fetched headlines are above. Call get_dollar_index, get_yields, "
                         f"get_economic_calendar, and optionally get_news/get_global_news for additional data. "
                         f"Then synthesize your report with macro regime classification."),
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

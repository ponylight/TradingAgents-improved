from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news, get_global_news
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = """You are a senior news analyst specializing in extracting actionable trading signals from global news flow. Your task is to analyze recent news and macroeconomic trends and produce a report that directly informs trading decisions.

## News Categorization Framework
Classify each major news item into one of these categories:
- **Earnings/Guidance**: Company results, revenue/EPS surprises, forward guidance changes
- **M&A/Corporate Actions**: Mergers, acquisitions, spinoffs, buybacks, insider transactions
- **Regulatory/Legal**: Government actions, lawsuits, antitrust, policy changes
- **Macro/Geopolitical**: Interest rates, inflation data, trade policy, geopolitical events, commodity shocks
- **Sector/Industry**: Industry-wide trends, competitive dynamics, supply chain disruptions

## Impact Assessment
For each significant news item, assess:
1. **Timeframe**: Immediate (hours/days), short-term (weeks), or structural (months+)?
2. **Magnitude**: Minor (1-2% move), moderate (3-5%), or major (5%+ potential impact)?
3. **Direction**: Positive, negative, or ambiguous for the target stock?
4. **Confidence**: How certain is the directional impact?

## Macro-to-Micro Analysis
1. Start with the big picture — what is the macro environment doing (rates, inflation, growth, risk appetite)?
2. How does the macro backdrop affect the target company's sector?
3. What company-specific news amplifies or dampens the macro trend?

## Instructions
- Use get_news(query, start_date, end_date) for company-specific and targeted news searches
- Use get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news
- Do NOT state trends are "mixed" without specifics — provide directional assessment with magnitude and confidence
- Identify the single most important news catalyst and explain why it matters most
- Append a Markdown summary table at the end organizing findings by category, impact, and direction"""

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
                    "For your reference, the current date is {current_date}. We are looking at the company {ticker}",
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

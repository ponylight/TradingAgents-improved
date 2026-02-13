from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_news,
        ]

        system_message = """You are a senior social media and sentiment analyst specializing in extracting trading signals from public discourse. Your objective is to analyze social media posts, news sentiment, and public opinion for the target company over the past week.

## Sentiment Scoring Framework
Rate each source/event on a -2 to +2 scale:
- **+2 Strong Bullish**: Major positive catalyst, widespread enthusiasm, institutional endorsements
- **+1 Mildly Bullish**: Positive tone, minor good news, cautious optimism
- **0 Neutral**: Factual reporting, no clear directional sentiment
- **-1 Mildly Bearish**: Concerns raised, minor negative news, cautious language
- **-2 Strong Bearish**: Major negative catalyst, widespread fear, institutional criticism

## Analysis Framework
1. **Volume-Weighted Sentiment**: Weight sentiment by the reach/engagement of each source. A viral bearish tweet matters more than a low-engagement blog post
2. **Source Credibility Tiers**: Tier 1 (institutional analysts, verified executives, major outlets) > Tier 2 (industry experts, established financial bloggers) > Tier 3 (retail traders, anonymous social media). Weight accordingly
3. **Sentiment Velocity**: Is sentiment improving, deteriorating, or stable day-over-day? Track the direction of change, not just the level — sentiment momentum often precedes price momentum
4. **Narrative Clustering**: Group related discussions into themes (e.g., "earnings expectations", "product concerns", "management changes") and assess the strength and direction of each narrative

## Instructions
- Use get_news(query, start_date, end_date) to search company-specific news and social media discussions
- Search multiple queries (company name, ticker, key products, CEO name) to capture full picture
- Do NOT state trends are "mixed" without specifics — provide directional sentiment with confidence level
- Identify the dominant narrative driving current sentiment and whether it's accelerating or fading
- Append a Markdown summary table at the end organizing key findings by source, sentiment score, and impact"""

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
                    "For your reference, the current date is {current_date}. The current company we want to analyze is {ticker}",
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

    return social_media_analyst_node

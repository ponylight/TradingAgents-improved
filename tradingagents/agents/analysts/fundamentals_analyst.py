from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement, get_insider_transactions
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = """You are a senior fundamentals analyst specializing in equity valuation and financial statement analysis. Your task is to produce a comprehensive fundamental analysis that directly informs trading decisions.

## Key Ratio Thresholds (Flag Deviations)
- **P/E Ratio**: < 15 potentially undervalued, 15-25 fairly valued, > 25 growth premium or overvalued. Compare to sector median
- **P/B Ratio**: < 1.0 potential deep value or distressed, 1-3 normal, > 5 high growth premium
- **Debt-to-Equity (D/E)**: < 0.5 conservative, 0.5-1.5 moderate, > 2.0 high leverage risk (sector-dependent)
- **Return on Equity (ROE)**: > 15% strong, 10-15% adequate, < 10% weak (adjust for leverage via DuPont decomposition)
- **Current Ratio**: < 1.0 liquidity concern, 1.0-2.0 healthy, > 3.0 potentially inefficient capital allocation
- **Free Cash Flow Yield**: > 5% attractive, 3-5% fair, < 3% expensive or capital-intensive

## Red Flags Checklist
Flag any of the following if found:
- Revenue growing but FCF declining (earnings quality concern)
- Accounts receivable growing faster than revenue (potential revenue recognition issues)
- Rising debt with declining interest coverage ratio
- Inventory buildup exceeding revenue growth
- Frequent one-time adjustments or non-GAAP reconciliation gaps
- Insider selling while company repurchases shares

## Analysis Framework
1. **Profitability**: Gross/operating/net margins and their trend direction over recent quarters
2. **Growth**: Revenue, earnings, and FCF growth rates — accelerating or decelerating?
3. **Balance Sheet Health**: Liquidity, leverage, and debt maturity profile
4. **Cash Flow Quality**: Operating cash flow vs net income ratio, capex intensity, FCF generation
5. **Valuation**: Current multiples vs historical range and sector peers

## Instructions
- Use get_fundamentals for comprehensive company analysis
- Use get_balance_sheet, get_cashflow, and get_income_statement for specific financial statements
- Do NOT state trends are "mixed" without specifics — provide directional assessment of fundamental health
- Explicitly rate the company's fundamental strength as Strong / Adequate / Weak with supporting evidence
- Append a Markdown summary table at the end organizing key ratios, their values, and status (healthy/warning/concern)"""

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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node

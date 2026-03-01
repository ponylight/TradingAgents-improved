import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a senior trading agent responsible for converting an investment recommendation into a concrete, executable trade plan. Your decisions directly affect portfolio performance.

## Position Sizing Framework
Risk per trade is fixed at 1% of equity (mechanical, ATR-based). You control MARGIN ALLOCATION:
- **High Conviction (8-10/10)**: 1-2% allocation (results in ~10-12x leverage). You are very confident.
- **Medium Conviction (5-7/10)**: 3-5% allocation (results in ~4-5x leverage). Reasonable setup.
- **Low Conviction (1-4/10)**: 8-12% allocation (results in ~1-2x leverage), or NEUTRAL/pass.

Lower allocation = higher leverage = less margin locked but same dollar risk.
Higher allocation = lower leverage = more margin locked but same dollar risk.
Risk is always 1% regardless of your allocation choice.

## Trade Plan Requirements
Your output MUST include:
1. **Decision**: BUY, SELL, or HOLD with a confidence rating (1-10)
2. **Entry Criteria**: Specific price level or condition for entry (e.g., "Buy at market" or "Buy on pullback to $X support")
3. **Position Size**: Percentage of portfolio, justified by conviction level
4. **Stop-Loss**: Specific exit level if the trade goes against you (required for all BUY/SELL decisions)
5. **Take-Profit Target**: Price target or condition for taking profits
6. **Risk-Reward Ratio**: Must be at least 2:1 for BUY/SELL decisions. If R:R < 2:1, default to HOLD
7. **Time Horizon**: How long you expect to hold — intraday, swing (days-weeks), or position (weeks-months)

## Decision Rules
- Do NOT default to HOLD out of indecision. HOLD should only be recommended when the risk-reward is genuinely unattractive
- A strong bull case with poor entry timing = HOLD (wait for better entry)
- A moderate case with excellent entry timing = BUY with smaller size
- Integrate past lessons from similar situations to avoid repeating mistakes

## Past Reflections
{past_memory_str}

Always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation.""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")

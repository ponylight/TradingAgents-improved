import functools
from tradingagents.agents.utils.report_context import get_agent_context
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

        # Portfolio context for thesis persistence
        portfolio_context = state.get("portfolio_context", {})
        current_position = portfolio_context.get("position", "none")
        position_pnl_pct = portfolio_context.get("pnl_pct", 0)
        hours_held = portfolio_context.get("hours_held", 0)
        last_decision = portfolio_context.get("last_decision", "HOLD")
        last_decision_reasoning = portfolio_context.get("last_decision_reasoning", "No prior reasoning available.")
        last_decision_time = portfolio_context.get("last_decision_time", "Unknown")

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        budgeted_context = get_agent_context(state, "trader")

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nAnalyst Reports (role-weighted context):\n{budgeted_context}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a senior trading agent responsible for converting an investment recommendation into a concrete, executable trade plan. Your decisions directly affect portfolio performance.

## Current Portfolio State
- Position: {current_position}
- Unrealized P&L: {position_pnl_pct:+.2f}%
- Hours held: {hours_held:.1f}
- Last decision: {last_decision} (at {last_decision_time})

## Your Prior Thesis (WHY we are in this position)
{last_decision_reasoning}

## CRITICAL: Material Change Requirement
You OWN the thesis. If we already have a position, you must ask: "WHAT HAS MATERIALLY CHANGED since my last thesis?"

Material change examples:
- Price hit stop-loss or take-profit level
- Major unexpected news event (NOT "same news, different angle")
- Technical structure broke (key support/resistance violated)
- Fundamental shift (regulatory action, major event)

"Same data reasoned differently" is NOT material change. "LLM vibes" is NOT material change.

If nothing material changed: propose HOLD with your existing thesis reaffirmed.
If something material changed: cite EXACTLY what changed, then propose your new direction.

### Anti-flip-flop Rules (HARD):
- Position < 12 hours old: propose HOLD unless stop-loss hit or genuine emergency
- If you are reversing direction: you MUST cite the specific material change
- Confidence must be >= 8 to reverse a position

## Position Sizing Framework
Risk per trade is fixed at 1% of equity (mechanical, ATR-based). You control MARGIN ALLOCATION:
- **High Conviction (8-10/10)**: 1-2% allocation (~10-12x leverage). Very confident.
- **Medium Conviction (5-7/10)**: 3-5% allocation (~4-5x leverage). Reasonable setup.
- **Low Conviction (1-4/10)**: 8-12% allocation (~1-2x leverage), or NEUTRAL/pass.

## Trade Plan Requirements
Your output MUST include:
1. **Decision**: BUY, SELL, or HOLD with confidence (1-10)
2. **Entry Criteria**: Specific price level or condition
3. **Position Size**: % of portfolio, justified by conviction
4. **Stop-Loss**: Specific exit level (required for BUY/SELL)
5. **Take-Profit Target**: Price target or condition
6. **Risk-Reward Ratio**: Must be >= 2:1 for BUY/SELL. If R:R < 2:1, default to HOLD
7. **Time Horizon**: Intraday, swing, or position
8. **Material Change**: What new information justifies this decision (or "N/A — reaffirming existing thesis")
9. **Thesis**: 2-3 sentence summary of WHY

## Decision Rules
- Do NOT default to HOLD out of indecision. HOLD is for when R:R is genuinely unattractive OR nothing has changed
- A strong case with poor entry = HOLD (wait for better entry)
- A moderate case with great entry = BUY/SELL with smaller size
- When already in a position with no material change: HOLD and reaffirm thesis
- Integrate past lessons from similar situations

## Past Reflections
{past_memory_str}

Always conclude with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'""",
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

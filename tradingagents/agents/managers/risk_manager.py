from tradingagents.agents.utils.report_context import get_agent_context

import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        budgeted_context = get_agent_context(state, "risk_manager")
        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""You are the Risk Manager. Your role is to EVALUATE and GATE risk — NOT to decide trade direction.

## Your Authority
- You can APPROVE, RESIZE, or VETO a trade
- You CANNOT change the direction (BUY→SELL or vice versa)
- You CANNOT propose trades — that is the Portfolio Manager's job
- If risk is unacceptable, you VETO. The Portfolio Manager decides what to do next.

## Position Risk Scoring
Score each dimension 1-5 (5 = highest risk):
- **Market Risk**: Sensitivity to broad market movements and volatility regime
- **Liquidity Risk**: Can the position be exited quickly without significant slippage?
- **Concentration Risk**: Does this position create excessive portfolio concentration?
- **Event Risk**: Are there upcoming binary events that could gap the asset?
- **Timing Risk**: Is the entry point technically sound, or are we chasing?

## Hard Risk Limits (Non-Negotiable)
- **2% Max Loss Rule**: No single trade should risk more than 2% of total portfolio value. If stop-loss implies more → RESIZE (reduce position size, do NOT change direction)
- **Exit Strategy Required**: Every trade MUST have stop-loss and take-profit. No exit strategy → VETO
- **Risk-Reward Minimum**: Only approve trades with at least 2:1 reward-to-risk ratio. Below 2:1 → VETO

## Past Reflections on Mistakes
{past_memory_str}

## Analyst Reports (role-weighted context)
{budgeted_context}

## Trader's Proposed Plan
{trader_plan}

## Risk Debate History
{history}

## Required Output
1. Score each risk dimension
2. Calculate composite risk score
3. Check each hard limit
4. Issue your verdict:

RISK VERDICT: APPROVED / APPROVED_WITH_RESIZE / VETOED
- If APPROVED: state the risk parameters you are comfortable with
- If APPROVED_WITH_RESIZE: specify the maximum position size and why
- If VETOED: state which hard limit was breached and why

Do NOT recommend a trade direction. Do NOT say BUY, SELL, or HOLD. Your job is risk assessment only."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node

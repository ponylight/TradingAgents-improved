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

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Risk Management Judge, your goal is to evaluate the three-way risk debate and produce a final, risk-adjusted trading decision.

## Position Risk Scoring
Evaluate the proposed trade on each dimension (1-5, where 5 = highest risk):
- **Market Risk**: Sensitivity to broad market movements and volatility regime
- **Liquidity Risk**: Can the position be exited quickly without significant slippage?
- **Concentration Risk**: Does this position create excessive portfolio concentration in a sector/theme?
- **Event Risk**: Are there upcoming binary events (earnings, FDA decisions, etc.) that could gap the stock?
- **Timing Risk**: Is the entry point technically sound, or are we chasing?

## Portfolio Impact Assessment
- How does this trade affect overall portfolio beta?
- Does it add diversification or increase correlation with existing positions?
- What is the maximum portfolio drawdown if this trade hits its stop-loss?

## Risk Limits (Enforce These)
- **2% Max Loss Rule**: No single trade should risk more than 2% of total portfolio value. If the proposed stop-loss implies more, reduce position size
- **Exit Strategy Required**: Every BUY/SELL decision MUST have a defined stop-loss and take-profit. Reject any plan without these
- **Risk-Reward Minimum**: Only approve trades with at least 2:1 reward-to-risk ratio

## Decision Framework
1. Start with the trader's original plan: **{trader_plan}**
2. Evaluate how each risk analyst's arguments should modify the plan
3. Apply the risk limits above — adjust position size, stops, or reject if risk criteria aren't met
4. Produce a final recommendation that balances opportunity with capital preservation

## Past Reflections on Mistakes
{past_memory_str}

## Analysts Debate History
{history}

Produce a clear, actionable recommendation (BUY, SELL, or HOLD) with specific risk parameters. Be decisive — ambiguity in risk management leads to losses."""

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

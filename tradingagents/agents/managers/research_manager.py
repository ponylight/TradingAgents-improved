import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Research Manager and debate judge, your role is to evaluate the bull/bear debate and produce a decisive investment recommendation for the trader.

## Evidence Weighting Framework
1. **Data-Backed Arguments (High Weight)**: Claims supported by specific numbers, ratios, or data points from analyst reports
2. **Logical Reasoning (Medium Weight)**: Sound analytical frameworks applied to available information
3. **Speculative Arguments (Low Weight)**: Claims based on assumptions without supporting evidence
4. **Contradicted Claims (Zero Weight)**: Arguments that were effectively rebutted by the opposing side with stronger evidence

## Conviction Scoring (1-10)
Rate your conviction in the final recommendation:
- **9-10**: Overwhelming evidence in one direction, debate was one-sided
- **7-8**: Strong evidence favoring one side, minor valid counterpoints
- **5-6**: Balanced but one side has a slight edge
- **3-4**: Highly uncertain, proceed with caution
- **1-2**: Insufficient information to make a confident call

## Anti-HOLD Default Rule
HOLD is NOT a neutral fallback. You must choose HOLD only when:
- The risk-reward is genuinely unattractive in BOTH directions
- There is a specific upcoming catalyst that makes waiting the optimal strategy
- Both bull and bear cases are equally strong AND equally well-evidenced
If in doubt between BUY and HOLD, lean toward a smaller BUY. Indecision costs money through missed opportunities.

## Your Deliverables
1. **Key Arguments Summary**: 2-3 strongest points from each side
2. **Decisive Recommendation**: BUY, SELL, or HOLD with conviction score
3. **Rationale**: Why the winning arguments are more compelling
4. **Investment Plan for Trader**: Concrete strategic actions to implement

## Past Reflections on Mistakes
"{past_memory_str}"

## Debate History
{history}

Present your analysis conversationally. Be decisive — the trader needs a clear direction, not a balanced summary."""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node

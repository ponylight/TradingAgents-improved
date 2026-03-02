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

        prompt = f"""You are the Research Manager and debate judge for a Bitcoin trading desk. Evaluate the bull/bear debate and produce a decisive investment recommendation.

## Evidence Weighting
1. **Data-Backed (High Weight)**: Arguments with specific numbers from reports — hash rates, RSI values, funding rates, price levels, volume data
2. **Logical Reasoning (Medium Weight)**: Sound frameworks applied to available data
3. **Speculative (Low Weight)**: Claims without supporting evidence from reports
4. **Contradicted (Zero Weight)**: Arguments effectively rebutted with stronger evidence

## Conviction Scoring (1-10)
- **9-10**: Overwhelming evidence, debate was one-sided. High-conviction trade.
- **7-8**: Strong evidence favoring one side, minor valid counterpoints
- **5-6**: Balanced, slight edge. Trade with reduced size.
- **3-4**: Highly uncertain. Default to HOLD unless current position warrants exit.
- **1-2**: Insufficient data. HOLD.

## Minimum Conviction to Trade
- Opening a NEW position: conviction >= 6
- HOLDING existing position: conviction >= 4 (lower bar to stay in)
- REVERSING a position: conviction >= 8 (high bar to flip)
If conviction is below threshold: recommend HOLD regardless of debate outcome.

## Anti-HOLD Default
HOLD is not a neutral fallback. Choose HOLD only when:
- Risk-reward is unattractive in BOTH directions
- A specific catalyst makes waiting optimal
- Both sides are equally strong AND equally evidenced
- Conviction is below the minimum threshold

## Deliverables
1. **Strongest Bull Points**: 2-3 key arguments with data citations
2. **Strongest Bear Points**: 2-3 key arguments with data citations
3. **Winner**: Which side presented the stronger data-backed case?
4. **Conviction Score**: 1-10 with justification
5. **Recommendation**: BUY (long), SELL (short), or HOLD
6. **Investment Plan for Trader**: Concrete direction with rationale

## Debate History
{history}

## Past Reflections
{past_memory_str}

Be decisive. The trader needs a clear direction backed by the stronger argument, not a balanced summary."""

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

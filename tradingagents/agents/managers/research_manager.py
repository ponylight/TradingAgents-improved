"""
Research Manager — Debate judge for bull/bear research team.

Evaluates debate, picks a winner, outputs a conviction-scored
investment recommendation using standardized LONG/NEUTRAL/SHORT.
"""

from tradingagents.agents.utils.report_context import get_agent_context
from tradingagents.agents.utils.trading_context import build_trading_context


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

        budgeted_context = get_agent_context(state, "research_manager")
        trading_ctx = build_trading_context(state)

        prompt = f"""You are the Research Manager and debate judge for a Bitcoin perpetual futures trading desk.

## Trading Mode
{trading_ctx['mode_instructions']}

Current position: {trading_ctx['current_position']}

{trading_ctx['position_logic']}

## Evidence Weighting
1. **Data-Backed (High Weight)**: Arguments with specific numbers — hash rates, RSI values, funding rates, price levels, volume data
2. **Logical Reasoning (Medium Weight)**: Sound frameworks applied to available data
3. **Speculative (Low Weight)**: Claims without supporting evidence
4. **Contradicted (Zero Weight)**: Arguments effectively rebutted with stronger evidence

## Conviction Scoring (1-10)
- **9-10**: Overwhelming evidence, debate was one-sided
- **7-8**: Strong evidence favoring one side, minor valid counterpoints
- **5-6**: Balanced with slight edge. Trade with reduced size.
- **3-4**: Highly uncertain. Default to NEUTRAL.
- **1-2**: Insufficient data. NEUTRAL.

## Minimum Conviction to Trade
- Opening a NEW position: conviction >= 6
- HOLDING existing position: conviction >= 4
- REVERSING a position: conviction >= 8
If conviction is below threshold → recommend NEUTRAL.

## Anti-NEUTRAL Default
NEUTRAL is not a fence-sit. Choose NEUTRAL only when:
- Setup is unattractive in BOTH directions
- Both sides are equally strong AND equally evidenced
- Conviction is below threshold
- A specific catalyst makes waiting optimal

## Chronic HOLD Warning
If the last_decision provided is HOLD and there's been no material market change,
you MUST explicitly address: "Why is this specific moment still not actionable?"
If you cannot give a SPECIFIC answer (not "market uncertainty" or "mixed signals"),
you should reconsider — chronic inaction is itself a risk. Sitting in cash while a
trend is underway is also a loss. Account for the opportunity cost of inaction.

**The bar for NEUTRAL should be: "I genuinely see no edge in either direction."
NOT: "I see some risk on both sides."** Some risk always exists. That's markets.

## Deliverables
1. **Strongest Bull Points**: 2-3 key arguments with data citations
2. **Strongest Bear Points**: 2-3 key arguments with data citations
3. **Winner**: Which side presented the stronger data-backed case?
4. **Conviction Score**: 1-10 with justification
5. **Recommendation**: {trading_ctx['actions']}
6. **Investment Plan for Trader**: Concrete direction with rationale

## Debate History
{history}

## Past Reflections
{past_memory_str}

Conclude with: {trading_ctx['final_format']}

Be decisive. The Trader needs a clear direction, not a balanced summary."""

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

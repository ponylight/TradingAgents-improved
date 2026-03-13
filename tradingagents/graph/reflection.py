"""
Reflection system — per-agent post-trade learning.

After each closed trade, every agent reflects on what went right/wrong.
Reflections are stored as (situation, recommendation) pairs in BM25 memory.
On future runs, similar situations retrieve past reflections to avoid repeating mistakes.
"""

from typing import Dict, Any


class Reflector:
    """Handles reflection on decisions and updating memory."""

    def __init__(self, quick_thinking_llm):
        self.quick_thinking_llm = quick_thinking_llm
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        return """You are reviewing a BTC perpetual futures trading decision to extract lessons for future trades.

## Your Task
Given the trade outcome (P&L), the agent's analysis/decision, and market context, produce a concise reflection.

## Framework
1. **Outcome Assessment**: Was the decision correct? A correct decision increased returns; incorrect decreased them.

2. **Factor Analysis** — Which inputs were most/least useful?
   - Technical indicators (trend, momentum, key levels, volume)
   - On-chain data (hash rate, addresses, supply dynamics)
   - Sentiment (Fear & Greed, funding rates, social mood)
   - News/macro (events, DXY, yields, regulatory)
   - Market structure (OI, liquidation levels, orderbook)

3. **What Went Right**: Specific factors that led to correct assessment. Cite data points.

4. **What Went Wrong**: Specific factors that were misleading or ignored. Cite what was missed.

5. **Lesson** (CRITICAL — this is what gets stored for future retrieval):
   - One clear, actionable lesson in under 200 words
   - Pattern-matching format: "When [situation], [what happened], so next time [do this]"
   - Example: "When funding rate was >0.05% and RSI >70, going long was wrong because the crowded positioning unwound. Next time, fade extreme funding even if trend looks strong."

Be specific. Vague lessons like "be more careful" are useless. Cite numbers, indicators, and conditions."""

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """Extract the current market situation from the state."""
        reports = []
        for key in ["market_report", "sentiment_report", "news_report", "fundamentals_report"]:
            report = current_state.get(key, "")
            if report:
                # Truncate each report to keep reflection context manageable
                reports.append(report[:1500])
        return "\n\n".join(reports)

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, returns_losses: str
    ) -> str:
        """Generate reflection for a component."""
        messages = [
            ("system", self.reflection_system_prompt),
            (
                "human",
                f"Agent Role: {component_type}\n\nTrade Outcome: {returns_losses}\n\nAgent's Analysis/Decision:\n{report[:2000]}\n\nMarket Context:\n{situation[:3000]}",
            ),
        ]
        result = self.quick_thinking_llm.invoke(messages).content
        return result

    def reflect_bull_researcher(self, current_state, returns_losses, bull_memory):
        situation = self._extract_current_situation(current_state)
        bull_debate_history = current_state["investment_debate_state"].get("bull_history", "")
        result = self._reflect_on_component("BULL RESEARCHER", bull_debate_history, situation, returns_losses)
        bull_memory.add_situations([(situation, result)])

    def reflect_bear_researcher(self, current_state, returns_losses, bear_memory):
        situation = self._extract_current_situation(current_state)
        bear_debate_history = current_state["investment_debate_state"].get("bear_history", "")
        result = self._reflect_on_component("BEAR RESEARCHER", bear_debate_history, situation, returns_losses)
        bear_memory.add_situations([(situation, result)])

    def reflect_trader(self, current_state, returns_losses, trader_memory):
        situation = self._extract_current_situation(current_state)
        trader_decision = current_state.get("trader_investment_plan", "")
        result = self._reflect_on_component("TRADER", trader_decision, situation, returns_losses)
        trader_memory.add_situations([(situation, result)])

    def reflect_invest_judge(self, current_state, returns_losses, invest_judge_memory):
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["investment_debate_state"].get("judge_decision", "")
        result = self._reflect_on_component("RESEARCH MANAGER", judge_decision, situation, returns_losses)
        invest_judge_memory.add_situations([(situation, result)])

    def reflect_risk_manager(self, current_state, returns_losses, risk_manager_memory):
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["risk_debate_state"].get("judge_decision", "")
        result = self._reflect_on_component("RISK JUDGE", judge_decision, situation, returns_losses)
        risk_manager_memory.add_situations([(situation, result)])

    def reflect_fund_manager(self, current_state, returns_losses, fund_manager_memory):
        situation = self._extract_current_situation(current_state)
        fund_decision = current_state.get("fund_manager_decision", "")
        result = self._reflect_on_component("FUND MANAGER", fund_decision, situation, returns_losses)
        fund_manager_memory.add_situations([(situation, result)])

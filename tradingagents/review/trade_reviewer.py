"""
Trade Reviewer (A) — Post-trade decision chain analysis.

Reviews the full pipeline for each closed trade:
  Analyst reports → Research debate → Trader plan → Risk assessment → FM approval

Identifies WHERE in the chain the decision went right or wrong.
Outputs a structured review with per-agent accountability.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("trade_reviewer")


class TradeReviewer:
    """Analyzes closed trades to identify pipeline breakdowns."""

    def __init__(self, llm, project_dir: str = "."):
        self.llm = llm
        self.project_dir = Path(project_dir)
        self.logs_dir = self.project_dir / "logs"
        self.eval_dir = self.project_dir / "eval_results" / "BTC_USDT" / "CryptoTradingAgents_logs"

    def load_state_for_date(self, date_str: str) -> Dict:
        """Load full agent state for a trading date."""
        state_file = self.eval_dir / f"full_states_log_{date_str}.json"
        if not state_file.exists():
            return {}
        with open(state_file) as f:
            data = json.load(f)
        # State logs are keyed by date
        return data.get(date_str, data)

    def load_trade_history(self) -> List[Dict]:
        """Load closed trades from executor state."""
        state_file = self.logs_dir / "executor_state.json"
        if not state_file.exists():
            return []
        with open(state_file) as f:
            state = json.load(f)
        return [t for t in state.get("trades", []) if t.get("status") == "closed"]

    def review_trade(self, trade: Dict, agent_state: Dict) -> Dict:
        """Review a single trade through the full decision chain.
        
        Returns a structured review with per-agent assessment.
        """
        pnl_pct = trade.get("pnl_pct", 0)
        side = trade.get("side", "unknown")
        outcome = "PROFITABLE" if pnl_pct > 0 else "LOSS" if pnl_pct < 0 else "BREAKEVEN"

        # Extract each agent's contribution
        chain = {
            "technical_analyst": agent_state.get("market_report", "")[:1500],
            "sentiment_analyst": agent_state.get("sentiment_report", "")[:1500],
            "news_analyst": agent_state.get("news_report", "")[:1500],
            "fundamentals_analyst": agent_state.get("fundamentals_report", "")[:1500],
            "research_debate": agent_state.get("investment_debate_state", {}).get("history", "")[:2000],
            "research_manager": agent_state.get("investment_plan", "")[:1500],
            "trader": agent_state.get("trader_investment_plan", "")[:1500],
            "risk_debate": agent_state.get("risk_debate_state", {}).get("history", "")[:1500],
            "risk_judge": agent_state.get("risk_debate_state", {}).get("judge_decision", "")[:1500],
            "fund_manager": agent_state.get("fund_manager_decision", agent_state.get("final_trade_decision", ""))[:1500],
        }

        # Build the chain summary for LLM review
        chain_text = ""
        for agent, report in chain.items():
            if report:
                chain_text += f"\n### {agent.replace('_', ' ').title()}\n{report}\n"

        prompt = f"""Review this BTC perpetual futures trade and identify where the decision chain succeeded or failed.

## Trade Outcome
- Direction: {side.upper()}
- P&L: {pnl_pct:+.2f}%
- Result: {outcome}

## Full Decision Chain
{chain_text}

## Your Analysis

For each agent in the chain, assess:
1. **Signal Quality**: Was their analysis/recommendation correct given what we now know?
2. **Data Usage**: Did they use their data effectively or miss key signals?
3. **Influence**: How much did their output influence the final decision?

Then identify:
- **Primary Failure Point**: Which agent's output most contributed to the loss (if loss)?
- **Primary Success Point**: Which agent's output most contributed to the win (if win)?
- **Chain Coherence**: Did information flow correctly or was something lost/distorted between stages?
- **Missed Signals**: What data was available but not used effectively?

## Required Output Format
```json
{{
    "outcome": "{outcome}",
    "pnl_pct": {pnl_pct},
    "agent_scores": {{
        "technical_analyst": {{"signal_correct": true/false, "influence": "high/medium/low", "notes": "..."}},
        "sentiment_analyst": {{"signal_correct": true/false, "influence": "high/medium/low", "notes": "..."}},
        "news_analyst": {{"signal_correct": true/false, "influence": "high/medium/low", "notes": "..."}},
        "fundamentals_analyst": {{"signal_correct": true/false, "influence": "high/medium/low", "notes": "..."}},
        "research_manager": {{"signal_correct": true/false, "influence": "high/medium/low", "notes": "..."}},
        "trader": {{"signal_correct": true/false, "influence": "high/medium/low", "notes": "..."}},
        "risk_judge": {{"sizing_appropriate": true/false, "notes": "..."}},
        "fund_manager": {{"approval_justified": true/false, "notes": "..."}}
    }},
    "primary_failure_point": "agent_name or 'none'",
    "primary_success_point": "agent_name or 'none'",
    "chain_coherence": "good/degraded/broken",
    "missed_signals": ["signal 1", "signal 2"],
    "lesson": "One actionable lesson for the system"
}}
```
Respond with ONLY the JSON."""

        try:
            result = self.llm.invoke(prompt)
            text = result.content if hasattr(result, 'content') else str(result)
            # Strip markdown fences
            text = re.sub(r'^```json?\s*', '', text.strip())
            text = re.sub(r'\s*```$', '', text.strip())
            review = json.loads(text)
            return review
        except Exception as e:
            log.error(f"Trade review failed: {e}")
            return {"error": str(e), "outcome": outcome, "pnl_pct": pnl_pct}

    def review_all_trades(self) -> List[Dict]:
        """Review all closed trades."""
        trades = self.load_trade_history()
        reviews = []

        for trade in trades:
            open_time = trade.get("open_time", "")
            if open_time:
                date_str = open_time[:10]
            else:
                continue

            agent_state = self.load_state_for_date(date_str)
            if not agent_state:
                log.warning(f"No agent state for {date_str}, skipping")
                continue

            review = self.review_trade(trade, agent_state)
            review["trade_date"] = date_str
            review["trade_id"] = trade.get("id", date_str)
            reviews.append(review)

        return reviews

    def generate_summary(self, reviews: List[Dict]) -> Dict:
        """Generate aggregate summary across all reviewed trades."""
        if not reviews:
            return {"error": "No reviews to summarize"}

        agent_stats = {}
        total_trades = len(reviews)
        wins = sum(1 for r in reviews if r.get("pnl_pct", 0) > 0)
        losses = sum(1 for r in reviews if r.get("pnl_pct", 0) < 0)
        total_pnl = sum(r.get("pnl_pct", 0) for r in reviews)

        failure_points = {}
        success_points = {}
        all_lessons = []

        for review in reviews:
            # Count failure/success points
            fp = review.get("primary_failure_point", "none")
            sp = review.get("primary_success_point", "none")
            if fp != "none":
                failure_points[fp] = failure_points.get(fp, 0) + 1
            if sp != "none":
                success_points[sp] = success_points.get(sp, 0) + 1

            # Aggregate agent scores
            for agent, scores in review.get("agent_scores", {}).items():
                if agent not in agent_stats:
                    agent_stats[agent] = {"correct": 0, "incorrect": 0, "total": 0, "high_influence": 0}
                agent_stats[agent]["total"] += 1
                if scores.get("signal_correct", False):
                    agent_stats[agent]["correct"] += 1
                else:
                    agent_stats[agent]["incorrect"] += 1
                if scores.get("influence") == "high":
                    agent_stats[agent]["high_influence"] += 1

            lesson = review.get("lesson", "")
            if lesson:
                all_lessons.append(lesson)

        # Calculate accuracy
        for agent, stats in agent_stats.items():
            stats["accuracy_pct"] = round(stats["correct"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total_trades * 100, 1) if total_trades > 0 else 0,
            "total_pnl_pct": round(total_pnl, 2),
            "agent_accuracy": agent_stats,
            "most_common_failure_point": max(failure_points, key=failure_points.get) if failure_points else "none",
            "most_common_success_point": max(success_points, key=success_points.get) if success_points else "none",
            "failure_distribution": failure_points,
            "success_distribution": success_points,
            "lessons": all_lessons,
        }

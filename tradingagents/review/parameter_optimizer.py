"""
Parameter Optimizer (B) — Data-driven tuning of system parameters.

Uses accumulated trade reviews and agent scores to recommend or apply
parameter changes:

1. Context budget weights — shift budget toward accurate agents
2. Conviction thresholds — tighten if false positives, loosen if missed trades
3. Debate rounds — increase if decisions are low quality, decrease if wasting tokens
4. Agent prompt hints — inject performance-based instructions

Does NOT auto-apply changes. Generates recommendations that can be
reviewed and applied manually, or auto-applied with explicit approval.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("parameter_optimizer")


# Default parameters (baseline)
DEFAULT_PARAMS = {
    "context_budget_weights": {
        "technical_analyst": 1.0,
        "sentiment_analyst": 1.0,
        "news_analyst": 1.0,
        "fundamentals_analyst": 1.0,
    },
    "conviction_thresholds": {
        "new_position": 6,
        "hold_position": 4,
        "reverse_position": 8,
    },
    "debate_rounds": {
        "research": 1,
        "risk": 1,
    },
    "risk_limits": {
        "max_loss_pct": 1.0,
        "max_leverage_swing": 15,
        "max_leverage_position": 10,
        "min_rr_ratio": 2.0,
    },
}


class ParameterOptimizer:
    """Generates parameter optimization recommendations from performance data."""

    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.params_file = self.project_dir / "logs" / "optimized_params.json"
        self.recommendations_file = self.project_dir / "logs" / "optimization_recommendations.json"
        self.current_params = self._load_params()

    def _load_params(self) -> Dict:
        """Load current parameters."""
        if self.params_file.exists():
            with open(self.params_file) as f:
                return json.load(f)
        return DEFAULT_PARAMS.copy()

    def save_params(self):
        """Persist current parameters."""
        self.params_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.params_file, "w") as f:
            json.dump(self.current_params, f, indent=2)

    def optimize_context_weights(self, agent_weights: Dict[str, float]) -> Dict[str, Any]:
        """Adjust context budget weights based on agent accuracy.
        
        Agents with higher accuracy get more of the context budget.
        This means the bull/bear researchers and trader see MORE from
        the accurate analysts and LESS from the inaccurate ones.
        
        Args:
            agent_weights: From PerformanceScorer.get_agent_weights()
        
        Returns:
            Recommendation dict
        """
        analyst_agents = ["technical_analyst", "sentiment_analyst", "news_analyst", "fundamentals_analyst"]
        
        old_weights = self.current_params.get("context_budget_weights", {})
        new_weights = {}
        changes = []

        for agent in analyst_agents:
            perf_weight = agent_weights.get(agent, 1.0)
            old_w = old_weights.get(agent, 1.0)
            
            # Blend: 70% performance-based, 30% current (smooth transitions)
            new_w = round(0.7 * perf_weight + 0.3 * old_w, 2)
            # Clamp to [0.3, 2.0] — never fully mute an agent
            new_w = max(0.3, min(2.0, new_w))
            new_weights[agent] = new_w
            
            if abs(new_w - old_w) > 0.05:
                direction = "↑" if new_w > old_w else "↓"
                changes.append(f"{agent}: {old_w:.2f} → {new_w:.2f} {direction}")

        return {
            "type": "context_weights",
            "old": old_weights,
            "new": new_weights,
            "changes": changes,
            "reason": "Adjusted based on agent directional accuracy",
        }

    def optimize_conviction_thresholds(self, trade_reviews: List[Dict]) -> Dict[str, Any]:
        """Adjust conviction thresholds based on trade outcomes.
        
        - Too many losing trades from low conviction → raise thresholds
        - Missing profitable trades (lots of HOLDs before moves) → lower thresholds
        """
        if not trade_reviews:
            return {"type": "conviction", "changes": [], "reason": "Insufficient data"}

        losses = [r for r in trade_reviews if r.get("pnl_pct", 0) < 0]
        wins = [r for r in trade_reviews if r.get("pnl_pct", 0) > 0]
        
        current = self.current_params.get("conviction_thresholds", DEFAULT_PARAMS["conviction_thresholds"])
        new_thresholds = current.copy()
        changes = []

        # If win rate < 40%, tighten new position threshold
        total = len(trade_reviews)
        win_rate = len(wins) / total if total > 0 else 0.5

        if win_rate < 0.4 and total >= 5:
            new_val = min(current["new_position"] + 1, 8)
            if new_val != current["new_position"]:
                changes.append(f"new_position: {current['new_position']} → {new_val} (win rate {win_rate:.0%} too low)")
                new_thresholds["new_position"] = new_val

        elif win_rate > 0.65 and total >= 5:
            new_val = max(current["new_position"] - 1, 4)
            if new_val != current["new_position"]:
                changes.append(f"new_position: {current['new_position']} → {new_val} (win rate {win_rate:.0%} — can be more aggressive)")
                new_thresholds["new_position"] = new_val

        # Check if reversals are profitable
        # (would need trade metadata to identify reversals — flag for future)

        return {
            "type": "conviction",
            "old": current,
            "new": new_thresholds,
            "changes": changes,
            "win_rate": round(win_rate, 2),
            "sample_size": total,
            "reason": "Based on win rate analysis",
        }

    def optimize_debate_rounds(self, trade_reviews: List[Dict]) -> Dict[str, Any]:
        """Assess whether debate quality justifies current round count.
        
        - If chain_coherence is frequently "broken" → increase rounds
        - If decisions are consistently correct → current rounds are fine
        - If token cost is high and accuracy is good → consider reducing
        """
        if not trade_reviews:
            return {"type": "debate_rounds", "changes": [], "reason": "Insufficient data"}

        coherence_scores = [r.get("chain_coherence", "good") for r in trade_reviews]
        broken_count = coherence_scores.count("broken")
        degraded_count = coherence_scores.count("degraded")
        total = len(coherence_scores)

        current = self.current_params.get("debate_rounds", DEFAULT_PARAMS["debate_rounds"])
        new_rounds = current.copy()
        changes = []

        # If >30% broken coherence, suggest more debate rounds
        if total >= 5 and (broken_count / total) > 0.3:
            for key in ["research", "risk"]:
                new_val = min(current[key] + 1, 3)
                if new_val != current[key]:
                    changes.append(f"{key}: {current[key]} → {new_val} ({broken_count}/{total} broken coherence)")
                    new_rounds[key] = new_val

        # If everything is good and we're using >1 round, could reduce
        elif total >= 10 and broken_count == 0 and degraded_count <= 1:
            for key in ["research", "risk"]:
                if current[key] > 1:
                    new_val = current[key] - 1
                    changes.append(f"{key}: {current[key]} → {new_val} (consistently good coherence, save tokens)")
                    new_rounds[key] = new_val

        return {
            "type": "debate_rounds",
            "old": current,
            "new": new_rounds,
            "changes": changes,
            "coherence_breakdown": {
                "good": coherence_scores.count("good"),
                "degraded": degraded_count,
                "broken": broken_count,
            },
            "reason": "Based on decision chain coherence analysis",
        }

    def generate_agent_hints(self, scorer_report: Dict) -> Dict[str, str]:
        """Generate performance-based hints to inject into agent prompts.
        
        Underperformers get corrective hints.
        Stars get reinforcement.
        """
        hints = {}
        
        for ranking in scorer_report.get("rankings", []):
            agent = ranking["agent"]
            accuracy = ranking["accuracy"]
            streak = ranking["streak"]
            status = ranking["status"]

            if status == "underperformer":
                hints[agent] = (
                    f"⚠️ PERFORMANCE ALERT: Your recent accuracy is {accuracy:.0f}% "
                    f"(streak: {streak}). Review your analytical framework — "
                    f"your signals have been unreliable. Be MORE conservative "
                    f"in your confidence scores until accuracy improves."
                )
            elif status == "star":
                hints[agent] = (
                    f"✅ Your accuracy has been strong ({accuracy:.0f}%). "
                    f"Your signals carry extra weight in the current pipeline. "
                    f"Maintain your analytical rigor."
                )

        return hints

    def run_full_optimization(self, trade_reviews: List[Dict], scorer_report: Dict) -> Dict:
        """Run all optimizations and generate a complete recommendation set.
        
        Returns all recommendations WITHOUT applying them.
        """
        agent_weights = {}
        for agent, data in scorer_report.get("agent_details", {}).items():
            if data["total_calls"] >= MIN_TRADES_FOR_RELIABILITY:
                agent_weights[agent] = round(0.5 + (data["accuracy_pct"] / 100.0), 2)
            else:
                agent_weights[agent] = 1.0

        recommendations = {
            "generated_at": datetime.now().isoformat(),
            "trades_analyzed": len(trade_reviews),
            "context_weights": self.optimize_context_weights(agent_weights),
            "conviction": self.optimize_conviction_thresholds(trade_reviews),
            "debate_rounds": self.optimize_debate_rounds(trade_reviews),
            "agent_hints": self.generate_agent_hints(scorer_report),
        }

        # Count total changes
        total_changes = sum(
            len(recommendations[k].get("changes", []))
            for k in ["context_weights", "conviction", "debate_rounds"]
        )
        total_hints = len(recommendations["agent_hints"])
        recommendations["summary"] = {
            "total_parameter_changes": total_changes,
            "total_agent_hints": total_hints,
            "action_required": total_changes > 0 or total_hints > 0,
        }

        # Save recommendations
        self.recommendations_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.recommendations_file, "w") as f:
            json.dump(recommendations, f, indent=2)

        log.info(f"Optimization: {total_changes} parameter changes, {total_hints} agent hints")
        return recommendations

    def apply_recommendations(self, recommendations: Dict):
        """Apply recommended parameter changes.
        
        Call this only after human review / explicit approval.
        """
        # Context weights
        cw = recommendations.get("context_weights", {})
        if cw.get("changes"):
            self.current_params["context_budget_weights"] = cw["new"]
            log.info(f"Applied context weight changes: {cw['changes']}")

        # Conviction thresholds
        ct = recommendations.get("conviction", {})
        if ct.get("changes"):
            self.current_params["conviction_thresholds"] = ct["new"]
            log.info(f"Applied conviction changes: {ct['changes']}")

        # Debate rounds
        dr = recommendations.get("debate_rounds", {})
        if dr.get("changes"):
            self.current_params["debate_rounds"] = dr["new"]
            log.info(f"Applied debate round changes: {dr['changes']}")

        self.save_params()
        log.info("✅ All recommendations applied")


# Import from scorer for the constant
MIN_TRADES_FOR_RELIABILITY = 5

"""
Performance Scorer (C) — Agent-level performance attribution.

Tracks each agent's directional accuracy over time.
Computes rolling metrics: accuracy, influence, consistency.
Flags underperformers and star performers.

Runs after each trade close and accumulates over time.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("performance_scorer")


# Minimum trades before an agent's score is considered reliable
MIN_TRADES_FOR_RELIABILITY = 5

# Thresholds
UNDERPERFORMER_ACCURACY = 40.0  # Below this = flagged
STAR_PERFORMER_ACCURACY = 70.0  # Above this = starred
INFLUENCE_DECAY = 0.9  # Exponential decay for rolling metrics


class PerformanceScorer:
    """Tracks and scores agent performance over time."""

    AGENTS = [
        "technical_analyst", "sentiment_analyst", "news_analyst",
        "fundamentals_analyst", "research_manager", "trader",
        "risk_judge", "fund_manager",
        "bull_researcher", "bear_researcher",
    ]

    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.scores_file = self.project_dir / "logs" / "agent_scores.json"
        self.scores = self._load_scores()

    def _load_scores(self) -> Dict:
        """Load accumulated scores from disk."""
        if self.scores_file.exists():
            with open(self.scores_file) as f:
                return json.load(f)
        return self._init_scores()

    def _init_scores(self) -> Dict:
        """Initialize empty score structure."""
        scores = {
            "agents": {},
            "trades_scored": 0,
            "last_updated": None,
        }
        for agent in self.AGENTS:
            scores["agents"][agent] = {
                "correct_calls": 0,
                "incorrect_calls": 0,
                "neutral_calls": 0,
                "total_calls": 0,
                "accuracy_pct": 0.0,
                "rolling_accuracy": 0.0,  # EMA-based
                "pnl_when_correct": 0.0,
                "pnl_when_incorrect": 0.0,
                "high_influence_count": 0,
                "primary_failure_count": 0,
                "primary_success_count": 0,
                "streak": 0,  # Positive = correct streak, negative = incorrect
                "last_10": [],  # Last 10 results: 1=correct, 0=incorrect, -1=neutral
                "status": "new",  # new, reliable, underperformer, star
            }
        return scores

    def save(self):
        """Persist scores to disk."""
        self.scores["last_updated"] = datetime.now().isoformat()
        self.scores_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.scores_file, "w") as f:
            json.dump(self.scores, f, indent=2)

    def record_trade(self, trade_review: Dict):
        """Record a single trade review into cumulative scores.
        
        Args:
            trade_review: Output from TradeReviewer.review_trade()
        """
        pnl = trade_review.get("pnl_pct", 0)
        agent_scores = trade_review.get("agent_scores", {})
        failure_point = trade_review.get("primary_failure_point", "none")
        success_point = trade_review.get("primary_success_point", "none")

        for agent_name, assessment in agent_scores.items():
            if agent_name not in self.scores["agents"]:
                continue

            agent = self.scores["agents"][agent_name]
            agent["total_calls"] += 1

            # Determine correctness
            correct = assessment.get("signal_correct", False)
            # Risk judge and FM use different keys
            if agent_name == "risk_judge":
                correct = assessment.get("sizing_appropriate", False)
            elif agent_name == "fund_manager":
                correct = assessment.get("approval_justified", False)

            if correct:
                agent["correct_calls"] += 1
                agent["pnl_when_correct"] += abs(pnl)
                agent["last_10"].append(1)
                agent["streak"] = max(1, agent["streak"] + 1)
            else:
                agent["incorrect_calls"] += 1
                agent["pnl_when_incorrect"] += abs(pnl)
                agent["last_10"].append(0)
                agent["streak"] = min(-1, agent["streak"] - 1)

            # Trim last_10
            agent["last_10"] = agent["last_10"][-10:]

            # Influence tracking
            if assessment.get("influence") == "high":
                agent["high_influence_count"] += 1

            # Accuracy
            total_decisive = agent["correct_calls"] + agent["incorrect_calls"]
            agent["accuracy_pct"] = round(
                agent["correct_calls"] / total_decisive * 100, 1
            ) if total_decisive > 0 else 0.0

            # Rolling accuracy (EMA over last 10)
            recent = agent["last_10"][-10:]
            if recent:
                agent["rolling_accuracy"] = round(
                    sum(r for r in recent if r >= 0) / max(len([r for r in recent if r >= 0]), 1) * 100, 1
                )

            # Status classification
            if agent["total_calls"] < MIN_TRADES_FOR_RELIABILITY:
                agent["status"] = "new"
            elif agent["accuracy_pct"] < UNDERPERFORMER_ACCURACY:
                agent["status"] = "underperformer"
            elif agent["accuracy_pct"] >= STAR_PERFORMER_ACCURACY:
                agent["status"] = "star"
            else:
                agent["status"] = "reliable"

        # Failure/success point tracking
        if failure_point in self.scores["agents"]:
            self.scores["agents"][failure_point]["primary_failure_count"] += 1
        if success_point in self.scores["agents"]:
            self.scores["agents"][success_point]["primary_success_count"] += 1

        self.scores["trades_scored"] += 1
        self.save()

    def get_report(self) -> Dict:
        """Generate a performance report."""
        agents = self.scores["agents"]
        
        # Rank by accuracy (only reliable+ agents)
        ranked = sorted(
            [(name, data) for name, data in agents.items() if data["total_calls"] >= MIN_TRADES_FOR_RELIABILITY],
            key=lambda x: x[1]["accuracy_pct"],
            reverse=True,
        )

        underperformers = [name for name, data in agents.items() if data["status"] == "underperformer"]
        stars = [name for name, data in agents.items() if data["status"] == "star"]

        # Most common failure point
        failure_counts = {name: data["primary_failure_count"] for name, data in agents.items()}
        worst_offender = max(failure_counts, key=failure_counts.get) if any(failure_counts.values()) else "none"

        return {
            "trades_scored": self.scores["trades_scored"],
            "last_updated": self.scores["last_updated"],
            "rankings": [
                {"agent": name, "accuracy": data["accuracy_pct"], "rolling": data["rolling_accuracy"],
                 "calls": data["total_calls"], "streak": data["streak"], "status": data["status"]}
                for name, data in ranked
            ],
            "underperformers": underperformers,
            "stars": stars,
            "worst_failure_point": worst_offender,
            "agent_details": agents,
        }

    def get_agent_weights(self) -> Dict[str, float]:
        """Calculate performance-based weights for each agent.
        
        Used by ParameterOptimizer to adjust context budgets.
        Higher accuracy = higher weight = more context budget.
        """
        weights = {}
        agents = self.scores["agents"]

        for name, data in agents.items():
            if data["total_calls"] < MIN_TRADES_FOR_RELIABILITY:
                weights[name] = 1.0  # Default weight for new agents
            else:
                # Map accuracy to weight: 0-100% → 0.5-1.5
                accuracy = data["accuracy_pct"]
                weights[name] = round(0.5 + (accuracy / 100.0), 2)

        return weights

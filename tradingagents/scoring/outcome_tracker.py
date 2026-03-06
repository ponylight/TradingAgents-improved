"""
Outcome Tracking System

Records every trade decision and tracks whether it was correct.
Feeds accuracy data back into agent memory for learning.

Stored in: logs/outcome_history.json
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

log = logging.getLogger("outcome_tracker")

DEFAULT_PATH = Path(__file__).parent.parent.parent / "logs" / "outcome_history.json"


class OutcomeTracker:
    """Tracks trade decisions and their outcomes for feedback learning."""

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path
        self._history = self._load()

    def _load(self) -> List[Dict]:
        try:
            if self.path.exists():
                return json.loads(self.path.read_text())
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Failed to load outcome history: {e}")
        return []

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._history, indent=2))
        except IOError as e:
            log.warning(f"Failed to save outcome history: {e}")

    def record_decision(
        self,
        decision: str,
        confidence: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        objective_score: Optional[float] = None,
        objective_signal: Optional[str] = None,
        was_overridden: bool = False,
        override_reason: str = "",
        agent_reports: Optional[Dict] = None,
        trade_params: Optional[Dict] = None,
        validation_warnings: Optional[List[str]] = None,
    ) -> int:
        """
        Record a trade decision for later outcome evaluation.

        Returns: record ID (index in history)
        """
        record = {
            "id": len(self._history),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "confidence": confidence,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "objective_score": objective_score,
            "objective_signal": objective_signal,
            "was_overridden": was_overridden,
            "override_reason": override_reason,
            "validation_warnings": validation_warnings or [],
            # Outcome fields — filled in later
            "outcome": None,  # "correct", "incorrect", "partial", "expired"
            "exit_price": None,
            "return_pct": None,
            "max_favorable_pct": None,
            "max_adverse_pct": None,
            "exit_reason": None,  # "tp_hit", "sl_hit", "trailing_stop", "reversal", "manual"
            "evaluated_at": None,
            "held_hours": None,
        }

        self._history.append(record)
        self._save()

        log.info(
            f"📝 Decision recorded #{record['id']}: {decision} @ ${entry_price:,.0f} "
            f"(conf={confidence}, obj_score={objective_score})"
        )
        return record["id"]

    def record_outcome(
        self,
        record_id: int,
        exit_price: float,
        exit_reason: str,
        max_favorable_pct: float = 0.0,
        max_adverse_pct: float = 0.0,
    ):
        """
        Record the outcome of a previously recorded decision.

        Args:
            record_id: ID from record_decision()
            exit_price: price at exit
            exit_reason: why the trade was closed
            max_favorable_pct: best unrealized PnL during trade
            max_adverse_pct: worst unrealized PnL during trade
        """
        if record_id < 0 or record_id >= len(self._history):
            log.warning(f"Invalid record ID: {record_id}")
            return

        record = self._history[record_id]
        entry = record.get("entry_price", 0)
        decision = record.get("decision", "")

        if entry <= 0:
            log.warning(f"Record {record_id} has no entry price")
            return

        # Calculate return
        if decision in ("BUY", "OPEN_LONG"):
            return_pct = ((exit_price - entry) / entry) * 100
        elif decision in ("SELL", "OPEN_SHORT"):
            return_pct = ((entry - exit_price) / entry) * 100
        else:
            return_pct = 0.0

        # Determine if correct
        if return_pct > 0.5:
            outcome = "correct"
        elif return_pct < -0.5:
            outcome = "incorrect"
        else:
            outcome = "partial"  # Basically breakeven

        # Time held
        try:
            opened = datetime.fromisoformat(record["timestamp"])
            held_hours = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
        except (ValueError, TypeError):
            held_hours = None

        record.update({
            "outcome": outcome,
            "exit_price": exit_price,
            "return_pct": round(return_pct, 2),
            "max_favorable_pct": round(max_favorable_pct, 2),
            "max_adverse_pct": round(max_adverse_pct, 2),
            "exit_reason": exit_reason,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "held_hours": round(held_hours, 1) if held_hours else None,
        })

        self._save()

        emoji = "✅" if outcome == "correct" else "❌" if outcome == "incorrect" else "➖"
        log.info(
            f"{emoji} Outcome #{record_id}: {outcome} | {decision} @ ${entry:,.0f} → ${exit_price:,.0f} "
            f"| return: {return_pct:+.2f}% | reason: {exit_reason} "
            f"| MFE: {max_favorable_pct:+.2f}% MAE: {max_adverse_pct:+.2f}%"
        )

    def get_accuracy_stats(self, last_n: int = 50) -> Dict[str, Any]:
        """
        Get accuracy statistics for recent decisions.

        Returns dict with win_rate, avg_return, etc.
        """
        evaluated = [r for r in self._history if r.get("outcome") is not None]
        recent = evaluated[-last_n:] if len(evaluated) > last_n else evaluated

        if not recent:
            return {"trades": 0, "win_rate": None, "avg_return": None}

        correct = sum(1 for r in recent if r["outcome"] == "correct")
        incorrect = sum(1 for r in recent if r["outcome"] == "incorrect")
        returns = [r["return_pct"] for r in recent if r.get("return_pct") is not None]

        # Override accuracy
        overridden = [r for r in recent if r.get("was_overridden")]
        override_correct = sum(1 for r in overridden if r["outcome"] == "correct")

        # Agent accuracy (non-overridden)
        agent_only = [r for r in recent if not r.get("was_overridden")]
        agent_correct = sum(1 for r in agent_only if r["outcome"] == "correct")

        stats = {
            "trades": len(recent),
            "correct": correct,
            "incorrect": incorrect,
            "win_rate": round(correct / len(recent) * 100, 1) if recent else 0,
            "avg_return": round(sum(returns) / len(returns), 2) if returns else 0,
            "total_return": round(sum(returns), 2) if returns else 0,
            "best_trade": round(max(returns), 2) if returns else 0,
            "worst_trade": round(min(returns), 2) if returns else 0,
            "avg_held_hours": round(
                sum(r["held_hours"] for r in recent if r.get("held_hours")) /
                max(1, sum(1 for r in recent if r.get("held_hours"))),
                1
            ),
            "override_trades": len(overridden),
            "override_win_rate": round(
                override_correct / len(overridden) * 100, 1
            ) if overridden else None,
            "agent_only_win_rate": round(
                agent_correct / len(agent_only) * 100, 1
            ) if agent_only else None,
        }

        log.info(
            f"📊 Accuracy stats (last {len(recent)}): "
            f"win_rate={stats['win_rate']}% avg_return={stats['avg_return']}% "
            f"agent_win={stats['agent_only_win_rate']}% override_win={stats['override_win_rate']}%"
        )

        return stats

    def get_feedback_for_agents(self, last_n: int = 10) -> str:
        """
        Generate a feedback summary that can be injected into agent prompts.
        
        Returns a natural language summary of recent performance.
        """
        stats = self.get_accuracy_stats(last_n)
        if stats["trades"] == 0:
            return "No historical trade outcomes available yet."

        lines = [
            f"## Recent Performance (last {stats['trades']} trades)",
            f"- Win rate: {stats['win_rate']}%",
            f"- Average return: {stats['avg_return']}%",
            f"- Best: {stats['best_trade']}% | Worst: {stats['worst_trade']}%",
        ]

        if stats.get("override_win_rate") is not None:
            lines.append(
                f"- Override accuracy: {stats['override_win_rate']}% "
                f"(vs agent-only: {stats.get('agent_only_win_rate', 'N/A')}%)"
            )

        # Recent errors to learn from
        recent_errors = [
            r for r in self._history[-last_n:]
            if r.get("outcome") == "incorrect"
        ]
        if recent_errors:
            lines.append("\n### Recent Incorrect Decisions:")
            for err in recent_errors[-3:]:
                lines.append(
                    f"- {err['decision']} @ ${err.get('entry_price', 0):,.0f} → "
                    f"return {err.get('return_pct', 0):+.2f}% "
                    f"(reason: {err.get('exit_reason', 'unknown')})"
                )

        return "\n".join(lines)

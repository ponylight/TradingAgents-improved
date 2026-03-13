"""
Decision Validation Layer

Post-LLM sanity check: validates agent decisions against objective data.
Catches absurd decisions before they reach the executor.

Returns: validated decision, confidence adjustment, and warnings.
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
from .objective_score import ScoreBreakdown

log = logging.getLogger("decision_validator")


class ValidationResult:
    """Result of decision validation."""

    def __init__(self):
        self.original_decision: str = ""
        self.validated_decision: str = ""
        self.confidence_adjustment: float = 0.0  # -1.0 to +1.0
        self.warnings: List[str] = []
        self.overridden: bool = False
        self.override_reason: str = ""

    @property
    def should_alert(self) -> bool:
        return self.overridden or len(self.warnings) > 0


def validate_decision(
    agent_decision: str,
    agent_confidence: int,
    objective_score: ScoreBreakdown,
    trade_params: Dict[str, Any],
    current_position: str = "NEUTRAL",  # NEUTRAL, LONG, SHORT
    geopolitical_severity: float = 0.0,
    geopolitical_events: Optional[List[str]] = None,
) -> ValidationResult:
    """
    Validate an agent decision against objective data.

    Args:
        agent_decision: BUY/SELL/HOLD from agents
        agent_confidence: 0-10 confidence
        objective_score: ScoreBreakdown from calculate_objective_score()
        trade_params: parsed trade params (SL, TP, etc.)
        current_position: current position state
        geopolitical_severity: 0 to -100 from geopolitical scanner
        geopolitical_events: list of detected events

    Returns:
        ValidationResult with validated decision and any warnings/overrides
    """
    result = ValidationResult()
    result.original_decision = agent_decision
    result.validated_decision = agent_decision

    decision = agent_decision.upper().strip()
    obj_signal = objective_score.signal
    obj_strength = objective_score.strength

    # --- Rule 1: Strong conflict between objective score and agent decision ---
    if objective_score.conflicts_with(decision):
        if obj_strength == "STRONG":
            # Strong conflict — override
            result.validated_decision = obj_signal
            result.overridden = True
            result.override_reason = (
                f"STRONG objective score ({objective_score.overall:+.1f}) conflicts with "
                f"agent decision ({decision}). Overriding to {obj_signal}."
            )
            result.confidence_adjustment = -0.3
            log.warning(f"🚨 OVERRIDE: {result.override_reason}")

        elif obj_strength == "MODERATE":
            # Moderate conflict — warn and reduce confidence for directional trades.
            # HOLD vs moderate BUY/SELL should stay a soft warning only.
            result.warnings.append(
                f"Objective score ({objective_score.overall:+.1f} = {obj_signal}) "
                f"conflicts with agent decision ({decision}). Score strength: {obj_strength}."
            )
            if decision not in ("HOLD", "STAY_NEUTRAL"):
                result.confidence_adjustment = -0.2
            log.warning(f"⚠️ CONFLICT: objective={obj_signal} vs agent={decision} (moderate)")

        elif obj_strength == "WEAK":
            # Weak conflict — just note it
            result.warnings.append(
                f"Minor divergence: objective score {objective_score.overall:+.1f} vs agent {decision}"
            )

    # --- Rule 2: Geopolitical risk override ---
    if geopolitical_severity <= -50:
        # Severe geopolitical event — block new longs, prefer shorts or cash
        if decision in ("BUY", "OPEN_LONG"):
            result.validated_decision = "HOLD"
            result.overridden = True
            result.override_reason = (
                f"SEVERE geopolitical risk ({geopolitical_severity:.0f}): "
                f"blocking new long. Events: {(geopolitical_events or ['unknown'])[:2]}"
            )
            result.confidence_adjustment = -0.4
            log.warning(f"🌍 GEOPOLITICAL OVERRIDE: {result.override_reason}")

        elif decision == "HOLD" and current_position == "LONG":
            result.warnings.append(
                f"Holding LONG during severe geopolitical risk ({geopolitical_severity:.0f}). "
                f"Consider reducing exposure."
            )
            result.confidence_adjustment = -0.2

    elif geopolitical_severity <= -25:
        # Moderate geopolitical risk — just warn
        if decision in ("BUY", "OPEN_LONG"):
            result.warnings.append(
                f"Moderate geopolitical risk ({geopolitical_severity:.0f}) detected. "
                f"Proceeding with caution."
            )
            result.confidence_adjustment = -0.1

    # --- Rule 3: Momentum divergence check ---
    # If momentum is strongly against the decision
    mom = objective_score.momentum
    if decision in ("BUY", "OPEN_LONG") and mom < -40:
        result.warnings.append(
            f"Strong bearish momentum ({mom:+.1f}) contradicts BUY. "
            f"RSI/Stoch/MACD all bearish."
        )
        result.confidence_adjustment = min(result.confidence_adjustment, -0.15)
    elif decision in ("SELL", "OPEN_SHORT") and mom > 40:
        result.warnings.append(
            f"Strong bullish momentum ({mom:+.1f}) contradicts SELL. "
            f"RSI/Stoch/MACD all bullish."
        )
        result.confidence_adjustment = min(result.confidence_adjustment, -0.15)

    # --- Rule 4: Volume confirmation ---
    vol = objective_score.volume
    if abs(vol) < 5 and decision in ("BUY", "SELL", "OPEN_LONG", "OPEN_SHORT"):
        result.warnings.append(
            f"Low volume score ({vol:+.1f}) — move may lack conviction."
        )

    # --- Rule 5: Sanity check on trade params ---
    sl = trade_params.get("stop_loss")
    tp1 = trade_params.get("take_profit_1")
    entry = trade_params.get("entry_price")

    if sl and tp1 and entry:
        if decision in ("BUY", "OPEN_LONG"):
            risk = abs(entry - sl)
            reward = abs(tp1 - entry)
            if risk > 0 and reward / risk < 1.0:
                result.warnings.append(
                    f"Poor R:R ratio ({reward/risk:.2f}:1) for long. "
                    f"Risk ${risk:,.0f} > Reward ${reward:,.0f}."
                )
        elif decision in ("SELL", "OPEN_SHORT"):
            risk = abs(sl - entry)
            reward = abs(entry - tp1)
            if risk > 0 and reward / risk < 1.0:
                result.warnings.append(
                    f"Poor R:R ratio ({reward/risk:.2f}:1) for short. "
                    f"Risk ${risk:,.0f} > Reward ${reward:,.0f}."
                )

    # Log summary
    if result.should_alert:
        status = "OVERRIDDEN" if result.overridden else "WARNINGS"
        log.info(
            f"🔍 Validation [{status}]: {result.original_decision} → {result.validated_decision} "
            f"| confidence adj: {result.confidence_adjustment:+.1f} "
            f"| warnings: {len(result.warnings)}"
        )
    else:
        log.info(f"✅ Validation passed: {decision} aligned with objective score")

    return result

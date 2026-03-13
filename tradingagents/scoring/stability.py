"""
Trading System Stability Scorer

Evaluates overall health of our trading system on a 0-100 scale across 5 dimensions.
Run weekly or on-demand to track system quality over time.

Dimensions (inspired by ClawQuant's scorer):
  - Quality (30%): Risk-adjusted returns (Sharpe, Sortino equivalents)
  - Risk (30%): Max drawdown, worst trade, stop-loss discipline
  - Robustness (20%): Win rate, profit factor, consistency
  - Data Quality (10%): Are indicators producing sane values?
  - Decision Quality (10%): Agent vs override accuracy, validation hit rate
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("stability_scorer")

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _clamp(value: float, lo: float = 0, hi: float = 100) -> float:
    return max(lo, min(hi, value))


def _score_quality(trades: List[Dict]) -> float:
    """Score based on risk-adjusted returns."""
    if len(trades) < 3:
        return 50.0  # Not enough data

    returns = [t.get("return_pct", 0) for t in trades if t.get("return_pct") is not None]
    if not returns:
        return 50.0

    avg_ret = sum(returns) / len(returns)
    std_ret = (sum((r - avg_ret) ** 2 for r in returns) / len(returns)) ** 0.5

    # Pseudo-Sharpe (return / risk) — negative Sharpe penalizes instead of clamping to 0
    sharpe_like = avg_ret / std_ret if std_ret > 0 else 0
    sharpe_score = _clamp(sharpe_like * 33.3, lo=-30)

    # Average return contribution
    ret_score = _clamp(50 + avg_ret * 10)  # 0% = 50, +5% = 100, -5% = 0

    return sharpe_score * 0.6 + ret_score * 0.4


def _score_risk(trades: List[Dict]) -> float:
    """Score based on risk management."""
    if not trades:
        return 50.0

    returns = [t.get("return_pct", 0) for t in trades if t.get("return_pct") is not None]
    if not returns:
        return 50.0

    # Worst trade
    worst = min(returns)
    worst_score = _clamp(100 + worst * 5)  # -20% = 0, 0% = 100

    # Max adverse excursion (how deep did losses get)
    maes = [t.get("max_adverse_pct", 0) for t in trades if t.get("max_adverse_pct") is not None]
    avg_mae = sum(maes) / len(maes) if maes else 0
    mae_score = _clamp(100 + avg_mae * 5)  # Deep drawdowns = low score

    # SL discipline: was SL hit vs worse outcomes?
    sl_hits = sum(1 for t in trades if t.get("exit_reason") == "sl_hit")
    worse_than_sl = sum(1 for t in trades if t.get("return_pct", 0) < -10)
    discipline_score = 80 if worse_than_sl == 0 else _clamp(80 - worse_than_sl * 20)

    return worst_score * 0.3 + mae_score * 0.4 + discipline_score * 0.3


def _score_robustness(trades: List[Dict]) -> float:
    """Score based on consistency."""
    if len(trades) < 3:
        return 50.0

    correct = sum(1 for t in trades if t.get("outcome") == "correct")
    win_rate = correct / len(trades) if trades else 0
    wr_score = _clamp(win_rate * 100)  # 50% = 50, 70% = 70 (linear)

    # Profit factor: gross wins / gross losses
    wins = sum(t.get("return_pct", 0) for t in trades if t.get("return_pct", 0) > 0)
    losses = abs(sum(t.get("return_pct", 0) for t in trades if t.get("return_pct", 0) < 0))
    pf = wins / losses if losses > 0 else (100 if wins > 0 else 50)
    pf_score = _clamp(pf * 33.3)

    # Penalize too few trades
    count_mult = 0.5 if len(trades) < 5 else 0.8 if len(trades) < 10 else 1.0

    return (wr_score * 0.4 + pf_score * 0.6) * count_mult


def _score_data_quality(brief=None) -> float:
    """Score based on whether indicators produce sane values."""
    if brief is None:
        return 50.0

    score = 100.0
    penalties = []

    n_tf = max(len(brief.timeframes), 1)
    penalty_per_tf = 30.0 / n_tf  # Proportional: total penalty scales with broken timeframes

    for tf in brief.timeframes:
        # VWAP z-score should be -3 to +3
        z = tf.vwap_state.zscore_distance
        if abs(z) > 5:
            score -= penalty_per_tf * 0.5
            penalties.append(f"{tf.timeframe}: VWAP z={z:.1f} (abnormal)")

        # Volume ratio should be 0.1 to 5.0
        vr = tf.volume.vol_ma_ratio
        if vr < 0.05 or vr > 10:
            score -= penalty_per_tf * 0.33
            penalties.append(f"{tf.timeframe}: vol_ratio={vr:.2f} (abnormal)")

        # RSI should be 0-100
        rsi = tf.momentum.rsi_value
        if rsi < 0 or rsi > 100:
            score -= penalty_per_tf * 0.67
            penalties.append(f"{tf.timeframe}: RSI={rsi:.1f} (invalid)")

    if penalties:
        log.warning(f"Data quality issues: {penalties}")

    return _clamp(score)


def _score_decision_quality(trades: List[Dict]) -> float:
    """Score based on agent decision accuracy vs override accuracy."""
    if not trades:
        return 50.0

    # How often did validation warnings fire?
    warned = sum(1 for t in trades if t.get("validation_warnings"))
    warn_rate = warned / len(trades) if trades else 0

    # Overrides
    overridden = [t for t in trades if t.get("was_overridden")]
    override_correct = sum(1 for t in overridden if t.get("outcome") == "correct")
    agent_only = [t for t in trades if not t.get("was_overridden")]
    agent_correct = sum(1 for t in agent_only if t.get("outcome") == "correct")

    # If overrides are more accurate, the scoring system is working well
    override_wr = override_correct / len(overridden) if overridden else None
    agent_wr = agent_correct / len(agent_only) if agent_only else None

    if override_wr is not None and agent_wr is not None:
        if override_wr > agent_wr:
            score = 80  # Guardrails are helping
        else:
            score = 60  # Guardrails might need tuning
    else:
        score = 50

    # Penalize if too many warnings (system too noisy)
    if warn_rate > 0.8:
        score -= 15  # Almost every decision gets warned — thresholds too tight

    return _clamp(score)


def compute_system_stability(brief=None) -> Dict[str, Any]:
    """
    Compute the full stability score for the trading system.

    Args:
        brief: Optional CryptoTechnicalBrief for data quality check

    Returns:
        Dict with total score, breakdown, and recommendations.
    """
    # Load outcome history
    outcome_path = PROJECT_ROOT / "logs" / "outcome_history.json"
    trades = []
    if outcome_path.exists():
        try:
            trades = json.loads(outcome_path.read_text())
        except (json.JSONDecodeError, IOError):
            pass

    # Only use evaluated trades
    evaluated = [t for t in trades if t.get("outcome") is not None]

    quality = _score_quality(evaluated)
    risk = _score_risk(evaluated)
    robustness = _score_robustness(evaluated)
    data_quality = _score_data_quality(brief)
    decision_quality = _score_decision_quality(evaluated)

    total = (
        quality * 0.30
        + risk * 0.30
        + robustness * 0.20
        + data_quality * 0.10
        + decision_quality * 0.10
    )

    # Generate recommendations
    recommendations = []
    if quality < 40:
        recommendations.append("⚠️ Poor risk-adjusted returns. Review entry timing and position sizing.")
    if risk < 40:
        recommendations.append("⚠️ Risk management needs attention. Check SL discipline and max drawdown.")
    if robustness < 40:
        recommendations.append("⚠️ Low consistency. Win rate or profit factor too low.")
    if data_quality < 60:
        recommendations.append("🔧 Data quality issues detected. Check indicator calculations.")
    if decision_quality < 40:
        recommendations.append("🔍 Decision guardrails may need tuning.")

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": round(total, 1),
        "breakdown": {
            "quality": round(quality, 1),
            "risk": round(risk, 1),
            "robustness": round(robustness, 1),
            "data_quality": round(data_quality, 1),
            "decision_quality": round(decision_quality, 1),
        },
        "trades_evaluated": len(evaluated),
        "trades_total": len(trades),
        "recommendations": recommendations,
    }

    log.info(
        f"📊 System Stability: {result['total']}/100 "
        f"(Q={quality:.0f} R={risk:.0f} Rob={robustness:.0f} "
        f"DQ={data_quality:.0f} DecQ={decision_quality:.0f})"
    )

    return result

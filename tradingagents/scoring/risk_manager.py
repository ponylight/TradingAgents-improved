"""
Consolidated Risk Manager (DEPRECATED)

This module is superseded by the LLM-based Risk Judge in
tradingagents/agents/managers/risk_manager.py, which handles risk evaluation
as part of the agent pipeline. This class is retained for backward compatibility
with the live executor's pre-order checks but should not be extended.

Returns a list of risk actions: ALLOW, REDUCE, BLOCK, FLATTEN.
"""

import warnings
warnings.warn(
    "tradingagents.scoring.risk_manager is deprecated; "
    "risk evaluation is handled by agents/managers/risk_manager.py",
    DeprecationWarning,
    stacklevel=2,
)

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("risk_manager")


class RiskAction:
    """A risk check result."""

    def __init__(self, action: str, reason: str, severity: str = "INFO"):
        self.action = action  # ALLOW, REDUCE, BLOCK, FLATTEN
        self.reason = reason
        self.severity = severity  # INFO, WARNING, CRITICAL

    def __repr__(self):
        return f"RiskAction({self.action}: {self.reason})"


class RiskManager:
    """
    Consolidated risk checks. Call check_all() before ANY order.

    Config:
        max_position_pct: max margin as % of equity (default 10%)
        max_leverage: max leverage (default 20x)
        max_daily_loss_pct: max daily loss before halting (default 3%)
        max_drawdown_pct: max drawdown from peak before halting (default 10%)
        cooldown_minutes: min time between trades (default 30)
        max_trades_per_day: max trades in 24h (default 6)
        max_correlation_exposure: max exposure to correlated assets (default 15%)
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.max_position_pct = cfg.get("max_position_pct", 0.10)
        self.max_leverage = cfg.get("max_leverage", 15)          # reduced from 20 — swing default
        self.max_leverage_swing = cfg.get("max_leverage_swing", 15)   # hard limit for swing trades
        self.max_leverage_position = cfg.get("max_leverage_position", 10)  # hard limit for position trades (>24h)
        self.max_daily_loss_pct = cfg.get("max_daily_loss_pct", 0.03)
        self.max_drawdown_pct = cfg.get("max_drawdown_pct", 0.10)
        self.cooldown_minutes = cfg.get("cooldown_minutes", 30)
        self.max_trades_per_day = cfg.get("max_trades_per_day", 6)

    def check_all(
        self,
        equity: float,
        proposed_direction: str,
        proposed_alloc_pct: float,
        proposed_leverage: float,
        current_position: Optional[Dict] = None,
        daily_pnl: float = 0.0,
        peak_equity: float = 0.0,
        trades_today: int = 0,
        last_trade_time: Optional[datetime] = None,
        objective_score: float = 0.0,
        geopolitical_severity: float = 0.0,
        validation_result=None,
    ) -> List[RiskAction]:
        """
        Run all risk checks. Returns list of RiskActions.
        If any action is BLOCK or FLATTEN, the order should NOT proceed.
        """
        actions = []

        # 1. Daily loss limit
        daily_loss_pct = abs(daily_pnl / equity) if equity > 0 and daily_pnl < 0 else 0
        if daily_loss_pct >= self.max_daily_loss_pct:
            actions.append(RiskAction(
                "BLOCK",
                f"Daily loss limit reached: {daily_loss_pct:.1%} >= {self.max_daily_loss_pct:.1%}",
                "CRITICAL",
            ))
            return actions  # Hard stop

        # 2. Max drawdown from peak
        if peak_equity > 0:
            drawdown_pct = (peak_equity - equity) / peak_equity
            if drawdown_pct >= self.max_drawdown_pct:
                actions.append(RiskAction(
                    "FLATTEN",
                    f"Max drawdown breached: {drawdown_pct:.1%} >= {self.max_drawdown_pct:.1%}. "
                    f"Peak: ${peak_equity:,.0f}, Current: ${equity:,.0f}",
                    "CRITICAL",
                ))
                return actions  # Hard stop

        # 3. Position size limit
        if proposed_alloc_pct > self.max_position_pct:
            actions.append(RiskAction(
                "REDUCE",
                f"Position size {proposed_alloc_pct:.1%} exceeds max {self.max_position_pct:.1%}. "
                f"Reducing to {self.max_position_pct:.1%}.",
                "WARNING",
            ))

        # 4. Leverage limit — soft REDUCE (hard gates enforced before calling check_all)
        if proposed_leverage > self.max_leverage_swing:
            actions.append(RiskAction(
                "REDUCE",
                f"Leverage {proposed_leverage}x exceeds swing max {self.max_leverage_swing}x. Consider reducing allocation.",
                "WARNING",
            ))
        elif proposed_leverage > self.max_leverage:
            actions.append(RiskAction(
                "REDUCE",
                f"Leverage {proposed_leverage}x exceeds default max {self.max_leverage}x.",
                "WARNING",
            ))

        # 5. Trade frequency
        if trades_today >= self.max_trades_per_day:
            actions.append(RiskAction(
                "BLOCK",
                f"Daily trade limit reached: {trades_today}/{self.max_trades_per_day}",
                "WARNING",
            ))

        # 6. Cooldown
        if last_trade_time:
            minutes_since = (datetime.now(timezone.utc) - last_trade_time).total_seconds() / 60
            if minutes_since < self.cooldown_minutes:
                actions.append(RiskAction(
                    "BLOCK",
                    f"Cooldown: {minutes_since:.0f}/{self.cooldown_minutes} minutes since last trade",
                    "INFO",
                ))

        # 7. Geopolitical risk — reduce size for new positions
        if geopolitical_severity <= -50 and not current_position:
            actions.append(RiskAction(
                "REDUCE",
                f"Severe geopolitical risk ({geopolitical_severity:.0f}). Halving position size.",
                "WARNING",
            ))

        # 8. Low confidence — reduce size
        if validation_result and validation_result.confidence_adjustment < -0.2:
            actions.append(RiskAction(
                "REDUCE",
                f"Validation reduced confidence by {validation_result.confidence_adjustment:+.1f}. "
                f"Consider smaller position.",
                "INFO",
            ))

        # If nothing blocked, allow
        if not any(a.action in ("BLOCK", "FLATTEN") for a in actions):
            actions.append(RiskAction("ALLOW", "All risk checks passed", "INFO"))

        # Log
        for a in actions:
            if a.severity == "CRITICAL":
                log.critical(f"🛑 RISK: {a}")
            elif a.severity == "WARNING":
                log.warning(f"⚠️ RISK: {a}")
            else:
                log.info(f"✅ RISK: {a}")

        return actions

    def get_adjusted_allocation(self, proposed_pct: float, actions: List[RiskAction]) -> float:
        """Apply REDUCE actions to get final allocation percentage."""
        pct = proposed_pct
        for a in actions:
            if a.action == "REDUCE":
                if "halving" in a.reason.lower():
                    pct *= 0.5
                elif "exceeds max" in a.reason.lower():
                    pct = min(pct, self.max_position_pct)
        return pct

"""
trading_context.py — Builds per-agent trading mode context for prompt injection.

Section 4 of the paper requires a structured communication protocol where
each agent uses consistent terminology. This module ensures all agents
speak LONG/NEUTRAL/SHORT consistently, with position-aware transition logic.
"""

from tradingagents.agents.utils.agent_trading_modes import (
    get_trading_mode_context,
    get_position_transition,
)


def build_trading_context(state: dict) -> dict:
    """Build trading mode context from graph state.
    
    Returns a dict with keys:
        mode_instructions: str — inject into system prompt
        current_position: str — LONG/SHORT/NEUTRAL
        position_logic: str — transition matrix for current position
        decision_format: str — "LONG/NEUTRAL/SHORT"
        final_format: str — "FINAL TRANSACTION PROPOSAL: **LONG/NEUTRAL/SHORT**"
    """
    portfolio_context = state.get("portfolio_context", {})
    
    # Map executor position names to TradingModeConfig positions
    pos_raw = portfolio_context.get("position", "none").upper()
    if "LONG" in pos_raw or "BUY" in pos_raw:
        current_position = "LONG"
    elif "SHORT" in pos_raw or "SELL" in pos_raw:
        current_position = "SHORT"
    else:
        current_position = "NEUTRAL"
    
    # Always use trading mode (we trade BTC perps with shorts)
    ctx = get_trading_mode_context(
        config={"allow_shorts": True},
        current_position=current_position,
    )
    
    return {
        "mode_instructions": ctx["instructions"],
        "current_position": current_position,
        "position_logic": ctx.get("position_logic", ""),
        "decision_format": ctx["decision_format"],
        "final_format": ctx["final_format"],
        "actions": ctx["actions"],
    }

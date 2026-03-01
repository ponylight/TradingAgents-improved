"""Fund Manager — Final gate before execution.

Translates the risk-adjusted market view into a position action
based on current portfolio state. Prevents flip-flopping by requiring
minimum conviction to change direction.

Input: Risk Judge decision + current position state
Output: OPEN_LONG, OPEN_SHORT, CLOSE, HOLD + parameters
"""

import re
import json
import logging

log = logging.getLogger("fund_manager")


def create_fund_manager(llm):
    """Create the Fund Manager node."""

    def fund_manager_node(state) -> dict:
        risk_decision = state.get("final_trade_decision", "")
        trader_plan = state.get("trader_investment_plan", "")
        
        # Portfolio context injected via state
        portfolio_context = state.get("portfolio_context", {})
        current_position = portfolio_context.get("position", "none")  # "long", "short", "none"
        position_pnl_pct = portfolio_context.get("pnl_pct", 0)
        position_age_bars = portfolio_context.get("age_bars", 0)
        equity = portfolio_context.get("equity", 0)
        consecutive_same = portfolio_context.get("consecutive_same_direction", 0)
        last_decision = portfolio_context.get("last_decision", "HOLD")

        prompt = f"""You are the Fund Manager — the final decision-maker before trade execution. 
Your job is to translate the Risk Judge's market view into a concrete position action, 
considering what we CURRENTLY hold.

## Current Portfolio State
- Position: {current_position}
- Unrealized P&L: {position_pnl_pct:+.2f}%
- Position age: {position_age_bars} bars (4H)
- Account equity: ${equity:,.0f}
- Last decision: {last_decision}
- Consecutive same-direction signals: {consecutive_same}

## Risk Judge's Decision
{risk_decision}

## Trader's Plan
{trader_plan}

## Your Decision Framework

### When we have NO position:
- Strong signal (confidence ≥7) → OPEN position in signal direction
- Weak signal (confidence <7) → HOLD, wait for stronger setup
- Conflicting analysts → HOLD

### When we ALREADY have a position:
- Same direction signal → HOLD (confirm), maybe tighten trail if profitable
- Opposing signal with HIGH confidence (≥8) → CLOSE current, OPEN opposite
- Opposing signal with MEDIUM confidence (5-7) → CLOSE current only, don't reverse
- Opposing signal with LOW confidence (<5) → HOLD current position (noise)
- NEUTRAL signal → HOLD current position

### Anti-flip-flop Rules:
- If position is <3 bars old and opposing signal confidence <8, HOLD (too early to flip)
- If we flipped direction last decision, require confidence ≥9 to flip again
- If position is profitable (>2%), require opposing confidence ≥8 to close

## Required Output Format
Provide your reasoning, then end with:

```
---FUND_DECISION---
ACTION: OPEN_LONG/OPEN_SHORT/CLOSE/HOLD
CONFIDENCE: <1-10>
POSITION_SIZE: <percentage or "unchanged">
STOP_LOSS: <price or "unchanged">  
TAKE_PROFIT: <price or "unchanged">
REASON: <one-line summary>
---END_FUND_DECISION---
```
"""

        response = llm.invoke(prompt)
        fund_decision = response.content

        # Parse the structured block
        parsed = _parse_fund_decision(fund_decision)
        
        log.info(f"🏛️ Fund Manager: {parsed.get('action', 'UNKNOWN')} (confidence: {parsed.get('confidence', '?')})")
        log.info(f"   Reason: {parsed.get('reason', 'N/A')}")

        return {
            "fund_manager_decision": fund_decision,
            "fund_manager_parsed": parsed,
            "final_trade_decision": fund_decision,  # Override risk judge's decision
        }

    return fund_manager_node


def _parse_fund_decision(text: str) -> dict:
    """Parse the ---FUND_DECISION--- block."""
    parsed = {}
    
    block_match = re.search(r'---FUND_DECISION---(.*?)---END_FUND_DECISION---', text, re.DOTALL)
    if not block_match:
        # Fallback: try to extract action from prose
        text_upper = text.upper()
        for action in ["OPEN_LONG", "OPEN_SHORT", "CLOSE", "HOLD"]:
            if action in text_upper:
                parsed["action"] = action
                break
        return parsed

    block = block_match.group(1)
    
    fields = {
        "ACTION": ("action", str),
        "CONFIDENCE": ("confidence", int),
        "POSITION_SIZE": ("position_size", str),
        "STOP_LOSS": ("stop_loss", str),
        "TAKE_PROFIT": ("take_profit", str),
        "REASON": ("reason", str),
    }
    
    for key, (param_name, converter) in fields.items():
        m = re.search(rf'{key}:\s*(.+)', block)
        if m:
            val = m.group(1).strip()
            try:
                parsed[param_name] = converter(val)
            except (ValueError, TypeError):
                parsed[param_name] = val

    return parsed

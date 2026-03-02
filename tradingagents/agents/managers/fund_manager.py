"""Portfolio Manager (Fund Manager) — Final approver.

In a real trading firm, the Fund Manager:
- Reviews the Trader's proposal and Risk Manager's assessment
- Approves, rejects, or resizes at a PORTFOLIO level
- Does NOT re-analyze the market or second-guess direction
- Considers portfolio-wide exposure, correlation, capital allocation

The Trader owns the thesis. The Risk Manager gates the risk.
The Fund Manager says "yes, go" or "no, not now" from a portfolio perspective.
"""

import re
import json
import logging

log = logging.getLogger("fund_manager")


def create_fund_manager(llm, memory=None):
    """Create the Fund Manager (Portfolio Manager) node."""

    def fund_manager_node(state) -> dict:
        risk_decision = state.get("final_trade_decision", "")
        trader_plan = state.get("trader_investment_plan", "")
        
        # Portfolio context
        portfolio_context = state.get("portfolio_context", {})
        current_position = portfolio_context.get("position", "none")
        position_pnl_pct = portfolio_context.get("pnl_pct", 0)
        hours_held = portfolio_context.get("hours_held", 0)
        equity = portfolio_context.get("equity", 0)
        last_decision = portfolio_context.get("last_decision", "HOLD")

        # Get past memories if available
        past_memory_str = ""
        if memory:
            try:
                past_memories = memory.get_memories(trader_plan[:500], n_matches=2)
                for rec in past_memories:
                    past_memory_str += rec["recommendation"] + "\n\n"
            except Exception:
                pass

        prompt = f"""You are the Fund Manager — the final approver for trade execution.

## Your Role
- You APPROVE or REJECT the Trader's proposal at a portfolio level
- You do NOT re-analyze the market or change the trade direction
- You do NOT second-guess the Trader's thesis — that is their job
- You consider: portfolio exposure, capital allocation, timing, risk assessment

## Current Portfolio
- Position: {current_position}
- P&L: {position_pnl_pct:+.2f}%
- Hours held: {hours_held:.1f}
- Equity: ${equity:,.0f}
- Last decision: {last_decision}

## Trader's Proposal
{trader_plan[:2000]}

## Risk Manager's Assessment
{risk_decision[:1500]}

## Past Reflections
{past_memory_str if past_memory_str else "None."}

## Decision Framework
1. If Risk Manager VETOED → you MUST reject (do not override risk veto)
2. If Risk Manager APPROVED and Trader proposal is sound → APPROVE
3. If Risk Manager APPROVED WITH RESIZE → approve with the resized parameters
4. You may reject if:
   - Portfolio is already overexposed to this asset/sector
   - Capital allocation limits would be breached
   - Timing is poor (e.g., right before a major known event)

## Required Output
State your decision, then end with:

FINAL TRANSACTION PROPOSAL: **BUY/SELL/HOLD**

And include:
```
---FUND_DECISION---
ACTION: OPEN_LONG/OPEN_SHORT/CLOSE/HOLD
CONFIDENCE: <1-10>
STOP_LOSS: <price or "as_proposed">
TAKE_PROFIT_1: <price or "as_proposed">
REASON: <one-line>
---END_FUND_DECISION---
```
"""

        response = llm.invoke(prompt)
        fund_decision = response.content

        parsed = _parse_fund_decision(fund_decision)
        log.info(f"Fund Manager: {parsed.get('action', 'UNKNOWN')} — {parsed.get('reason', 'N/A')}")

        return {
            "fund_manager_decision": fund_decision,
            "fund_manager_parsed": parsed,
            "final_trade_decision": fund_decision,
        }

    return fund_manager_node


def _parse_fund_decision(text):
    parsed = {}
    block = re.search(r'---FUND_DECISION---(.*?)---END_FUND_DECISION---', text, re.DOTALL)
    if not block:
        upper = text.upper()
        for action in ["OPEN_LONG", "OPEN_SHORT", "CLOSE", "HOLD"]:
            if action in upper:
                parsed["action"] = action
                break
        return parsed

    for key, name in [("ACTION","action"),("CONFIDENCE","confidence"),("STOP_LOSS","stop_loss"),
                      ("TAKE_PROFIT_1","take_profit_1"),("REASON","reason")]:
        m = re.search(rf'{key}:\s*(.+)', block.group(1))
        if m:
            val = m.group(1).strip()
            parsed[name] = int(val) if key == "CONFIDENCE" and val.isdigit() else val
    return parsed

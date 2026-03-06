"""
Fund Manager (Portfolio Manager) — Final approver for trade execution.

In a real trading firm:
- Reviews Trader's proposal + Risk Judge's assessment
- Approves, rejects, or defers from a PORTFOLIO perspective
- Does NOT re-analyze the market or second-guess direction
- The Trader owns the thesis. The Risk Judge gates the risk. The FM says GO/NO-GO.
"""

import re
from tradingagents.agents.utils.trading_context import build_trading_context
import logging

log = logging.getLogger("fund_manager")


def create_fund_manager(llm, memory=None):

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
        open_orders = portfolio_context.get("open_orders", 0)
        performance_feedback = portfolio_context.get("performance_feedback", "No historical trade outcomes available yet.")

        # Past memories
        past_memory_str = ""
        if memory:
            try:
                past_memories = memory.get_memories(trader_plan[:500], n_matches=2)
                for rec in past_memories:
                    past_memory_str += rec["recommendation"] + "\n\n"
            except Exception:
                pass

        trading_ctx = build_trading_context(state)

        prompt = f"""You are the Fund Manager — the final authority on whether a trade executes.

## Your Role (STRICT)
You are the last checkpoint before capital moves. You do NOT:
- Re-analyze the market (analysts did that)
- Second-guess the Trader's thesis (they own it)
- Override the Risk Judge's veto (they gate risk)

You DO:
- Verify the decision chain is coherent (analyst → researcher → trader → risk judge)
- Check portfolio-level constraints
- Approve or reject from a capital allocation perspective
- Ensure the overall process was followed

## Current Portfolio
- Committee Position: {current_position} | P&L: {position_pnl_pct:+.2f}% | Held: {hours_held:.1f}h
- Equity: ${equity:,.0f}
- Last decision: {last_decision}
- Open orders: {open_orders}
{portfolio_context.get("positions_summary", "")}

## Multi-Position Awareness
You manage the FULL portfolio across all position sources (committee trades + green lane scalps).
If positions conflict directionally (e.g. committee short + green lane long), you MUST flag this
and either: (a) reject the new trade, or (b) close the conflicting position first.
Net exposure should always be intentional, never accidental.

## Decision Rules

### Automatic REJECT:
- Risk Judge VETOED → you MUST reject. No override. Ever.
- No stop-loss defined → reject
- Missing confidence score or thesis → reject (process failure)

### Automatic APPROVE:
- Risk Judge APPROVED + Trader HOLD with reaffirmed thesis → approve HOLD
- Risk Judge APPROVED + Trader proposal with clear thesis → approve

### Judgment Calls:
- Risk Judge APPROVED_WITH_ADJUSTMENTS → approve with the adjusted parameters
- If Trader's confidence is borderline (5-6) and setup is mediocre → may downgrade to HOLD
- If we already have a profitable position and Trader wants to add → scrutinize carefully (don't chase)
- Capital allocation: never commit >20% of equity to a single position (margin + unrealized)

### Chronic HOLD Accountability
If last_decision is HOLD and this is also a HOLD, your REASON field must specifically state:
"Waiting for [SPECIFIC CONDITION] — expected in [TIMEFRAME]."
Vague HOLD reasons like "no edge" or "mixed signals" are not acceptable without citing
what specific condition would change that. You are managing real capital sitting idle.
Each HOLD is a capital allocation decision — own it with the same rigor as a trade.

## Crypto-Specific Portfolio Considerations
- **24/7 Exposure**: Position stays open through weekends/holidays. Account for reduced liquidity.
- **Funding Carry**: If holding a position costs >0.1%/day in funding, factor that into approval.
- **Exchange Risk**: Single exchange (Bybit). No hedge on another venue. Accept this but size accordingly.
- **Drawdown Budget**: If account is down >5% from peak, reduce max position size by 50%.

## Historical Performance (Your Track Record)
{performance_feedback}

## Past Reflections
{past_memory_str if past_memory_str else "None."}

## Trader's Proposal
{trader_plan[:2500]}

## Risk Judge's Assessment
{risk_decision[:2000]}

## Required Output

Brief assessment (3-5 sentences max), then:

{trading_ctx["final_format"]}

```
---FUND_DECISION---
ACTION: OPEN_LONG/OPEN_SHORT/CLOSE/HOLD
CONFIDENCE: <1-10>
STOP_LOSS: <price or "as_proposed">
TAKE_PROFIT_1: <price or "as_proposed">
TAKE_PROFIT_2: <price or "as_proposed">
MARGIN_ALLOCATION: <percentage>
REASON: <one-line summary>
---END_FUND_DECISION---
```

Keep it short. You are an approver, not an analyst. The work is already done — you just stamp it."""

        response = llm.invoke(prompt)
        fund_decision = response.content

        parsed = _parse_fund_decision(fund_decision)
        parsed = _validate_fund_decision(fund_decision, parsed)
        log.info(f"Fund Manager: {parsed.get('action', 'UNKNOWN')} — {parsed.get('reason', 'N/A')}")

        return {
            "fund_manager_decision": fund_decision,
            "fund_manager_parsed": parsed,
            "final_trade_decision": fund_decision,
        }

    return fund_manager_node


def _parse_fund_decision(text):
    """Parse the structured decision block from FM output."""
    parsed = {}
    block = re.search(r'---FUND_DECISION---(.*?)---END_FUND_DECISION---', text, re.DOTALL)
    if not block:
        # Fallback: extract action from text
        upper = text.upper()
        for action in ["OPEN_LONG", "OPEN_SHORT", "CLOSE", "HOLD"]:
            if action in upper:
                parsed["action"] = action
                break
        return parsed

    fields = [
        ("ACTION", "action"), ("CONFIDENCE", "confidence"),
        ("STOP_LOSS", "stop_loss"), ("TAKE_PROFIT_1", "take_profit_1"),
        ("TAKE_PROFIT_2", "take_profit_2"), ("MARGIN_ALLOCATION", "margin_allocation"),
        ("REASON", "reason"),
    ]
    for key, name in fields:
        m = re.search(rf'{key}:\s*(.+)', block.group(1))
        if m:
            val = m.group(1).strip()
            if key == "CONFIDENCE" and val.isdigit():
                parsed[name] = int(val)
            else:
                parsed[name] = val
    return parsed


def _validate_fund_decision(full_text, parsed):
    """Validate consistency between FM reasoning and structured decision block.
    
    If the reasoning text clearly says HOLD but the structured block says
    OPEN_LONG/OPEN_SHORT (or vice versa), downgrade to HOLD as a safety measure.
    Contradictory FM output = process failure = no trade.
    """
    action = parsed.get('action', '').upper()
    if action not in ('OPEN_LONG', 'OPEN_SHORT'):
        return parsed  # Only validate trade-opening actions
    
    # Extract reasoning text (everything before the structured block)
    import re
    block_start = re.search(r'---FUND_DECISION---', full_text)
    if not block_start:
        return parsed
    reasoning = full_text[:block_start.start()].upper()
    
    # Check for contradiction: reasoning says HOLD but block says trade
    hold_signals = ['HOLD', 'WAIT', 'DEFER', 'REJECT', 'NO TRADE', 'STAY FLAT', 'REMAIN NEUTRAL']
    trade_signals = ['APPROVE', 'OPEN_LONG', 'OPEN_SHORT', 'GO AHEAD', 'EXECUTE']
    
    has_hold = any(s in reasoning for s in hold_signals)
    has_trade = any(s in reasoning for s in trade_signals)
    
    if has_hold and not has_trade:
        log.warning(
            f"FM CONTRADICTION DETECTED: reasoning says HOLD but block says {action}. "
            f"Downgrading to HOLD for safety. Raw reasoning excerpt: {full_text[:200]}..."
        )
        parsed['action'] = 'HOLD'
        parsed['reason'] = f'OVERRIDDEN: reasoning/block contradiction (block said {action}, reasoning said HOLD)'
        parsed['_contradiction_detected'] = True
    
    return parsed

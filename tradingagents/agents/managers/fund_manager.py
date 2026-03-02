"""Portfolio Manager (Fund Manager) — Final decision-maker.

Translates analyst research + risk assessment into a position action
based on current portfolio state and thesis persistence.

In real trading firms:
- Analysts research → Trader proposes → Risk Manager gates → PM decides
- The PM owns the thesis and only reverses on MATERIAL NEW INFORMATION
- The Risk Manager can veto but never decides direction

Input: All reports + Risk assessment + current position state
Output: OPEN_LONG, OPEN_SHORT, CLOSE, HOLD + parameters
"""

import re
import json
import logging

log = logging.getLogger("fund_manager")


def create_fund_manager(llm, memory=None):
    """Create the Portfolio Manager (Fund Manager) node."""

    def fund_manager_node(state) -> dict:
        risk_decision = state.get("final_trade_decision", "")
        trader_plan = state.get("trader_investment_plan", "")
        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        # Portfolio context injected via state
        portfolio_context = state.get("portfolio_context", {})
        current_position = portfolio_context.get("position", "none")
        position_pnl_pct = portfolio_context.get("pnl_pct", 0)
        position_age_bars = portfolio_context.get("age_bars", 0)
        hours_held = portfolio_context.get("hours_held", 0)
        equity = portfolio_context.get("equity", 0)
        consecutive_same = portfolio_context.get("consecutive_same_direction", 0)
        last_decision = portfolio_context.get("last_decision", "HOLD")
        last_decision_reasoning = portfolio_context.get("last_decision_reasoning", "No prior reasoning available.")
        last_decision_time = portfolio_context.get("last_decision_time", "Unknown")

        # Get past memories if available
        past_memory_str = ""
        if memory:
            curr_situation = f"{market_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
            past_memories = memory.get_memories(curr_situation, n_matches=2)
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""You are the Portfolio Manager — the FINAL decision-maker. You own the thesis and the position.

## Your Role (as in a real trading firm)
- Analysts research, Trader proposes, Risk Manager gates — but YOU decide
- You own the THESIS behind every position
- You only reverse on MATERIAL NEW INFORMATION — not reinterpretation of same data
- A Risk Manager VETO means you cannot open that specific trade. You may HOLD or propose a modified trade instead.

## Current Portfolio State
- Position: {current_position}
- Unrealized P&L: {position_pnl_pct:+.2f}%
- Position age: {position_age_bars} bars (4H) / {hours_held:.1f} hours
- Account equity: ${equity:,.0f}
- Last decision: {last_decision} (at {last_decision_time})
- Consecutive same-direction signals: {consecutive_same}

## Your Prior Thesis (WHY we are in this position)
{last_decision_reasoning}

## Analyst Reports
Market: {market_report[:800]}

Sentiment: {sentiment_report[:500]}

News: {news_report[:500]}

Macro: {fundamentals_report[:500]}

## Trader's Proposed Plan
{trader_plan}

## Risk Manager's Assessment
{risk_decision}

## Past Reflections
{past_memory_str if past_memory_str else "No past reflections."}

## Decision Framework

### CRITICAL: Material Change Requirement
To REVERSE a position, you must identify SPECIFIC NEW INFORMATION that was NOT available when the prior thesis was formed. Examples of material change:
- Price hit stop-loss or take-profit level
- Major unexpected news event (not just "same news, different spin")
- Technical structure broke (key support/resistance violated)
- Fundamentals shifted (earnings miss, regulatory action, etc.)

"The same data viewed from a different angle" is NOT material change. "The LLM reasoned differently this time" is NOT material change.

### When we have NO position:
- If Risk Manager APPROVED and Trader has conviction >= 7: OPEN position in proposed direction
- If Risk Manager VETOED: HOLD (do not open)
- If Trader conviction < 7: HOLD (wait for stronger setup)

### When we ALREADY have a position:
- Same direction signal: HOLD (confirm thesis). Maybe tighten trail if profitable.
- Opposing signal: Ask yourself "WHAT CHANGED since I opened this position?"
  - If you can cite specific material new information: CLOSE (and optionally reverse if conviction >= 8)
  - If you cannot cite material change: HOLD current position. The analysts are noise.
- NEUTRAL signal: HOLD current position
- Position profitable > 3%: Require very strong conviction (>= 9) to close early

### Anti-flip-flop Rules (HARD):
- Position < 12 hours old: NEVER reverse unless stop-loss hit
- If last trade was a reversal: require conviction >= 9 AND material change to reverse again
- Maximum 2 direction changes per 48 hours

## Required Output Format
Provide your reasoning (especially cite what material information changed, if reversing), then end with:

FINAL TRANSACTION PROPOSAL: **BUY/SELL/HOLD**

And include a structured block:
```
---FUND_DECISION---
ACTION: OPEN_LONG/OPEN_SHORT/CLOSE/HOLD
CONFIDENCE: <1-10>
POSITION_SIZE: <percentage or "unchanged">
STOP_LOSS: <price or "unchanged">
TAKE_PROFIT_1: <price or "unchanged">
TAKE_PROFIT_2: <price or "unchanged">
THESIS: <2-3 sentence summary of WHY this position>
MATERIAL_CHANGE: <what new info justified this decision, or "N/A" if holding>
REASON: <one-line summary>
---END_FUND_DECISION---
```
"""

        response = llm.invoke(prompt)
        fund_decision = response.content

        # Parse the structured block
        parsed = _parse_fund_decision(fund_decision)

        log.info(f"Portfolio Manager: {parsed.get('action', 'UNKNOWN')} (confidence: {parsed.get('confidence', '?')})")
        log.info(f"   Thesis: {parsed.get('thesis', 'N/A')}")
        log.info(f"   Material Change: {parsed.get('material_change', 'N/A')}")
        log.info(f"   Reason: {parsed.get('reason', 'N/A')}")

        return {
            "fund_manager_decision": fund_decision,
            "fund_manager_parsed": parsed,
            "final_trade_decision": fund_decision,  # Override risk manager's assessment
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
        "TAKE_PROFIT_1": ("take_profit_1", str),
        "TAKE_PROFIT_2": ("take_profit_2", str),
        "THESIS": ("thesis", str),
        "MATERIAL_CHANGE": ("material_change", str),
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

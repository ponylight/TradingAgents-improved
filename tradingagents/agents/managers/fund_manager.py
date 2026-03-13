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


def _extract_risk_decision(risk_decision: str, fallback_chars: int = 4000) -> str:
    """
    Extract the structured ---RISK_DECISION--- block from the risk judge output,
    always including it in full. Falls back to first `fallback_chars` chars if
    no structured block is found. This prevents critical blocking signals from
    being truncated at an arbitrary char limit.
    """
    if not risk_decision:
        return ""
    # Try to find the structured block
    match = re.search(r"(---RISK_DECISION---.*?---END---)", risk_decision, re.DOTALL)
    if match:
        structured = match.group(1)
        # Include preamble up to fallback_chars, always append full structured block
        pre = risk_decision[:risk_decision.index("---RISK_DECISION---")]
        preamble = pre[:fallback_chars - len(structured)] if len(structured) < fallback_chars else ""
        return (preamble + structured).strip()
    # No structured block — truncate at fallback_chars
    return risk_decision[:fallback_chars]


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
- Missing ---RISK_SIZING--- block from Risk Judge → reject (process failure)
- account_risk_pct > 1.0% → reject (hard limit breach)
- implied_leverage > 15x → reject (hard limit breach)

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
{_extract_risk_decision(risk_decision)}

## Risk Sizing Validation (YOU MUST CHECK)
Before approving any trade, verify the Risk Judge provided a ---RISK_SIZING--- block with:
- account_risk_pct ≤ 1.0%
- implied_leverage ≤ 15x (swing) / 10x (position)
- liquidation_buffer_pct > 1.5× stop_distance_pct
- position_size_btc is reasonable for current equity
If ANY field is missing or breaches limits, REJECT with reason "risk sizing incomplete/invalid".

## Required Output — Mandatory Sections

Your response MUST include ALL three of the following before the structured block:

**1. INDEPENDENT OBSERVATION** (required): State at least ONE insight or concern NOT mentioned by any upstream agent (researchers, trader, risk judge). This must be your own analysis — not a restatement of what the trader said. Examples: macro calendar risk, exchange-specific liquidity, funding carry cost at current rate, order book depth at the proposed entry, correlation with equities, weekend/holiday liquidity risk, etc.

**2. ORDERBOOK / LIQUIDITY CHECK** (required): Cross-check the trader's proposed entry price against current market conditions. Is there sufficient liquidity to fill the proposed size? Are there large walls nearby that could cause slippage? State your finding explicitly (e.g. "Entry at $X is well-supported with $Y in bids within 0.5%" or "Thin book at proposed entry — slippage risk").

**3. EXPLICIT AGREEMENT / DISAGREEMENT** (required): Clearly state: "I AGREE with the trader's thesis because [reason]" OR "I DISAGREE with the trader's thesis because [reason]". Do NOT just summarize the trader's view — you must take a position. Vague or noncommittal statements are not acceptable.

After these three sections, provide your brief overall assessment (2-3 sentences), then:

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
        parsed = _validate_required_sections(fund_decision, parsed)

        # Validate risk sizing from upstream Risk Judge
        risk_sizing = _parse_risk_sizing(risk_decision)
        if risk_sizing:
            parsed["risk_sizing"] = risk_sizing
            sizing_issue = _validate_risk_sizing(risk_sizing)
            if sizing_issue and parsed.get("action") in ("OPEN_LONG", "OPEN_SHORT"):
                log.warning(f"FM RISK SIZING REJECT: {sizing_issue}")
                parsed["action"] = "HOLD"
                parsed["reason"] = f"OVERRIDDEN: risk sizing invalid — {sizing_issue}"
                parsed["_sizing_rejected"] = True
        elif parsed.get("action") in ("OPEN_LONG", "OPEN_SHORT"):
            log.warning("FM: No ---RISK_SIZING--- block found in Risk Judge output — trade blocked")
            parsed["_sizing_missing"] = True
            # Don't auto-reject here — the LLM prompt already instructs rejection

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


def _validate_required_sections(full_text: str, parsed: dict) -> dict:
    """Validate that FM output includes all required reasoning sections.

    Only enforced for trade-opening actions — HOLD doesn't need full reasoning.
    """
    action = parsed.get('action', '').upper()
    if action not in ('OPEN_LONG', 'OPEN_SHORT'):
        return parsed

    upper = full_text.upper()
    missing = []
    if 'INDEPENDENT OBSERVATION' not in upper:
        missing.append('INDEPENDENT OBSERVATION')
    if 'ORDERBOOK' not in upper and 'LIQUIDITY CHECK' not in upper:
        missing.append('ORDERBOOK/LIQUIDITY CHECK')
    if 'I AGREE' not in upper and 'I DISAGREE' not in upper:
        missing.append('EXPLICIT AGREEMENT/DISAGREEMENT')

    if missing:
        log.warning(f"FM missing required sections: {', '.join(missing)} — downgrading to HOLD")
        parsed['action'] = 'HOLD'
        parsed['reason'] = f'OVERRIDDEN: missing required sections: {", ".join(missing)}'
        parsed['_missing_sections'] = missing

    return parsed


def _parse_risk_sizing(risk_text: str) -> dict:
    """Extract ---RISK_SIZING--- block from Risk Judge output."""
    if not risk_text:
        return {}
    block = re.search(r'---RISK_SIZING---(.*?)---END_RISK_SIZING---', risk_text, re.DOTALL)
    if not block:
        return {}
    sizing = {}
    fields = [
        ("account_risk_pct", float),
        ("stop_distance_pct", float),
        ("implied_leverage", float),
        ("liquidation_buffer_pct", float),
        ("position_size_btc", float),
    ]
    for key, cast in fields:
        m = re.search(rf'{key}:\s*([\d.]+)', block.group(1))
        if m:
            try:
                sizing[key] = cast(m.group(1))
            except (ValueError, TypeError):
                pass
    return sizing


def _validate_risk_sizing(sizing: dict) -> str:
    """Validate risk sizing values. Returns error string or empty string if OK."""
    if not sizing:
        return "no sizing fields parsed"
    required = ["account_risk_pct", "stop_distance_pct", "implied_leverage", "liquidation_buffer_pct", "position_size_btc"]
    missing = [f for f in required if f not in sizing]
    if missing:
        return f"missing fields: {', '.join(missing)}"
    if sizing["account_risk_pct"] > 1.05:  # 5% tolerance for rounding
        return f"account_risk_pct={sizing['account_risk_pct']:.1f}% exceeds 1% hard limit"
    if sizing["implied_leverage"] > 15.5:  # 3% tolerance for rounding
        return f"implied_leverage={sizing['implied_leverage']:.1f}x exceeds 15x hard limit"
    liq_buf = sizing.get("liquidation_buffer_pct", 0)
    stop_dist = sizing.get("stop_distance_pct", 0)
    if liq_buf > 0 and stop_dist > 0 and liq_buf < 1.5 * stop_dist:
        return f"liquidation_buffer={liq_buf:.1f}% < 1.5× stop_distance={stop_dist:.1f}%"
    return ""

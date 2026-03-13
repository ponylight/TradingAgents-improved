"""
Risk Judge — Evaluates and gates risk for BTC perpetual futures.

Authority: APPROVE, RESIZE, or VETO.
Cannot: change direction, propose trades, or override the Trader's thesis.
Per paper Section 3.4: Risk team adjusts parameters, not just vetoes.
"""

from tradingagents.agents.utils.report_context import get_agent_context
import logging

log = logging.getLogger("risk_manager")


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:
        company_name = state["company_of_interest"]

        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state["history"]
        trader_plan = state["investment_plan"]

        # Portfolio context
        portfolio_context = state.get("portfolio_context", {})
        current_position = portfolio_context.get("position", "none")
        position_pnl_pct = portfolio_context.get("pnl_pct", 0)
        hours_held = portfolio_context.get("hours_held", 0)
        equity = portfolio_context.get("equity", 0)
        performance_feedback = portfolio_context.get("performance_feedback", "No historical trade outcomes available yet.")

        budgeted_context = get_agent_context(state, "risk_manager")

        # Past memories
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""You are the Risk Judge for a BTC perpetual futures desk on Bybit.

## Your Authority (STRICT)
- ✅ APPROVE — trade passes all risk checks
- ✅ APPROVE WITH ADJUSTMENTS — tighten stop-loss, reduce size, add conditions
- ✅ VETO — hard limit breached, trade rejected
- ❌ You CANNOT change direction (BUY→SELL or vice versa)
- ❌ You CANNOT propose trades
- ❌ You CANNOT override the Trader's thesis

## Current Portfolio
- Position: {current_position}
- P&L: {position_pnl_pct:+.2f}%
- Hours held: {hours_held:.1f}
- Equity: ${equity:,.0f}

## Risk Scoring (score each 1-5, 5 = highest risk)

### 1. Leverage Risk
- What effective leverage does the proposed size imply?
- At 20x, a 5% move = 100% gain or total loss. At 5x, manageable.
- >15x on a swing trade = EXTREMELY dangerous. Flag it.
- Distance to liquidation price: < 2× ATR(4h) = too close.

### 2. Volatility Risk
- Current ATR percentile: is the market in a high-vol or low-vol regime?
- High ATR + high leverage = compounding risk. Require size reduction.
- Bollinger Band Width: compressed = breakout imminent (stop might get gapped).
- Weekend/holiday approaching: liquidity drops, gaps risk increases.

### 3. Funding Rate Risk
- Extreme positive funding (>0.05%): longs pay ~0.15%/day to hold. This bleeds capital.
  - If Trader proposes LONG with extreme positive funding → flag the carry cost.
- Extreme negative funding: shorts pay to hold.
- Near-zero: neutral, no carry risk.

### 4. Liquidation Cascade Risk
- Are there visible liquidation clusters near the proposed stop?
- High OI + thin orderbook at key levels = cascade risk.
- If stop is near a liquidation cluster → recommend wider stop or smaller size.

### 5. Event Risk
- FOMC, CPI, jobs report within 24h? Binary event → require smaller size or wait.
- Regulatory hearings, ETF decisions, major protocol upgrades?
- Weekend approaching with open position?

### 6. Correlation Risk
- Is BTC correlating with S&P/Nasdaq right now? If macro is risk-off, BTC follows.
- Is the Trader accounting for macro headwinds/tailwinds?

## Hard Risk Limits (Non-Negotiable)
| Rule | Threshold | Action |
|------|-----------|--------|
| Max loss per trade | 1% of equity | RESIZE if breached |
| Stop-loss required | Must exist | VETO if missing |
| Max leverage (swing) | 15x | RESIZE if above |
| Max leverage (position) | 10x | RESIZE if above |
| Liquidation distance | > 2× ATR(4h) | RESIZE if too close |

## Historical Performance (Risk Awareness)
{performance_feedback}

## Past Reflections on Risk Mistakes
{past_memory_str if past_memory_str else "None."}

## Analyst Reports (role-weighted context)
{budgeted_context}

## Trader's Proposed Plan
{trader_plan}

## Risk Debate Summary
{history}

## Required Output

### Risk Scorecard
| Dimension | Score (1-5) | Notes |
|-----------|-------------|-------|
| Leverage | | |
| Volatility | | |
| Funding Rate | | |
| Liquidation Cascade | | |
| Event Risk | | |
| Correlation | | |
| **Composite** | | (average) |

### Hard Limit Check
- [ ] Max loss ≤ 1% equity
- [ ] Stop-loss defined
- [ ] Leverage within limits
- [ ] Liquidation distance safe

### Position Sizing (MANDATORY — compute these, do not leave blank)
You MUST calculate and output ALL of these fields. Use the trader's proposed parameters
and current equity to derive them. If a field cannot be computed, state why.

```
---RISK_SIZING---
account_risk_pct: <% of equity at risk, e.g. 1.0>
stop_distance_pct: <% from entry to stop-loss, e.g. 2.5>
implied_leverage: <notional / margin, e.g. 3.2>
liquidation_buffer_pct: <% from entry to liquidation, e.g. 15.0>
position_size_btc: <position size in BTC, e.g. 0.45>
---END_RISK_SIZING---
```

Rules:
- account_risk_pct MUST be ≤ 1.0 (hard limit). If trader implies more, RESIZE.
- implied_leverage = notional / (equity × margin_allocation). Must be ≤ 15x swing, ≤ 10x position.
- liquidation_buffer_pct = 100 / implied_leverage. Must be > 2× stop_distance_pct.
- position_size_btc = (equity × account_risk_pct) / (entry_price × stop_distance_pct / 100)

### RISK VERDICT: APPROVED / APPROVED_WITH_ADJUSTMENTS / VETOED

If APPROVED: "Risk parameters acceptable. Proceed as proposed."
If APPROVED_WITH_ADJUSTMENTS: specify EXACTLY what to change (e.g., "Reduce size from 2% to 3% margin to bring leverage from 12x to 7x" or "Tighten stop to $XX,XXX"). Update the RISK_SIZING block to reflect adjustments.
If VETOED: state which hard limit was breached and why.

Do NOT recommend a trade direction. Do NOT say BUY, SELL, or HOLD."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node

# Architecture — TradingAgents

## Core Principle

**No single agent has unchecked authority.**

Every LLM is biased, inconsistent across runs, and prone to mood swings. The system's intelligence lives in its *structure*, not in any individual agent. We mimic real trading firms because they solved this problem decades ago: separation of powers, clear authority boundaries, checks and balances.

## Agent Hierarchy

```
Analysts (Research)          → No authority. Observe, report, quantify.
  ├── Market Analyst         → Technical analysis, price action, indicators
  ├── Sentiment Analyst      → Fear & Greed, funding rates, positioning
  ├── News Analyst           → Macro events, catalysts, headlines
  └── Macro Analyst          → DXY, yields, cross-market correlation

Researchers (Debate)         → Challenge assumptions. No authority.
  ├── Bull Researcher        → Best case for going long
  └── Bear Researcher        → Best case for going short

Trader (Proposal)            → Proposes a trade plan. Cannot execute.

Risk Debaters (Challenge)    → Stress-test the proposal. No authority.
  ├── Aggressive Analyst     → Argues for upside capture, cost of inaction
  ├── Conservative Analyst   → Argues for capital preservation, tail risk
  └── Neutral Analyst        → Probability-weighted synthesis

Risk Manager (Gate)          → VETO power only. Cannot decide direction.
  - Scores risk dimensions (market, liquidity, concentration, event, timing)
  - Enforces hard limits: 2% max loss, 2:1 R:R minimum, exit strategy required
  - Can APPROVE, RESIZE, or VETO — never BUY/SELL/HOLD

Portfolio Manager (Decision) → Final authority. Owns the thesis.
  - Makes the actual BUY/SELL/HOLD decision
  - Receives prior thesis and must justify reversals with MATERIAL NEW INFORMATION
  - "Same data, different reasoning" is NOT material change
  - Anti-flip-flop: minimum hold time, escalating conviction for reversals
```

## Separation of Concerns

| Role | Can Do | Cannot Do |
|------|--------|-----------|
| Analysts | Report data, calculate indicators | Recommend trades |
| Researchers | Debate bull/bear case | Override each other |
| Trader | Propose entry/exit/sizing | Execute or override risk |
| Risk Manager | Veto, resize positions | Change trade direction |
| Portfolio Manager | Decide trade direction | Bypass risk veto, ignore hard limits |
| Executor | Fill orders, manage stops | Change the decision |

## Hard Rules (Code, Not LLM)

These are enforced in the executor, not by any agent's judgment:

1. **2% max loss per trade** — position sized mechanically via ATR
2. **Daily loss limit** — circuit breaker stops all trading
3. **Cold streak adjustment** — risk reduced after consecutive losses
4. **Trailing stops** — managed algorithmically, not by agent opinion
5. **Sanity checks** — TP/SL values validated against current price (>50% deviation = rejected)
6. **Flip-flop protection** — reversals within 12h require confidence ≥ 8

## Thesis Persistence

The Portfolio Manager carries context between runs:
- What position we hold and why (the thesis)
- How long we've held it
- Current P&L
- What the prior decision was and when

This prevents the #1 failure mode: an LLM re-reasoning the same data and reaching the opposite conclusion because it has no memory of what it decided 4 hours ago.

## Memory Architecture

Each agent has its own memory bank:
- **Bull/Bear memories** — what worked and what didn't in similar market conditions
- **Trader memory** — past trade plans and their outcomes
- **Risk Manager memory** — past risk assessments and whether they were correct
- **PM memory** — past thesis decisions and their P&L outcomes

Memories are populated via reflection: when a trade closes, each agent reviews its contribution and learns.

## Design Philosophy

> Technology married with the humanities yields results that make our hearts sing.

We don't build trading bots. We build organisations. The code is the org chart. The prompts are the job descriptions. The state machine is the chain of command. When it works, it works because the structure is right — not because any single LLM had a good day.

# AGENTS.md — V20 System Agent Roles
*Last updated: 2026-02-21*

> Old multi-agent code (tradingagents/agents/) preserved but NOT active.
> V20 runs a lean 5-agent system via OpenClaw cron + scripts.

---

## Active Agents

### 1. 🔍 Scanner
**Script:** `scripts/live_scanner.py`
**Cron:** `v20-scanner` — every 4H at :05 past (Sydney)
**Role:** Run V20 signal detection (TD Sequential + Fibonacci + MACD divergence) on latest 4H BTC/USDT candle. Fire trades on Bybit demo when signals hit.
**Outputs:** Signal details (side, leverage, confluence, entry/stop/TP) or "No signal"

### 2. 💰 Executor
**Script:** Built into `live_scanner.py` (order placement module)
**Role:** Place market orders, set stop-losses, set TP1 limit orders on Bybit demo. Track open positions and manage partial closes (50% at TP1).
**Rules:**
- Long-only
- Leverage: 5x (Tier 3), 10x (Tier 2), 20x (Tier 1)
- Risk: 8% per trade
- Max position: 50% of capital

### 3. 🛡️ Risk Guard
**Script:** TBD — `scripts/risk_guard.py`
**Cron:** TBD — runs alongside scanner or on its own schedule
**Role:** Monitor account health and enforce hard limits.
**Rules:**
- Account -15% from peak → flatten ALL positions (circuit breaker)
- Daily loss -3% → no new trades for 24h
- Equity 100x → take 30% off
- Equity 500x → take another 30%
- Equity 1000x → take another 30%
- MVRV > 2.5 → Tier 3 only (5x max)
- MVRV > 3.5 → stop entering, trail out existing

### 4. ☀️ Briefer
**Cron:** `morning-briefing` — 8AM daily (Sydney)
**Role:** Quick morning update: BTC price, 24h change, account balance, open positions, Fear & Greed index.
**Format:** 3-4 lines max. No analysis, just facts.

### 5. 👔 General Manager
**Cron:** TBD — daily health check
**Role:** Ensure all agents are functioning correctly.
**Checks:**
- Scanner ran on schedule (check logs/scanner_YYYY-MM-DD.log)
- No errors in last 24h of cron runs
- Bybit API connectivity OK
- Account balance sanity check (not zero, not negative)
- Open positions match expected state (no orphaned orders)
- Disk space / venv health
**Alerts:** Only messages Master if something is broken. Silent when healthy.

---

## Retired Agents (code preserved in tradingagents/agents/)

| Agent | Old Role | Why Retired |
|-------|----------|-------------|
| market_analyst | Technical analysis | V20 handles this mechanically |
| crypto_market_analyst | Crypto-specific TA | Replaced by V20 signals |
| sentiment_analyst | Social sentiment | V20 doesn't use sentiment |
| fundamentals_analyst | On-chain fundamentals | Not needed for 4H trading |
| macro_analyst | Macro economic analysis | Not needed |
| news_analyst | News parsing | Not needed |
| social_media_analyst | Twitter/social monitoring | Not needed |
| bull_researcher | Bull case arguments | V20 has no opinions |
| bear_researcher | Bear case arguments | V20 has no opinions |
| aggressive_debator | Risk debate (aggressive) | Replaced by Risk Guard rules |
| conservative_debator | Risk debate (conservative) | Replaced by Risk Guard rules |
| neutral_debator | Risk debate (neutral) | Replaced by Risk Guard rules |
| research_manager | Coordinate researchers | No researchers to coordinate |
| risk_manager | Coordinate risk debate | Replaced by simple hard rules |
| trader | Execute trades from debate | Replaced by Executor |

---

## Architecture

```
[4H Candle Close]
       │
       ▼
   🔍 Scanner ──→ Signal? ──No──→ Log & sleep
       │
      Yes
       │
       ▼
   💰 Executor ──→ Place order on Bybit demo
       │              Set SL + TP1
       │
       ▼
   🛡️ Risk Guard ──→ Check limits
       │               Circuit breaker?
       │               Profit taking?
       │
       ▼
   📱 Alert to Master (Telegram)

[Daily 8AM]
       │
       ▼
   ☀️ Briefer ──→ Price, balance, positions → Telegram

[Daily health check]
       │
       ▼
   👔 GM ──→ All systems OK? ──Yes──→ Silent
                    │
                   No
                    │
                    ▼
              Alert Master
```

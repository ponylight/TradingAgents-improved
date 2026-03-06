# ROADMAP.md — Trading System Evolution
*Last updated: 2026-03-07*

## Current State
- BTC/USDT perpetual on Bybit Demo, 4H timeframe
- Multi-agent decision system (market, sentiment, news, fundamentals analysts → trader → fund manager)
- 2 TP + native trailing stop, conditional orders
- Stability score: ~55/100
- Trades completed: ~5

---

## Phase 1: Foundation (NOW)
**Goal:** Reliable execution, accurate bookkeeping, no silent failures.

- [x] Fix tpslMode=Full → multi-TP conditional orders (Mar 7)
- [x] Fix reconcile: record closed trades with PnL (Mar 7)
- [x] Native trailing stop via Bybit API (Mar 7)
- [x] TP1 hit detection + state update (Mar 7)
- [x] Codex audit — all critical issues fixed (Mar 6-7)
- [x] Objective scoring guardrail (Mar 6)
- [x] Geopolitical event detection (Mar 6)
- [x] Risk manager consolidation (Mar 6)
- [x] Tiered analyst scheduling — ~60% token savings (Mar 6)
- [ ] Verify multi-TP works on next live trade
- [ ] Verify native trailing stop activates correctly
- [ ] 20 trades with clean closed_trades data

---

## Phase 2: Evaluation (~20 trades)
**Goal:** Data-driven strategy selection and agent improvement.

- [ ] **TP strategy comparison**: 1TP vs 1TP+TS vs 2TP vs 2TP+TS — backtest with closed_trades
- [ ] **Agent accuracy audit**: outcome_tracker win rates per agent, override vs agent performance
- [ ] **Curated few-shot examples**: select best/worst trades as in-context examples for agents
- [ ] **Position scaling**: add-on-pullback when confidence ≥ 8/10 (see docs/ROADMAP.md)
  - Prerequisite: stability > 70, consistent win rate
- [ ] **Monthly Codex audit**: automated via cron

---

## Phase 3: Intelligence (~50 trades)
**Goal:** Agents learn from their own history.

- [ ] **LoRA fine-tune**: small model for analyst roles using trade history
- [ ] **Smart money data**: integrate Binance web3 API (free, no-auth)
  - Smart Money Inflow Rank
  - Trading Signals (direction + confidence)
  - Token audit for risk scoring
- [ ] **Chinese social sentiment**: union-search-skill for Xiaohongshu/Douyin/Bilibili crypto sentiment
- [ ] **Multi-timeframe**: 1H scalps alongside 4H positions

---

## Phase 4: Scale
**Goal:** Production-ready, multi-exchange, continuous learning.

- [ ] **Go live**: Bybit mainnet with real capital (small allocation)
- [ ] **Multi-exchange**: Binance futures as backup/redundancy
- [ ] **DeFi integration**: OKX OnchainOS for on-chain execution
- [ ] **ANE local training**: Apple Neural Engine for continuous model improvement (Mac Studio)
- [ ] **Portfolio expansion**: ETH, SOL perpetuals

---

## Deferred / Parking Lot
- Pyramiding/scaling — revisit Phase 2
- ClawQuant as execution engine — stays as agent tool only
- Green lane auto-trading from price sentinel — needs more testing
- TP1 fill price from order history (currently estimated)
- Scrapling (github.com/D4Vinci/Scrapling) — adaptive web scraper with anti-bot bypass, MCP support. Install when we need to scrape resistant sites for news/sentiment.

---

## Principles
1. **Fix fundamentals before adding complexity**
2. **Data over intuition** — every decision should be backed by closed_trades analysis
3. **Test on demo until proven** — no real money until Phase 4
4. **Monthly audits** — Codex review of executor changes
5. **One position at a time** — until scaling is proven safe

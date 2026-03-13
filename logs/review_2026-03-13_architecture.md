# Full Architecture & Logic Review — 2026-03-13

## Summary

Comprehensive review of the TradingAgents-improved system covering graph orchestration, live executor, light decision layer, fund/risk managers, and scoring/validation. **43 issues identified** across all subsystems.

| Severity | Count |
|----------|-------|
| CRITICAL | 8 |
| HIGH     | 13 |
| MEDIUM   | 16 |
| LOW      | 6 |

---

## 1. Graph Orchestration & Pipeline

Files reviewed: `crypto_trading_graph.py`, `conditional_logic.py`, `signal_processing.py`, `propagation.py`, `green_lane.py`, `reflection.py`

### CRITICAL

**1.1 — `crypto_trading_graph.py:351` — Invalid `add_edge()` with list argument**

`analyst_done_nodes` is a Python list passed to `add_edge()`, which only accepts single strings. The crypto trading graph cannot compile.

```python
workflow.add_edge(analyst_done_nodes, "Analyst Cleanup")  # ❌ list not valid
```

**Fix:** Loop over list:
```python
for node in analyst_done_nodes:
    workflow.add_edge(node, "Analyst Cleanup")
```

---

### HIGH

**1.2 — `propagation.py:22-54` — Missing state field initializations**

Fields `investment_plan`, `trader_investment_plan`, `portfolio_context`, `fund_manager_decision`, `fund_manager_parsed` are never initialized in `create_initial_state()`. Causes `KeyError` crashes in `_log_state()` (lines 486, 494) and `risk_manager.py:21`.

**Fix:** Add all fields to initial state dict with empty defaults.

**1.3 — `risk_manager.py:21` (agents/managers) — Hard state access without fallback**

```python
trader_plan = state["investment_plan"]  # KeyError if missing
```

Contrast with fund_manager.py which uses `.get()`. Fix: `state.get("investment_plan", "")`.

---

### MEDIUM

**1.4 — `conditional_logic.py:14-44` — Dead code methods**

`should_continue_market/social/news/fundamentals` are never called by `crypto_trading_graph.py`, which reimplements the logic locally with a safer `hasattr` check. The stock trading flow in `setup.py` uses the unsafe version without `hasattr`.

**1.5 — `signal_processing.py:26-29` — Greedy pattern matching without word boundaries**

```python
for action in ["LONG", "SHORT", "NEUTRAL", "BUY", "SELL", "HOLD"]:
    if action in tail:
```

"NEUTRAL" matches inside "EMOTIONALLY_NEUTRAL". Fix: use `re.search(rf"\b{action}\b", tail)`.

**1.6 — `green_lane.py:220` — Invalid date format in audit call**

```python
results = ta_graph.propagate(symbol, symbol)  # symbol passed as date
```

Creates filename with `/` in path. Fix: pass `datetime.now().strftime("%Y-%m-%d")` as second arg.

**1.7 — `reflection.py:102` — Cascading fallback conflates different agents**

Fund manager reflection falls back to `final_trade_decision` (Risk Judge output) when `fund_manager_decision` is missing. These are different agents; should not be conflated.

**1.8 — `conditional_logic.py:50` — Misleading comment on debate round formula**

Comment says "3 rounds" but formula is `2 * max_debate_rounds`. Formula is correct; comment is wrong.

---

## 2. Live Executor (`scripts/live_executor.py`)

### CRITICAL

**2.1 — Lines 2875→2889 — Undefined `new_direction` variable on fresh opens**

`new_direction` is defined inside the REVERSE block (line 2875). For `OPEN_LONG`/`OPEN_SHORT` (non-reverse), execution skips to line 2889 where `new_direction` is referenced but never defined. Causes `NameError` crash.

**Fix:** Define `new_direction` unconditionally before the conditional block:
```python
new_direction = "BUY" if "LONG" in action else "SELL"
```

**2.2 — Lines 2779-2816, 2839-2872 — Missing `save_state()` after position close**

After `close_position()` and `active_trade` pop, no state save occurs. If process crashes between exchange close and next save, state misaligns with exchange.

**Fix:** Add `save_state(executor_state)` immediately after lines 2816 and 2872.

**2.3 — Lines 1317-1356 — Protection retry creates duplicate TP orders**

`needs_protection_retry` only clears if ALL protections succeed. Partial success (TP placed, SL fails) leaves flag set. Next retry re-places TP orders without dedup, creating duplicates on exchange.

**Fix:** Track which protections succeeded individually; only retry failed ones.

**2.4 — Lines 3093-3165 — Failed market order not saved to state**

If `open_position()` returns `None`, the trade record is not appended to state or saved. Silent failure can lead to double-open on next cycle.

**Fix:** Save failed trade record with `record["action"] = "failed"` and `save_state()`.

---

### HIGH

**2.5 — Lines 2944-2971 — Leverage cap recalculation uses pre-cap value**

After capping `alloc_pct`, the check at line 2955 uses `_eff_lev` computed before the cap was applied. Conservative (fails closed) but error message is incorrect.

**2.6 — Lines 2092-2106 — State corruption risk on SIGTERM**

Signal handler sets `_shutdown_requested = True` but doesn't call `save_state()`. If signal arrives during temp file write (before `os.replace()`), state file remains stale.

**2.7 — Line 1456 — TP1 hit detection uses percentage threshold, not fill data**

Detects TP1 hit if position size shrinks by >10%. Partial fills, slippage, or manual adjustments trigger false TP1_HIT. Should validate against actual TP order fills.

---

### MEDIUM

**2.8 — Lines 1516-1530 — Time exit closes marginally profitable trades**

Closes after 24 bars (4 days) if `abs(pnl_pct) < 1%`. A trade at +0.5% after 4 days closes as "stale" without considering slippage/fees.

**2.9 — Line 1112 — Orphan cancel has type confusion in string matching**

`str(reduce_only).lower()` compared against set including boolean `True`. Minor but sloppy.

**2.10 — Line 2365 — Pending entry TP exception not caught**

If `set_take_profit()` throws after pending limit fills, trade opens unprotected and `save_state()` never runs. Pattern inconsistent with other TP placements that use try/except.

---

## 3. Light Decision Layer (`scripts/light_decision.py`)

### HIGH

**3.1 — Line 314 — Green lane action is "LONG"/"SHORT" not "BUY"/"SELL"**

`signal.direction.upper()` produces "LONG"/"SHORT" but executor expects "BUY"/"SELL". Executor normalizes at lines 2483-2486, but schema is inconsistent.

**Fix:** `action = "BUY" if signal.direction.lower() == "long" else "SELL"`

---

### MEDIUM

**3.2 — Lines 300-301 — Silent exception in cooldown check**

Parsing `last_light_override_time` failure is silently caught with `pass`. Corrupted timestamp bypasses cooldown entirely. Should log warning.

**3.3 — Lines 436-445 — Lock file handle not guaranteed closed on exception**

If exception occurs during tmp file operations, lock fd may not be properly closed. Should use try/finally.

**3.4 — Line 59 — MIN_REPORT_FRESHNESS_HOURS = 6 too generous**

With 4H executor cycle, reports could be 10 hours old. Consider reducing to 4-5 hours.

---

## 4. Fund Manager & Risk Manager

Files: `agents/managers/fund_manager.py`, `agents/managers/risk_manager.py`, `scoring/risk_manager.py`

### CRITICAL

**4.1 — `risk_manager.py:156` (agents/managers) — Position size formula 100× oversizing**

The prompt instructs:
```
position_size_btc = (equity × account_risk_pct) / (entry_price × stop_distance_pct / 100)
```

With equity=$50K, risk=1%, entry=$95K, stop=2.5%:
- Formula gives: 50,000 / (95,000 × 0.025) = **21.05 BTC** ($2M notional)
- Correct: (50,000 × 0.01) / (95,000 × 0.025) = **0.21 BTC** ($20K notional)

The `/100` is applied to `stop_distance_pct` but NOT to `account_risk_pct`, making positions 100× too large.

**Fix:** Formula should be:
```
position_size_btc = (equity × account_risk_pct / 100) / (entry_price × stop_distance_pct / 100)
```

**4.2 — `fund_manager.py:307-310` — Hard limit tolerance 50% above stated limit**

Validation allows `account_risk_pct > 1.5` (50% above "1% hard limit") and `implied_leverage > 16` (6.7% above "15x hard limit"). Comments say "for rounding" but 50% is not rounding error.

**Fix:** Tighten to `> 1.05` and `> 15.5`, or update prompt to match actual limits.

---

### HIGH

**4.3 — `fund_manager.py:303` — Missing `liquidation_buffer_pct` in required fields**

Validation checks 4 fields but omits `liquidation_buffer_pct`, which risk_manager.py:155 requires. Incomplete risk data passes validation.

**4.4 — `scoring/risk_manager.py` — Entire class is dead code**

`RiskManager` class in scoring module is never instantiated or called in the agent pipeline. Two parallel risk systems exist with potentially different limits, creating confusion.

**4.5 — Multiple files — Three different leverage definitions**

- Agent risk_manager: `notional / (equity × margin_allocation)`
- Live executor: `notional / margin`
- Scoring risk_manager: accepts `proposed_leverage` as input

No canonical definition. Could produce different leverage values for the same position.

**4.6 — `fund_manager.py:149-178` — Required output sections not enforced**

FM prompt requires INDEPENDENT OBSERVATION, ORDERBOOK CHECK, and EXPLICIT AGREEMENT sections, but parser never validates their presence. LLM can skip reasoning and output just the decision block.

---

### MEDIUM

**4.7 — `risk_manager.py:155` — Liquidation buffer 2× rule is overly strict**

Rule `liquidation_buffer_pct > 2× stop_distance_pct` makes many valid setups impossible. 10x leverage → 10% buffer → stop must be <5%. Tight stops get hit by noise.

**4.8 — `risk_manager.py:158` — No structured RISK_VERDICT block**

Risk Judge outputs narrative verdicts (APPROVED/VETO/ADJUSTMENTS) but no structured block for deterministic parsing. Fund Manager must infer intent from free text.

**4.9 — `risk_manager.py:155-156` — LLM not instructed how to calculate liquidation buffer**

Prompt says to compute `liquidation_buffer_pct = 100 / implied_leverage` but gives no example. LLM may compute it differently.

---

## 5. Scoring & Validation System

Files: `decision_validator.py`, `objective_score.py`, `geopolitical.py`, `stability.py`

### HIGH

**5.1 — `objective_score.py:64-65` — `conflicts_with()` blind to current position**

Method checks if signal conflicts with agent decision but has no access to `current_position`. HOLD SHORT during STRONG BUY should NOT override to BUY (protecting short), but the method treats all HOLDs as conflicts with STRONG signals.

**5.2 — `objective_score.py:285-292` — Daily timeframe double-weighted**

Daily candles get 3× weight in technical sub-score, then technical gets 25% overall weight. A single bullish daily candle can override weak 4H/1H momentum. Creates over-reliance on daily signals.

**5.3 — `geopolitical.py:87-90` — Actor amplification breaks severity ordering**

TIER3 + 2 actors = -10 × 1.5 = -15, which is LESS severe than TIER2 + 0 actors = -20. Amplification should not cause tier inversion.

**Fix:** Apply amplification after tier comparison, or cap amplified score at next-higher tier minimum.

---

### MEDIUM

**5.4 — `objective_score.py:134-141` — RSI 60 penalizes normal bullish momentum**

RSI staying 60-70 in a strong uptrend is healthy, but scores -10 per timeframe. Introduces bearish bias.

**5.5 — `objective_score.py:218-225` — Funding rate swings ±20 points (20% of scale)**

Single metric can swing 40 points on the -100 to +100 scale. Disproportionate weight.

**5.6 — `decision_validator.py:23` — Confidence adjustments only negative**

All adjustments are -0.15 to -0.4. No positive adjustments when objective score validates the decision.

**5.7 — `stability.py:45` — Negative Sharpe clamped to 0 instead of penalized**

Sharpe < 0 (losing money) scores 0, same as "no data." Should indicate poor quality with negative score.

**5.8 — `stability.py:84-91` — Win rate formula: 50% = 65/100 (above neutral)**

`win_rate * 130` means a coin-flip system scores 65/100. Not intuitive; `win_rate * 100` would be more linear.

**5.9 — `geopolitical.py:40-46` — Actor regex patterns prone to false positives**

"Gulf Stream", "American stocks", "EU" in "continue" could trigger false actor detection.

**5.10 — `stability.py:108-125` — Data quality penalties absolute, not proportional**

Invalid RSI on all 3 timeframes = -60. Should be proportional to % of broken indicators.

**5.11 — `stability.py:161-164` — Penalizes <10% warning rate**

Assumes low warning rate means thresholds are too loose. Could be correct calibration.

---

## Priority Fix Order

### Immediate (blocks execution or causes crashes)
1. **1.1** — `crypto_trading_graph.py:351` add_edge with list (BLOCKER)
2. **2.1** — Undefined `new_direction` in live_executor (NameError crash)
3. **4.1** — Position size formula 100× oversizing (account blowup risk)

### Urgent (data loss, state corruption, wrong trades)
4. **2.2** — Missing save_state after position close
5. **2.3** — Duplicate TP orders from protection retry
6. **2.4** — Failed market order not logged
7. **1.2** — Missing state field initializations
8. **1.3** — Hard state access in risk_manager

### High (incorrect behavior, inconsistencies)
9. **4.2** — Hard limit tolerance 50% above stated
10. **5.1** — conflicts_with() blind to position
11. **5.2** — Daily timeframe double-weighted
12. **5.3** — Geopolitical severity ordering broken
13. **3.1** — Green lane action schema mismatch
14. **4.3** — Missing liquidation_buffer validation
15. **4.5** — Three different leverage definitions

### Medium (scoring accuracy, robustness)
16-31. Remaining MEDIUM issues as listed above.

### Low (style, clarity)
32-43. Remaining LOW issues.

---

*Review conducted 2026-03-13. All line references verified against current codebase.*

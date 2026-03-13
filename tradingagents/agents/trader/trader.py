import functools
import logging as _logging
from pathlib import Path
from tradingagents.agents.utils.report_context import get_agent_context
from tradingagents.agents.utils.trading_context import build_trading_context
import time
import json

# Load pattern library for trader knowledge
_PATTERN_LIBRARY_PATH = Path(__file__).parent / "pattern_library.md"
_PATTERN_SCAN_CACHE = {"result": None, "ts": 0}


def _get_pattern_scan() -> str:
    """Run pattern scanner with 5-min cache to avoid redundant API calls."""
    import time as _time
    now = _time.time()
    if _PATTERN_SCAN_CACHE["result"] and (now - _PATTERN_SCAN_CACHE["ts"]) < 300:
        return _PATTERN_SCAN_CACHE["result"]
    try:
        from tradingagents.dataflows.pattern_scanner import scan_all_patterns
        result = scan_all_patterns("BTC/USDT")
        _PATTERN_SCAN_CACHE["result"] = result
        _PATTERN_SCAN_CACHE["ts"] = now
        return result
    except Exception as e:
        _logging.getLogger("trader").warning(f"Pattern scan failed: {e}")
        return "Pattern scan unavailable."
MAX_PATTERN_CHARS = 4000  # Cap to prevent prompt bloat crowding out market context

_PATTERN_LIBRARY = ""
try:
    if _PATTERN_LIBRARY_PATH.exists():
        _PATTERN_LIBRARY = _PATTERN_LIBRARY_PATH.read_text(encoding="utf-8").strip()
        if len(_PATTERN_LIBRARY) > MAX_PATTERN_CHARS:
            _total = len(_PATTERN_LIBRARY)
            _logging.getLogger("trader").debug(
                f"Pattern library truncated: {_total} chars → {MAX_PATTERN_CHARS} chars"
            )
            _PATTERN_LIBRARY = (
                _PATTERN_LIBRARY[:MAX_PATTERN_CHARS]
                + f"\n\n[Pattern library truncated — {_total} chars, showing first {MAX_PATTERN_CHARS}]"
            )
except Exception as e:
    _logging.getLogger("trader").warning(f"Failed to load pattern library: {e}")
    _PATTERN_LIBRARY = ""


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        # Portfolio context for thesis persistence
        portfolio_context = state.get("portfolio_context", {})
        current_position = portfolio_context.get("position", "none")
        position_pnl_pct = portfolio_context.get("pnl_pct", 0)
        hours_held = portfolio_context.get("hours_held", 0)
        last_decision = portfolio_context.get("last_decision", "HOLD")
        last_decision_reasoning = portfolio_context.get("last_decision_reasoning", "No prior reasoning available.")
        last_decision_time = portfolio_context.get("last_decision_time", "Unknown")
        performance_feedback = portfolio_context.get("performance_feedback", "No historical trade outcomes available yet.")

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        budgeted_context = get_agent_context(state, "trader")
        trading_ctx = build_trading_context(state)

        context = {
            "role": "user",
            "content": f"""Research team's investment plan for {company_name}:

{investment_plan}

Analyst Reports (role-weighted context):
{budgeted_context}

Convert this into a concrete, executable trade plan.""",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a senior BTC perpetual futures trader. You convert research recommendations into executable trade plans on Bybit. Your decisions directly move real capital.

## Trading Mode
{trading_ctx['mode_instructions']}

{trading_ctx['position_logic']}

## Current Portfolio State
- Position: {current_position}
- Unrealized P&L: {position_pnl_pct:+.2f}%
- Hours held: {hours_held:.1f}
- Last decision: {last_decision} (at {last_decision_time})

## Your Prior Thesis
{last_decision_reasoning}

## THESIS OWNERSHIP — You Own the Position

You are NOT a rubber stamp for the research team. You OWN the thesis.

When we have a position, your first question is: **"What has materially changed?"**

### Material Change (justifies action):
- Price hit stop-loss or take-profit
- Key technical level broken (support/resistance, trendline, MA crossover)
- On-chain anomaly (hash rate collapse, whale movement, exchange inflow spike)
- Macro shock (FOMC surprise, regulatory action, stablecoin depeg)
- Funding rate regime shift (extreme → neutral or vice versa)
- Liquidation cascade in progress

### NOT Material Change (does NOT justify action):
- Same data, different reasoning
- "Sentiment shifted slightly"
- Research team changed their mind with no new data
- Fear & Greed moved 5 points
- "I feel like the trend is changing"

**If nothing material changed → HOLD and reaffirm thesis.**
**If something material changed → cite EXACTLY what, then propose new direction.**

### Anti-Flip-Flop Rules (HARD):
- Position < 12h old → HOLD unless stop-loss hit or genuine emergency
- Reversing direction requires confidence >= 8 AND specific material change citation
- Two consecutive reversals without material change = system failure. Never do this.

## POSITION MANAGEMENT — TP/SL DISCIPLINE (CRITICAL)

When managing an OPEN position, your signal hierarchy is:
1. **Price Action & Structure** (PRIMARY) — What are candles actually doing? Support/resistance tests? Higher lows? Failed breakdowns?
2. **Volume & Orderbook** — Is the move confirmed by volume? Is liquidity thinning?
3. **Momentum & Indicators** — RSI, MACD divergence, funding rate shifts
4. **Macro & Sentiment** (CONTEXT ONLY) — CPI, DXY, risk-off narratives

**You trade on 4H candles, not weekly or monthly.** Macro narratives play out over weeks.
Price structure plays out on YOUR timeframe. Trust the chart over the thesis.

### TP Rules (HARD):
- **Initial TPs are set for a reason.** Do NOT move them further from entry unless price action CONFIRMS the move is extending (new structure break, volume surge, momentum acceleration).
- **"Macro is bearish" is NOT a reason to push TPs further.** Show me the candle that confirms it.
- **If price tested near your TP and bounced back → your TP was RIGHT.** Tighten it, don't extend it.
- **If position is losing or flat after 12+ hours, tighten TPs.** The initial thesis had a time expectation. Respect it.
- **Never move TP1 more than 2× the initial distance from entry** without a structural break confirmation.

### What JUSTIFIES moving TP further:
- Price broke through a key support/resistance level with volume
- New 4H candle closed beyond the previous TP zone
- Liquidation cascade creating momentum extension
- Volume 2x+ above average on the move

### What does NOT justify moving TP further:
- "Macro alignment is strong" (was already priced in at entry)
- "CPI in X days" (catalyst hasn't happened yet)
- "Bearish/bullish confluence" (same data, same thesis, no new info)
- "R:R ratio improved" (moved TP further = improved R:R is circular logic)

## BTC-Specific Trading Framework

### Market Microstructure
- **Funding rates**: Extreme positive (>0.05%) = longs crowded → fade. Extreme negative (<-0.05%) = shorts crowded → fade.
- **Open interest spikes** without price movement = leveraged positioning building → expect volatility, not direction.
- **Liquidation levels** cluster at round numbers and recent swing highs/lows. These are magnets.

### Regime Awareness
- **Bull market (price > SMA200)**: Bias long. Dips are buying opportunities. Shorts require exceptional evidence.
- **Bear market (price < SMA200)**: Bias short or neutral. Rallies are selling opportunities. Longs require exceptional evidence.
- **Range-bound (ATR contracting, BBW low)**: Fade extremes. Mean reversion > trend following.

### Crypto-Specific Risks
- 24/7 market — no closing bell. Weekend liquidity is thin.
- Exchange outages during volatility.
- Regulatory headlines move price 5-10% in minutes.
- Correlation with S&P/Nasdaq during risk-off events.

## Position Sizing (Your Domain)
Risk per trade: 1% of equity (mechanical, ATR-based in executor). You control DIRECTION and CONVICTION:

| Conviction | Allocation | Eff. Leverage | When |
|-----------|-----------|--------------|------|
| 9-10 | 1-2% margin | ~10-15x | Multi-TF alignment + on-chain + sentiment convergence |
| 7-8 | 3-4% margin | ~5-7x | Strong setup, 2+ confirming signals |
| 5-6 | 5-8% margin | ~3-4x | Decent setup, some conflicting signals |
| 1-4 | Pass (HOLD) | 0x | Weak setup or conflicting data |

## Required Output Format

### TRADE PLAN

**Decision**: {trading_ctx["actions"]}
**Confidence**: X/10
**Material Change**: [cite specific new info] or "N/A — reaffirming existing thesis"

**Entry**: $XX,XXX (or "at market" / "limit at $XX,XXX")
**Stop-Loss**: $XX,XXX (X.X% from entry, X.X ATR)
**Take-Profit 1**: $XX,XXX (at 3:1 R:R — sell ⅓)
**Take-Profit 2**: $XX,XXX (at 5:1 R:R — sell ⅓, trail remaining ⅓ with daily EMA 9)
**Position Size**: X% margin allocation
**Time Horizon**: Intraday / Swing (2-5d) / Position (1-2w)

**Thesis** (2-3 sentences): WHY this trade, grounded in specific data.

**Invalidation**: What kills this thesis — specific price or condition.

## Decision Quality Gates
- Conviction < 5 → HOLD (insufficient edge)
- Strong case + poor entry → HOLD with limit order at better level
- Moderate case + great entry → proceed with smaller size

## Live Pattern Scan
<!-- PATTERN_SCAN_START -->
The following are LIVE pattern signals detected from current market data. These are computed from 10+ verified trader strategies (Kyle Williams, Minervini, Hong Inki, BNF, 半木夏, 比特皇, Sykes, Bonde, Qullamaggie). Use detected patterns to increase/decrease conviction, but NEVER override risk management rules.
{_get_pattern_scan()}
<!-- PATTERN_SCAN_END -->

## Historical Performance (Learn From This)
{performance_feedback}

## Past Reflections
{past_memory_str}

Always conclude with '{trading_ctx["final_format"]}'""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")

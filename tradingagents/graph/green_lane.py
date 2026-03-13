"""
Green Lane Router — Phase 2
Fast execution path for high-confidence deterministic setups.

Flow:
  1. check_green_lane()        → run scanner, gate on quality
  2. execute_green_lane()      → Trader formats order, Risk Manager approves
  3. trigger_analyst_audit()   → fire-and-forget full pipeline (post-entry)
  4. format_green_lane_alert() → Telegram-friendly notification
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

from tradingagents.dataflows.ta_schema import GreenLaneSignal
from tradingagents.dataflows.green_lane_scanner import scan_green_lane

log = logging.getLogger("green_lane")


# ─────────────────────────────────────────────────────────────────────────────
# 2a. Scanner Integration
# ─────────────────────────────────────────────────────────────────────────────

def check_green_lane(symbol: str = "BTC/USDT", min_quality: int = 7) -> Optional[GreenLaneSignal]:
    """
    Run the deterministic scanner and return the signal only if
    it triggered and quality >= min_quality.  Returns None otherwise.
    """
    log.info(f"check_green_lane: scanning {symbol} (min_quality={min_quality})")
    try:
        signal = scan_green_lane(symbol)
    except Exception as exc:
        log.error(f"Scanner error for {symbol}: {exc}", exc_info=True)
        return None

    if not signal.triggered:
        log.info(f"check_green_lane: no setup detected for {symbol}")
        return None

    if signal.quality_score < min_quality:
        log.info(
            f"check_green_lane: setup found but quality {signal.quality_score} < {min_quality} — skipping"
        )
        return None

    log.info(
        f"check_green_lane: GREEN LANE triggered — {symbol} "
        f"dir={signal.direction} q={signal.quality_score} entry={signal.entry_price}"
    )
    return signal


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Fast Execution Path
# ─────────────────────────────────────────────────────────────────────────────

def execute_green_lane(
    signal: GreenLaneSignal,
    llm_quick,
    llm_deep,
    portfolio_context: Optional[dict] = None,
) -> dict:
    """
    Fast path: 2 LLM calls only.
      Call 1 (llm_quick) — Trader validates and formats the order.
      Call 2 (llm_deep)  — Risk Manager approves / rejects / resizes.

    Returns:
        {
          "approved": bool,
          "trader_plan": str,
          "risk_verdict": str,
          "signal": GreenLaneSignal,
        }
    """
    portfolio_context = portfolio_context or {}

    direction_upper = signal.direction.upper()
    stop_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100 if signal.entry_price else 0
    risk = abs(signal.entry_price - signal.stop_loss)
    tp1_rr = abs(signal.tp1 - signal.entry_price) / risk if risk > 0 else 0
    tp2_rr = abs(signal.tp2 - signal.entry_price) / risk if risk > 0 else 0

    # ── Call 1: Trader formats the order ─────────────────────────────────────
    trader_prompt = f"""You are a BTC perpetual futures trader. A deterministic scanner has detected a high-confidence Green Lane setup. Your job is to validate it makes logical sense and output a clean, formatted trade plan.

## Green Lane Signal
Direction: {direction_upper}
Entry: ${signal.entry_price:,.2f}
Stop Loss: ${signal.stop_loss:,.2f} (-{stop_pct:.2f}%)
TP1: ${signal.tp1:,.2f} ({tp1_rr:.1f}:1 R:R)
TP2: ${signal.tp2:,.2f} ({tp2_rr:.1f}:1 R:R)
Trail: EMA 9 (Daily)

## Scanner Evidence
Quality Score: {signal.quality_score}/10
AVWAP Pinch: {"Active (" + str(signal.pinch_width_pct) + "%)" if signal.pinch_active else "Not active"}
Zone Width: {signal.zone_width_pct:.2f}%
Sweep Depth: {signal.sweep_depth_pct:.2f}%
V-Reversal Velocity: {signal.reversal_velocity:.4f}%/bar
Volume Ratio: {signal.volume_ratio:.2f}x
MTF Alignment: {signal.mtf_alignment}
Reasoning: {signal.reasoning}

## Instructions
1. Confirm the setup is logically coherent (entry above sweep low for long, stop below sweep wick)
2. Confirm R:R ratios are acceptable (3:1 minimum)
3. Output a formatted trade plan with order type, entry, stop, TP1, TP2, trail logic
4. Flag any concerns briefly
5. End with "PLAN CONFIRMED" or "PLAN REJECTED: <reason>"

Be concise. This is a fast-path execution."""

    log.info("execute_green_lane: calling Trader LLM...")
    try:
        trader_messages = [{"role": "user", "content": trader_prompt}]
        trader_response = llm_quick.invoke(trader_messages)
        trader_plan = trader_response.content if hasattr(trader_response, "content") else str(trader_response)
        log.info(f"Trader plan received ({len(trader_plan)} chars)")
    except Exception as exc:
        log.error(f"Trader LLM call failed: {exc}", exc_info=True)
        trader_plan = f"ERROR: Trader LLM failed — {exc}"

    # ── Call 2: Risk Manager evaluates ───────────────────────────────────────
    equity = portfolio_context.get("equity", 0)
    current_position = portfolio_context.get("position", "none")
    current_leverage = portfolio_context.get("leverage", 0)
    pnl_pct = portfolio_context.get("pnl_pct", 0)
    open_positions = portfolio_context.get("open_positions", [])
    max_portfolio_risk_pct = portfolio_context.get("max_portfolio_risk_pct", 2.0)

    risk_prompt = f"""You are the Risk Judge for a BTC perpetual futures desk on Bybit.

A Green Lane trade is about to be executed. Evaluate it NOW — fast path, no debate.

## Trader's Plan
{trader_plan}

## Green Lane Signal
Direction: {direction_upper}
Entry: ${signal.entry_price:,.2f}
Stop Loss: ${signal.stop_loss:,.2f} (-{stop_pct:.2f}%)
TP1: ${signal.tp1:,.2f}   TP2: ${signal.tp2:,.2f}
Quality: {signal.quality_score}/10

## Portfolio Context
Current Equity: ${equity:,.2f}
Current Position: {current_position}
Current Leverage: {current_leverage}x
Position PnL: {pnl_pct:.2f}%
Open Positions: {json.dumps(open_positions, indent=2) if open_positions else "None"}
Max Risk Per Trade: {max_portfolio_risk_pct}%

## Risk Checks (evaluate each)
1. Stop loss reasonable? (1.5%–3.0% from entry) — stop is {stop_pct:.2f}%
2. R:R acceptable? (min 3:1) — TP1 is {tp1_rr:.1f}:1, TP2 is {tp2_rr:.1f}:1
3. Combined exposure with existing positions
4. Leverage limits (max 5x recommended, 10x hard cap)
5. Liquidation distance (stop should be >=1.5% away)
6. Max loss per trade (risk <={max_portfolio_risk_pct}% of equity)

## Your Response Format
VERDICT: APPROVED | APPROVED WITH ADJUSTMENTS | VETO

If APPROVED: state position size recommendation (% of equity)
If APPROVED WITH ADJUSTMENTS: specify exact adjustments
If VETO: state the hard limit breached

Be decisive. One paragraph max."""

    log.info("execute_green_lane: calling Risk Manager LLM...")
    try:
        risk_messages = [{"role": "user", "content": risk_prompt}]
        risk_response = llm_deep.invoke(risk_messages)
        risk_verdict = risk_response.content if hasattr(risk_response, "content") else str(risk_response)
        log.info(f"Risk verdict received ({len(risk_verdict)} chars)")
    except Exception as exc:
        log.error(f"Risk Manager LLM call failed: {exc}", exc_info=True)
        risk_verdict = f"ERROR: Risk LLM failed — {exc}"

    verdict_upper = risk_verdict.upper()
    approved = "VETO" not in verdict_upper and (
        "APPROVED" in verdict_upper or "APPROVE" in verdict_upper
    )

    log.info(
        f"execute_green_lane complete: approved={approved} "
        f"trader_plan_len={len(trader_plan)} risk_verdict_len={len(risk_verdict)}"
    )

    return {
        "approved": approved,
        "trader_plan": trader_plan,
        "risk_verdict": risk_verdict,
        "signal": signal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2c. Parallel Analyst Wake-up
# ─────────────────────────────────────────────────────────────────────────────

def trigger_analyst_audit(symbol: str, signal: GreenLaneSignal, ta_graph) -> None:
    """
    Fire-and-forget: spawn the full CryptoTradingAgentsGraph pipeline in a
    background thread for post-entry audit.  Results saved to
    logs/green_lane_audit_YYYY-MM-DD.json.  Never blocks the caller.
    """

    def _run_audit():
        log.info(f"trigger_analyst_audit: starting background audit for {symbol}")
        try:
            results = ta_graph.propagate(symbol, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            # Resolve logs dir relative to project root
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            )
            log_dir = os.path.join(project_root, "logs")
            os.makedirs(log_dir, exist_ok=True)
            audit_path = os.path.join(log_dir, f"green_lane_audit_{date_str}.json")

            audit_record = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": signal.model_dump(),
                "analyst_results": results if isinstance(results, dict) else str(results),
            }

            existing = []
            if os.path.exists(audit_path):
                try:
                    with open(audit_path, "r") as f:
                        existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
                except Exception:
                    existing = []

            existing.append(audit_record)
            with open(audit_path, "w") as f:
                json.dump(existing, f, indent=2, default=str)

            log.info(f"trigger_analyst_audit: audit saved to {audit_path}")

        except Exception as exc:
            log.error(f"trigger_analyst_audit: background audit failed: {exc}", exc_info=True)

    t = threading.Thread(target=_run_audit, daemon=True, name=f"green_lane_audit_{symbol}")
    t.start()
    log.info(f"trigger_analyst_audit: background thread started for {symbol}")


# ─────────────────────────────────────────────────────────────────────────────
# 2d. Notification Helper
# ─────────────────────────────────────────────────────────────────────────────

def format_green_lane_alert(signal: GreenLaneSignal, risk_verdict: str) -> str:
    """Format a Telegram-friendly Green Lane alert message."""

    direction_upper = signal.direction.upper()
    direction_emoji = "📈" if signal.direction == "long" else "📉"

    stop_pct = (
        abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
        if signal.entry_price > 0 else 0
    )

    risk = abs(signal.entry_price - signal.stop_loss)
    tp1_rr = abs(signal.tp1 - signal.entry_price) / risk if risk > 0 else 0
    tp2_rr = abs(signal.tp2 - signal.entry_price) / risk if risk > 0 else 0

    pinch_str = f"✅ ({signal.pinch_width_pct:.1f}%)" if signal.pinch_active else "❌"

    trail_display = {
        "daily_ema9": "Daily EMA 9",
        "ema9": "EMA 9",
        "ema21": "EMA 21",
    }.get(signal.trail_ema, signal.trail_ema)

    mtf_display = signal.mtf_alignment.title() if signal.mtf_alignment else "Mixed"

    verdict_upper = risk_verdict.upper()
    if "VETO" in verdict_upper:
        verdict_label = "VETOED ❌"
    elif "ADJUSTMENTS" in verdict_upper:
        verdict_label = "APPROVED WITH ADJUSTMENTS ⚠️"
    elif "APPROVED" in verdict_upper or "APPROVE" in verdict_upper:
        verdict_label = "APPROVED ✅"
    else:
        verdict_label = risk_verdict.split("\n")[0].strip()

    alert = (
        f"🟢 GREEN LANE TRIGGERED\n"
        f"\n"
        f"{direction_emoji} Direction: {direction_upper}\n"
        f"Entry:  ${signal.entry_price:,.2f}\n"
        f"Stop:   ${signal.stop_loss:,.2f} (-{stop_pct:.1f}%)\n"
        f"TP1:    ${signal.tp1:,.2f} ({tp1_rr:.0f}:1)\n"
        f"TP2:    ${signal.tp2:,.2f} ({tp2_rr:.0f}:1)\n"
        f"Trail:  {trail_display}\n"
        f"\n"
        f"Quality: {signal.quality_score}/10\n"
        f"AVWAP Pinch: {pinch_str}\n"
        f"MTF Alignment: {mtf_display}\n"
        f"\n"
        f"Risk Verdict: {verdict_label}\n"
        f"\n"
        f"⚠️ Full analyst audit running — investigate"
    )

    return alert

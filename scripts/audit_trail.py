#!/usr/bin/env python3
"""Audit Trail & Performance Attribution Module

Provides:
1. Audit Trail: Every decision with full agent reasoning chain
2. Performance Attribution: Which agents contribute alpha vs destroy it

Integrates with existing dashboard or runs standalone.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SYDNEY_TZ = ZoneInfo("Australia/Sydney")
PROJECT_DIR = Path(__file__).parent.parent
LOGS_DIR = PROJECT_DIR / "logs"
EVAL_DIR = PROJECT_DIR / "eval_results" / "BTC_USDT" / "CryptoTradingAgents_logs"
STATE_FILE = LOGS_DIR / "executor_state.json"
ATTRIBUTION_FILE = LOGS_DIR / "performance_attribution.json"


def load_executor_state():
    """Load trade history from executor state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"trades": []}


def load_agent_reports(date_str):
    """Load agent reports for a given date."""
    report_file = LOGS_DIR / f"agent_reports_{date_str}.json"
    if report_file.exists():
        with open(report_file) as f:
            return json.load(f)
    return {}


def load_full_state_log(date_str):
    """Load full state log from eval results."""
    state_file = EVAL_DIR / f"full_states_log_{date_str}.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {}


def extract_agent_signals(reports):
    """Extract what each agent recommended from their reports."""
    signals = {}
    
    for agent_name in ["market", "sentiment", "news"]:
        report = reports.get(agent_name, "")
        if not report:
            signals[agent_name] = {"direction": "UNKNOWN", "summary": "No report"}
            continue
        
        # Try to find directional recommendation
        upper = report.upper()
        direction = "NEUTRAL"
        for d in ["SELL", "SHORT", "BEARISH"]:
            if d in upper[-300:]:
                direction = "BEARISH"
                break
        for d in ["BUY", "LONG", "BULLISH"]:
            if d in upper[-300:]:
                direction = "BULLISH"
                break
        
        # Extract first 200 chars as summary
        clean = re.sub(r'\s+', ' ', report).strip()
        signals[agent_name] = {
            "direction": direction,
            "summary": clean[:200] + "..." if len(clean) > 200 else clean,
        }
    
    # Extract final decision
    final = reports.get("final", "")
    if final:
        upper = final.upper()
        for d in ["SELL", "SHORT"]:
            if f"**{d}**" in upper or f"PROPOSAL: **{d}" in upper:
                signals["final_direction"] = "SHORT"
                break
        else:
            for d in ["BUY", "LONG"]:
                if f"**{d}**" in upper or f"PROPOSAL: **{d}" in upper:
                    signals["final_direction"] = "LONG"
                    break
            else:
                signals["final_direction"] = "NEUTRAL"
    
    return signals


def build_audit_trail():
    """Build complete audit trail from all available data."""
    state = load_executor_state()
    trades = state.get("trades", [])
    
    trail = []
    for trade in trades:
        timestamp = trade.get("timestamp", "")
        date_str = timestamp[:10] if timestamp else ""
        
        reports = load_agent_reports(date_str)
        agent_signals = extract_agent_signals(reports) if reports else {}
        
        # Load full state logs for detailed agent reasoning
        full_state = load_full_state_log(date_str)
        agent_details = {}
        if full_state:
            # Full states are keyed by date
            for dt_key, state_data in full_state.items():
                agent_details = {
                    "market_report": state_data.get("market_report", ""),
                    "sentiment_report": state_data.get("sentiment_report", ""),
                    "news_report": state_data.get("news_report", ""),
                    "fundamentals_report": state_data.get("fundamentals_report", ""),
                    "bull_case": state_data.get("investment_debate_state", {}).get("bull_history", ""),
                    "bear_case": state_data.get("investment_debate_state", {}).get("bear_history", ""),
                    "research_verdict": state_data.get("investment_debate_state", {}).get("judge_decision", ""),
                    "trader_plan": state_data.get("trader_investment_decision", state_data.get("trader_investment_plan", "")),
                    "risk_debate": {
                        "aggressive": state_data.get("risk_debate_state", {}).get("aggressive_history", ""),
                        "conservative": state_data.get("risk_debate_state", {}).get("conservative_history", ""),
                        "neutral": state_data.get("risk_debate_state", {}).get("neutral_history", ""),
                    },
                    "risk_verdict": state_data.get("risk_debate_state", {}).get("judge_decision", ""),
                    "final_decision": state_data.get("final_trade_decision", ""),
                }
        
        entry = {
            "timestamp": timestamp,
            "timestamp_sydney": _to_sydney(timestamp),
            "decision": trade.get("decision", "UNKNOWN"),
            "action": trade.get("action", trade.get("transition", "UNKNOWN")),
            "price": trade.get("price"),
            "amount": trade.get("amount"),
            "stop_loss": trade.get("stop_loss"),
            "tp1": trade.get("tp1"),
            "pnl_pct": trade.get("close_pnl_pct"),
            "equity": trade.get("equity"),
            "params": trade.get("params", {}),
            "agent_signals": agent_signals,
            "agent_details": agent_details,
            "flip_flop_blocked": trade.get("flip_flop_blocked", False),
        }
        trail.append(entry)
    
    return trail


def calculate_attribution():
    """Calculate which agents are adding vs destroying alpha.
    
    For each closed trade, we check:
    - What did each agent recommend?
    - Was the trade profitable?
    - Did the agent's recommendation align with the profitable direction?
    
    Agents who consistently recommend the right direction get positive attribution.
    Agents who consistently recommend the wrong direction get negative attribution.
    """
    state = load_executor_state()
    trades = state.get("trades", [])
    
    # Load existing attribution or start fresh
    if ATTRIBUTION_FILE.exists():
        with open(ATTRIBUTION_FILE) as f:
            attribution = json.load(f)
    else:
        attribution = {
            "agents": {
                "market": {"correct": 0, "incorrect": 0, "neutral": 0, "total_pnl_aligned": 0.0},
                "sentiment": {"correct": 0, "incorrect": 0, "neutral": 0, "total_pnl_aligned": 0.0},
                "news": {"correct": 0, "incorrect": 0, "neutral": 0, "total_pnl_aligned": 0.0},
                "risk_judge": {"correct": 0, "incorrect": 0, "neutral": 0, "total_pnl_aligned": 0.0},
                "portfolio_manager": {"correct": 0, "incorrect": 0, "neutral": 0, "total_pnl_aligned": 0.0},
            },
            "trades_analyzed": 0,
            "total_pnl_pct": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "last_updated": None,
        }
    
    # Find closed trades (those with close_pnl_pct)
    closed_trades = [t for t in trades if t.get("close_pnl_pct") is not None]
    
    # Only analyze trades we haven't seen yet
    already_analyzed = attribution.get("trades_analyzed", 0)
    new_trades = closed_trades[already_analyzed:]
    
    for trade in new_trades:
        pnl = trade["close_pnl_pct"]
        decision = trade.get("decision", "").upper()
        date_str = trade.get("timestamp", "")[:10]
        
        # Determine if the trade was profitable
        profitable = pnl > 0
        trade_direction = "BULLISH" if decision in ("BUY", "LONG") else "BEARISH" if decision in ("SELL", "SHORT") else "NEUTRAL"
        
        # What direction would have been correct?
        correct_direction = trade_direction if profitable else ("BEARISH" if trade_direction == "BULLISH" else "BULLISH")
        
        # Load agent reports and check alignment
        reports = load_agent_reports(date_str)
        if reports:
            signals = extract_agent_signals(reports)
            for agent_name in ["market", "sentiment", "news"]:
                agent_dir = signals.get(agent_name, {}).get("direction", "NEUTRAL")
                if agent_dir == "NEUTRAL":
                    attribution["agents"][agent_name]["neutral"] += 1
                elif agent_dir == correct_direction:
                    attribution["agents"][agent_name]["correct"] += 1
                    attribution["agents"][agent_name]["total_pnl_aligned"] += abs(pnl)
                else:
                    attribution["agents"][agent_name]["incorrect"] += 1
                    attribution["agents"][agent_name]["total_pnl_aligned"] -= abs(pnl)
        
        attribution["total_pnl_pct"] += pnl
        if profitable:
            attribution["winning_trades"] += 1
        else:
            attribution["losing_trades"] += 1
    
    attribution["trades_analyzed"] = len(closed_trades)
    attribution["last_updated"] = datetime.now(timezone.utc).isoformat()
    
    # Calculate accuracy percentages
    for agent_name, stats in attribution["agents"].items():
        total = stats["correct"] + stats["incorrect"]
        stats["accuracy_pct"] = round(stats["correct"] / total * 100, 1) if total > 0 else 0
        stats["total_calls"] = total + stats["neutral"]
    
    # Save
    with open(ATTRIBUTION_FILE, "w") as f:
        json.dump(attribution, f, indent=2)
    
    return attribution


def _to_sydney(iso_str):
    """Convert ISO timestamp to Sydney time string."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(SYDNEY_TZ).strftime("%Y-%m-%d %H:%M %Z")
    except (ValueError, TypeError):
        return iso_str


def get_attribution_summary():
    """Get a human-readable attribution summary."""
    attr = calculate_attribution()
    
    lines = [
        f"📊 Performance Attribution — {attr['trades_analyzed']} trades analyzed",
        f"   Total P&L: {attr['total_pnl_pct']:+.2f}% | W: {attr['winning_trades']} L: {attr['losing_trades']}",
        "",
    ]
    
    # Sort agents by accuracy
    sorted_agents = sorted(
        attr["agents"].items(),
        key=lambda x: x[1].get("accuracy_pct", 0),
        reverse=True,
    )
    
    for agent_name, stats in sorted_agents:
        total = stats["total_calls"]
        if total == 0:
            continue
        acc = stats["accuracy_pct"]
        pnl = stats["total_pnl_aligned"]
        emoji = "🟢" if acc >= 60 else "🟡" if acc >= 40 else "🔴"
        lines.append(f"   {emoji} {agent_name:20s} | Accuracy: {acc:5.1f}% | Alpha: {pnl:+.2f}% | Calls: {total}")
    
    return "\n".join(lines)


# ── Flask routes (register with existing dashboard) ──

def register_audit_routes(app):
    """Register audit trail and attribution routes with a Flask app."""
    from flask import jsonify, render_template_string
    
    @app.route("/api/audit-trail")
    def api_audit_trail():
        return jsonify(build_audit_trail())
    
    @app.route("/api/attribution")
    def api_attribution():
        return jsonify(calculate_attribution())
    
    @app.route("/audit")
    def audit_page():
        trail = build_audit_trail()
        attr = calculate_attribution()
        return render_template_string(AUDIT_HTML, trail=trail, attribution=attr)


AUDIT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Audit Trail & Attribution</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0; padding: 20px; }
  h1, h2, h3 { color: #fff; margin-bottom: 12px; }
  h1 { font-size: 24px; border-bottom: 1px solid #333; padding-bottom: 12px; margin-bottom: 24px; }
  h2 { font-size: 18px; margin-top: 24px; }
  
  .card { background: #141420; border: 1px solid #2a2a3a; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
  
  .attribution-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
  .agent-card { background: #1a1a2e; border: 1px solid #2a2a3a; border-radius: 8px; padding: 16px; }
  .agent-name { font-weight: 600; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #888; }
  .agent-accuracy { font-size: 32px; font-weight: 700; margin: 8px 0; }
  .green { color: #00ff88; }
  .yellow { color: #ffaa00; }
  .red { color: #ff4444; }
  .agent-detail { font-size: 13px; color: #888; margin-top: 4px; }
  
  .summary-row { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 24px; }
  .summary-stat { text-align: center; }
  .summary-stat .value { font-size: 28px; font-weight: 700; }
  .summary-stat .label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
  
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .badge-long { background: #00ff8820; color: #00ff88; }
  .badge-short { background: #ff444420; color: #ff4444; }
  .badge-hold { background: #ffaa0020; color: #ffaa00; }
  .badge-blocked { background: #ff000030; color: #ff6666; }
  
  .pnl-positive { color: #00ff88; }
  .pnl-negative { color: #ff4444; }
  
  /* Trade Decision Cards */
  .trade-card { background: #141420; border: 1px solid #2a2a3a; border-radius: 12px; margin-bottom: 20px; overflow: hidden; }
  .trade-header { display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; cursor: pointer; transition: background 0.2s; }
  .trade-header:hover { background: #1a1a2e; }
  .trade-meta { display: flex; align-items: center; gap: 12px; }
  .trade-stats { display: flex; gap: 20px; font-size: 13px; color: #aaa; }
  .trade-stats span { white-space: nowrap; }
  
  /* Agent Reasoning Panels */
  .trade-body { display: none; border-top: 1px solid #2a2a3a; }
  .trade-body.open { display: block; }
  
  .agent-tabs { display: flex; flex-wrap: wrap; gap: 0; border-bottom: 1px solid #2a2a3a; background: #0e0e18; }
  .agent-tab { padding: 10px 18px; cursor: pointer; font-size: 13px; font-weight: 500; color: #888; border-bottom: 2px solid transparent; transition: all 0.2s; }
  .agent-tab:hover { color: #ccc; background: #1a1a2e; }
  .agent-tab.active { color: #fff; border-bottom-color: #6366f1; }
  
  .agent-panel { display: none; padding: 20px; max-height: 600px; overflow-y: auto; }
  .agent-panel.active { display: block; }
  .agent-panel .md-content { font-size: 14px; line-height: 1.7; }
  .agent-panel .md-content h1, .agent-panel .md-content h2, .agent-panel .md-content h3 { margin-top: 16px; margin-bottom: 8px; }
  .agent-panel .md-content table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  .agent-panel .md-content th, .agent-panel .md-content td { padding: 6px 10px; border: 1px solid #333; text-align: left; }
  .agent-panel .md-content th { background: #1a1a2e; color: #aaa; }
  .agent-panel .md-content ul, .agent-panel .md-content ol { padding-left: 24px; margin: 8px 0; }
  .agent-panel .md-content li { margin: 4px 0; }
  .agent-panel .md-content strong { color: #fff; }
  .agent-panel .md-content code { background: #1a1a2e; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
  .agent-panel .md-content blockquote { border-left: 3px solid #6366f1; padding-left: 12px; color: #aaa; margin: 12px 0; }
  
  .expand-icon { font-size: 18px; color: #666; transition: transform 0.2s; }
  .trade-card.open .expand-icon { transform: rotate(180deg); }
  
  /* Pipeline visualization */
  .pipeline { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; padding: 12px 20px; background: #0e0e18; border-bottom: 1px solid #2a2a3a; font-size: 12px; }
  .pipeline-step { padding: 4px 10px; border-radius: 4px; background: #1a1a2e; color: #888; }
  .pipeline-step.bullish { background: #00ff8815; color: #00ff88; }
  .pipeline-step.bearish { background: #ff444415; color: #ff4444; }
  .pipeline-step.neutral { background: #ffaa0015; color: #ffaa00; }
  .pipeline-arrow { color: #444; }
</style>
</head>
<body>
<h1>🔍 Audit Trail & Performance Attribution</h1>

<!-- Attribution Summary -->
<div class="card">
<h2>📊 Agent Performance Attribution</h2>
<div class="summary-row">
  <div class="summary-stat">
    <div class="value">{{ attribution.trades_analyzed }}</div>
    <div class="label">Trades Analyzed</div>
  </div>
  <div class="summary-stat">
    <div class="value {% if attribution.total_pnl_pct >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
      {{ "%.2f"|format(attribution.total_pnl_pct) }}%
    </div>
    <div class="label">Total P&L</div>
  </div>
  <div class="summary-stat">
    <div class="value green">{{ attribution.winning_trades }}</div>
    <div class="label">Winners</div>
  </div>
  <div class="summary-stat">
    <div class="value red">{{ attribution.losing_trades }}</div>
    <div class="label">Losers</div>
  </div>
</div>

<div class="attribution-grid">
{% for agent_name, stats in attribution.agents.items() %}
{% if stats.total_calls > 0 %}
<div class="agent-card">
  <div class="agent-name">{{ agent_name }}</div>
  <div class="agent-accuracy {% if stats.accuracy_pct >= 60 %}green{% elif stats.accuracy_pct >= 40 %}yellow{% else %}red{% endif %}">
    {{ stats.accuracy_pct }}%
  </div>
  <div class="agent-detail">{{ stats.correct }} correct / {{ stats.incorrect }} incorrect / {{ stats.neutral }} neutral</div>
  <div class="agent-detail">Alpha: <span class="{% if stats.total_pnl_aligned >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">{{ "%.2f"|format(stats.total_pnl_aligned) }}%</span></div>
</div>
{% endif %}
{% endfor %}
</div>
</div>

<!-- Decision Trail -->
<div class="card">
<h2>📋 Decision Trail — Full Agent Reasoning</h2>

{% for entry in trail|reverse %}
<div class="trade-card" id="trade-{{ loop.index }}">
  <div class="trade-header" onclick="toggleTrade({{ loop.index }})">
    <div class="trade-meta">
      <span class="expand-icon">▼</span>
      <strong>{{ entry.timestamp_sydney }}</strong>
      {% if entry.decision in ['BUY', 'LONG'] %}
      <span class="badge badge-long">{{ entry.decision }}</span>
      {% elif entry.decision in ['SELL', 'SHORT'] %}
      <span class="badge badge-short">{{ entry.decision }}</span>
      {% else %}
      <span class="badge badge-hold">{{ entry.decision }}</span>
      {% endif %}
      {% if entry.flip_flop_blocked %}
      <span class="badge badge-blocked">FLIP-FLOP BLOCKED</span>
      {% endif %}
    </div>
    <div class="trade-stats">
      <span>{{ entry.action or '' }}</span>
      {% if entry.price %}<span>@ {{ "${:,.0f}".format(entry.price) }}</span>{% endif %}
      {% if entry.amount %}<span>{{ entry.amount }} BTC</span>{% endif %}
      {% if entry.pnl_pct is not none %}<span class="{% if entry.pnl_pct >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">{{ "{:+.2f}%".format(entry.pnl_pct) }}</span>{% endif %}
      {% if entry.params and entry.params.get('confidence') %}<span>Conf: {{ entry.params.confidence }}/10</span>{% endif %}
    </div>
  </div>
  
  {% if entry.agent_details %}
  <!-- Pipeline visualization -->
  <div class="trade-body" id="trade-body-{{ loop.index }}">
    <div class="pipeline">
      <span class="pipeline-step {% if 'BEARISH' in entry.agent_signals.get('market', {}).get('direction', '') %}bearish{% elif 'BULLISH' in entry.agent_signals.get('market', {}).get('direction', '') %}bullish{% else %}neutral{% endif %}">📈 Market</span>
      <span class="pipeline-arrow">→</span>
      <span class="pipeline-step {% if 'BEARISH' in entry.agent_signals.get('sentiment', {}).get('direction', '') %}bearish{% elif 'BULLISH' in entry.agent_signals.get('sentiment', {}).get('direction', '') %}bullish{% else %}neutral{% endif %}">💭 Sentiment</span>
      <span class="pipeline-arrow">→</span>
      <span class="pipeline-step {% if 'BEARISH' in entry.agent_signals.get('news', {}).get('direction', '') %}bearish{% elif 'BULLISH' in entry.agent_signals.get('news', {}).get('direction', '') %}bullish{% else %}neutral{% endif %}">📰 News</span>
      <span class="pipeline-arrow">→</span>
      <span class="pipeline-step">🐂 Bull</span>
      <span class="pipeline-arrow">⚔️</span>
      <span class="pipeline-step">🐻 Bear</span>
      <span class="pipeline-arrow">→</span>
      <span class="pipeline-step">📊 Trader</span>
      <span class="pipeline-arrow">→</span>
      <span class="pipeline-step">⚖️ Risk Debate</span>
      <span class="pipeline-arrow">→</span>
      <span class="pipeline-step">🛡️ Risk Mgr</span>
      <span class="pipeline-arrow">→</span>
      <span class="pipeline-step {% if entry.decision in ['BUY','LONG'] %}bullish{% elif entry.decision in ['SELL','SHORT'] %}bearish{% else %}neutral{% endif %}">🏛️ PM → {{ entry.decision }}</span>
    </div>
    
    <div class="agent-tabs" id="tabs-{{ loop.index }}">
      <div class="agent-tab active" onclick="showPanel({{ loop.index }}, 'market')">📈 Market</div>
      <div class="agent-tab" onclick="showPanel({{ loop.index }}, 'sentiment')">💭 Sentiment</div>
      <div class="agent-tab" onclick="showPanel({{ loop.index }}, 'news')">📰 News</div>
      <div class="agent-tab" onclick="showPanel({{ loop.index }}, 'bull')">🐂 Bull</div>
      <div class="agent-tab" onclick="showPanel({{ loop.index }}, 'bear')">🐻 Bear</div>
      <div class="agent-tab" onclick="showPanel({{ loop.index }}, 'research')">🔬 Research Verdict</div>
      <div class="agent-tab" onclick="showPanel({{ loop.index }}, 'trader')">📊 Trader</div>
      <div class="agent-tab" onclick="showPanel({{ loop.index }}, 'risk_aggressive')">⚡ Aggressive</div>
      <div class="agent-tab" onclick="showPanel({{ loop.index }}, 'risk_conservative')">🛡️ Conservative</div>
      <div class="agent-tab" onclick="showPanel({{ loop.index }}, 'risk_neutral')">⚖️ Neutral</div>
      <div class="agent-tab" onclick="showPanel({{ loop.index }}, 'risk_verdict')">🔒 Risk Verdict</div>
      <div class="agent-tab" onclick="showPanel({{ loop.index }}, 'final')">🏛️ Final Decision</div>
    </div>
    
    <div class="agent-panel active" id="panel-{{ loop.index }}-market"><div class="md-content" data-md="{{ entry.agent_details.get('market_report', 'No report available.')|e }}"></div></div>
    <div class="agent-panel" id="panel-{{ loop.index }}-sentiment"><div class="md-content" data-md="{{ entry.agent_details.get('sentiment_report', 'No report available.')|e }}"></div></div>
    <div class="agent-panel" id="panel-{{ loop.index }}-news"><div class="md-content" data-md="{{ entry.agent_details.get('news_report', 'No report available.')|e }}"></div></div>
    <div class="agent-panel" id="panel-{{ loop.index }}-bull"><div class="md-content" data-md="{{ entry.agent_details.get('bull_case', 'No report available.')|e }}"></div></div>
    <div class="agent-panel" id="panel-{{ loop.index }}-bear"><div class="md-content" data-md="{{ entry.agent_details.get('bear_case', 'No report available.')|e }}"></div></div>
    <div class="agent-panel" id="panel-{{ loop.index }}-research"><div class="md-content" data-md="{{ entry.agent_details.get('research_verdict', 'No report available.')|e }}"></div></div>
    <div class="agent-panel" id="panel-{{ loop.index }}-trader"><div class="md-content" data-md="{{ entry.agent_details.get('trader_plan', 'No report available.')|e }}"></div></div>
    <div class="agent-panel" id="panel-{{ loop.index }}-risk_aggressive"><div class="md-content" data-md="{{ entry.agent_details.get('risk_debate', {}).get('aggressive', 'No report available.')|e }}"></div></div>
    <div class="agent-panel" id="panel-{{ loop.index }}-risk_conservative"><div class="md-content" data-md="{{ entry.agent_details.get('risk_debate', {}).get('conservative', 'No report available.')|e }}"></div></div>
    <div class="agent-panel" id="panel-{{ loop.index }}-risk_neutral"><div class="md-content" data-md="{{ entry.agent_details.get('risk_debate', {}).get('neutral', 'No report available.')|e }}"></div></div>
    <div class="agent-panel" id="panel-{{ loop.index }}-risk_verdict"><div class="md-content" data-md="{{ entry.agent_details.get('risk_verdict', 'No report available.')|e }}"></div></div>
    <div class="agent-panel" id="panel-{{ loop.index }}-final"><div class="md-content" data-md="{{ entry.agent_details.get('final_decision', 'No report available.')|e }}"></div></div>
  </div>
  {% endif %}
</div>
{% endfor %}

</div>

<script>
function toggleTrade(id) {
  const card = document.getElementById('trade-' + id);
  const body = document.getElementById('trade-body-' + id);
  if (body) {
    card.classList.toggle('open');
    body.classList.toggle('open');
    // Render markdown on first open
    if (body.classList.contains('open')) {
      body.querySelectorAll('.md-content[data-md]').forEach(el => {
        if (!el.dataset.rendered) {
          el.innerHTML = marked.parse(el.dataset.md);
          el.dataset.rendered = '1';
        }
      });
    }
  }
}

function showPanel(tradeId, panel) {
  // Hide all panels and deactivate all tabs for this trade
  document.querySelectorAll('#trade-body-' + tradeId + ' .agent-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('#tabs-' + tradeId + ' .agent-tab').forEach(t => t.classList.remove('active'));
  
  // Show selected panel
  const panelEl = document.getElementById('panel-' + tradeId + '-' + panel);
  if (panelEl) {
    panelEl.classList.add('active');
    // Render markdown
    panelEl.querySelectorAll('.md-content[data-md]').forEach(el => {
      if (!el.dataset.rendered) {
        el.innerHTML = marked.parse(el.dataset.md);
        el.dataset.rendered = '1';
      }
    });
  }
  
  // Activate tab
  event.target.classList.add('active');
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    # Standalone: print summary
    print(get_attribution_summary())
    print()
    trail = build_audit_trail()
    print(f"📋 Audit Trail: {len(trail)} decisions")
    for entry in trail:
        ts = entry["timestamp_sydney"]
        dec = entry["decision"]
        action = entry["action"]
        price = f"${entry['price']:,.0f}" if entry.get("price") else "-"
        pnl = f"{entry['pnl_pct']:+.2f}%" if entry.get("pnl_pct") is not None else "-"
        print(f"   {ts} | {dec:5s} | {action:20s} | {price:>10s} | P&L: {pnl}")

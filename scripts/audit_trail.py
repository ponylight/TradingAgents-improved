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
        return dt.astimezone(SYDNEY_TZ).strftime("%Y-%m-%d %H:%M AEDT")
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
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0; padding: 20px; }
  h1, h2 { color: #fff; margin-bottom: 16px; }
  h1 { font-size: 24px; border-bottom: 1px solid #333; padding-bottom: 12px; margin-bottom: 24px; }
  h2 { font-size: 18px; margin-top: 32px; }
  
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
  
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 10px 8px; border-bottom: 2px solid #333; color: #888; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 1px; }
  td { padding: 10px 8px; border-bottom: 1px solid #1a1a2a; }
  tr:hover { background: #1a1a2e; }
  
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .badge-long { background: #00ff8820; color: #00ff88; }
  .badge-short { background: #ff444420; color: #ff4444; }
  .badge-hold { background: #ffaa0020; color: #ffaa00; }
  .badge-blocked { background: #ff000030; color: #ff6666; }
  
  .pnl-positive { color: #00ff88; }
  .pnl-negative { color: #ff4444; }
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
  <div class="agent-detail">Alpha contribution: <span class="{% if stats.total_pnl_aligned >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">{{ "%.2f"|format(stats.total_pnl_aligned) }}%</span></div>
</div>
{% endif %}
{% endfor %}
</div>
</div>

<!-- Audit Trail -->
<div class="card">
<h2>📋 Decision Audit Trail</h2>
<table>
<thead>
<tr>
  <th>Time (AEDT)</th>
  <th>Decision</th>
  <th>Action</th>
  <th>Price</th>
  <th>Size</th>
  <th>SL</th>
  <th>TP1</th>
  <th>P&L</th>
  <th>Confidence</th>
</tr>
</thead>
<tbody>
{% for entry in trail|reverse %}
<tr>
  <td>{{ entry.timestamp_sydney }}</td>
  <td>
    {% if entry.decision in ['BUY', 'LONG'] %}
    <span class="badge badge-long">{{ entry.decision }}</span>
    {% elif entry.decision in ['SELL', 'SHORT'] %}
    <span class="badge badge-short">{{ entry.decision }}</span>
    {% else %}
    <span class="badge badge-hold">{{ entry.decision }}</span>
    {% endif %}
    {% if entry.flip_flop_blocked %}
    <span class="badge badge-blocked">BLOCKED</span>
    {% endif %}
  </td>
  <td>{{ entry.action or '-' }}</td>
  <td>{{ "${:,.0f}".format(entry.price) if entry.price else '-' }}</td>
  <td>{{ entry.amount or '-' }}</td>
  <td>{{ "${:,.0f}".format(entry.stop_loss) if entry.stop_loss else '-' }}</td>
  <td>{{ "${:,.0f}".format(entry.tp1) if entry.tp1 else '-' }}</td>
  <td class="{% if entry.pnl_pct and entry.pnl_pct >= 0 %}pnl-positive{% elif entry.pnl_pct %}pnl-negative{% endif %}">
    {{ "{:+.2f}%".format(entry.pnl_pct) if entry.pnl_pct is not none else '-' }}
  </td>
  <td>{{ entry.params.get('confidence', '-') if entry.params else '-' }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

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

#!/usr/bin/env python3
"""Trading Agents Dashboard — single-file Flask app on port 5555.

Usage:
    python scripts/dashboard.py
"""

import json
import glob
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
EVAL_DIR = BASE_DIR / "eval_results" / "BTC_USDT" / "CryptoTradingAgents_logs"
MEMORY_DIR = BASE_DIR / "memory"

AEST = timezone(timedelta(hours=11))
CRON_HOURS = [3, 7, 11, 15, 19, 23]
CRON_MINUTE = 5

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _latest_full_state():
    pattern = str(EVAL_DIR / "full_states_log_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return _load_json(files[-1])


def _next_cron_run():
    now = datetime.now(AEST)
    for h in CRON_HOURS:
        candidate = now.replace(hour=h, minute=CRON_MINUTE, second=0, microsecond=0)
        if candidate > now:
            return candidate
    tomorrow = now + timedelta(days=1)
    return tomorrow.replace(hour=CRON_HOURS[0], minute=CRON_MINUTE, second=0, microsecond=0)


def _daily_reviews():
    pattern = str(LOGS_DIR / "daily_review_*.json")
    files = sorted(glob.glob(pattern))
    reviews = []
    for f in files:
        d = _load_json(f)
        if d:
            reviews.append(d)
    return reviews


def _fetch_btc_price():
    try:
        import ccxt
        ex = ccxt.bybit()
        ticker = ex.fetch_ticker("BTC/USDT")
        return ticker.get("last")
    except Exception:
        return None

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    state = _load_json(LOGS_DIR / "executor_state.json") or {}
    sentinel = _load_json(LOGS_DIR / "sentinel_state.json") or {}
    nxt = _next_cron_run()
    now = datetime.now(AEST)
    diff = (nxt - now).total_seconds()

    btc_price = _fetch_btc_price() or sentinel.get("last_price")

    active = state.get("active_trade")
    unrealized_pnl = None
    unrealized_pnl_pct = None
    if active and btc_price:
        entry = active.get("price", 0)
        amt = active.get("amount", 0)
        side = active.get("side", active.get("action", ""))
        if "buy" in str(side).lower() or "long" in str(side).lower():
            unrealized_pnl = (btc_price - entry) * amt
        else:
            unrealized_pnl = (entry - btc_price) * amt
        if entry:
            unrealized_pnl_pct = (btc_price - entry) / entry * 100
            if "sell" in str(side).lower() or "short" in str(side).lower():
                unrealized_pnl_pct = -unrealized_pnl_pct

    equity = None
    trades = state.get("trades", [])
    if trades:
        equity = trades[-1].get("equity")

    # Multi-position support
    positions = state.get("positions", {})

    return jsonify({
        "last_decision": state.get("last_decision"),
        "last_decision_time": state.get("last_decision_time"),
        "last_decision_reasoning": state.get("last_decision_reasoning"),
        "equity": equity,
        "active_trade": active,
        "positions": positions,
        "closed_trades": state.get("closed_trades", []),
        "btc_price": btc_price,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "next_cron_seconds": int(diff),
        "next_cron_time": nxt.isoformat(),
        "sentinel": sentinel,
    })


@app.route("/api/chain")
def api_chain():
    state = _latest_full_state()
    if not state:
        return jsonify({"error": "No full state log found"})

    dates = sorted(state.keys())
    if not dates:
        return jsonify({"error": "Empty state log"})
    latest = state[dates[-1]]

    agent_order = [
        ("Technical Analyst", "market_report"),
        ("Sentiment Analyst", "sentiment_report"),
        ("News Analyst", "news_report"),
        ("Fundamentals Analyst", "fundamentals_report"),
        ("Bull Researcher", "investment_debate_state.bull_history"),
        ("Bear Researcher", "investment_debate_state.bear_history"),
        ("Research Manager", "investment_debate_state.judge_decision"),
        ("Trader", "trader_investment_decision"),
        ("Aggressive Risk", "risk_debate_state.aggressive_history"),
        ("Conservative Risk", "risk_debate_state.conservative_history"),
        ("Neutral Risk", "risk_debate_state.neutral_history"),
        ("Risk Judge", "risk_debate_state.judge_decision"),
        ("Fund Manager", "final_trade_decision"),
    ]

    chain = []
    for name, path in agent_order:
        val = latest
        for key in path.split("."):
            if isinstance(val, dict):
                val = val.get(key)
            else:
                val = None
                break
        chain.append({
            "agent": name,
            "output": val if val else None,
            "has_data": bool(val),
        })

    return jsonify({"date": dates[-1], "chain": chain})


@app.route("/api/scores")
def api_scores():
    scores = _load_json(LOGS_DIR / "agent_scores.json") or {}
    attr = _load_json(LOGS_DIR / "performance_attribution.json") or {}
    return jsonify({"scores": scores, "attribution": attr})


@app.route("/api/trades")
def api_trades():
    state = _load_json(LOGS_DIR / "executor_state.json") or {}
    trades = state.get("trades", [])
    reviews = _daily_reviews()
    return jsonify({"trades": trades, "reviews": reviews})


@app.route("/api/optimization")
def api_optimization():
    recs = _load_json(LOGS_DIR / "optimization_recommendations.json") or {}
    return jsonify(recs)


@app.route("/api/apply-optimization", methods=["POST"])
def api_apply_optimization():
    recs = _load_json(LOGS_DIR / "optimization_recommendations.json")
    if not recs:
        return jsonify({"error": "No recommendations found"}), 404

    action = request.json.get("action") if request.json else None
    if action == "dismiss":
        recs["summary"]["action_required"] = False
        with open(LOGS_DIR / "optimization_recommendations.json", "w") as f:
            json.dump(recs, f, indent=2)
        return jsonify({"status": "dismissed"})

    applied = {
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "context_weights": recs.get("context_weights", {}).get("new"),
        "conviction": recs.get("conviction", {}).get("new"),
        "debate_rounds": recs.get("debate_rounds", {}).get("new"),
    }
    with open(LOGS_DIR / "applied_optimization.json", "w") as f:
        json.dump(applied, f, indent=2)
    recs["summary"]["action_required"] = False
    with open(LOGS_DIR / "optimization_recommendations.json", "w") as f:
        json.dump(recs, f, indent=2)
    return jsonify({"status": "applied", "applied": applied})

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Agents Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
:root{--bg:#1a1a2e;--card:#16213e;--accent:#0f3460;--text:#e0e0e0;--green:#00c853;--red:#ff5252;--yellow:#ffd740;--blue:#448aff;--border:#233554}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}

/* header */
.header{background:var(--accent);padding:12px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border);flex-wrap:wrap;gap:8px}
.header h1{font-size:1.3rem;font-weight:600}
.header .live-dot{width:8px;height:8px;background:var(--green);border-radius:50%;display:inline-block;margin-right:8px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* tabs */
.tabs{display:flex;background:var(--card);border-bottom:1px solid var(--border);overflow-x:auto}
.tab{padding:12px 24px;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;font-size:.9rem;transition:all .2s;user-select:none}
.tab:hover{background:var(--accent)}
.tab.active{border-bottom-color:var(--blue);color:var(--blue)}

/* layout */
.content{padding:20px;max-width:1400px;margin:0 auto}
.panel{display:none}.panel.active{display:block}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:20px;margin-bottom:16px}
.card h3{margin-bottom:12px;font-size:1rem;color:var(--blue)}
.grid{display:grid;gap:16px}
.grid-2{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.grid-3{grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
.grid-4{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}

/* big indicator */
.big-indicator{font-size:3rem;font-weight:700;text-align:center;padding:24px}
.big-indicator.buy,.big-indicator.long{color:var(--green)}
.big-indicator.sell,.big-indicator.short{color:var(--red)}
.big-indicator.hold,.big-indicator.neutral,.big-indicator.flat{color:var(--yellow)}

/* stats */
.stat-label{font-size:.75rem;text-transform:uppercase;color:#888;margin-bottom:4px}
.stat-value{font-size:1.4rem;font-weight:600}
.stat-value.positive{color:var(--green)}.stat-value.negative{color:var(--red)}

/* table */
table{width:100%;border-collapse:collapse;font-size:.85rem}
th{text-align:left;padding:10px 12px;border-bottom:2px solid var(--border);color:var(--blue);font-weight:600}
td{padding:8px 12px;border-bottom:1px solid var(--border)}
tr:hover{background:rgba(68,138,255,.05)}

/* badges */
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.75rem;font-weight:600}
.badge.star{background:#ffd74033;color:var(--yellow)}
.badge.reliable{background:#00c85333;color:var(--green)}
.badge.underperformer{background:#ff525233;color:var(--red)}
.badge.new{background:#448aff33;color:var(--blue)}

/* accordion */
.accordion{border:1px solid var(--border);border-radius:8px;margin-bottom:8px;overflow:hidden}
.accordion-header{padding:12px 16px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;background:var(--accent);transition:background .2s;user-select:none}
.accordion-header:hover{background:#0f3460cc}
.accordion-header .arrow{transition:transform .3s;font-size:.8rem}
.accordion-header.open .arrow{transform:rotate(180deg)}
.accordion-body{padding:16px;display:none;white-space:pre-wrap;font-size:.85rem;line-height:1.6;max-height:600px;overflow-y:auto;background:var(--card)}
.accordion-body.open{display:block}

/* dots */
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:8px;flex-shrink:0}
.dot.green{background:var(--green)}.dot.red{background:var(--red)}

/* reasoning */
.reasoning{white-space:pre-wrap;font-size:.85rem;line-height:1.6;max-height:300px;overflow-y:auto;padding:12px;background:var(--bg);border-radius:6px;margin-top:8px}

/* buttons */
.btn{padding:8px 20px;border:none;border-radius:6px;cursor:pointer;font-size:.85rem;font-weight:600;transition:all .2s}
.btn-primary{background:var(--blue);color:#fff}.btn-primary:hover{background:#5c9aff}
.btn-danger{background:var(--red);color:#fff}.btn-danger:hover{background:#ff7070}
.btn-group{display:flex;gap:8px;margin-top:12px}

/* misc */
.countdown{font-size:2rem;font-weight:700;font-variant-numeric:tabular-nums}
.chart-container{position:relative;height:300px;margin-top:12px}
.param-row{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)}
.param-key{color:#888}.param-val{font-weight:600}
.change-arrow{color:var(--yellow);font-weight:700}

/* positions */
.badge-committee{background:#3b82f633;color:#3b82f6;display:inline-block;padding:2px 8px;border-radius:10px;font-size:.72rem;font-weight:700}
.position-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;margin-top:8px}
.position-card{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:14px}
.exposure-bar{background:var(--accent);border-radius:6px;padding:10px 14px;margin-bottom:12px;display:flex;gap:20px;flex-wrap:wrap;font-size:.85rem}

@media(max-width:768px){
  .grid-2,.grid-3,.grid-4{grid-template-columns:1fr}
  .big-indicator{font-size:2rem}
  .content{padding:12px}
  .tab{padding:10px 16px;font-size:.8rem}
}
</style>
</head>
<body>
<div class="header">
  <h1><span class="live-dot"></span>Trading Agents Dashboard</h1>
  <span id="clock" style="font-size:.85rem;color:#888"></span>
</div>
<div class="tabs">
  <div class="tab active" data-tab="status">Live Status</div>
  <div class="tab" data-tab="chain">Decision Chain</div>
  <div class="tab" data-tab="performance">Agent Performance</div>
  <div class="tab" data-tab="trades">Trade History</div>
  <div class="tab" data-tab="optimization">Optimization</div>
</div>
<div class="content">

<!-- ═══════════ TAB 1: Live Status ═══════════ -->
<div class="panel active" id="panel-status">
  <div class="grid grid-2">
    <div class="card">
      <h3>Position</h3>
      <div id="position-indicator" class="big-indicator flat">FLAT</div>
      <div style="text-align:center;margin-top:8px">
        <div class="stat-label">Last Decision</div>
        <div id="last-decision-label" style="font-size:1.1rem;font-weight:600">—</div>
        <div id="last-decision-time" style="font-size:.75rem;color:#888;margin-top:4px">—</div>
      </div>
    </div>
    <div class="card">
      <h3>BTC Price</h3>
      <div id="btc-price" class="big-indicator" style="color:var(--blue)">—</div>
      <div class="grid grid-2" style="margin-top:8px">
        <div><div class="stat-label">Equity</div><div id="equity" class="stat-value">—</div></div>
        <div><div class="stat-label">Unrealized P&L</div><div id="unrealized-pnl" class="stat-value">—</div></div>
      </div>
    </div>
  </div>
  <div class="grid grid-3" style="margin-top:0">
    <div class="card">
      <h3>Next Cron Run</h3>
      <div id="countdown" class="countdown" style="text-align:center">—</div>
      <div id="next-cron-label" style="text-align:center;font-size:.8rem;color:#888;margin-top:4px">—</div>
      <div style="text-align:center;font-size:.7rem;color:#666;margin-top:8px">Cron: :05 past 3,7,11,15,19,23 AEST</div>
    </div>
    <div class="card">
      <h3>Sentinel Status</h3>
      <div style="display:grid;gap:8px">
        <div><div class="stat-label">Last Price Check</div><div id="sentinel-price" class="stat-value">—</div></div>
        <div><div class="stat-label">4H Close</div><div id="sentinel-4h" class="stat-value">—</div></div>
        <div><div class="stat-label">Change from 4H Close</div><div id="sentinel-change" class="stat-value">—</div></div>
        <div><div class="stat-label">Last Check</div><div id="sentinel-time" style="font-size:.8rem;color:#888">—</div></div>
      </div>
    </div>
  </div>

  <!-- Position Grid -->
  <div class="card">
    <h3>Active Positions</h3>
    <div id="exposure-bar" class="exposure-bar" style="display:none"></div>
    <div id="position-grid" class="position-grid"></div>
    <div id="no-positions" style="color:#888;font-size:.9rem;padding:8px 0">No active positions — FLAT</div>
  </div>

  <div class="card">
    <h3>Fund Manager Reasoning</h3>
    <div id="reasoning" class="reasoning">No reasoning available.</div>
  </div>
</div>

<!-- ═══════════ TAB 2: Decision Chain ═══════════ -->
<div class="panel" id="panel-chain">
  <div class="card">
    <h3>Latest Decision Chain <span id="chain-date" style="font-size:.8rem;color:#888;margin-left:8px"></span></h3>
    <div id="chain-list">Loading...</div>
  </div>
</div>

<!-- ═══════════ TAB 3: Agent Performance ═══════════ -->
<div class="panel" id="panel-performance">
  <div class="card">
    <h3>Agent Scores</h3>
    <div style="overflow-x:auto">
      <table id="scores-table">
        <thead><tr><th>Agent</th><th>Accuracy</th><th>Rolling</th><th>Calls</th><th>Streak</th><th>Status</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
  <div class="card">
    <h3>Accuracy by Agent</h3>
    <div class="chart-container"><canvas id="accuracy-chart"></canvas></div>
  </div>
</div>

<!-- ═══════════ TAB 4: Trade History ═══════════ -->
<div class="panel" id="panel-trades">
  <div class="grid grid-4">
    <div class="card"><div class="stat-label">Total Trades</div><div id="stat-total" class="stat-value">0</div></div>
    <div class="card"><div class="stat-label">Win Rate</div><div id="stat-winrate" class="stat-value">—</div></div>
    <div class="card"><div class="stat-label">Avg Win</div><div id="stat-avgwin" class="stat-value positive">—</div></div>
    <div class="card"><div class="stat-label">Avg Loss</div><div id="stat-avgloss" class="stat-value negative">—</div></div>
  </div>
  <div class="card">
    <h3>Equity Curve</h3>
    <div class="chart-container"><canvas id="equity-chart"></canvas></div>
  </div>
  <div class="card">
    <h3>Trade Log</h3>
    <div style="overflow-x:auto">
      <table id="trades-table">
        <thead><tr><th>Time</th><th>Decision</th><th>Action</th><th>Price</th><th>P&L %</th><th>Equity</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
  <div class="card">
    <h3>Daily Review Grades</h3>
    <div id="reviews-list" style="font-size:.85rem">No reviews yet.</div>
  </div>
</div>

<!-- ═══════════ TAB 5: Optimization ═══════════ -->
<div class="panel" id="panel-optimization">
  <div class="grid grid-2">
    <div class="card">
      <h3>Context Weights</h3>
      <div id="opt-weights"></div>
    </div>
    <div class="card">
      <h3>Conviction Thresholds</h3>
      <div id="opt-conviction"></div>
    </div>
  </div>
  <div class="grid grid-2">
    <div class="card">
      <h3>Debate Rounds</h3>
      <div id="opt-debate"></div>
    </div>
    <div class="card">
      <h3>Agent Hints</h3>
      <div id="opt-hints" style="font-size:.85rem">No hints.</div>
    </div>
  </div>
  <div class="card">
    <h3>Pending Recommendations</h3>
    <div id="opt-summary"></div>
    <div id="opt-actions" class="btn-group" style="display:none">
      <button class="btn btn-primary" onclick="applyOpt('apply')">Apply Changes</button>
      <button class="btn btn-danger" onclick="applyOpt('dismiss')">Dismiss</button>
    </div>
  </div>
</div>

</div>

<script>
/* ─── Tab switching ─── */
document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('panel-'+t.dataset.tab).classList.add('active');
  });
});

/* ─── Clock ─── */
function updateClock(){
  document.getElementById('clock').textContent=new Date().toLocaleString('en-AU',{timeZone:'Australia/Sydney',hour12:false});
}
setInterval(updateClock,1000);updateClock();

/* ─── Helpers ─── */
function fmt(n,d=2){return n!=null?Number(n).toFixed(d):'—'}
function fmtUsd(n){return n!=null?'$'+Number(n).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}):'—'}
function fmtPct(n){return n!=null?(n>=0?'+':'')+Number(n).toFixed(2)+'%':'—'}
function agentLabel(k){return k.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}
function escHtml(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}

/* ─── Countdown ─── */
let countdownTarget=null;
function updateCountdown(){
  if(!countdownTarget){document.getElementById('countdown').textContent='—';return}
  let diff=Math.max(0,Math.floor((countdownTarget-Date.now())/1000));
  const h=Math.floor(diff/3600);diff%=3600;
  const m=Math.floor(diff/60);const s=diff%60;
  document.getElementById('countdown').textContent=
    String(h).padStart(2,'0')+':'+String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
}
setInterval(updateCountdown,1000);

/* ─── Chart instances ─── */
let accuracyChart=null, equityChart=null;

/* ═══════════ Load Status ═══════════ */
async function loadStatus(){
  try{
    const r=await fetch('/api/status');const d=await r.json();

    /* position indicator */
    const active=d.active_trade;
    let pos='FLAT';
    if(active){
      const act=(active.action||'').toLowerCase();
      if(act.includes('buy')||act.includes('long'))pos='LONG';
      else if(act.includes('sell')||act.includes('short'))pos='SHORT';
    }
    const posEl=document.getElementById('position-indicator');
    posEl.textContent=pos;
    posEl.className='big-indicator '+pos.toLowerCase();

    /* last decision */
    document.getElementById('last-decision-label').textContent=d.last_decision||'—';
    if(d.last_decision_time){
      document.getElementById('last-decision-time').textContent=
        new Date(d.last_decision_time).toLocaleString('en-AU',{timeZone:'Australia/Sydney',hour12:false});
    }

    /* btc price */
    document.getElementById('btc-price').textContent=fmtUsd(d.btc_price);

    /* equity */
    document.getElementById('equity').textContent=fmtUsd(d.equity);

    /* unrealized P&L */
    const pnlEl=document.getElementById('unrealized-pnl');
    if(d.unrealized_pnl!=null){
      pnlEl.textContent=fmtUsd(d.unrealized_pnl)+' ('+fmtPct(d.unrealized_pnl_pct)+')';
      pnlEl.className='stat-value '+(d.unrealized_pnl>=0?'positive':'negative');
    }else{pnlEl.textContent='—';pnlEl.className='stat-value'}

    /* countdown */
    if(d.next_cron_seconds){
      countdownTarget=Date.now()+d.next_cron_seconds*1000;
      document.getElementById('next-cron-label').textContent=
        'Next: '+new Date(d.next_cron_time).toLocaleString('en-AU',{timeZone:'Australia/Sydney',hour12:false})+' AEST';
    }

    /* sentinel */
    const s=d.sentinel||{};
    document.getElementById('sentinel-price').textContent=fmtUsd(s.last_price);
    document.getElementById('sentinel-4h').textContent=fmtUsd(s.last_4h_close);
    const chEl=document.getElementById('sentinel-change');
    chEl.textContent=fmtPct(s.last_change_pct);
    chEl.className='stat-value '+((s.last_change_pct||0)>=0?'positive':'negative');
    if(s.last_check){
      document.getElementById('sentinel-time').textContent=
        new Date(s.last_check).toLocaleString('en-AU',{timeZone:'Australia/Sydney',hour12:false});
    }

    /* ── Position Grid ── */
    const positions=d.positions||{};
    const posEntries=Object.entries(positions);
    const posGrid=document.getElementById('position-grid');
    const noPosEl=document.getElementById('no-positions');
    const expBar=document.getElementById('exposure-bar');
    posGrid.innerHTML='';
    if(posEntries.length===0){
      noPosEl.style.display='block';expBar.style.display='none';
    }else{
      noPosEl.style.display='none';
      expBar.style.display='flex';
      const totalMargin=posEntries.reduce((s,[,p])=>s+(p.margin_pct||0),0);
      expBar.innerHTML=`<span><strong>Positions:</strong> ${posEntries.length}</span><span><strong>Total Margin:</strong> ${fmt(totalMargin,1)}%</span><span><strong>Active:</strong> ${posEntries.map(([id])=>id).join(', ')}</span>`;
      const now=Date.now();
      posEntries.forEach(([posId,pos])=>{
        const badge='<span class="badge-committee">COMMITTEE</span>';
        const dir=(pos.direction||pos.side||'').toUpperCase();
        const entry=pos.entry_price||pos.price||0;
        let pnlPct=pos.pnl_pct;
        if(pnlPct==null&&d.btc_price&&entry){
          pnlPct=(d.btc_price-entry)/entry*100;
          if(dir.includes('SHORT'))pnlPct=-pnlPct;
        }
        const pnlColor=pnlPct==null?'':pnlPct>=0?'color:#00c853':'color:#ff5252';
        const tp1Hit=pos.tp1_hit?'✅':'';const tp2Hit=pos.tp2_hit?'✅':'';
        let dur='—';
        if(pos.opened_at){const ms=now-new Date(pos.opened_at).getTime();const h=Math.floor(ms/3600000);const m=Math.floor((ms%3600000)/60000);dur=h>0?`${h}h ${m}m`:`${m}m`;}
        posGrid.insertAdjacentHTML('beforeend',`
          <div class="position-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
              ${badge}<span style="font-size:.8rem;color:#888">${dur}</span>
            </div>
            <div class="param-row"><span class="param-key">Direction</span><span class="param-val" style="color:${dir.includes('LONG')?'var(--green)':'var(--red)'}">${dir||'—'}</span></div>
            <div class="param-row"><span class="param-key">Entry</span><span class="param-val">${fmtUsd(entry||null)}</span></div>
            <div class="param-row"><span class="param-key">PnL%</span><span class="param-val" style="${pnlColor}">${fmtPct(pnlPct)}</span></div>
            <div class="param-row"><span class="param-key">Stop Loss</span><span class="param-val">${fmtUsd(pos.stop_loss)}</span></div>
            <div class="param-row"><span class="param-key">TP1</span><span class="param-val">${fmtUsd(pos.tp1)} ${tp1Hit}</span></div>
            <div class="param-row"><span class="param-key">TP2</span><span class="param-val">${fmtUsd(pos.tp2)} ${tp2Hit}</span></div>
            <div class="param-row"><span class="param-key">Trail Stop</span><span class="param-val">${fmtUsd(pos.trail_stop)}</span></div>
          </div>`);
      });
    }

    /* reasoning */
    document.getElementById('reasoning').textContent=d.last_decision_reasoning||'No reasoning available.';
  }catch(e){console.error('Status error:',e)}
}

/* ═══════════ Load Decision Chain ═══════════ */
async function loadChain(){
  try{
    const r=await fetch('/api/chain');const d=await r.json();
    if(d.error){document.getElementById('chain-list').textContent=d.error;return}
    document.getElementById('chain-date').textContent=d.date||'';
    const container=document.getElementById('chain-list');
    container.innerHTML='';
    (d.chain||[]).forEach(item=>{
      const acc=document.createElement('div');
      acc.className='accordion';
      const dot=item.has_data?'<span class="dot green"></span>':'<span class="dot red"></span>';
      acc.innerHTML=`
        <div class="accordion-header" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">
          <span>${dot}${escHtml(item.agent)}</span>
          <span class="arrow">&#9660;</span>
        </div>
        <div class="accordion-body">${item.output?escHtml(item.output):'<em style="color:#888">No data available</em>'}</div>`;
      container.appendChild(acc);
    });
  }catch(e){console.error('Chain error:',e)}
}

/* ═══════════ Load Agent Performance ═══════════ */
async function loadScores(){
  try{
    const r=await fetch('/api/scores');const d=await r.json();
    const agents=(d.scores||{}).agents||{};
    const tbody=document.querySelector('#scores-table tbody');
    tbody.innerHTML='';
    const labels=[],accData=[];

    /* find primary failure agent */
    let maxFail=0,failAgent='';
    Object.entries(agents).forEach(([name,a])=>{
      if((a.primary_failure_count||0)>maxFail){maxFail=a.primary_failure_count;failAgent=name}
    });

    Object.entries(agents).forEach(([name,a])=>{
      const streakStr=a.streak>0?'+'+a.streak:String(a.streak);
      const isPrimary=name===failAgent&&maxFail>0;
      const row=document.createElement('tr');
      row.innerHTML=`
        <td>${agentLabel(name)}${isPrimary?' <span style="color:var(--red);font-size:.7rem">PRIMARY FAILURE</span>':''}</td>
        <td>${fmt(a.accuracy_pct,1)}%</td>
        <td>${fmt(a.rolling_accuracy,1)}%</td>
        <td>${a.total_calls}</td>
        <td style="color:${a.streak>0?'var(--green)':a.streak<0?'var(--red)':'var(--text)'}">${streakStr}</td>
        <td><span class="badge ${a.status||'new'}">${a.status||'new'}</span></td>`;
      tbody.appendChild(row);
      labels.push(agentLabel(name));
      accData.push(a.accuracy_pct);
    });

    /* chart */
    const ctx=document.getElementById('accuracy-chart').getContext('2d');
    if(accuracyChart)accuracyChart.destroy();
    accuracyChart=new Chart(ctx,{
      type:'bar',
      data:{labels,datasets:[{
        label:'Accuracy %',data:accData,
        backgroundColor:accData.map(v=>v>=60?'#00c85388':v>=40?'#ffd74088':'#ff525288'),
        borderColor:accData.map(v=>v>=60?'#00c853':v>=40?'#ffd740':'#ff5252'),
        borderWidth:1
      }]},
      options:{
        responsive:true,maintainAspectRatio:false,
        scales:{y:{beginAtZero:true,max:100,ticks:{color:'#888'},grid:{color:'#233554'}},
                x:{ticks:{color:'#888',maxRotation:45},grid:{display:false}}},
        plugins:{legend:{display:false}}
      }
    });
  }catch(e){console.error('Scores error:',e)}
}

/* ═══════════ Load Trades ═══════════ */
async function loadTrades(){
  try{
    const r=await fetch('/api/trades');const d=await r.json();
    const trades=d.trades||[];

    /* stats */
    const closed=trades.filter(t=>t.close_pnl_pct!=null);
    const wins=closed.filter(t=>t.close_pnl_pct>0);
    const losses=closed.filter(t=>t.close_pnl_pct<=0);

    document.getElementById('stat-total').textContent=trades.length;
    document.getElementById('stat-winrate').textContent=
      closed.length?fmt(wins.length/closed.length*100,1)+'%':'—';
    document.getElementById('stat-avgwin').textContent=
      wins.length?fmtPct(wins.reduce((s,t)=>s+t.close_pnl_pct,0)/wins.length):'—';
    document.getElementById('stat-avgloss').textContent=
      losses.length?fmtPct(losses.reduce((s,t)=>s+t.close_pnl_pct,0)/losses.length):'—';

    /* profit factor */
    const totalWin=wins.reduce((s,t)=>s+Math.abs(t.close_pnl_pct),0);
    const totalLoss=losses.reduce((s,t)=>s+Math.abs(t.close_pnl_pct),0);

    /* table */
    const tbody=document.querySelector('#trades-table tbody');
    tbody.innerHTML='';
    trades.forEach(t=>{
      const pnl=t.close_pnl_pct;
      const pnlStr=pnl!=null?fmtPct(pnl):'—';
      const pnlClass=pnl!=null?(pnl>0?'color:var(--green)':'color:var(--red)'):'';
      const row=document.createElement('tr');
      row.innerHTML=`
        <td>${t.timestamp?new Date(t.timestamp).toLocaleString('en-AU',{timeZone:'Australia/Sydney',hour12:false,month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'—'}</td>
        <td>${t.decision||'—'}</td>
        <td>${t.action||t.transition||'—'}</td>
        <td>${fmtUsd(t.price)}</td>
        <td style="${pnlClass}">${pnlStr}</td>
        <td>${fmtUsd(t.equity)}</td>`;
      tbody.appendChild(row);
    });

    /* equity chart */
    const eqData=trades.filter(t=>t.equity).map(t=>({
      x:t.timestamp,
      y:t.equity,
      label:new Date(t.timestamp).toLocaleDateString('en-AU',{timeZone:'Australia/Sydney',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false})
    }));
    const ctx=document.getElementById('equity-chart').getContext('2d');
    if(equityChart)equityChart.destroy();
    if(eqData.length>0){
      equityChart=new Chart(ctx,{
        type:'line',
        data:{
          labels:eqData.map(p=>p.label),
          datasets:[{
            label:'Equity',data:eqData.map(p=>p.y),
            borderColor:'#448aff',backgroundColor:'#448aff22',fill:true,tension:.3,
            pointRadius:3,pointBackgroundColor:'#448aff'
          }]
        },
        options:{
          responsive:true,maintainAspectRatio:false,
          scales:{y:{ticks:{color:'#888',callback:v=>'$'+v.toLocaleString()},grid:{color:'#233554'}},
                  x:{ticks:{color:'#888',maxRotation:45},grid:{display:false}}},
          plugins:{legend:{display:false}}
        }
      });
    }

    /* reviews */
    const reviews=d.reviews||[];
    const revEl=document.getElementById('reviews-list');
    if(reviews.length){
      revEl.innerHTML=reviews.map(rv=>{
        const grade=rv.overall_quality||rv.error||'unknown';
        const color=grade==='good'?'var(--green)':grade==='degraded'?'var(--yellow)':'var(--red)';
        return `<div class="param-row"><span class="param-key">${rv.date||'—'}</span><span class="param-val" style="color:${color}">${grade}</span></div>`;
      }).join('');
    }
  }catch(e){console.error('Trades error:',e)}
}

/* ═══════════ Load Optimization ═══════════ */
async function loadOptimization(){
  try{
    const r=await fetch('/api/optimization');const d=await r.json();

    /* weights */
    const wEl=document.getElementById('opt-weights');
    const wNew=(d.context_weights||{}).new||{};
    const wOld=(d.context_weights||{}).old||{};
    wEl.innerHTML=Object.entries(wNew).map(([k,v])=>{
      const old=wOld[k];const changed=old!=null&&old!==v;
      return `<div class="param-row"><span class="param-key">${agentLabel(k)}</span><span class="param-val">${v}${changed?' <span class="change-arrow">&larr; '+old+'</span>':''}</span></div>`;
    }).join('')||'<span style="color:#888">No data</span>';

    /* conviction */
    const cEl=document.getElementById('opt-conviction');
    const cNew=(d.conviction||{}).new||{};
    const cOld=(d.conviction||{}).old||{};
    cEl.innerHTML=Object.entries(cNew).map(([k,v])=>{
      const old=cOld[k];const changed=old!=null&&old!==v;
      return `<div class="param-row"><span class="param-key">${agentLabel(k)}</span><span class="param-val">${v}${changed?' <span class="change-arrow">&larr; '+old+'</span>':''}</span></div>`;
    }).join('')||'<span style="color:#888">No data</span>';

    /* debate rounds */
    const dEl=document.getElementById('opt-debate');
    const dNew=(d.debate_rounds||{}).new||{};
    const dOld=(d.debate_rounds||{}).old||{};
    dEl.innerHTML=Object.entries(dNew).map(([k,v])=>{
      const old=dOld[k];const changed=old!=null&&old!==v;
      return `<div class="param-row"><span class="param-key">${agentLabel(k)}</span><span class="param-val">${v}${changed?' <span class="change-arrow">&larr; '+old+'</span>':''}</span></div>`;
    }).join('')||'<span style="color:#888">No data</span>';

    /* hints */
    const hints=d.agent_hints||{};
    const hEl=document.getElementById('opt-hints');
    const hEntries=Object.entries(hints);
    if(hEntries.length){
      hEl.innerHTML=hEntries.map(([k,v])=>`<div class="param-row"><span class="param-key">${agentLabel(k)}</span><span class="param-val">${escHtml(String(v))}</span></div>`).join('');
    }else{hEl.textContent='No hints.'}

    /* summary */
    const sum=d.summary||{};
    document.getElementById('opt-summary').innerHTML=`
      <div class="param-row"><span class="param-key">Parameter Changes</span><span class="param-val">${sum.total_parameter_changes||0}</span></div>
      <div class="param-row"><span class="param-key">Agent Hints</span><span class="param-val">${sum.total_agent_hints||0}</span></div>
      <div class="param-row"><span class="param-key">Action Required</span><span class="param-val" style="color:${sum.action_required?'var(--yellow)':'var(--green)'}">${sum.action_required?'Yes':'No'}</span></div>`;
    document.getElementById('opt-actions').style.display=sum.action_required?'flex':'none';
  }catch(e){console.error('Optimization error:',e)}
}

async function applyOpt(action){
  try{
    const r=await fetch('/api/apply-optimization',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action})
    });
    const d=await r.json();
    alert(d.status==='applied'?'Optimization applied!':d.status==='dismissed'?'Dismissed.':'Error: '+(d.error||'unknown'));
    loadOptimization();
  }catch(e){alert('Error: '+e.message)}
}

/* ═══════════ Initial load ═══════════ */
loadStatus();
loadChain();
loadScores();
loadTrades();
loadOptimization();

/* auto-refresh status tab every 60s */
setInterval(loadStatus,60000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return DASHBOARD_HTML


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("  Trading Agents Dashboard")
    print("=" * 50)
    print("  http://localhost:5555")
    print("=" * 50)
    print()
    app.run(host="0.0.0.0", port=5555, debug=True)

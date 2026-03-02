#!/usr/bin/env python3
"""
Review & Optimization Runner

Runs after trades close (or on-demand) to:
A. Review each trade through the full decision chain
B. Score agent performance and flag underperformers
C. Generate parameter optimization recommendations

Usage:
    python scripts/run_review.py                 # Review all trades
    python scripts/run_review.py --report        # Just show current scores
    python scripts/run_review.py --apply         # Apply latest recommendations
"""

import sys
import os
import json
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.review import TradeReviewer, PerformanceScorer, ParameterOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("review_runner")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_llm():
    """Get the review LLM via OpenClaw gateway (same as trading agents)."""
    from tradingagents.llm_clients.factory import create_llm_client
    client = create_llm_client(provider="anthropic", model="claude-sonnet-4-20250514")
    return client.get_llm()


def run_full_review():
    """Run the complete review pipeline."""
    llm = get_llm()
    
    reviewer = TradeReviewer(llm, PROJECT_DIR)
    scorer = PerformanceScorer(PROJECT_DIR)
    optimizer = ParameterOptimizer(PROJECT_DIR)

    # A. Review all trades
    log.info("📋 Running trade reviews...")
    reviews = reviewer.review_all_trades()
    
    if not reviews:
        log.info("No closed trades to review yet.")
        return
    
    log.info(f"Reviewed {len(reviews)} trades")
    
    # Save reviews
    reviews_file = os.path.join(PROJECT_DIR, "logs", "trade_reviews.json")
    with open(reviews_file, "w") as f:
        json.dump(reviews, f, indent=2)

    # B. Score agents
    log.info("📊 Scoring agent performance...")
    for review in reviews:
        scorer.record_trade(review)
    
    report = scorer.get_report()
    log.info(f"Agent scores updated ({report['trades_scored']} total trades)")
    
    # Print rankings
    if report["rankings"]:
        log.info("\n📈 Agent Rankings:")
        for r in report["rankings"]:
            status_icon = {"star": "⭐", "reliable": "✅", "underperformer": "⚠️", "new": "🆕"}.get(r["status"], "")
            log.info(f"  {status_icon} {r['agent']}: {r['accuracy']:.0f}% accuracy ({r['calls']} calls, streak {r['streak']:+d})")
    
    if report["underperformers"]:
        log.warning(f"⚠️ Underperformers: {', '.join(report['underperformers'])}")
    if report["stars"]:
        log.info(f"⭐ Stars: {', '.join(report['stars'])}")

    # C. Generate optimization recommendations
    log.info("🔧 Generating optimization recommendations...")
    summary = reviewer.generate_summary(reviews)
    recommendations = optimizer.run_full_optimization(reviews, report)
    
    # Print recommendations
    for key in ["context_weights", "conviction", "debate_rounds"]:
        rec = recommendations.get(key, {})
        changes = rec.get("changes", [])
        if changes:
            log.info(f"\n{'='*50}")
            log.info(f"📌 {key.upper()} recommendations:")
            for change in changes:
                log.info(f"  → {change}")

    hints = recommendations.get("agent_hints", {})
    if hints:
        log.info(f"\n{'='*50}")
        log.info("📝 Agent hints:")
        for agent, hint in hints.items():
            log.info(f"  {agent}: {hint[:100]}...")

    action_required = recommendations.get("summary", {}).get("action_required", False)
    if action_required:
        log.info("\n⚡ Action required! Review recommendations in logs/optimization_recommendations.json")
        log.info("   Apply with: python scripts/run_review.py --apply")
    else:
        log.info("\n✅ No parameter changes recommended at this time.")

    # Print trade summary
    log.info(f"\n{'='*50}")
    log.info(f"📊 Trade Summary:")
    log.info(f"  Total: {summary['total_trades']} | Wins: {summary['wins']} | Losses: {summary['losses']}")
    log.info(f"  Win rate: {summary['win_rate']:.0f}% | Total P&L: {summary['total_pnl_pct']:+.2f}%")
    if summary.get("most_common_failure_point") != "none":
        log.info(f"  Most common failure: {summary['most_common_failure_point']}")
    
    if summary.get("lessons"):
        log.info(f"\n📚 Lessons learned:")
        for lesson in summary["lessons"][:5]:
            log.info(f"  • {lesson[:120]}")


def show_report():
    """Show current agent performance report."""
    scorer = PerformanceScorer(PROJECT_DIR)
    report = scorer.get_report()
    
    if report["trades_scored"] == 0:
        print("No trades scored yet. Run a full review first.")
        return

    print(f"\n{'='*60}")
    print(f"AGENT PERFORMANCE REPORT ({report['trades_scored']} trades)")
    print(f"Last updated: {report['last_updated']}")
    print(f"{'='*60}")
    
    if report["rankings"]:
        print(f"\n{'Agent':<25} {'Accuracy':>8} {'Rolling':>8} {'Calls':>6} {'Streak':>7} {'Status':>14}")
        print("-" * 70)
        for r in report["rankings"]:
            print(f"{r['agent']:<25} {r['accuracy']:>7.0f}% {r['rolling']:>7.0f}% {r['calls']:>6} {r['streak']:>+7} {r['status']:>14}")
    
    if report["underperformers"]:
        print(f"\n⚠️  Underperformers: {', '.join(report['underperformers'])}")
    if report["stars"]:
        print(f"⭐ Stars: {', '.join(report['stars'])}")
    print(f"🔴 Most common failure point: {report['worst_failure_point']}")


def apply_recommendations():
    """Apply the latest optimization recommendations."""
    optimizer = ParameterOptimizer(PROJECT_DIR)
    rec_file = os.path.join(PROJECT_DIR, "logs", "optimization_recommendations.json")
    
    if not os.path.exists(rec_file):
        print("No recommendations file found. Run a full review first.")
        return

    with open(rec_file) as f:
        recommendations = json.load(f)

    print(f"Recommendations generated: {recommendations.get('generated_at', 'unknown')}")
    
    total_changes = recommendations.get("summary", {}).get("total_parameter_changes", 0)
    if total_changes == 0:
        print("No parameter changes to apply.")
        return

    print(f"\n{total_changes} parameter changes to apply:")
    for key in ["context_weights", "conviction", "debate_rounds"]:
        for change in recommendations.get(key, {}).get("changes", []):
            print(f"  → {change}")

    optimizer.apply_recommendations(recommendations)
    print("\n✅ Recommendations applied.")




def run_daily_review(date_str: str = None):
    """Review all agent decisions from a specific day, regardless of trade outcome.
    
    Unlike run_full_review() which only reviews CLOSED trades, this reviews
    every decision cycle from the day — including HOLDs and open positions.
    Useful for catching bad reasoning even when no trade triggered.
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    
    SYDNEY_TZ = ZoneInfo("Australia/Sydney")
    
    if not date_str:
        # Default to yesterday
        yesterday = datetime.now(SYDNEY_TZ) - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")
    
    llm = get_llm()
    reviewer = TradeReviewer(llm, PROJECT_DIR)
    
    # Load agent state for the day
    agent_state = reviewer.load_state_for_date(date_str)
    if not agent_state:
        log.info(f"No agent state found for {date_str}")
        return None
    
    # Load agent reports if available
    reports_file = os.path.join(PROJECT_DIR, "logs", f"agent_reports_{date_str}.json")
    agent_reports = {}
    if os.path.exists(reports_file):
        with open(reports_file) as f:
            agent_reports = json.load(f)
    
    # Load executor log for context
    exec_log_file = os.path.join(PROJECT_DIR, "logs", f"executor_{date_str}.log")
    exec_log_tail = ""
    if os.path.exists(exec_log_file):
        with open(exec_log_file) as f:
            lines = f.readlines()
            exec_log_tail = "".join(lines[-50:])  # Last 50 lines
    
    # Extract decisions from the day
    decision = agent_state.get("final_trade_decision", "")[:2000]
    fm_decision = agent_state.get("fund_manager_decision", "")[:1500]
    trader_plan = agent_state.get("trader_investment_plan", "")[:1500]
    investment_plan = agent_state.get("investment_plan", "")[:1500]
    
    # Build chain for review
    chain = {
        "market_report": agent_state.get("market_report", "")[:1000],
        "sentiment_report": agent_state.get("sentiment_report", "")[:1000],
        "news_report": agent_state.get("news_report", "")[:1000],
        "fundamentals_report": agent_state.get("fundamentals_report", "")[:1000],
        "research_debate": agent_state.get("investment_debate_state", {}).get("history", "")[:1500],
        "research_manager": investment_plan,
        "trader": trader_plan,
        "risk_debate": agent_state.get("risk_debate_state", {}).get("history", "")[:1500],
        "risk_judge": agent_state.get("risk_debate_state", {}).get("judge_decision", "")[:1000],
        "fund_manager": fm_decision or decision,
    }
    
    chain_text = ""
    for agent, report in chain.items():
        if report:
            chain_text += f"\n### {agent.replace('_', ' ').title()}\n{report[:1000]}\n"
    
    prompt = f"""Review the trading system's decisions from {date_str}. This is a DAILY REVIEW — 
we're checking decision quality regardless of whether a trade was placed.

## Decision Chain
{chain_text}

## Executor Log (last 50 lines)
{exec_log_tail[:2000]}

## Your Analysis

Evaluate:
1. **Data Quality**: Did analysts produce useful, actionable reports? Any data gaps or failures?
2. **Debate Quality**: Was the bull/bear debate substantive? Did it add value or just pad tokens?
3. **Decision Logic**: Was the trader's reasoning sound? Did they follow the material change framework?
4. **Risk Assessment**: Did the risk team catch real risks or just rubber-stamp?
5. **Final Decision**: Was the FM's approval/rejection justified?
6. **Missed Opportunities**: Should the system have acted but didn't? Or did it correctly hold?

## Required Output
```json
{{
    "date": "{date_str}",
    "overall_quality": "excellent|good|adequate|poor|critical",
    "decision_made": "LONG|SHORT|HOLD|UNKNOWN",
    "data_quality": {{
        "technical": "good|degraded|failed",
        "sentiment": "good|degraded|failed",
        "news": "good|degraded|failed",
        "fundamentals": "good|degraded|failed"
    }},
    "debate_added_value": true,
    "trader_reasoning_sound": true,
    "risk_assessment_thorough": true,
    "fm_justified": true,
    "issues": ["issue 1", "issue 2"],
    "improvements": ["improvement 1", "improvement 2"],
    "summary": "2-3 sentence summary of the day's decision quality"
}}
```
Respond with ONLY the JSON."""

    try:
        result = llm.invoke(prompt)
        import re
        text = result.content if hasattr(result, 'content') else str(result)
        text = re.sub(r'^```json?\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text.strip())
        daily = json.loads(text)
    except Exception as e:
        log.error(f"Daily review failed: {e}")
        daily = {"date": date_str, "error": str(e), "overall_quality": "unknown"}
    
    # Save daily review
    daily_file = os.path.join(PROJECT_DIR, "logs", f"daily_review_{date_str}.json")
    with open(daily_file, "w") as f:
        json.dump(daily, f, indent=2)
    
    # Print summary
    quality = daily.get("overall_quality", "unknown")
    quality_icon = {"excellent": "🌟", "good": "✅", "adequate": "🟡", "poor": "⚠️", "critical": "🔴"}.get(quality, "❓")
    
    log.info(f"\n{'='*60}")
    log.info(f"DAILY REVIEW: {date_str} {quality_icon} {quality.upper()}")
    log.info(f"{'='*60}")
    log.info(f"Decision: {daily.get('decision_made', 'UNKNOWN')}")
    
    dq = daily.get("data_quality", {})
    for analyst, status in dq.items():
        icon = {"good": "✅", "degraded": "⚠️", "failed": "🔴"}.get(status, "❓")
        log.info(f"  {icon} {analyst}: {status}")
    
    log.info(f"Debate value: {'✅' if daily.get('debate_added_value') else '❌'}")
    log.info(f"Trader reasoning: {'✅' if daily.get('trader_reasoning_sound') else '❌'}")
    log.info(f"Risk thorough: {'✅' if daily.get('risk_assessment_thorough') else '❌'}")
    log.info(f"FM justified: {'✅' if daily.get('fm_justified') else '❌'}")
    
    issues = daily.get("issues", [])
    if issues:
        log.info(f"\n🔴 Issues:")
        for issue in issues:
            log.info(f"  • {issue}")
    
    improvements = daily.get("improvements", [])
    if improvements:
        log.info(f"\n💡 Improvements:")
        for imp in improvements:
            log.info(f"  • {imp}")
    
    summary = daily.get("summary", "")
    if summary:
        log.info(f"\n📋 {summary}")
    
    return daily


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade Review & Optimization")
    parser.add_argument("--report", action="store_true", help="Show current agent scores")
    parser.add_argument("--apply", action="store_true", help="Apply latest recommendations")
    parser.add_argument("--daily", nargs="?", const="", default=None, metavar="YYYY-MM-DD", help="Daily decision review (default: yesterday)")
    args = parser.parse_args()

    if args.report:
        show_report()
    elif args.apply:
        apply_recommendations()
    elif args.daily is not None:
        run_daily_review(args.daily if args.daily else None)
    else:
        run_full_review()

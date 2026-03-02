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
    """Get the review LLM (quick-thinking model)."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_DIR, ".env"))
    
    backend_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:18789/v1")
    
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="claude-sonnet-4-20250514",
        base_url=backend_url,
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        temperature=0.1,
    )


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade Review & Optimization")
    parser.add_argument("--report", action="store_true", help="Show current agent scores")
    parser.add_argument("--apply", action="store_true", help="Apply latest recommendations")
    args = parser.parse_args()

    if args.report:
        show_report()
    elif args.apply:
        apply_recommendations()
    else:
        run_full_review()

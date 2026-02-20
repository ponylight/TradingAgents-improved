"""
Entry point for running the Crypto Trading Agents on BTC/USDT.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from tradingagents.graph.crypto_trading_graph import CryptoTradingAgentsGraph, CRYPTO_DEFAULT_CONFIG


def main():
    # Configure for Anthropic
    config = CRYPTO_DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["deep_think_llm"] = "claude-sonnet-4-20250514"
    config["quick_think_llm"] = "claude-sonnet-4-20250514"

    print("🚀 Initializing Crypto Trading Agents...")
    print(f"   Symbol: BTC/USDT")
    print(f"   LLM: {config['llm_provider']} / {config['deep_think_llm']}")
    print()

    # Initialize the graph
    ta = CryptoTradingAgentsGraph(
        selected_analysts=["market", "sentiment", "macro", "news"],
        config=config,
        debug=True,
    )

    # Run analysis for today
    trade_date = "2026-02-20"
    print(f"📊 Running analysis for {trade_date}...")
    print("=" * 60)

    final_state, decision = ta.propagate("BTC/USDT", trade_date)

    print("\n" + "=" * 60)
    print(f"🎯 FINAL DECISION: {decision}")
    print("=" * 60)

    # Print reports
    print("\n📈 Market Report:")
    print(final_state.get("market_report", "N/A")[:500])
    
    print("\n😊 Sentiment Report:")
    print(final_state.get("sentiment_report", "N/A")[:500])
    
    print("\n🌍 Macro Report:")
    print(final_state.get("news_report", "N/A")[:500])

    print("\n📋 Final Trade Decision:")
    print(final_state.get("final_trade_decision", "N/A"))


if __name__ == "__main__":
    main()

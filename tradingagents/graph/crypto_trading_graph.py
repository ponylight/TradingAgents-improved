"""
CryptoTradingAgentsGraph — Adapted trading graph for BTC/USDT on Bybit.
Uses crypto-specific analysts instead of stock-focused ones.
"""

import os
import time as _time
import logging as _logging
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

_graph_log = _logging.getLogger("executor.graph")

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.fallback import FallbackLLM
from tradingagents.agents import *
from tradingagents.agents.managers.fund_manager import create_fund_manager
from tradingagents.utils.telemetry import wrap_with_telemetry
from tradingagents.agents.analysts.crypto_market_analyst import create_crypto_market_analyst
from tradingagents.agents.analysts.crypto_sentiment_analyst import create_crypto_sentiment_analyst
from tradingagents.agents.analysts.macro_analyst import create_macro_analyst
from tradingagents.agents.analysts.crypto_fundamentals_analyst import create_crypto_fundamentals_analyst
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.dataflows.config import set_config

# Crypto tools
from tradingagents.agents.utils.crypto_tools import (
    get_crypto_price_data,
    get_crypto_technical_indicators,
    get_funding_rate,
    get_open_interest,
    get_oi_timeseries,
    get_orderbook_depth,
    get_crypto_fear_greed,
    get_liquidation_info,
    get_macro_signal_radar,
    get_stablecoin_peg_health,
    )

# Macro tools
from tradingagents.agents.utils.macro_tools import (
    get_dollar_index,
    get_yields,
    get_sp500,
    get_economic_data,
    get_economic_calendar,
)
from tradingagents.agents.analysts.crypto_fundamentals_analyst import get_onchain_fundamentals
from tradingagents.dataflows.crypto_technical_brief import build_crypto_technical_brief

# News tools (reuse existing for crypto news)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_global_news,
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


# Default config for crypto trading
CRYPTO_DEFAULT_CONFIG = {
    **DEFAULT_CONFIG,
    "llm_provider": "anthropic",
    "deep_think_llm": "claude-opus-4-6",
    "quick_think_llm": "claude-sonnet-4-20250514",
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    # Crypto-specific settings
    "crypto": {
        "symbol": "BTC/USDT",
        "exchange": "bybit",
        "timeframe": "4h",
        "max_leverage": 10,
        "leverage_tiers": {
            "low": {"leverage": 2, "position_pct": 5},
            "medium": {"leverage": 5, "position_pct": 10},
            "high": {"leverage": 10, "position_pct": 15},
        },
        "risk_limits": {
            "max_daily_loss_pct": 5,
            "max_drawdown_pct": 15,
            "max_consecutive_losses": 3,
            "cooldown_after_liquidation_hours": 48,
        },
    },
}


class CryptoTradingAgentsGraph:
    """Main class that orchestrates the crypto trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "sentiment", "fundamentals", "news"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        self.debug = debug
        self.config = config or CRYPTO_DEFAULT_CONFIG
        self.callbacks = callbacks or []

        set_config(self.config)

        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs
        llm_kwargs = {}
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        deep_llm = deep_client.get_llm()
        quick_llm = quick_client.get_llm()

        # Wrap with fallback to OpenRouter/Minimax M2.5 on rate limits
        fallback_provider = self.config.get("fallback_provider")
        fallback_model = self.config.get("fallback_model")
        if fallback_provider and fallback_model:
            fallback_client = create_llm_client(
                provider=fallback_provider,
                model=fallback_model,
                **llm_kwargs,
            )
            fb_llm = fallback_client.get_llm()
            fb_label = f"{fallback_provider}/{fallback_model}"
            deep_llm = FallbackLLM(primary=deep_llm, fallback=fb_llm, fallback_label=fb_label)
            quick_llm = FallbackLLM(primary=quick_llm, fallback=fb_llm, fallback_label=fb_label)

        self.deep_thinking_llm = wrap_with_telemetry(deep_llm, "deep/opus")
        self.quick_thinking_llm = wrap_with_telemetry(quick_llm, "quick/sonnet")

        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)
        self.fund_manager_memory = FinancialSituationMemory("fund_manager_memory", self.config)

        # Create crypto-specific tool nodes
        self.tool_nodes = self._create_crypto_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic()

        # Use the standard GraphSetup but with crypto tool nodes
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}

        # Set up the crypto graph
        self.graph = self._setup_crypto_graph(selected_analysts)

    def _create_crypto_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create crypto-specific tool nodes with clean ownership.
        
        Technical Analyst: TechnicalBrief (pre-computed, minimal tools)
        Sentiment Analyst: positioning + social (funding, OI, Fear & Greed)
        Fundamentals Analyst: on-chain data
        News Analyst: news + macro (DXY, yields, CPI, economic calendar)
        """
        return {
            "market": ToolNode([
                get_crypto_price_data,
                get_orderbook_depth,
            ]),
            "sentiment": ToolNode([
                get_crypto_fear_greed,
                get_funding_rate,
                get_open_interest,
                get_oi_timeseries,
            ]),
            "fundamentals": ToolNode([
                get_onchain_fundamentals,
                get_macro_signal_radar,
                get_stablecoin_peg_health,
                # GDELT CII removed — too many 429s, low decision value
            ]),
            "news": ToolNode([
                get_news,
                get_global_news,
                get_dollar_index,
                get_yields,
                get_sp500,
                get_economic_data,
                get_economic_calendar,
            ]),
        }

    def _setup_crypto_graph(self, selected_analysts):
        """Set up the crypto trading agent graph."""
        from langgraph.graph import END, StateGraph, START

        # Create analyst nodes mapping
        analyst_creators = {
            "market": create_crypto_market_analyst,
            "sentiment": create_crypto_sentiment_analyst,
            "fundamentals": create_crypto_fundamentals_analyst,
            "news": create_news_analyst,
        }

        # Report field mapping — each analyst writes to its correct field
        report_fields = {
            "market": "market_report",
            "sentiment": "sentiment_report",
            "fundamentals": "fundamentals_report",
            "news": "news_report",
        }

        analyst_nodes = {}
        tool_nodes = {}

        for analyst_type in selected_analysts:
            if analyst_type in analyst_creators:
                analyst_nodes[analyst_type] = analyst_creators[analyst_type](
                    self.quick_thinking_llm
                )
                tool_nodes[analyst_type] = self.tool_nodes[analyst_type]

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(
            self.deep_thinking_llm, self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self.deep_thinking_llm, self.bear_memory
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory
        )
        trader_node = create_trader(self.deep_thinking_llm, self.trader_memory)

        # Risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        risk_manager_node = create_risk_manager(
            self.deep_thinking_llm, self.risk_manager_memory
        )
        # Fund manager can use a separate LLM (e.g. GPT-5.4 for financial reasoning)
        fund_manager_llm = self.deep_thinking_llm
        fm_model = self.config.get("fund_manager_llm")
        if fm_model:
            fm_provider = self.config.get("fund_manager_llm_provider", "openrouter")
            import logging as _logging
            _logging.getLogger(__name__).info(f"🏦 Fund manager using dedicated LLM: {fm_model} via {fm_provider}")
            from tradingagents.llm_clients.factory import create_llm_client
            fm_client = create_llm_client(fm_provider, fm_model)
            fund_manager_llm = fm_client.get_llm()
            fund_manager_llm = wrap_with_telemetry(fund_manager_llm, f"fund_manager/{fm_model.split('/')[-1]}")
        fund_manager_node = create_fund_manager(
            fund_manager_llm, self.fund_manager_memory
        )

        # Build workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes (no per-branch Msg Clear — single cleanup after fan-in)
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Single post-fan-in cleanup node (avoids parallel RemoveMessage conflicts)
        workflow.add_node("Analyst Cleanup", create_msg_delete())

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Risk Judge", risk_manager_node)
        workflow.add_node("Portfolio Manager", fund_manager_node)

        # Wire edges — analysts run in PARALLEL (fan-out from START, join before cleanup)
        analyst_done_nodes = []
        for analyst_type in selected_analysts:
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            done_node_name = f"Done {analyst_type.capitalize()}"
            analyst_done_nodes.append(done_node_name)

            # Lightweight pass-through node to signal analyst completion
            workflow.add_node(done_node_name, lambda state: {})

            # Fan-out: START → each analyst in parallel
            workflow.add_edge(START, current_analyst)

            # Create dynamic conditional function for tool continuation
            def make_should_continue(tools_name, done_name):
                def should_continue(state):
                    messages = state["messages"]
                    last_message = messages[-1]
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        return tools_name
                    return done_name
                return should_continue

            workflow.add_conditional_edges(
                current_analyst,
                make_should_continue(current_tools, done_node_name),
                [current_tools, done_node_name],
            )
            workflow.add_edge(current_tools, current_analyst)

        # Fan-in: ALL analysts must complete → single cleanup → Bull Researcher
        workflow.add_edge(analyst_done_nodes, "Analyst Cleanup")
        workflow.add_edge("Analyst Cleanup", "Bull Researcher")

        # Research debate
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")

        # Risk debate
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Risk Judge": "Risk Judge",
            },
        )

        workflow.add_edge("Risk Judge", "Portfolio Manager")
        workflow.add_edge("Portfolio Manager", END)

        return workflow.compile()

    def propagate(self, symbol, trade_date):
        """Run the crypto trading agents graph."""
        self.ticker = symbol

        init_agent_state = self.propagator.create_initial_state(symbol, trade_date)
        # Inject portfolio context if available
        if hasattr(self, "portfolio_context") and self.portfolio_context:
            init_agent_state["portfolio_context"] = self.portfolio_context
        # Inject cached reports for analysts not running this cycle
        if hasattr(self, "_cached_reports"):
            for field, report in self._cached_reports.items():
                init_agent_state[field] = report
        args = dict(self.propagator.get_graph_args())  # defensive copy
        # Override stream_mode from args — we need dual mode for node monitoring
        args.pop("stream_mode", None)

        # Use stream(updates) for node-level monitoring + stream(values) for full state.
        # Fall back to graph.invoke() if the stream API is unavailable or incompatible.
        self._last_completed_node = None
        self._node_timings = {}
        node_start = _time.monotonic()
        final_state = None

        try:
            for mode, chunk in self.graph.stream(
                init_agent_state, stream_mode=["updates", "values"], **args
            ):
                if mode == "updates":
                    # chunk is {node_name: state_update}
                    for node_name in chunk:
                        elapsed = _time.monotonic() - node_start
                        self._last_completed_node = node_name
                        self._node_timings[node_name] = elapsed
                        _graph_log.info(f"✅ Node '{node_name}' completed ({elapsed:.1f}s)")
                        node_start = _time.monotonic()

                        if self.debug:
                            update = chunk[node_name]
                            if isinstance(update, dict) and update.get("messages"):
                                update["messages"][-1].pretty_print()

                elif mode == "values":
                    # chunk is the full accumulated state after each node
                    final_state = chunk

        except (TypeError, ValueError) as _stream_err:
            # LangGraph stream API may change between versions — fall back to invoke()
            _graph_log.warning(
                f"⚠️ Stream mode failed ({type(_stream_err).__name__}: {_stream_err}), "
                "falling back to invoke()"
            )
            final_state = self.graph.invoke(init_agent_state, **args)
            # No per-node granularity in fallback — mark the last node as the graph itself
            elapsed = _time.monotonic() - node_start
            self._last_completed_node = "invoke_fallback"
            self._node_timings["invoke_fallback"] = elapsed

        if final_state is None:
            raise RuntimeError("Graph stream produced no output — pipeline may be misconfigured")

        self.curr_state = final_state
        self._log_state(trade_date, final_state)

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"]["current_response"],
                "judge_decision": final_state["investment_debate_state"]["judge_decision"],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        directory = Path(f"eval_results/{self.ticker.replace('/', '_')}/CryptoTradingAgents_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker.replace('/', '_')}/CryptoTradingAgents_logs/full_states_log_{trade_date}.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory."""
        self.reflector.reflect_bull_researcher(self.curr_state, returns_losses, self.bull_memory)
        self.reflector.reflect_bear_researcher(self.curr_state, returns_losses, self.bear_memory)
        self.reflector.reflect_trader(self.curr_state, returns_losses, self.trader_memory)
        self.reflector.reflect_invest_judge(self.curr_state, returns_losses, self.invest_judge_memory)
        self.reflector.reflect_risk_manager(self.curr_state, returns_losses, self.risk_manager_memory)
        self.reflector.reflect_fund_manager(self.curr_state, returns_losses, self.fund_manager_memory)

    def process_signal(self, full_signal):
        """Process signal to extract decision."""
        return self.signal_processor.process_signal(full_signal)

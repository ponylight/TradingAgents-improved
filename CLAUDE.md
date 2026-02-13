# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradingAgents is a multi-agent LLM financial trading framework (v0.2.0). It uses LangGraph to orchestrate a hierarchy of specialized AI agents that collaborate to produce BUY/HOLD/SELL trading decisions for stocks.

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt
# Or editable install
pip install -e .

# Run via Python API (see main.py for example)
python main.py

# Run via CLI (interactive prompts for ticker, date, analysts, LLM provider)
python -m cli.main
# Or after pip install -e .:
tradingagents
```

There is no formal test suite (pytest/unittest). `test.py` is a manual validation script for dataflow functions.

## Environment Variables

LLM provider API key is required based on configured provider:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, `OPENROUTER_API_KEY`
- `ALPHA_VANTAGE_API_KEY` — only if using alpha_vantage data vendor (yfinance is default, no key needed)
- `TRADINGAGENTS_RESULTS_DIR` — optional override for results directory

## Architecture

### Agent Pipeline (execution order)

The system runs a sequential pipeline via a LangGraph `StateGraph`:

1. **Analyst Team** (sequential, each uses tool-calling with data APIs):
   - Market Analyst → technical indicators (MACD, RSI, Bollinger Bands, etc.)
   - Social Media Analyst → news sentiment
   - News Analyst → news + insider transactions + global macro
   - Fundamentals Analyst → balance sheet, cash flow, income statement

2. **Research Team** (bull/bear debate, configurable rounds via `max_debate_rounds`):
   - Bull Researcher ↔ Bear Researcher (alternating debate)
   - Research Manager → synthesizes into investment recommendation

3. **Trader** — converts recommendation into specific trade plan (uses BM25 memory of past situations)

4. **Risk Management Team** (3-way debate, configurable rounds via `max_risk_discuss_rounds`):
   - Aggressive ↔ Conservative ↔ Neutral Analyst (rotating debate)
   - Risk Manager → final risk-adjusted decision

The graph is defined in `tradingagents/graph/setup.py` (node wiring) with conditional routing in `tradingagents/graph/conditional_logic.py`.

### Key Entry Points

| File | Purpose |
|------|---------|
| `tradingagents/graph/trading_graph.py` | `TradingAgentsGraph` — main orchestrator class |
| `cli/main.py` | Typer CLI app with Rich real-time display |
| `main.py` | Minimal Python API usage example |

### Package Structure

- **`tradingagents/agents/`** — Agent factory functions organized by role:
  - `analysts/` — market, social_media, news, fundamentals analysts
  - `researchers/` — bull_researcher, bear_researcher
  - `managers/` — research_manager, risk_manager
  - `risk_mgmt/` — aggressive, conservative, neutral debators
  - `trader/` — trader
  - `utils/` — `agent_states.py` (TypedDict state), `memory.py` (BM25 memory), `agent_utils.py` (tool definitions)

- **`tradingagents/dataflows/`** — Data fetching with vendor abstraction:
  - `interface.py` — routes tool calls to correct vendor based on config
  - `y_finance.py` / `yfinance_news.py` — yfinance implementation (default)
  - `alpha_vantage*.py` — Alpha Vantage implementation (alternative)
  - `config.py` — runtime config for data vendor routing
  - Data is cached in `tradingagents/dataflows/data_cache/`

- **`tradingagents/llm_clients/`** — Multi-provider LLM support:
  - `factory.py` — `create_llm_client()` factory function
  - `base_client.py` → `openai_client.py` (also handles xAI, Ollama, OpenRouter), `anthropic_client.py`, `google_client.py`
  - Providers: openai, anthropic, google, xai, ollama, openrouter

- **`tradingagents/graph/`** — LangGraph workflow:
  - `setup.py` — `GraphSetup.setup_graph()` builds the `StateGraph`
  - `conditional_logic.py` — routing functions for debate rounds and tool calls
  - `propagation.py` — creates initial state for graph execution
  - `reflection.py` — post-trade memory update
  - `signal_processing.py` — extracts BUY/HOLD/SELL from LLM output

- **`cli/`** — Interactive CLI with Rich panels, questionary prompts, stats tracking

### State Management

`AgentState` (in `agents/utils/agent_states.py`) is a `TypedDict` extending LangGraph's `MessagesState`. It carries reports from each analyst, debate histories (`InvestDebateState`, `RiskDebateState`), and the final trade decision through the graph.

### Memory System

`FinancialSituationMemory` (in `agents/utils/memory.py`) uses BM25 lexical matching (no API calls) to find similar past market situations. Memories persist to disk and are updated via `reflect_and_remember()` after observing trade outcomes.

### Configuration

`tradingagents/default_config.py` defines `DEFAULT_CONFIG` dict with:
- `llm_provider` / `deep_think_llm` / `quick_think_llm` — LLM selection
- `max_debate_rounds` / `max_risk_discuss_rounds` — debate depth
- `data_vendors` — category-level vendor routing (core_stock_apis, technical_indicators, fundamental_data, news_data)
- `tool_vendors` — per-tool vendor overrides (takes precedence over category)
- Provider-specific: `google_thinking_level`, `openai_reasoning_effort`

## Python API Usage

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-5-mini"
config["quick_think_llm"] = "gpt-5-mini"

ta = TradingAgentsGraph(config=config, debug=True)
final_state, decision = ta.propagate("NVDA", "2024-05-10")
# decision is the processed BUY/HOLD/SELL signal

# After observing returns, update memory:
# ta.reflect_and_remember(returns_losses=1000)
```

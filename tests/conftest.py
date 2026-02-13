"""Shared test fixtures for TradingAgents test suite.

Sets up module stubs to allow testing pure-Python logic without requiring
the full LLM/LangGraph runtime (which may not be available in CI or on
all Python versions).
"""

import sys
import types
import pytest


def _make_stub(name, attrs=None):
    """Create a stub module with optional attributes."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


def _make_class_stub(name):
    """Create a stub class."""
    return type(name, (), {"__init__": lambda self, *a, **kw: None})


# Build comprehensive stubs BEFORE any tradingagents modules are imported.
# This must happen at conftest load time (before test collection).

_ToolNode = _make_class_stub("ToolNode")
_ChatOpenAI = _make_class_stub("ChatOpenAI")
_HumanMessage = _make_class_stub("HumanMessage")
_AIMessage = _make_class_stub("AIMessage")
_RemoveMessage = _make_class_stub("RemoveMessage")
_ChatPromptTemplate = type("ChatPromptTemplate", (), {
    "from_messages": classmethod(lambda cls, *a, **kw: type("_Prompt", (), {
        "partial": lambda self, **kw: self,
        "__or__": lambda self, other: other,
    })()),
})
_MessagesPlaceholder = _make_class_stub("MessagesPlaceholder")
_MessagesState = type("MessagesState", (dict,), {})
_END = "END"
_START = "START"
_StateGraph = _make_class_stub("StateGraph")


def _tool_decorator(fn=None, *args, **kwargs):
    if fn is None:
        return lambda f: f
    return fn


_ChatAnthropic = _make_class_stub("ChatAnthropic")
_ChatGoogleGenerativeAI = _make_class_stub("ChatGoogleGenerativeAI")

_STUBS = {
    # langgraph
    "langgraph": _make_stub("langgraph"),
    "langgraph.prebuilt": _make_stub("langgraph.prebuilt", {"ToolNode": _ToolNode}),
    "langgraph.graph": _make_stub("langgraph.graph", {
        "END": _END, "START": _START, "StateGraph": _StateGraph, "MessagesState": _MessagesState,
    }),
    "langgraph.graph.message": _make_stub("langgraph.graph.message", {"MessagesState": _MessagesState}),
    "langgraph.checkpoint": _make_stub("langgraph.checkpoint"),
    "langgraph.checkpoint.memory": _make_stub("langgraph.checkpoint.memory", {
        "MemorySaver": _make_class_stub("MemorySaver"),
    }),
    # langchain_openai
    "langchain_openai": _make_stub("langchain_openai", {"ChatOpenAI": _ChatOpenAI}),
    # langchain_anthropic
    "langchain_anthropic": _make_stub("langchain_anthropic", {"ChatAnthropic": _ChatAnthropic}),
    # langchain_google_genai
    "langchain_google_genai": _make_stub("langchain_google_genai", {
        "ChatGoogleGenerativeAI": _ChatGoogleGenerativeAI,
    }),
    # langchain_core and submodules
    "langchain_core": _make_stub("langchain_core"),
    "langchain_core._api": _make_stub("langchain_core._api"),
    "langchain_core._api.deprecation": _make_stub("langchain_core._api.deprecation"),
    "langchain_core.messages": _make_stub("langchain_core.messages", {
        "HumanMessage": _HumanMessage,
        "AIMessage": _AIMessage,
        "RemoveMessage": _RemoveMessage,
    }),
    "langchain_core.prompts": _make_stub("langchain_core.prompts", {
        "ChatPromptTemplate": _ChatPromptTemplate,
        "MessagesPlaceholder": _MessagesPlaceholder,
    }),
    "langchain_core.tools": _make_stub("langchain_core.tools", {"tool": _tool_decorator}),
}

# Force stubs into sys.modules, overriding any partially-loaded real packages
for mod_name, stub in _STUBS.items():
    sys.modules[mod_name] = stub


# --- Fixtures ---

@pytest.fixture
def test_config():
    """Minimal config dict for testing."""
    return {
        "llm_provider": "openai",
        "deep_think_llm": "test-model",
        "quick_think_llm": "test-model",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "data_vendors": {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
        },
        "tool_vendors": {},
        "cache_ttl_hours": 24,
        "request_timeout": 30,
    }


@pytest.fixture
def sample_risk_debate_state():
    """Sample risk debate state for testing."""
    return {
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "history": "",
        "latest_speaker": "",
        "current_aggressive_response": "",
        "current_conservative_response": "",
        "current_neutral_response": "",
        "judge_decision": "",
        "count": 0,
    }


@pytest.fixture
def sample_invest_debate_state():
    """Sample investment debate state for testing."""
    return {
        "bull_history": "",
        "bear_history": "",
        "history": "",
        "current_response": "",
        "judge_decision": "",
        "count": 0,
    }


@pytest.fixture
def sample_agent_state(sample_invest_debate_state, sample_risk_debate_state):
    """Full agent state for testing."""
    return {
        "messages": [],
        "sender": "",
        "company_of_interest": "AAPL",
        "trade_date": "2024-05-10",
        "market_report": "Market is trending up with strong momentum.",
        "sentiment_report": "Social media sentiment is bullish.",
        "news_report": "No major negative news.",
        "fundamentals_report": "Strong balance sheet with growing revenue.",
        "investment_debate_state": sample_invest_debate_state,
        "risk_debate_state": sample_risk_debate_state,
        "investment_plan": "Buy AAPL with moderate position size.",
        "trader_investment_plan": "Buy 100 shares of AAPL at market price.",
        "final_trade_decision": "",
    }

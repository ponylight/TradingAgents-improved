"""Tests for ConditionalLogic graph routing."""

import pytest
from unittest.mock import MagicMock

from tradingagents.graph.conditional_logic import ConditionalLogic


@pytest.fixture
def logic():
    return ConditionalLogic(max_debate_rounds=2, max_risk_discuss_rounds=2)


class TestDebateRouting:
    def test_routes_to_bear_after_bull(self, logic, sample_agent_state):
        sample_agent_state["investment_debate_state"]["current_response"] = "Bull Analyst: ..."
        sample_agent_state["investment_debate_state"]["count"] = 1
        result = logic.should_continue_debate(sample_agent_state)
        assert result == "Bear Researcher"

    def test_routes_to_bull_after_bear(self, logic, sample_agent_state):
        sample_agent_state["investment_debate_state"]["current_response"] = "Bear Analyst: ..."
        sample_agent_state["investment_debate_state"]["count"] = 2
        result = logic.should_continue_debate(sample_agent_state)
        assert result == "Bull Researcher"

    def test_routes_to_manager_when_rounds_exhausted(self, logic, sample_agent_state):
        # 2 rounds * 2 agents = count of 4
        sample_agent_state["investment_debate_state"]["count"] = 4
        sample_agent_state["investment_debate_state"]["current_response"] = "Bull Analyst: ..."
        result = logic.should_continue_debate(sample_agent_state)
        assert result == "Research Manager"

    def test_single_round_debate(self, sample_agent_state):
        logic = ConditionalLogic(max_debate_rounds=1)
        sample_agent_state["investment_debate_state"]["count"] = 2
        result = logic.should_continue_debate(sample_agent_state)
        assert result == "Research Manager"


class TestRiskAnalysisRouting:
    def test_routes_conservative_after_aggressive(self, logic, sample_agent_state):
        sample_agent_state["risk_debate_state"]["latest_speaker"] = "Aggressive"
        sample_agent_state["risk_debate_state"]["count"] = 1
        result = logic.should_continue_risk_analysis(sample_agent_state)
        assert result == "Conservative Analyst"

    def test_routes_neutral_after_conservative(self, logic, sample_agent_state):
        sample_agent_state["risk_debate_state"]["latest_speaker"] = "Conservative"
        sample_agent_state["risk_debate_state"]["count"] = 2
        result = logic.should_continue_risk_analysis(sample_agent_state)
        assert result == "Neutral Analyst"

    def test_routes_aggressive_after_neutral(self, logic, sample_agent_state):
        sample_agent_state["risk_debate_state"]["latest_speaker"] = "Neutral"
        sample_agent_state["risk_debate_state"]["count"] = 3
        result = logic.should_continue_risk_analysis(sample_agent_state)
        assert result == "Aggressive Analyst"

    def test_routes_to_judge_when_rounds_exhausted(self, logic, sample_agent_state):
        # 2 rounds * 3 agents = count of 6
        sample_agent_state["risk_debate_state"]["count"] = 6
        sample_agent_state["risk_debate_state"]["latest_speaker"] = "Neutral"
        result = logic.should_continue_risk_analysis(sample_agent_state)
        assert result == "Risk Judge"


class TestAnalystToolRouting:
    def test_market_continues_with_tool_calls(self, logic):
        mock_msg = MagicMock()
        mock_msg.tool_calls = [{"name": "get_stock_data"}]
        state = {"messages": [mock_msg]}
        assert logic.should_continue_market(state) == "tools_market"

    def test_market_stops_without_tool_calls(self, logic):
        mock_msg = MagicMock()
        mock_msg.tool_calls = []
        state = {"messages": [mock_msg]}
        assert logic.should_continue_market(state) == "Msg Clear Market"

    def test_social_continues_with_tool_calls(self, logic):
        mock_msg = MagicMock()
        mock_msg.tool_calls = [{"name": "get_news"}]
        state = {"messages": [mock_msg]}
        assert logic.should_continue_social(state) == "tools_social"

    def test_news_stops_without_tool_calls(self, logic):
        mock_msg = MagicMock()
        mock_msg.tool_calls = []
        state = {"messages": [mock_msg]}
        assert logic.should_continue_news(state) == "Msg Clear News"

    def test_fundamentals_continues_with_tool_calls(self, logic):
        mock_msg = MagicMock()
        mock_msg.tool_calls = [{"name": "get_fundamentals"}]
        state = {"messages": [mock_msg]}
        assert logic.should_continue_fundamentals(state) == "tools_fundamentals"

from tradingagents.agents.utils.report_context import get_agent_context

"""Shared factories for debate agent nodes.

Eliminates code duplication across risk debators (aggressive/conservative/neutral)
and investment researchers (bull/bear) by extracting the common state-management
pattern into reusable factory functions.
"""


# Maps each risk debate role to its state keys
RISK_ROLES = {
    "Aggressive": {
        "history_key": "aggressive_history",
        "response_key": "current_aggressive_response",
        "others": ["current_conservative_response", "current_neutral_response"],
    },
    "Conservative": {
        "history_key": "conservative_history",
        "response_key": "current_conservative_response",
        "others": ["current_aggressive_response", "current_neutral_response"],
    },
    "Neutral": {
        "history_key": "neutral_history",
        "response_key": "current_neutral_response",
        "others": ["current_aggressive_response", "current_conservative_response"],
    },
}

INVEST_ROLES = {
    "Bull": {
        "history_key": "bull_history",
        "other_history_key": "bear_history",
    },
    "Bear": {
        "history_key": "bear_history",
        "other_history_key": "bull_history",
    },
}


def create_risk_debate_node(llm, role_label, prompt_builder):
    """Factory for risk debate agent graph nodes.

    Args:
        llm: Language model instance
        role_label: "Aggressive", "Conservative", or "Neutral"
        prompt_builder: callable(trader_decision, reports, history, other_responses) -> str
    """
    role_config = RISK_ROLES[role_label]

    def risk_debate_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        own_history = risk_debate_state.get(role_config["history_key"], "")

        other_responses = {
            key: risk_debate_state.get(key, "")
            for key in role_config["others"]
        }

        reports = {
            "market": state["market_report"],
            "sentiment": state["sentiment_report"],
            "news": state["news_report"],
            "fundamentals": state["fundamentals_report"],
        }

        # Build budgeted context for this specific risk debate role
        risk_role_map = {"Aggressive": "aggressive_debator", "Conservative": "conservative_debator", "Neutral": "neutral_debator"}
        budgeted_context = get_agent_context(state, risk_role_map.get(role_label, "default"))
        reports["_budgeted_context"] = budgeted_context
        trader_decision = state["trader_investment_plan"]

        prompt = prompt_builder(trader_decision, reports, history, other_responses)
        response = llm.invoke(prompt)
        argument = f"{role_label} Analyst: {response.content}"

        # Build new state preserving all fields
        new_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": role_label,
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count": risk_debate_state["count"] + 1,
        }
        # Update own fields
        new_state[role_config["history_key"]] = own_history + "\n" + argument
        new_state[role_config["response_key"]] = argument

        return {"risk_debate_state": new_state}

    return risk_debate_node


def create_invest_debate_node(llm, memory, role_label, prompt_builder):
    """Factory for investment debate agent graph nodes (bull/bear researchers).

    Args:
        llm: Language model instance
        memory: FinancialSituationMemory instance
        role_label: "Bull" or "Bear"
        prompt_builder: callable(reports, history, opponent_response, past_memories_str) -> str
    """
    role_config = INVEST_ROLES[role_label]

    def invest_debate_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        own_history = investment_debate_state.get(role_config["history_key"], "")
        opponent_response = investment_debate_state.get("current_response", "")

        reports = {
            "market": state["market_report"],
            "sentiment": state["sentiment_report"],
            "news": state["news_report"],
            "fundamentals": state["fundamentals_report"],
        }

        # Build budgeted context for this specific risk debate role
        risk_role_map = {"Aggressive": "aggressive_debator", "Conservative": "conservative_debator", "Neutral": "neutral_debator"}
        budgeted_context = get_agent_context(state, risk_role_map.get(role_label, "default"))
        reports["_budgeted_context"] = budgeted_context

        # Build budgeted context for this specific agent role
        agent_role = "bull_researcher" if role_label == "Bull" else "bear_researcher"
        budgeted_context = get_agent_context(state, agent_role)
        reports["_budgeted_context"] = budgeted_context

        # Retrieve similar past situations
        curr_situation = "\n\n".join(v for k, v in reports.items() if not k.startswith("_"))
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = "\n\n".join(
            rec["recommendation"] for rec in past_memories
        ) or "No past memories found."

        prompt = prompt_builder(reports, history, opponent_response, past_memory_str)
        response = llm.invoke(prompt)
        argument = f"{role_label} Analyst: {response.content}"

        new_state = {
            "history": history + "\n" + argument,
            role_config["history_key"]: own_history + "\n" + argument,
            role_config["other_history_key"]: investment_debate_state.get(
                role_config["other_history_key"], ""
            ),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_state}

    return invest_debate_node

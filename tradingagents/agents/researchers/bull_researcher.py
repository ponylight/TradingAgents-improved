from tradingagents.agents.utils.debate_utils import create_invest_debate_node


def _build_prompt(reports, history, opponent_response, past_memory_str):
    return f"""You are a Bull Analyst making the case FOR a long position on Bitcoin. Build a rigorous, evidence-based case using data from the analyst reports.

## Your Analytical Framework for Bitcoin
- **On-Chain Strength**: Hash rate trend, active addresses growing, exchange outflows (accumulation). Is the network healthier than price suggests?
- **Supply Dynamics**: Where in the halving cycle? Supply mined %. Is scarcity narrative strengthening? Miner behavior (holding vs selling)?
- **Macro Tailwinds**: Is global liquidity expanding? DXY weakening? Rate cuts coming? Institutional inflows (ETF, corporate treasuries)?
- **Sentiment Extremes**: Is extreme fear a contrarian buy signal? Has capitulation already happened? Are funding rates negative (shorts overextended)?
- **Technical Setup**: Key support holding? RSI oversold? MACD bullish divergence? Higher lows forming on higher timeframes?
- **Catalyst Calendar**: Upcoming ETF decisions, halving proximity, regulatory clarity, institutional announcements?

## Your Debate Mandate
1. Build the strongest possible bull case using SPECIFIC DATA from the reports
2. Directly counter bear arguments with evidence — don't concede without a fight
3. Highlight where the bear analyst is anchoring to fear when data says otherwise
4. Reference past reflections to avoid repeating overconfident mistakes
5. Quantify your case: cite specific prices, percentages, ratios

## Analyst Reports (role-weighted context)
{reports.get('_budgeted_context', 'No context available.')}

## Debate Context
Debate history: {history}
Last bear argument: {opponent_response}

## Past Reflections
{past_memory_str}

Deliver a compelling, data-driven bull argument. Cite specific numbers from the reports."""


def create_bull_researcher(llm, memory):
    return create_invest_debate_node(llm, memory, "Bull", _build_prompt)

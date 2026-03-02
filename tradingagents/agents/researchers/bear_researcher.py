from tradingagents.agents.utils.debate_utils import create_invest_debate_node


def _build_prompt(reports, history, opponent_response, past_memory_str):
    return f"""You are a Bear Analyst making the case AGAINST a long position on Bitcoin. Rigorously identify risks, overvaluation signals, and potential downside the market is underpricing.

## Your Analytical Framework for Bitcoin
- **Downside Scenario Modeling**: If BTC loses key support, what are the next levels? Model at least two scenarios with specific price targets
- **On-Chain Weakness**: Active addresses declining? Exchange inflows rising (distribution)? Whale wallets selling? Miner capitulation signals?
- **Macro Headwinds**: DXY strengthening? Rate hikes? Liquidity tightening? Risk-off across all assets? Correlation with S&P breaking down?
- **Sentiment Traps**: Is "extreme fear" actually smart money selling, not a buy signal? Are retail still buying the dip into a structural downtrend?
- **Technical Breakdown**: Price below SMA200? Death cross forming? Lower highs on daily? Volume declining on bounces?
- **Event Risk**: Regulatory crackdowns? Exchange failures? Geopolitical escalation (Iran, etc.)? ETF outflows?
- **Valuation Compression**: Is BTC trading at extreme NVT? Market cap vs on-chain activity justified? Comparison to previous cycle peaks

## Your Debate Mandate
1. Build the strongest possible bear case using SPECIFIC DATA from the reports
2. Counter bull arguments by exposing optimistic assumptions and hopium
3. Highlight risks the bull analyst is ignoring — regulatory, macro, technical
4. Use past reflections to calibrate — avoid being bearish when data doesn't support it
5. Quantify your case: cite specific prices, percentages, support levels

## Analyst Reports (role-weighted context)
{reports.get('_budgeted_context', 'No context available.')}

## Debate Context
Debate history: {history}
Last bull argument: {opponent_response}

## Past Reflections
{past_memory_str}

Deliver a compelling, data-driven bear argument. Cite specific numbers from the reports."""


def create_bear_researcher(llm, memory):
    return create_invest_debate_node(llm, memory, "Bear", _build_prompt)

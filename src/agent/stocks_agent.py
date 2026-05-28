"""
stocks_agent.py — A LangGraph ReAct agent for stock market queries.

How to plug in your real data functions:
  1. Replace the three sample functions (get_stock_price, get_stock_info,
     get_stock_history) with your own implementations.
  2. Keep the function signatures identical — name, typed args, and str return.
  3. That's it. The agent picks up the docstrings as tool descriptions, so
     keep them accurate.
"""
import requests , os
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from pydantic import ValidationError

from src.agent.models import AgentConfig, StockQuoteResponse

STOCKS_AGENT_CONFIG = AgentConfig(
    name="stocks_agent",
    description="An agent that fetches real-time and historical stock data.",
    model="gemini-2.5-flash",
)

SYSTEM_PROMPT = (
    "You are a helpful stock market assistant. "
    "Use the available tools to fetch real-time prices, company information, "
    "and historical data for any stock the user asks about. "
    "Always present numbers clearly and confirm the data source/timestamp when available."
)

base_stocks_url = "https://api.stockdata.org"

# ── Sample tool functions ────────────────────────────────────────────────────
# Replace these three functions with your own. Keep the signatures the same.

@tool
def get_stock_price(tickers: str) -> str:
    """
    Fetch the current price and daily stats for one or more stock ticker symbols.

    Args:
        tickers: Comma-separated ticker symbols, e.g. 'AAPL' or 'AAPL,TSLA,MSFT'.

    Returns:
        A formatted string with price, day high, day low, and volume for each ticker.
    """
    symbol_list = [s.strip().upper() for s in tickers.split(",")]

    url = base_stocks_url + "/v1/data/quote"
    params = {
        "symbols": ",".join(symbol_list),
        "api_token": os.getenv("STOCK_DATA_API_KEY")
    }

    try:
        response = requests.get(url, params=params, timeout=5)
    except requests.RequestException:
        return "Failed to reach the stock data API. Please try again."

    if response.status_code != 200:
        return f"API returned status {response.status_code}."

    try:
        data = StockQuoteResponse(**response.json())
    except ValidationError:
        return "Failed to parse the stock data response."

    if not data.data:
        return f"No data found for: {', '.join(symbol_list)}."

    lines = []
    for quote in data.data:
        parts = [f"{quote.ticker}: ${quote.price:.2f}"]
        if quote.day_high is not None:
            parts.append(f"High: ${quote.day_high:.2f}")
        if quote.day_low is not None:
            parts.append(f"Low: ${quote.day_low:.2f}")
        if quote.volume is not None:
            parts.append(f"Volume: {quote.volume:,}")
        lines.append(" | ".join(parts))

    return "\n".join(lines)

# ── Agent builder ────────────────────────────────────────────────────────────

STOCKS_TOOLS = [get_stock_price]


def build_stocks_agent(api_key: str):
    """Build a LangGraph ReAct agent for stock market queries."""
    llm = ChatGoogleGenerativeAI(
        model=STOCKS_AGENT_CONFIG.model,
        google_api_key=api_key,
    )

    agent = create_react_agent(
        model=llm,
        tools=STOCKS_TOOLS,
        prompt=SYSTEM_PROMPT,
    )

    return agent
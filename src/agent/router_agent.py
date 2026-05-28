"""
router_agent.py — LLM-powered router that classifies user intent.

The router reads the latest user message and returns the name of the agent
that should handle it.  It does NOT answer the user — it only routes.

To add a new agent destination:
  1. Add its name to AGENT_DESTINATIONS below.
  2. Add a short description to the ROUTING_PROMPT so the LLM knows when to use it.
  3. Wire a new conditional branch in graph.py.
"""

import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from src.agent.models import GraphState

# ── Destinations the router knows about ──────────────────────────────────────
AGENT_DESTINATIONS = ["github_agent", "stocks_agent"]
DEFAULT_AGENT = "github_agent"

ROUTING_PROMPT = """You are a routing assistant. Your only job is to decide which agent should handle the user's message.

Available agents:
- github_agent: Handles anything related to GitHub — repositories, branches, pull requests, issues, commits, code search, forks, stars, file contents, GitHub Actions, etc.
- stocks_agent: Handles anything related to stocks, financial markets, stock prices, company financials, historical price data, market cap, P/E ratio, tickers, trading, investment queries, etc.

Rules:
- Respond with ONLY a JSON object in this exact format: {{"agent": "<agent_name>"}}
- No explanation, no markdown, no extra text.
- If the message is ambiguous or fits neither agent, default to: {{"agent": "github_agent"}}

User message:
\"\"\"{user_message}\"\"\"
"""


def build_router(api_key: str):
    """
    Return an async router node function ready to drop into a LangGraph graph.

    Usage in graph.py:
        router_node = build_router(gemini_api_key)
        graph.add_node("router", router_node)
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
    )

    async def router_node(state: GraphState) -> dict:
        """
        Inspect the latest human message and set state.active_agent
        to the appropriate destination agent name.
        """
        # Find the most recent human message
        user_message = ""
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        if not user_message:
            return {"active_agent": DEFAULT_AGENT}

        prompt = ROUTING_PROMPT.format(user_message=user_message)

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

            # Strip markdown fences if the LLM wraps in ```json ... ```
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

            parsed = json.loads(raw)
            agent_name = parsed.get("agent", DEFAULT_AGENT)

            if agent_name not in AGENT_DESTINATIONS:
                print(f"⚠️  Router returned unknown agent '{agent_name}', defaulting to '{DEFAULT_AGENT}'")
                agent_name = DEFAULT_AGENT

        except Exception as e:
            print(f"⚠️  Router error ({e}), defaulting to '{DEFAULT_AGENT}'")
            agent_name = DEFAULT_AGENT

        print(f"🔀 Router → {agent_name}")
        return {"active_agent": agent_name}

    return router_node


def route_to_agent(state: GraphState) -> str:
    """
    Conditional-edge function for LangGraph.

    Maps state.active_agent → the node name LangGraph should go to next.
    Add a new entry here whenever you wire in a new agent node.
    """
    routing_map = {
        "github_agent": "github_agent",
        "stocks_agent": "stocks_agent",
    }
    destination = routing_map.get(state.active_agent, DEFAULT_AGENT)
    return destination
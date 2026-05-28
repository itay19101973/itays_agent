"""
graph.py — LangGraph graph definition.

Current agents: github_agent, stocks_agent.
A router node classifies every message and directs it to the right agent.

To add a new agent:
  1. Build it in its own file following the pattern in stocks_agent.py.
  2. Import and register a node here (see "Add your node here" comment).
  3. Add its name to the conditional edge map in router_agent.route_to_agent.
  4. Add a description for it in router_agent.ROUTING_PROMPT.
"""

from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph

from src.agent.github_agent import build_github_agent
from src.agent.stocks_agent import build_stocks_agent
from src.agent.router_agent import build_router, route_to_agent
from src.agent.models import GraphState


def build_graph(tools: list[StructuredTool], gemini_api_key: str):
    """
    Construct and compile the multi-agent LangGraph graph.

    Flow:
        START → router → (github_agent | stocks_agent) → END
    """
    graph = StateGraph(GraphState)

    # ── Router ───────────────────────────────────────────────────────────────
    router_node = build_router(gemini_api_key)
    graph.add_node("router", router_node)

    # ── Agents ───────────────────────────────────────────────────────────────
    github_agent = build_github_agent(tools, gemini_api_key)

    async def github_node(state: GraphState) -> dict:
        result = await github_agent.ainvoke({"messages": state.messages})
        return {"messages": result["messages"]}

    graph.add_node("github_agent", github_node)

    # ── Add your node here ───────────────────────────────────────────────────
    stocks_agent = build_stocks_agent(gemini_api_key)

    async def stocks_node(state: GraphState) -> dict:
        result = await stocks_agent.ainvoke({"messages": state.messages})
        return {"messages": result["messages"]}

    graph.add_node("stocks_agent", stocks_node)
    # ── End add node ─────────────────────────────────────────────────────────

    # ── Edges ────────────────────────────────────────────────────────────────
    # Every conversation starts at the router
    graph.add_edge(START, "router")

    # Router decides the destination via the conditional edge
    graph.add_conditional_edges(
        "router",
        route_to_agent,          # reads state.active_agent, returns node name
        {
            "github_agent": "github_agent",
            "stocks_agent": "stocks_agent",
            # Add new agents here: "my_agent": "my_agent"
        },
    )

    # Both agents terminate the turn
    graph.add_edge("github_agent", END)
    graph.add_edge("stocks_agent", END)

    return graph.compile()
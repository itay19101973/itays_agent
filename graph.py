"""
graph.py — LangGraph graph definition.

To add a new agent:
  1. Build it in its own file (e.g. my_agent.py) following the same pattern as agent.py.
  2. Import it here and add a node + conditional edge below.
"""

from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph

from agent import build_github_agent
from models import GraphState


def build_graph(tools: list[StructuredTool], gemini_api_key: str):
    """
    Construct and compile the multi-agent LangGraph graph.

    Currently contains one agent: github_agent.
    Additional agents can be wired in as new nodes.
    """
    graph = StateGraph(GraphState)

    # ── Agents ──────────────────────────────────────────────────────────────
    github_agent = build_github_agent(tools, gemini_api_key)

    async def github_node(state: GraphState) -> dict:
        result = await github_agent.ainvoke({"messages": state.messages})
        return {"messages": result["messages"]}

    graph.add_node("github_agent", github_node)

    # ── Routing ──────────────────────────────────────────────────────────────
    # Right now every conversation goes straight to the github_agent.
    # Replace this with a router node when you add more agents.
    graph.add_edge(START, "github_agent")
    graph.add_edge("github_agent", END)

    return graph.compile()
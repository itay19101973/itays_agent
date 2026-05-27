"""
main.py — Entry point. Connects to all registered MCP servers and starts the chat loop.

To add a new MCP server, edit mcp_registry.py only.
"""

import asyncio
import os
from contextlib import AsyncExitStack

from langchain_core.messages import AIMessage, HumanMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from graph import build_graph
from mcp_registry import MCP_SERVERS
from mcp_tools import load_tools_from_session
from models import GraphState


def _get_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value


async def connect_all_mcps(stack: AsyncExitStack) -> list:
    """Open stdio connections for every server in MCP_SERVERS and return all tools."""
    all_tools = []

    for server in MCP_SERVERS:
        print(f"🔌 Connecting to MCP: {server.name}...")
        read, write = await stack.enter_async_context(stdio_client(server.params))
        session = await stack.enter_async_context(ClientSession(read, write))
        tools = await load_tools_from_session(session)
        print(f"   ✅ {len(tools)} tools loaded from '{server.name}': {[t.name for t in tools]}")
        all_tools.extend(tools)

    return all_tools


async def chat_loop(tools: list, gemini_api_key: str) -> None:
    """Run an interactive chat loop backed by the LangGraph agent graph."""
    graph = build_graph(tools, gemini_api_key)
    conversation: list = []

    print(f"\n🤖 Agent ready ({len(tools)} tools total). Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        conversation.append(HumanMessage(content=user_input))

        result = await graph.ainvoke(GraphState(messages=conversation))
        updated_messages: list = result["messages"]

        reply = next(
            (msg.content if isinstance(msg.content, str) else str(msg.content)
             for msg in reversed(updated_messages)
             if isinstance(msg, AIMessage) and msg.content),
            "",
        )

        conversation = updated_messages
        print(f"\n🧠 Agent: {reply}\n")


async def main() -> None:
    gemini_api_key = _get_env("GEMINI_API_KEY")

    async with AsyncExitStack() as stack:
        tools = await connect_all_mcps(stack)
        await chat_loop(tools, gemini_api_key)


if __name__ == "__main__":
    asyncio.run(main())
"""
main.py — Entry point. Connects to all registered MCP servers and starts the chat loop.

To add a new MCP server, edit mcp_registry.py only.
"""

import asyncio
import os
from contextlib import AsyncExitStack

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from src.agent.graph import build_graph
from src.mcp_servers.mcp_registry import MCP_SERVERS
from src.mcp_servers.mcp_tools import load_tools_from_session
from src.agent.models import GraphState


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


def _extract_reply(updated_messages: list) -> str:
    """Pull the final text reply out of the last AIMessage, handling both
    plain-string content (OpenAI style) and block-list content (Gemini style)."""
    for msg in reversed(updated_messages):
        if not isinstance(msg, AIMessage) or not msg.content:
            continue
        if isinstance(msg.content, str):
            return msg.content
        if isinstance(msg.content, list):
            texts = [
                block["text"]
                for block in msg.content
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
            ]
            if texts:
                return "\n".join(texts)
    return ""


def _print_tool_trace(new_messages: list) -> None:
    """Print every tool call and its result for messages added in this turn."""
    for msg in new_messages:
        # AIMessage with a list of blocks may contain tool_use blocks
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    print(f"\n🔧 Tool call : {block['name']}")
                    print(f"   Args      : {block.get('input', {})}")
        # ToolMessage carries the result back from the tool
        if isinstance(msg, ToolMessage):
            preview = str(msg.content)[:300]
            ellipsis = "..." if len(str(msg.content)) > 300 else ""
            print(f"   ↳ Result  : {preview}{ellipsis}")


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

        # Show tool calls/results for everything added this turn
        new_messages = updated_messages[len(conversation):]
        _print_tool_trace(new_messages)

        reply = _extract_reply(updated_messages)
        conversation = updated_messages
        print(f"\n🧠 Agent: {reply}\n")


async def main() -> None:
    gemini_api_key = _get_env("GEMINI_API_KEY")

    async with AsyncExitStack() as stack:
        tools = await connect_all_mcps(stack)
        await chat_loop(tools, gemini_api_key)


if __name__ == "__main__":
    asyncio.run(main())
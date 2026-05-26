import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage
from mcp.client.stdio import stdio_client
from mcp import ClientSession

from graph import build_graph
from models import GraphState
from mcp_tools import convert_mcp_tools_to_langchain, github_server_params


def _get_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value


async def chat_loop(session: ClientSession, gemini_api_key: str) -> None:
    """Run an interactive chat loop backed by the LangGraph agent graph."""

    print("Initialising MCP session...")
    await session.initialize()

    raw_tools = await session.list_tools()
    print(f"✅ {len(raw_tools.tools)} MCP tools loaded:")
    for t in raw_tools.tools:
        print(f"   - {t.name}")

    lc_tools = convert_mcp_tools_to_langchain(raw_tools, session)
    graph = build_graph(lc_tools, gemini_api_key)

    conversation: list = []

    print("\n🤖 GitHub Agent ready. Type 'exit' or 'quit' to stop.\n")

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

        state = GraphState(messages=conversation)
        result = await graph.ainvoke(state)

        updated_messages: list = result["messages"]

        # Find the last AI message to display
        reply = ""
        for msg in reversed(updated_messages):
            if isinstance(msg, AIMessage) and msg.content:
                reply = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        conversation = updated_messages
        print(f"\n🧠 Agent: {reply}\n")


async def main() -> None:
    gemini_api_key = _get_env("GEMINI_API_KEY")

    server_params = github_server_params()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await chat_loop(session, gemini_api_key)


if __name__ == "__main__":
    asyncio.run(main())
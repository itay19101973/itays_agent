from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent  # Keep this import
from models import AgentConfig

GITHUB_AGENT_CONFIG = AgentConfig(
    name="github_agent",
    description="An agent that can interact with GitHub via MCP tools.",
    model="gemini-2.5-flash",
)

SYSTEM_PROMPT = (
    "You are a helpful GitHub assistant. "
    "You have access to GitHub tools via MCP. "
    "Use them to answer the user's requests about repositories, branches, issues, and PRs. "
    "Always confirm what you did after completing an action."
)


def build_github_agent(tools: list[StructuredTool], api_key: str):
    """Build a LangGraph ReAct agent for GitHub interactions."""
    llm = ChatGoogleGenerativeAI(
        model=GITHUB_AGENT_CONFIG.model,
        google_api_key=api_key,
    )

    # create_react_agent configures everything out-of-the-box
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,  # Pass instructions directly here
    )

    return agent
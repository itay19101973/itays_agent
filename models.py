from typing import Annotated, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    name: str
    description: str
    model: str = "gemini-2.5-flash"


class GraphState(BaseModel):
    """Shared state passed through the LangGraph graph."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    active_agent: str = "github_agent"

    model_config = {"arbitrary_types_allowed": True}


class ToolCallResult(BaseModel):
    """Result from an MCP tool execution."""

    tool_name: str
    args: dict[str, Any]
    output: str
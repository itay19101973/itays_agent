from typing import Annotated, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import List, Optional

class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    name: str
    description: str
    model: str = "gemini-2.5-flash"


class GraphState(BaseModel):
    """
    Shared state passed through the LangGraph graph.

    Fields:
        messages:      Full conversation history. LangGraph merges new messages
                       into this list automatically via the add_messages reducer.
        active_agent:  Set by the router node on every turn. Downstream
                       conditional edges read this to pick the right agent.
                       Defaults to 'github_agent' so the graph is safe even if
                       the router is bypassed.
    """

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    active_agent: str = "github_agent"   # written by router, read by route_to_agent()

    model_config = {"arbitrary_types_allowed": True}


class ToolCallResult(BaseModel):
    """Result from an MCP tool execution."""

    tool_name: str
    args: dict[str, Any]
    output: str


class StockQuote(BaseModel):
    ticker: str
    price: float
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    volume: Optional[int] = None


class StockQuoteResponse(BaseModel):
    data: List[StockQuote]

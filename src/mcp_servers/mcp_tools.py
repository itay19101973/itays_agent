"""
mcp_tools.py — Converts raw MCP tool definitions into LangChain StructuredTools.

This module is MCP-server-agnostic. Pass in any ClientSession and it will
wrap whatever tools that server exposes.
"""

import json
from typing import Any

from mcp import ClientSession
from langchain_core.tools import StructuredTool
from pydantic import create_model, Field


def _mcp_type_to_python(mcp_type: str, prop_schema: dict) -> type:
    """Map MCP JSON-schema types to Python types."""

    type_map = {"integer": int, "number": int, "boolean": bool, "object": dict}
    return type_map.get(mcp_type, str)


def convert_mcp_tools_to_langchain(
    mcp_tools: Any,
    session: ClientSession,
) -> list[StructuredTool]:
    """Convert MCP tool definitions into LangChain StructuredTools."""
    langchain_tools: list[StructuredTool] = []

    for tool in mcp_tools.tools:
        raw_schema = tool.inputSchema or {}
        properties: dict[str, Any] = raw_schema.get("properties", {})
        required: list[str] = raw_schema.get("required", [])

        field_definitions: dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            python_type = _mcp_type_to_python(prop_schema.get("type", "string"), prop_schema)
            description = prop_schema.get("description", "")

            if prop_name in required:
                field_definitions[prop_name] = (python_type, Field(description=description))
            else:
                field_definitions[prop_name] = (python_type | None, Field(default=None, description=description))

        args_schema = create_model(f"{tool.name}_args", **field_definitions)

        def make_tool_fn(t_name: str, t_session: ClientSession):
            async def tool_fn(**kwargs: Any) -> str:
                # Strip explicit None args — some MCP servers reject null values
                clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

                print(f"\n🔧 TOOL CALL: {t_name}")
                print("Args:", json.dumps(clean_kwargs, indent=2, default=str))

                result = await t_session.call_tool(t_name, clean_kwargs)
                output = str(result.content if hasattr(result, "content") else result)

                print("📦 RESULT:", output[:500] + "..." if len(output) > 500 else output)
                return output

            return tool_fn

        langchain_tools.append(
            StructuredTool(
                name=tool.name,
                description=(tool.description or "")[:300],
                args_schema=args_schema,
                coroutine=make_tool_fn(tool.name, session),
            )
        )

    return langchain_tools


async def load_tools_from_session(session: ClientSession) -> list[StructuredTool]:
    """Initialise a session, list its tools, and return LangChain StructuredTools."""
    await session.initialize()
    raw_tools = await session.list_tools()
    return convert_mcp_tools_to_langchain(raw_tools, session)
"""
mcp_registry.py — Register all MCP servers here.

To add a new MCP:
  1. Add a new McpServerConfig entry to MCP_SERVERS.
  2. Add any required env vars to docker-compose.yml.
  3. Pre-install the npm package in the Dockerfile (optional but speeds up first run).

That's it — no other files need to change.
"""

import os
from dataclasses import dataclass, field

from mcp import StdioServerParameters


@dataclass
class McpServerConfig:
    name: str                          # Human-readable label (used in logs)
    params: StdioServerParameters      # How to launch the MCP server process


def _path() -> str:
    return os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")


MCP_SERVERS: list[McpServerConfig] = [
    # ── GitHub ────────────────────────────────────────────────────────────────
    McpServerConfig(
        name="github",
        params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", ""),
                "PATH": _path(),
            },
        ),
    ),

    # ── Filesystem ────────────────────────────────────────────────────────────
    # Gives the agent read access to /workspace inside the container.
    # Mount a host directory there via docker-compose volumes if needed.
    McpServerConfig(
        name="filesystem",
        params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/workspace" ],
            env={"PATH": _path()},
        ),
    ),


]
FROM python:3.12-slim

# Node.js is required for npx-based MCP servers
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/null/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-install MCP servers so the first run is fast.
# Add a line here whenever you add a new npx-based MCP to mcp_registry.py.
RUN npx -y @modelcontextprotocol/server-github --help 2>/dev/null || true
RUN npx -y @modelcontextprotocol/server-filesystem --help 2>/dev/null || true

# Workspace dir exposed to the filesystem MCP
RUN mkdir -p /workspace

COPY . .

CMD ["python", "main.py"]
FROM python:3.12-slim

# Node.js is required for the MCP GitHub server (npx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-install the MCP GitHub server so the first run is fast
RUN npx -y @modelcontextprotocol/server-github --help 2>/dev/null || true

COPY . .

CMD ["python", "main.py"]
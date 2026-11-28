# relock's HTTP/JSON API (and MCP server) in a container, for harnesses in any language.
#
#   docker run -p 8765:8765 -e RELOCK_TOKEN=change-me -v relock-data:/data ghcr.io/inboxpraveen/relock
#
# The server refuses to listen publicly without RELOCK_TOKEN. Data lives in /data (SQLite by
# default); set RELOCK_DB=postgresql://... to use Postgres instead.
FROM python:3.13-slim

LABEL org.opencontainers.image.source="https://github.com/inboxpraveen/relock" \
      org.opencontainers.image.description="relock: agent memory that fixes the rule that fired, in place and only on trusted evidence, so corrected mistakes don't come back." \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[postgres,mcp]" && cd / && rm -rf /build \
    && useradd --create-home --uid 10001 relock && mkdir -p /data && chown relock /data

USER relock
WORKDIR /data
ENV RELOCK_DB=/data/relock.db RELOCK_HOST=0.0.0.0 RELOCK_PORT=8765
VOLUME ["/data"]
EXPOSE 8765
# Probes the HTTP API only when PID 1 runs `serve`: `relock mcp` has no HTTP server, so it passes.
# A RELOCK_PORT that isn't a port number (Kubernetes sets RELOCK_PORT=tcp://... for a Service named
# relock) means 8765, as in the CLI. Set a custom port with RELOCK_PORT, not --port, so this finds it.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import os, urllib.request; c = '/proc/1/cmdline'; serving = not os.access(c, os.R_OK) or b'serve' in open(c, 'rb').read().split(bytes(1)); p = os.environ.get('RELOCK_PORT', '').strip(); p = p if p.isascii() and p.isdigit() and int(p) <= 65535 else '8765'; serving and urllib.request.urlopen('http://127.0.0.1:' + p + '/v1/health', timeout=2)"

ENTRYPOINT ["relock"]
CMD ["serve"]

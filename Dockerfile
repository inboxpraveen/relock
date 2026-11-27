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
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('RELOCK_PORT', '8765') + '/v1/health', timeout=2)"

ENTRYPOINT ["relock"]
CMD ["serve"]

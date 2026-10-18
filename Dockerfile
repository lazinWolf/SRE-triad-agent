# SRE triage worker + dashboard. Joins the OpenTelemetry Demo network (see compose.yaml).
FROM python:3.12-slim
RUN pip install --no-cache-dir uv==0.12.10
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev
COPY src ./src
COPY seeds ./seeds
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501
CMD ["sre-triage", "worker"]

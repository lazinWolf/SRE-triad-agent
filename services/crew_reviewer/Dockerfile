FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY crew.py server.py ./
ENV PATH="/app/.venv/bin:$PATH" CREW_REVIEWER_URL="http://crew_reviewer:8002/"
EXPOSE 8002
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8002"]

"""The investigation tools as two MCP servers (Model Context Protocol, streamable HTTP).

    telemetry  Prometheus, Jaeger and OpenSearch: health, timelines, traces, logs, alerts
    ops        Docker, flagd and the runbooks: changes, container state and logs, flags, runbooks

    uv run sre-triage mcp --port 8011      # telemetry on :8011/mcp, ops on :8012/mcp
    SRE_MCP_URL=http://127.0.0.1:8011      # the agents then use these servers

Without SRE_MCP_URL the same two servers start inside the app process (still real MCP over loopback HTTP), so
`sre-triage ui` works on its own. All tools are read-only. The agents send the incident window in the
`x-incident-window` header, so the model never handles timestamps; any other MCP client gets the last 10 minutes.
"""

from __future__ import annotations

import inspect
import json
import os
import socket
import threading
import time

os.environ.setdefault("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")  # ADK's MCP client otherwise probes for mTLS (12 s)

from mcp.server.mcpserver import Context, MCPServer  # noqa: E402

from .telemetry import Backends, Unavailable  # noqa: E402
from .tools import RESULT_MODELS, SERVER_TOOLS, Tools, Window  # noqa: E402

WINDOW_HEADER = "x-incident-window"
DESCRIPTIONS = {
    "telemetry": "Read-only telemetry of the demo shop (metrics, traces, logs, alerts), incident window vs baseline.",
    "ops": "Read-only operations view of the demo shop: configuration changes, containers, feature flags, runbooks.",
}


def _window(ctx: Context) -> Window:
    request = getattr(ctx.request_context, "request", None)
    raw = request.headers.get(WINDOW_HEADER) if request is not None else None
    if raw:
        return Window(**json.loads(raw))
    now = time.time()
    return Window(now - 600, now)


def _tool(backends: Backends, name: str):
    """A Tools method as an MCP tool function: same parameters and docstring, typed result."""
    method = getattr(Tools, name)
    sig = inspect.signature(method)
    params = [p.replace(annotation=str) for p in sig.parameters.values() if p.name != "self"]
    model = RESULT_MODELS[name]

    def tool(ctx: Context, **kwargs):
        try:
            return model.model_validate(getattr(Tools(backends, _window(ctx)), name)(**kwargs))
        except Unavailable as exc:
            raise RuntimeError(f"backend unavailable: {exc}") from exc

    tool.__name__ = name
    tool.__doc__ = method.__doc__
    context = inspect.Parameter("ctx", inspect.Parameter.KEYWORD_ONLY, annotation=Context)
    tool.__signature__ = sig.replace(parameters=[*params, context], return_annotation=model)
    tool.__annotations__ = {p.name: str for p in params} | {"ctx": Context, "return": model}
    return tool


def build(backends: Backends, kind: str) -> MCPServer:
    server = MCPServer(f"sre-{kind}", instructions=DESCRIPTIONS[kind], log_level="WARNING")
    for name in SERVER_TOOLS[kind]:
        server.add_tool(_tool(backends, name), name=name)
    return server


def _serve(backends: Backends, kind: str, host: str, port: int):
    import uvicorn

    app = build(backends, kind).streamable_http_app(stateless_http=True, json_response=True, host=host)
    return uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))


def serve(backends: Backends, host: str = "127.0.0.1", port: int = 8011) -> None:
    """Run both servers until interrupted: telemetry on `port`, ops on `port + 1`."""
    ops = _serve(backends, "ops", host, port + 1)
    threading.Thread(target=ops.run, daemon=True).start()
    _serve(backends, "telemetry", host, port).run()


def urls(base: str) -> dict[str, str]:
    """Server URLs from SRE_MCP_URL (the telemetry server; ops listens on the next port)."""
    head, _, port = base.rstrip("/").rpartition(":")
    return {"telemetry": f"{head}:{int(port)}/mcp", "ops": f"{head}:{int(port) + 1}/mcp"}


_embedded: dict[int, tuple[Backends, dict[str, str]]] = {}
_lock = threading.Lock()


def ensure(backends: Backends, base: str = "") -> dict[str, str]:
    """URLs of the two servers; starts them in this process (once per backends) when no external URL is set."""
    if base:
        return urls(base)
    with _lock:
        if id(backends) not in _embedded:
            out = {}
            for kind in SERVER_TOOLS:
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", 0))
                    port = probe.getsockname()[1]
                server = _serve(backends, kind, "127.0.0.1", port)
                threading.Thread(target=server.run, daemon=True, name=f"mcp-{kind}").start()
                while not server.started:
                    time.sleep(0.02)
                out[kind] = f"http://127.0.0.1:{port}/mcp"
            _embedded[id(backends)] = (backends, out)  # keep the backends alive so the id stays unique
        return _embedded[id(backends)][1]


def tools_listing(base_urls: dict[str, str]) -> dict[str, list[dict]]:
    """Tools each server advertises (name, description), read over MCP; for `doctor` and the dashboard."""
    import asyncio

    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def read(url: str) -> list[dict]:
        async with streamable_http_client(url) as streams, ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            return [{"name": t.name, "description": (t.description or "").strip()} for t in (await session.list_tools()).tools]

    async def all_() -> dict[str, list[dict]]:
        return {kind: await read(url) for kind, url in base_urls.items()}

    return asyncio.run(all_())

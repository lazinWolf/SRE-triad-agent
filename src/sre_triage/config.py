"""Settings, read once from the environment and the project's .env file."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Application services of the demo that agents investigate and remediation may touch.
SERVICES = ("frontend", "checkout", "cart", "product-catalog", "currency", "payment",
            "shipping", "quote", "email", "recommendation", "ad")
# Who calls whom (static architecture map of the demo).
DEPENDENCIES = {
    "frontend": ["ad", "cart", "checkout", "currency", "product-catalog", "recommendation", "shipping"],
    "checkout": ["cart", "currency", "email", "payment", "product-catalog", "shipping"],
    "cart": ["valkey-cart"],
    "recommendation": ["product-catalog"],
    "product-catalog": ["astronomy-db"],
    "shipping": ["quote"],
}
BASELINE_CPUS = 1.0  # every app container runs with 1 CPU (infra/sre/compose.sre.yaml)


class Settings(BaseModel):
    model: str = "gemini/gemini-3.5-flash"
    fallback_models: list[str] = []
    temperature: float = 0.1
    prometheus_url: str = "http://127.0.0.1:9090"
    jaeger_url: str = "http://127.0.0.1:16686/jaeger/ui"
    opensearch_url: str = "http://127.0.0.1:9200"
    docker_socket: str = "/var/run/docker.sock"
    demo_dir: Path = PROJECT_ROOT / "infra" / "otel-demo"
    data_dir: Path = PROJECT_ROOT / "data"
    baseline_seconds: float = 300     # healthy reference window just before the incident
    agent_steps: int = 5              # tool calls per specialist before it must report
    max_revisions: int = 1            # critic -> re-investigate loops
    verify_wait_seconds: float = 75   # settle time before checking recovery
    critic_a2a_url: str = ""          # e.g. http://127.0.0.1:8001 to use the Critic A2A service

    @property
    def flagd_file(self) -> Path:
        return self.demo_dir / "src" / "flagd" / "demo.flagd.json"


@lru_cache
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]  # LiteLLM's Gemini provider reads this
    env = os.environ
    values: dict = {}
    for field, var in (("model", "SRE_MODEL"), ("prometheus_url", "SRE_PROMETHEUS_URL"),
                       ("jaeger_url", "SRE_JAEGER_URL"), ("opensearch_url", "SRE_OPENSEARCH_URL"),
                       ("docker_socket", "SRE_DOCKER_SOCKET"), ("demo_dir", "SRE_DEMO_DIR"),
                       ("data_dir", "SRE_DATA_DIR"), ("verify_wait_seconds", "SRE_VERIFY_WAIT_SECONDS"),
                       ("critic_a2a_url", "SRE_CRITIC_A2A_URL")):
        if env.get(var):
            values[field] = env[var]
    if env.get("SRE_FALLBACK_MODELS"):
        values["fallback_models"] = [m.strip() for m in env["SRE_FALLBACK_MODELS"].split(",") if m.strip()]
    settings = Settings(**values)
    if not settings.data_dir.is_absolute():
        settings.data_dir = PROJECT_ROOT / settings.data_dir
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings

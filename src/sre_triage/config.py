"""Runtime settings, read from the environment (and .env) once per process."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Services the agents may query and the operator may remediate. Telemetry and
# infrastructure services are deliberately excluded from remediation.
APP_SERVICES = (
    "frontend", "frontend-proxy", "checkout", "cart", "product-catalog", "currency",
    "payment", "shipping", "quote", "email", "recommendation", "ad", "image-provider",
)
DATA_STORES = ("valkey-cart", "astronomy-db", "flagd")
REMEDIABLE_SERVICES = (
    "checkout", "cart", "product-catalog", "currency", "payment", "shipping",
    "quote", "email", "recommendation", "ad", "frontend",
)


class Budget(BaseModel):
    """Bounds shared by every investigation mode, so comparisons stay fair."""

    max_tool_calls: int = 40
    max_llm_calls: int = 40
    max_rounds: int = 3
    max_tasks_per_round: int = 3
    wall_seconds: float = 420


class RecoveryPolicy(BaseModel):
    settle_seconds: float = 45
    observe_seconds: float = 120
    latency_ratio: float = 1.5      # recovered p95 must be <= baseline * ratio + slack
    latency_slack_ms: float = 25
    error_rate_slack: float = 0.01  # absolute error-rate allowance above baseline


class Settings(BaseModel):
    prometheus_url: str = "http://127.0.0.1:9090"
    jaeger_url: str = "http://127.0.0.1:16686/jaeger/ui"
    opensearch_url: str = "http://127.0.0.1:9200"
    grafana_url: str = "http://127.0.0.1:8080/grafana"
    docker_base_url: str = "unix:///var/run/docker.sock"
    demo_dir: Path = PROJECT_ROOT / "infra" / "otel-demo"
    data_dir: Path = PROJECT_ROOT / "data"
    model: str = ""
    coordinator_model: str = ""
    specialist_model: str = ""
    fallback_models: list[str] = Field(default_factory=list)
    model_retries: int = 4              # LiteLLM retries with backoff (free tiers rate-limit bursts)
    model_timeout: float = 180
    model_temperature: float = 0.1
    model_max_tokens: int = 8192        # thinking models spend many tokens before calling a tool
    ollama_num_ctx: int = 24576         # Ollama's default context silently truncates investigation prompts
    baseline_seconds: float = 600
    budget: Budget = Field(default_factory=Budget)
    recovery: RecoveryPolicy = Field(default_factory=RecoveryPolicy)
    http_timeout: float = 10

    @property
    def flagd_file(self) -> Path:
        return self.demo_dir / "src" / "flagd" / "demo.flagd.json"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "triage.sqlite"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    env = os.environ
    values: dict = {}
    for field, var in (
        ("prometheus_url", "SRE_PROMETHEUS_URL"),
        ("jaeger_url", "SRE_JAEGER_URL"),
        ("opensearch_url", "SRE_OPENSEARCH_URL"),
        ("grafana_url", "SRE_GRAFANA_URL"),
        ("docker_base_url", "SRE_DOCKER_URL"),
        ("demo_dir", "SRE_DEMO_DIR"),
        ("data_dir", "SRE_DATA_DIR"),
        ("model", "SRE_MODEL"),
        ("coordinator_model", "SRE_COORDINATOR_MODEL"),
        ("specialist_model", "SRE_SPECIALIST_MODEL"),
        ("model_retries", "SRE_MODEL_RETRIES"),
        ("model_timeout", "SRE_MODEL_TIMEOUT"),
        ("model_temperature", "SRE_MODEL_TEMPERATURE"),
        ("model_max_tokens", "SRE_MODEL_MAX_TOKENS"),
        ("ollama_num_ctx", "SRE_OLLAMA_NUM_CTX"),
        ("baseline_seconds", "SRE_BASELINE_SECONDS"),
    ):
        if env.get(var):
            values[field] = env[var]
    if env.get("SRE_FALLBACK_MODELS"):
        values["fallback_models"] = [m.strip() for m in env["SRE_FALLBACK_MODELS"].split(",") if m.strip()]
    budget = {}
    for field, var in (
        ("max_tool_calls", "SRE_MAX_TOOL_CALLS"),
        ("max_llm_calls", "SRE_MAX_LLM_CALLS"),
        ("max_rounds", "SRE_MAX_ROUNDS"),
        ("wall_seconds", "SRE_WALL_SECONDS"),
    ):
        if env.get(var):
            budget[field] = env[var]
    if budget:
        values["budget"] = Budget(**budget)
    return Settings(**values)

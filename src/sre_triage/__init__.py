"""Multi-agent SRE incident triage for the OpenTelemetry Demo (Google ADK)."""

import warnings

# ADK announces every experimental feature it uses (A2A, MCP, Workflow) and the LiteLLM route on each call.
warnings.filterwarnings("ignore", message=r"\[(EXPERIMENTAL|GEMINI_VIA_LITELLM)\]")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"google\.adk")

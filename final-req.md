# Agentic AI Project: Minimum Requirements

1. **Context Sharing & State Updates:** Intermediate outputs from one agent must feed into another agent. Use shared state to pass context between them.

2. **Tool Integration using MCP:** Use at least two external tools via MCP (library or custom), e.g., web search API, database lookup, file parser, or calculator.

3. **Structured Output using Pydantic:** The final output must be validated with a Pydantic model and presented in a structured form (JSON, Markdown, or table).

4. **Agentic Design Patterns:** Implement design patterns suited to your project's goals (e.g., sequential, router, parallel, reflection).

5. **Task Monitoring & Logging using Callbacks:** Use callback functions to track intermediate outputs, execution flow, and errors for transparency and debugging.

6. **Agent-to-Agent Communication using A2A:** Agents must communicate via the A2A protocol, demonstrating interoperability between an ADK agent and a CrewAI agent.

7. **Frameworks:** You MUST use both CrewAI and ADK. Do NOT use any other agentic framework.

8. **Security & Guardrails:** Use guardrails as safeguards and callbacks for data validation, e.g., type-checking tool outputs and screening user prompts for toxic language or prompt injection.

9. **Observability & Tracing:** Implement tracing for auditability and provenance using a platform such as Langfuse.

10. **Evaluation:** Perform an end-to-end evaluation covering performance, cost, and running time estimates.

11. **Deployment:** Deploy the project on a cloud platform.

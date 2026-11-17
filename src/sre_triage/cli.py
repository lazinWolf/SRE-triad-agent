"""Command line: health checks, fault lab, investigations, approvals, demo runs and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import PROJECT_ROOT, get_settings
from .llm import has_key
from .simulator import CATALOG, clear_active, generate, inject, probe, reset, save_active
from .telemetry import Backends, Unavailable, read_flags
from .tools import Window, health_scan

COLORS = {"node": "\033[36m", "agent": "\033[36m", "tool": "\033[90m", "finding": "\033[32m", "report": "\033[1;35m",
          "critique": "\033[33m", "error": "\033[31m", "approval": "\033[1;34m", "action": "\033[1;34m",
          "guardrail": "\033[1;31m", "plan": "\033[35m", "a2a": "\033[33m"}


def _print_event(e: dict) -> None:
    color = COLORS.get(e["kind"], "")
    text = e["text"] if e["kind"] != "thought" else e["text"][:300]
    print(f"{color}{time.strftime('%H:%M:%S', time.localtime(e['t']))} {e['agent']:>13} | {text}\033[0m", flush=True)


def _run_live(triage, case_id: str, resume: dict | None = None) -> dict:
    """Run the graph in a thread and stream the case's activity log to the terminal."""
    result: dict = {}
    seen = len(triage.cases.events(case_id))
    worker = threading.Thread(target=lambda: result.update(triage.run(case_id, resume)), daemon=True)
    worker.start()
    while worker.is_alive() or seen < len(triage.cases.events(case_id)):
        events = triage.cases.events(case_id)
        for e in events[seen:]:
            _print_event(e)
        seen = len(events)
        worker.join(0.5)
    return result


def _decide_interactively(triage, case_id: str, auto: str | None) -> dict:
    """Answer the case's requests (a question from the coordinator, or an approval) until none is left."""
    while (request := triage.pending(case_id)) is not None:
        if request.get("kind") == "question":
            print(f"\n>>> The coordinator asks: {request['question']}")
            answer = "" if auto else input("Your answer (empty = no additional information): ").strip()
            _run_live(triage, case_id, {"answer": answer})
            continue
        options = request.get("options") or [request]
        print("\n>>> Proposed action" + (f" (review board risk: {request['review_risk']})" if request.get("review_risk") else ""))
        for i, option in enumerate(options, 1):
            print(f"    {i}. {option['description']}  [risk {option['risk']}]")
        if auto == "approve":
            answer = "y"
        elif auto == "reject":
            answer = "n"
        else:
            answer = input("Approve? [y]es / [2] for the alternative / [n]o / or type feedback to re-investigate: ").strip()
        if answer.lower() in ("y", "yes", "1"):
            _run_live(triage, case_id, {"approved": True, "feedback": "", "choice": 0})
        elif answer == "2" and len(options) > 1:
            _run_live(triage, case_id, {"approved": True, "feedback": "", "choice": 1})
        elif answer.lower() in ("n", "no", ""):
            _run_live(triage, case_id, {"approved": False})
        else:
            _run_live(triage, case_id, {"approved": False, "feedback": answer})
    return triage.state(case_id)


def _summary(triage, case_id: str) -> None:
    meta = triage.cases.get(case_id)
    report = meta.get("report") or {}
    print(f"\n=== Case {case_id}: {meta['status']} ===")
    print(json.dumps({k: report.get(k) for k in ("status", "root_cause_service", "category", "mechanism",
                                                  "confidence", "remediation")}, indent=2))
    print("stats:", json.dumps(meta.get("stats", {})))


def _alert_line(a: dict) -> str:
    age = f"{time.time() - a['since']:.0f}s" if a["since"] else "?"
    return f"{a['state']:8} {a['severity']:8} {a['service'] or '-':16} {a['name']:26} {age:>6}  {a['summary']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sre-triage")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check backends, MCP servers, review board, tracing and model keys")
    sub.add_parser("health", help="anomaly scan of the last 3 minutes vs the healthy baseline")
    p = sub.add_parser("alerts", help="pending and firing alerts (Prometheus rules)")
    p.add_argument("--watch", action="store_true", help="keep printing; offer to investigate new alerts")
    sub.add_parser("scenarios", help="the fault catalog")
    p = sub.add_parser("inject", help="inject a catalog scenario, or --random for a generated one")
    p.add_argument("scenario", nargs="?", choices=CATALOG)
    p.add_argument("--random", type=int, metavar="SEED", help="generate one scenario from this seed")
    p.add_argument("--difficulty", default="mixed", choices=["easy", "medium", "hard", "mixed"])
    sub.add_parser("reset", help="restore the standard configuration (flags, limits, stopped or paused containers)")
    p = sub.add_parser("probe", help="check which scenarios really raise an alert (injects each one, then resets)")
    p.add_argument("--scenarios", default=",".join(CATALOG))
    p.add_argument("--wait", type=float, default=180)
    p = sub.add_parser("investigate")
    p.add_argument("symptom")
    p.add_argument("--since", type=float, default=180, help="incident started this many seconds ago")
    p.add_argument("--mode", choices=["multi", "single"], default="multi")
    p.add_argument("--model")
    p.add_argument("--auto", choices=["approve", "reject"])
    p = sub.add_parser("decide", help="approve or reject a waiting case")
    p.add_argument("case")
    p.add_argument("--reject", action="store_true")
    p.add_argument("--feedback", default="")
    p.add_argument("--choice", type=int, default=1, help="1 = proposed action, 2 = the alternative")
    p = sub.add_parser("answer", help="answer a question the coordinator asked")
    p.add_argument("case")
    p.add_argument("text")
    p = sub.add_parser("demo", help="inject a fault, wait for the alert, investigate, approve, verify, reset")
    p.add_argument("scenario", choices=CATALOG)
    p.add_argument("--wait", type=float, default=120, help="longest wait for an alert")
    p.add_argument("--variant", default="multi", help="multi, single, multi+memory or multi-no-review")
    p = sub.add_parser("eval", help="run an evaluation suite and write the performance / cost / time report")
    p.add_argument("--suite", default="quick", help="smoke, quick or full")
    p.add_argument("--runs", help="explicit scenario:variant pairs, e.g. payment_failure:multi,cart_cpu_limit:single")
    p.add_argument("--generated", type=int, default=0, metavar="N", help="add N generated scenarios (multi)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--wait", type=float, default=120)
    p.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run in data/eval/RUN_ID")
    sub.add_parser("memory", help="list long-term incident memory (ADK SqliteMemoryService)")
    p = sub.add_parser("mcp", help="serve the MCP tool servers (telemetry on PORT, ops on PORT+1)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8011)
    p = sub.add_parser("reviewer", help="serve the CrewAI review board over A2A (services/crew_reviewer)")
    p.add_argument("--port", type=int, default=8002)
    sub.add_parser("graph", help="print the ADK workflow graph (Mermaid)")
    sub.add_parser("ui", help="start the Streamlit dashboard")
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.cmd == "reviewer":
        return subprocess.call(["uv", "run", "python", "server.py", "--port", str(args.port)],
                               cwd=PROJECT_ROOT / "services" / "crew_reviewer")
    if args.cmd == "ui":
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("ui.py")),
                                "--server.headless", "true", "--server.address", "127.0.0.1",  # local only:
                                "--browser.gatherUsageStats", "false"])                       # it can change containers
    backends = Backends(settings)
    if args.cmd == "mcp":
        from . import mcp_servers

        print(f"MCP servers: telemetry http://{args.host}:{args.port}/mcp   ops http://{args.host}:{args.port + 1}/mcp")
        mcp_servers.serve(backends, args.host, args.port)
        return 0
    if args.cmd == "doctor":
        return doctor(settings, backends)
    if args.cmd == "health":
        now = time.time()
        for svc, h in health_scan(backends, Window(now - 180, now, settings.baseline_seconds)).items():
            print(f"{svc:>16}  p95 {h['baseline']['p95_ms']} -> {h['incident']['p95_ms']} ms   "
                  f"err {h['baseline']['error_ratio']} -> {h['incident']['error_ratio']}   {'; '.join(h['anomalies'])}")
        return 0
    if args.cmd == "alerts" and not args.watch:
        print("\n".join(_alert_line(a) for a in backends.prometheus.alerts()) or "no alerts")
        return 0
    if args.cmd == "scenarios":
        for s in CATALOG.values():
            print(f"{s.id:26} {s.difficulty:7} {s.title}  (expected: {s.service}/{s.category})")
        return 0
    if args.cmd == "inject":
        if args.random is None and not args.scenario:
            parser.error("give a scenario id or --random SEED")
        scenario = generate(args.random, 1, args.difficulty)[0] if args.random is not None else CATALOG[args.scenario]
        reset(backends)
        save_active(settings.data_dir, inject(scenario, backends))
        print(f"injected {scenario.id} at {time.strftime('%H:%M:%S')}" + ("" if args.scenario else "  (blind: see data/sim/active.json)"))
        return 0
    if args.cmd == "reset":
        print("\n".join(reset(backends)) or "already at standard configuration")
        clear_active(settings.data_dir)
        return 0
    if args.cmd == "probe":
        out = settings.data_dir / "sim" / "calibration.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:  # keep the results of scenarios that are not probed this time
            rows = [r for r in json.loads(out.read_text()) if r.get("scenario") in CATALOG]
        except (OSError, ValueError):
            rows = []
        for sid in args.scenarios.split(","):
            result = {**probe(CATALOG[sid], backends, wait=args.wait), "at": time.time()}
            rows = [r for r in rows if r.get("scenario") != sid] + [result]
            seen = ", ".join(sorted({f"{a['name']}:{a['service']}" for a in result["alerts"]})) or "-"
            print(f"{sid:26} detected={str(result['detected']):5} after {result['seconds'] if result['detected'] else '-':>5}s"
                  f"{'' if result['clean_start'] else '  (older alerts were still active)'}   {seen}", flush=True)
            out.write_text(json.dumps(rows, indent=2, default=str))
            time.sleep(60)  # recovery time; the next probe also waits until every alert has cleared
        return 0
    if args.cmd == "memory":
        from .memory import Memory

        for m in Memory(settings.data_dir).all():
            print(f"{m['id']:28} {m.get('root_cause_service')}/{m.get('category')}  {m['title'][:70]}  "
                  f"[{m.get('source')}, feedback={m.get('feedback')}]")
        return 0

    from .workflow import Triage

    triage = Triage(settings, backends)
    if args.cmd == "graph":
        print("graph TD")
        for src, dst, route in triage.edges():
            print(f"  {src} -->{f'|{route}|' if route else ''} {dst}")
        return 0
    if args.cmd == "alerts":
        return watch_alerts(triage, backends)
    if args.cmd == "investigate":
        case_id = triage.open_case(args.symptom, start=time.time() - args.since, mode=args.mode, model=args.model)
        _run_live(triage, case_id)
        _decide_interactively(triage, case_id, args.auto)
        _summary(triage, case_id)
        return 0
    if args.cmd == "decide":
        _run_live(triage, args.case, {"approved": not args.reject, "feedback": args.feedback, "choice": args.choice - 1})
        _decide_interactively(triage, args.case, None)
        _summary(triage, args.case)
        return 0
    if args.cmd == "answer":
        _run_live(triage, args.case, {"answer": args.text})
        _decide_interactively(triage, args.case, None)
        _summary(triage, args.case)
        return 0
    return evaluate(args, settings, backends)


def watch_alerts(triage, backends: Backends) -> int:
    """Event trigger for the terminal: print new alerts and ask whether to investigate (never starts by itself)."""
    seen: set[tuple] = set()
    print("watching alerts (Ctrl-C to stop)")
    try:
        while True:
            try:
                alerts = backends.prometheus.alerts()
            except Unavailable as exc:  # Prometheus restarting: keep watching
                print(f"Prometheus not reachable ({exc}); retrying")
                alerts = []
            for a in alerts:
                key = (a["name"], a["service"])
                if a["state"] != "firing" or key in seen:
                    continue
                seen.add(key)
                print("\n" + _alert_line(a))
                if input("Investigate this alert? [y/N] ").strip().lower() in ("y", "yes"):
                    case_id = triage.open_case(a["summary"], start=(a["since"] or time.time()) - 60, source="alert")
                    _run_live(triage, case_id)
                    _decide_interactively(triage, case_id, None)
                    _summary(triage, case_id)
            time.sleep(15)
    except KeyboardInterrupt:
        return 0


def evaluate(args, settings, backends: Backends) -> int:
    """`demo` (one scenario) and `eval` (a suite): every run is inject -> alert -> investigate -> approve -> verify."""
    from . import evaluation, guardrails
    from .workflow import Triage

    out = (settings.data_dir / "eval" / args.resume if getattr(args, "resume", None)
           else evaluation.new_run_dir(settings.data_dir, args.scenario if args.cmd == "demo" else args.suite))

    def make_triage(variant):
        return Triage(evaluation.variant_settings(settings, variant, out), backends)

    def live(triage):
        return lambda case_id, resume: _run_live(triage, case_id, resume)

    if args.cmd == "demo":
        runs = [(CATALOG[args.scenario], evaluation.VARIANTS[args.variant])]
    else:
        pairs = ([tuple(r.split(":")) for r in args.runs.split(",")] if args.runs else evaluation.SUITES[args.suite])
        runs = [(CATALOG[sid], evaluation.VARIANTS[variant]) for sid, variant in pairs]
        runs += [(s, evaluation.VARIANTS["multi"]) for s in generate(args.seed, args.generated, "mixed")]
    print(f"{len(runs)} runs -> {out}")
    screen = guardrails.evaluate(PROJECT_ROOT / "tests" / "data" / "guardrail_prompts.jsonl")
    results = evaluation.run_suite(make_triage, runs, out, wait=args.wait, run_factory=live, guardrails=screen)
    print_eval(results)
    print(f"\nreport: {out / 'report.md'}")
    return 0


def print_eval(results: list[dict]) -> None:
    print(f"\n{'scenario':26} {'variant':16} {'correct':8} {'service':8} {'fix':5} {'status':16} {'s':>5} {'llm':>4} {'usd':>8}")
    for r in results:
        s, stats = r.get("score") or {}, r.get("stats") or {}
        print(f"{r['scenario']:26} {r['variant']:16} {str(s.get('correct')):8} {str(s.get('service')):8} "
              f"{str(s.get('remediation')):5} {str(r.get('status')):16} {(r.get('timings') or {}).get('total_s') or 0:5.0f} "
              f"{stats.get('llm_calls', 0):4} {stats.get('cost_usd', 0):8.4f}")


def doctor(settings, backends: Backends) -> int:
    ok = True

    def check(name: str, fn) -> None:
        nonlocal ok
        try:
            detail = fn()
            print(f"  ok   {name}: {detail}")
        except Exception as exc:  # report every failure, keep checking
            ok = False
            print(f"  FAIL {name}: {exc}")

    now = time.time()
    check("prometheus", lambda: f"checkout p95 {backends.prometheus.service_red('checkout', now - 120, now)['p95_ms']} ms")
    def container_metrics() -> str:
        cpu = backends.prometheus.container("cart", now - 120, now)["cpu_cores_used"]
        if cpu is None:
            raise Unavailable("no docker_stats metrics; run: docker restart otel-collector")
        return f"cart uses {cpu} cores"

    check("container metrics", container_metrics)
    check("jaeger", lambda: f"{len(backends.jaeger.search('checkout', now - 120, now, limit=5))} recent checkout traces")
    check("opensearch", lambda: f"{backends.logs.search(now - 600, now, limit=1)['total']} warning+ logs in 10 min")
    check("docker", lambda: f"cart cpu limit {backends.docker.inspect('cart')['cpu_limit']}")
    check("flagd file", lambda: f"{sum(1 for f in read_flags(settings.flagd_file).values() if f['variant'] != 'off')} flags on")
    check("alert rules", lambda: f"{len(backends.prometheus.alerts())} alerts pending or firing; cart baseline "
                                 f"{(backends.prometheus.service_baseline('cart', now) or {}).get('p95_ms')} ms")
    check("standard config", lambda: f"{len(backends.standard['services'])} containers, "
                                     f"{len(backends.standard['flags'].get('flags', {}))} flags"
          if backends.standard["services"] else (_ for _ in ()).throw(Unavailable("run: infra/sre/demo.sh standard")))

    def mcp() -> str:
        from . import mcp_servers

        urls = mcp_servers.ensure(backends, settings.mcp_url)
        listing = mcp_servers.tools_listing(urls)
        return ", ".join(f"{kind}: {len(tools)} tools" for kind, tools in listing.items()) \
            + (f" at {settings.mcp_url}" if settings.mcp_url else " (embedded)")

    check("MCP tool servers", mcp)
    if settings.reviewer_a2a_url:
        from .a2a_critic import agent_card

        check("review board (CrewAI, A2A)", lambda: f"{agent_card(settings.reviewer_a2a_url)['name']} at {settings.reviewer_a2a_url}")
    else:
        print("  --   review board (CrewAI, A2A): SRE_REVIEWER_A2A_URL not set; the local critic reviews reports")
    from . import tracing

    print(f"  {'ok  ' if tracing.enabled() else '--  '} Langfuse tracing: "
          + ("keys set" if tracing.enabled() else "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set"))
    chain = [settings.model] + settings.fallback_models
    check("model keys", lambda: ", ".join(f"{m}{'' if has_key(m) else ' (NO KEY)'}" for m in chain))
    if any(m.startswith("vertex_ai/") for m in chain):
        check("Vertex AI credentials", vertex_credentials)
    return 0 if ok else 1


def vertex_credentials() -> str:
    """Fetch a Google token the way LiteLLM will (the VM's service account, or a gcloud ADC login). No model is called."""
    import google.auth
    from google.auth.transport.requests import Request

    creds, adc_project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    project = os.getenv("VERTEXAI_PROJECT") or adc_project
    if not project:
        raise Unavailable("set VERTEXAI_PROJECT in .env")
    location = os.getenv("VERTEXAI_LOCATION") or "us-central1 (LiteLLM's default; Gemini 3.5 needs global)"
    return f"token ok ({type(creds).__module__.split('.')[-1]}), project {project}, location {location}"


if __name__ == "__main__":
    raise SystemExit(main())

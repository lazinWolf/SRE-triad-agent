"""Command line: health checks, fault lab, investigations, approvals, demo runs and evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import PROJECT_ROOT, get_settings
from .faults import SCENARIOS, inject, reset, score
from .llm import has_key
from .telemetry import Backends, Unavailable, read_flags
from .tools import Window, health_scan

COLORS = {"node": "\033[36m", "tool": "\033[90m", "finding": "\033[32m", "report": "\033[1;35m",
          "critique": "\033[33m", "error": "\033[31m", "approval": "\033[1;34m", "action": "\033[1;34m"}


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
    while (request := triage.pending(case_id)) is not None:
        print(f"\n>>> Proposed action: {request['description']}")
        if auto == "approve":
            answer = "y"
        elif auto == "reject":
            answer = "n"
        else:
            answer = input("Approve? [y]es / [n]o / or type feedback to re-investigate: ").strip()
        if answer.lower() in ("y", "yes"):
            _run_live(triage, case_id, {"approved": True})
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sre-triage")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check backends, flags and model keys")
    sub.add_parser("health", help="anomaly scan of the last 3 minutes vs the 5 before")
    sub.add_parser("scenarios")
    p = sub.add_parser("inject")
    p.add_argument("scenario", choices=SCENARIOS)
    sub.add_parser("reset", help="all flags off, all CPU limits back to 1.0")
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
    p = sub.add_parser("demo", help="inject a fault, investigate, approve, verify, reset")
    p.add_argument("scenario", choices=SCENARIOS)
    p.add_argument("--wait", type=float, default=100, help="seconds to let the fault develop")
    p.add_argument("--mode", choices=["multi", "single"], default="multi")
    p.add_argument("--model")
    p = sub.add_parser("eval", help="run scenarios in both modes and score them")
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--modes", default="multi,single")
    p.add_argument("--runs", help="explicit scenario:mode pairs, e.g. payment_failure:multi,cart_cpu_limit:single")
    p.add_argument("--wait", type=float, default=100)
    p.add_argument("--cooldown", type=float, default=330, help="healthy time between runs (>= baseline window)")
    sub.add_parser("memory", help="list long-term incident memory (ADK SqliteMemoryService)")
    p = sub.add_parser("a2a-critic", help="serve the Critic agent over A2A")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    sub.add_parser("graph", help="print the ADK workflow graph (Mermaid)")
    sub.add_parser("ui", help="start the Streamlit dashboard")
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.cmd == "a2a-critic":
        from .a2a_critic import serve

        print(f"Critic A2A service on http://{args.host}:{args.port}  "
              f"(agent card: http://{args.host}:{args.port}/.well-known/agent-card.json)")
        serve(settings, args.host, args.port)
        return 0
    if args.cmd == "ui":
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("ui.py")),
                                "--server.headless", "true", "--browser.gatherUsageStats", "false"])
    backends = Backends(settings)
    if args.cmd == "doctor":
        return doctor(settings, backends)
    if args.cmd == "health":
        now = time.time()
        for svc, h in health_scan(backends, Window(now - 180, now, settings.baseline_seconds)).items():
            print(f"{svc:>16}  p95 {h['baseline']['p95_ms']} -> {h['incident']['p95_ms']} ms   "
                  f"err {h['baseline']['error_ratio']} -> {h['incident']['error_ratio']}   {'; '.join(h['anomalies'])}")
        return 0
    if args.cmd == "scenarios":
        for s in SCENARIOS.values():
            print(f"{s.id:24} {s.title}  (expected: {s.service}/{s.category})")
        return 0
    if args.cmd == "inject":
        inject(SCENARIOS[args.scenario], backends)
        print(f"injected {args.scenario} at {time.strftime('%H:%M:%S')}")
        return 0
    if args.cmd == "reset":
        print("\n".join(reset(backends)) or "already at standard configuration")
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
    if args.cmd == "investigate":
        case_id = triage.open_case(args.symptom, start=time.time() - args.since, mode=args.mode, model=args.model)
        _run_live(triage, case_id)
        _decide_interactively(triage, case_id, args.auto)
        _summary(triage, case_id)
        return 0
    if args.cmd == "decide":
        _run_live(triage, args.case, {"approved": not args.reject, "feedback": args.feedback})
        _summary(triage, args.case)
        return 0
    if args.cmd == "demo":
        result = run_scenario(triage, backends, args.scenario, args.mode, args.wait, args.model)
        print("score:", json.dumps(result["score"]))
        return 0
    if args.cmd == "eval":
        results = []
        out = settings.data_dir / "eval.json"
        runs = ([tuple(r.split(":")) for r in args.runs.split(",")] if args.runs else
                [(s, m) for s in args.scenarios.split(",") for m in args.modes.split(",")])
        for i, (sid, mode) in enumerate(runs):
            if i:
                print(f"\n... cooling down {args.cooldown:.0f}s so the next baseline is healthy")
                time.sleep(args.cooldown)
            results.append(run_scenario(triage, backends, sid, mode, args.wait))
            out.write_text(json.dumps(results, indent=2))
        print_eval(results)
        return 0
    return 1


def run_scenario(triage, backends: Backends, scenario_id: str, mode: str, wait: float, model: str | None = None) -> dict:
    scenario = SCENARIOS[scenario_id]
    reset(backends)
    print(f"\n##### {scenario.id} ({mode}): {scenario.title}")
    started = time.time()
    inject(scenario, backends)
    print(f"injected; letting the fault develop for {wait:.0f}s")
    time.sleep(wait)
    case_id = triage.open_case(scenario.symptom, start=started, mode=mode, model=model, scenario=scenario.id)
    _run_live(triage, case_id)
    _decide_interactively(triage, case_id, "approve")
    _summary(triage, case_id)
    meta = triage.cases.get(case_id)
    result = {"case": case_id, "scenario": scenario.id, "mode": mode, "status": meta["status"],
              "score": score(scenario, meta.get("report")), "stats": meta.get("stats", {}),
              "report": meta.get("report")}
    triage.cases.update(case_id, score=result["score"])
    reset(backends)
    return result


def print_eval(results: list[dict]) -> None:
    print(f"\n{'scenario':24} {'mode':7} {'correct':8} {'service':8} {'fix':5} {'outcome':16} {'s':>5} {'llm':>4}")
    for r in results:
        s = r["score"]
        print(f"{r['scenario']:24} {r['mode']:7} {str(s['correct']):8} {str(s['service']):8} "
              f"{str(s['remediation']):5} {r['status']:16} {r['stats'].get('seconds', 0):5.0f} {r['stats'].get('llm_calls', 0):4}")


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
    if settings.critic_a2a_url:
        from .a2a_critic import agent_card

        check("critic A2A service", lambda: f"{agent_card(settings.critic_a2a_url)['name']} at {settings.critic_a2a_url}")
    chain = [settings.model] + settings.fallback_models
    check("model keys", lambda: ", ".join(f"{m}{'' if has_key(m) else ' (NO KEY)'}" for m in chain))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

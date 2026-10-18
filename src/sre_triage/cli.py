"""Command line: run the worker and UI, open and investigate cases, inject faults, evaluate."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time

from .config import PROJECT_ROOT, get_settings


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_worker(args) -> None:
    from .worker import Worker

    worker = Worker(get_settings(), auto_investigate=None if not args.no_auto else False, alerts=not args.no_alerts)
    print(f"worker running (model={worker.settings.model or 'unset'}, auto-investigate={worker.auto_investigate}, "
          f"alerts={worker.alerts_enabled}); Ctrl+C to stop", flush=True)
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        pass


def cmd_ui(args) -> None:
    app = PROJECT_ROOT / "src" / "sre_triage" / "ui" / "app.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
                              "--server.address", "127.0.0.1", "--server.headless", "true",
                              "--browser.gatherUsageStats", "false"]))


def cmd_open(args) -> None:
    from .ledger import Ledger
    from .models import CaseInput

    settings = get_settings()
    ledger = Ledger(settings.db_path)
    now = time.time()
    case = ledger.create_case(CaseInput(
        service=args.service, symptom=args.symptom, incident_start=now - args.minutes * 60, incident_end=now,
        baseline_seconds=args.baseline_minutes * 60, mode=args.mode, use_memory=not args.no_memory,
        model=args.model))
    print(case["id"])
    if args.investigate:
        args.case_id = case["id"]
        cmd_investigate(args)


def cmd_investigate(args) -> None:
    from .ledger import Ledger
    from .orchestrator import investigate

    settings = get_settings()
    ledger = Ledger(settings.db_path)
    result = asyncio.run(investigate(ledger, settings, args.case_id, on_progress=lambda m: print(m, flush=True)))
    _print({k: result[k] for k in ("case_id", "run_id", "status", "error", "stats")})
    _print(result["report"])


def cmd_feedback(args) -> None:
    from .ledger import Ledger

    ledger = Ledger(get_settings().db_path)
    ledger.add_feedback(args.case_id, args.text)
    print("feedback saved; the previous report is invalidated. Run `sre-triage investigate CASE_ID` again.")


def cmd_decide(args) -> None:
    from .ledger import Ledger

    ledger = Ledger(get_settings().db_path)
    action = ledger.decide_action(args.action_id, approved=args.approve, who="cli-operator", note=args.note)
    if args.approve:
        ledger.enqueue("execute_action", action["case_id"], {"action_id": action["id"]})
        print("approved; the worker will execute it and then verify recovery")
    else:
        ledger.set_case(action["case_id"], status="needs_review")
        print("rejected")


def cmd_review(args) -> None:
    from .ledger import Ledger
    from .memory import review_case
    from .models import ReviewInput

    ledger = Ledger(get_settings().db_path)
    print(review_case(ledger, args.case_id, ReviewInput(
        outcome=args.outcome, confirmed_service=args.service, confirmed_category=args.category,
        confirmed_cause=args.cause, notes=args.notes)))


def cmd_cases(args) -> None:
    from .ledger import Ledger

    ledger = Ledger(get_settings().db_path)
    for c in ledger.cases(args.limit):
        report = c.get("report") or {}
        print(f"{c['id']}  {time.strftime('%m-%d %H:%M', time.localtime(c['created_at']))}  {c['status']:<17} "
              f"{c['mode']:<12} {c['service']:<16} {report.get('diagnosis_status', '-'):<21} {c['symptom'][:60]}")


def cmd_inject(args) -> None:
    from .faults import SCENARIOS, inject

    if args.scenario not in SCENARIOS:
        sys.exit(f"unknown scenario; choose from: {', '.join(SCENARIOS)}")
    _print(inject(SCENARIOS[args.scenario], get_settings().flagd_file))


def cmd_reset(args) -> None:
    from .faults import reset

    _print(reset(get_settings().flagd_file))


def cmd_scenarios(args) -> None:
    from .faults import SCENARIOS

    for s in SCENARIOS.values():
        print(f"{s.id:<24} {s.difficulty:<8} alert={s.alert_service:<10} {s.title}")


def cmd_seed(args) -> None:
    from .ledger import Ledger
    from .memory import load_seeds

    print(f"loaded {load_seeds(Ledger(get_settings().db_path), PROJECT_ROOT / 'seeds' / 'incidents.json')} seed incidents")


def cmd_eval(args) -> None:
    from .evaluation import run_evaluation
    from .faults import SCENARIOS

    ids = args.scenarios or [s for s in SCENARIOS]
    unknown = [s for s in ids if s not in SCENARIOS]
    if unknown:
        sys.exit(f"unknown scenarios: {unknown}")
    modes = tuple(args.modes)
    result = asyncio.run(run_evaluation(get_settings(), ids, modes=modes, baseline_seconds=args.baseline,
                                        use_memory=not args.no_memory, repeats=args.repeats))
    _print({"eval_id": result["eval_id"], "out_dir": result["out_dir"], "summary": result["summary"]})


def cmd_models(args) -> None:
    from .llm import CATALOG, has_credentials, probe

    settings = get_settings()
    targets = args.models or [m.id for m in CATALOG]
    print(f"default SRE_MODEL={settings.model or '(unset)'}")
    if not args.probe:
        for m in CATALOG:
            print(f"{'✓' if has_credentials(m.id) else '✗ no key'}  {m.id:<55} {m.label}  {m.notes}")
        print("run with --probe to test tool calling for real")
        return

    async def run_all():
        return [await probe(settings, m) for m in targets]

    for r in asyncio.run(run_all()):
        status = "PASS" if r["ok"] else "FAIL"
        print(f"{status}  {r['model']:<55} {r.get('seconds', '-')}s  calls={r.get('llm_calls', '-')}  "
              f"tokens={r.get('tokens', '-')}  {r.get('error') or ''}"[:220])


def cmd_doctor(args) -> None:
    from .doctor import run_checks

    ok = asyncio.run(run_checks(get_settings()))
    sys.exit(0 if ok else 1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "LiteLLM", "litellm", "google_adk", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="sre-triage", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("worker", help="run the change audit, alert rules and job queue")
    p.add_argument("--no-auto", action="store_true", help="do not investigate alert cases automatically")
    p.add_argument("--no-alerts", action="store_true")
    p.set_defaults(func=cmd_worker)

    p = sub.add_parser("ui", help="run the Streamlit dashboard")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("open", help="open a case for the last N minutes")
    p.add_argument("service")
    p.add_argument("symptom")
    p.add_argument("--minutes", type=float, default=5)
    p.add_argument("--baseline-minutes", type=float, default=10)
    p.add_argument("--mode", choices=["multi_agent", "single_agent"], default="multi_agent")
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--model", default="", help="LiteLLM model for this case (default: SRE_MODEL)")
    p.add_argument("--investigate", action="store_true", help="investigate immediately in this process")
    p.set_defaults(func=cmd_open)

    p = sub.add_parser("investigate", help="investigate a case in this process")
    p.add_argument("case_id")
    p.set_defaults(func=cmd_investigate)

    p = sub.add_parser("feedback", help="add an operator correction (invalidates the report)")
    p.add_argument("case_id")
    p.add_argument("text")
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("decide", help="approve or reject a proposed action")
    p.add_argument("action_id")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--approve", action="store_true")
    g.add_argument("--reject", action="store_true")
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("review", help="record a reviewed outcome into incident memory")
    p.add_argument("case_id")
    p.add_argument("--outcome", choices=["resolved", "mitigated", "unresolved", "false_alarm"], required=True)
    p.add_argument("--service", default="")
    p.add_argument("--category", default="")
    p.add_argument("--cause", default="")
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("cases", help="list cases")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_cases)

    p = sub.add_parser("scenarios", help="list fault scenarios")
    p.set_defaults(func=cmd_scenarios)
    p = sub.add_parser("inject", help="apply a fault scenario (operator only)")
    p.add_argument("scenario")
    p.set_defaults(func=cmd_inject)
    p = sub.add_parser("reset", help="restore all flags and CPU limits")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("seed-memory", help="load seeded historical incidents")
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("eval", help="run scenarios in multi-agent and single-agent modes and score them")
    p.add_argument("scenarios", nargs="*")
    p.add_argument("--modes", nargs="+", default=["multi_agent", "single_agent"], choices=["multi_agent", "single_agent"])
    p.add_argument("--baseline", type=float, default=300, help="healthy baseline seconds before each fault")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--no-memory", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("models", help="list catalog models; --probe tests tool calling for real")
    p.add_argument("models", nargs="*")
    p.add_argument("--probe", action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("doctor", help="check backends, metric names and model configuration")
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args()
    if getattr(args, "reject", False):
        args.approve = False
    args.func(args)


if __name__ == "__main__":
    main()

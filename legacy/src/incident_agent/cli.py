"""Run cases, record corrections and serve the local operator interface."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .models import CaseInput, ReviewInput
from .storage import CaseStore


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.getenv("SRE_DATA_DIR", "data"))
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Investigate a JSON alert using live LLMs or scripted ADK replay")
    run.add_argument("alert", type=Path)
    run.add_argument("--mode", choices=["live", "replay"], default="replay")
    run.add_argument("--output", type=Path)
    rerun = commands.add_parser("investigate", help="Investigate an existing case after operator feedback")
    rerun.add_argument("case_id")
    feedback = commands.add_parser("feedback")
    feedback.add_argument("case_id")
    feedback.add_argument("text")
    review = commands.add_parser("review")
    review.add_argument("case_id")
    review.add_argument("--cause", default="", help="Confirmed cause, if known; recovery alone does not confirm one")
    review.add_argument("--outcome", choices=["unresolved", "mitigated", "resolved"], required=True)
    review.add_argument("--notes", default="")
    commands.add_parser("list")
    commands.add_parser("memory")
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    os.environ["SRE_DATA_DIR"] = args.data_dir
    if args.command == "serve":
        import uvicorn
        uvicorn.run("incident_agent.api:app", host="127.0.0.1", port=args.port)
        return
    store = CaseStore(Path(args.data_dir) / "cases.sqlite")
    if args.command in ("run", "investigate"):
        from .api import service_urls
        from .workflow import investigate
        if args.command == "run":
            item = CaseInput.model_validate({**json.loads(args.alert.read_text()), "mode": args.mode})
            case = store.create(item)
            case_id = case["id"]
        else:
            case_id = args.case_id
        result = asyncio.run(investigate(store, case_id, service_urls()))
        output = json.dumps(result, indent=2)
        if getattr(args, "output", None):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n")
        print(output)
        if result["error"]:
            raise SystemExit(1)
    elif args.command == "feedback":
        print(json.dumps(store.feedback(args.case_id, args.text), indent=2))
    elif args.command == "review":
        print(json.dumps(store.review(args.case_id, ReviewInput(confirmed_cause=args.cause, outcome=args.outcome, notes=args.notes)), indent=2))
    elif args.command == "memory":
        print(json.dumps(store.memory(limit=100), indent=2))
    else:
        print(json.dumps(store.list(), indent=2))


if __name__ == "__main__":
    main()

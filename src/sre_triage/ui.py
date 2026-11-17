"""Streamlit dashboard: alerts, investigations with the agent graph live, approvals, fault lab, ADK internals, evaluation."""

from __future__ import annotations

import asyncio
import os
import random
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `streamlit run` without installing

from sre_triage import adk_view, guardrails, mcp_servers, remediation, simulator, tracing  # noqa: E402
from sre_triage.a2a_critic import agent_card  # noqa: E402
from sre_triage.config import PROJECT_ROOT, get_settings  # noqa: E402
from sre_triage.llm import SUGGESTED_MODELS, has_key  # noqa: E402
from sre_triage.simulator import CATALOG, Fault, Scenario  # noqa: E402
from sre_triage.telemetry import Unavailable  # noqa: E402
from sre_triage.tools import SERVER_TOOLS, Window, health_scan  # noqa: E402
from sre_triage.workflow import Triage  # noqa: E402

ACTIVE = {"investigating", "remediating", "verifying"}
WAITING = {"awaiting_approval", "awaiting_answer"}
FINAL = {"resolved", "not_recovered", "rejected", "diagnosed_no_action", "inconclusive", "no_incident", "blocked",
         "action_failed", "failed"}
STATUS_ICON = {"investigating": "🔎", "awaiting_approval": "✋", "awaiting_answer": "❓", "remediating": "🛠️",
               "verifying": "⏱️", "resolved": "✅", "not_recovered": "⚠️", "inconclusive": "❔", "rejected": "🚫",
               "diagnosed_no_action": "📝", "no_incident": "🟢", "blocked": "🛡️", "failed": "❌", "action_failed": "❌"}
KIND_ICON = {"node": "▶", "agent": "🤖", "tool": "🔧", "thought": "💭", "finding": "📌", "plan": "🗺️", "report": "📄",
             "critique": "🧐", "approval": "✋", "action": "🛠️", "result": "📊", "a2a": "🔗", "guardrail": "🛡️",
             "error": "❌"}
GUARD_PROMPTS = PROJECT_ROOT / "tests" / "data" / "guardrail_prompts.jsonl"

st.set_page_config(page_title="SRE Incident Triage", page_icon="🚨", layout="wide")


@st.cache_resource
def triage() -> Triage:
    return Triage(get_settings())


def start_in_background(fn, *args) -> None:
    threading.Thread(target=fn, args=args, daemon=True).start()


def ago(ts: float | None) -> str:
    seconds = max(0, time.time() - ts) if ts else 0
    return f"{seconds:.0f} s ago" if seconds < 90 else f"{seconds / 60:.0f} min ago"


def active_scenario() -> str | None:
    return (simulator.load_active(get_settings().data_dir) or {}).get("scenario")


# ---------------------------------------------------------------------------- graph drawing
NODE_KINDS = {  # kind -> (shape, fill, border): the shape and colour tell what a step is
    "start": ("ellipse", "#e5e7eb", "#6b7280"),
    "llm": ("box", "#d8c8f7", "#6b3fc0"),
    "code": ("box", "#e5e7eb", "#6b7280"),
    "memory": ("cylinder", "#bfe0fb", "#2b6cb0"),
    "human": ("octagon", "#ffe8a3", "#b7791f"),
    "action": ("box", "#fbc4c4", "#c0392b"),
}
GRAPH_NODES = {  # workflow node -> (kind, what it is)
    "__START__": ("start", ""),
    "detect": ("code", "anomaly scan + alerts, no LLM"),
    "recall": ("memory", "search ADK memory"),
    "coordinator": ("llm", "decides: investigate, ask, conclude<BR/>Coordinator LlmAgent"),
    "specialists": ("llm", "metrics · traces · logs_changes<BR/>parallel LlmAgents (MCP tools)"),
    "collect": ("code", "join the findings"),
    "question": ("human", "the coordinator asks the operator"),
    "answer": ("code", "feed the answer back"),
    "review": ("llm", "CrewAI review board over A2A<BR/>or local critic LlmAgent"),
    "approval": ("human", "human approves the fix"),
    "decide": ("code", "route the operator's answer"),
    "remediate": ("action", "apply the approved fix"),
    "verify": ("code", "re-check metrics, roll back"),
    "learn": ("memory", "store verified fixes in ADK memory"),
}
ROUTE_COLOR = {"accept": "#2d6a4f", "approve": "#2d6a4f", "done": "#2d6a4f", "revise": "#e76f51", "feedback": "#e76f51",
               "reject": "#e76f51", "retry": "#e76f51"}


def graph_dot(events: list[dict], status: str) -> str:
    """The ADK workflow graph: shape and colour show what each step is, the border and fill show progress."""
    edges = triage().edges()
    visited = [e["agent"] for e in events if e["kind"] == "node"]
    current = visited[-1] if visited and (status in ACTIVE or status in WAITING) else None
    lines = ["digraph G {", "rankdir=TB; bgcolor=transparent; nodesep=0.45; ranksep=0.5; "
             "node [fontname=Helvetica fontsize=15 fontcolor=\"#1f2933\" margin=\"0.18,0.1\"]; "
             "edge [color=\"#888888\" fontname=Helvetica fontsize=12 fontcolor=\"#888888\"];"]
    names = list(dict.fromkeys([n for e in edges for n in e[:2]]))
    for name in names:
        kind, about = GRAPH_NODES.get(name, ("code", ""))
        shape, fill, border = NODE_KINDS[kind]
        count = visited.count(name)
        title = ("start" if name == "__START__" else name) + (f" ×{count}" if count > 1 and name != "specialists" else "")
        label = f"<B>{title}</B>" + (f'<BR/><FONT POINT-SIZE="12">{about}</FONT>' if about else "")
        width = 1.2
        if name in visited or name == "__START__":
            border, width = "#2d6a4f", 3
        if name == current:
            fill = "#ffd166" if status in WAITING else "#f4a261"
            width = 4
        style = "rounded,filled" if shape == "box" else "filled"
        extra = " peripheries=2" if name == "specialists" else ""
        lines.append(f'"{name}" [label=<{label}> shape={shape} style="{style}" fillcolor="{fill}" '
                     f'color="{border}" penwidth={width}{extra}];')
    for src, dst, route in edges:
        color = ROUTE_COLOR.get(route or "", "#888888")
        lines.append(f'"{src}" -> "{dst}"' + (f' [label="{route}" style=dashed color="{color}" fontcolor="{color}"]'
                                              if route else "") + ";")
    lines.append("}")
    return "\n".join(lines)


def graph_legend() -> None:
    st.caption("🟪 LLM agent (ADK `LlmAgent` / A2A) · ⬜ code step · 🟦 memory (cylinder) · 🟨 human in the loop (octagon) · "
               "🟥 acts on the system · double border = parallel agents · green border = done · "
               "orange fill = running now · orange arrows = loops back")


# ---------------------------------------------------------------------------- sidebar
def sidebar() -> tuple[str, str]:
    settings = get_settings()
    st.sidebar.header("Settings")
    options = [m for m in dict.fromkeys([settings.model, *settings.fallback_models, *SUGGESTED_MODELS]) if has_key(m)]
    model = st.sidebar.selectbox("Model (LiteLLM)", options or [settings.model],
                                 help="Rate-limited models fall back to SRE_FALLBACK_MODELS automatically.")
    mode = st.sidebar.radio("Mode", ["multi", "single"], format_func=lambda m: {
        "multi": "Multi-agent (coordinator + specialists + review board)", "single": "Single agent (baseline)"}[m])
    st.sidebar.caption("Inject faults on the Fault lab page (operator only, agents never see it).")
    observability_links()
    return model, mode


def observability_links() -> None:
    """Browser links. On the VM they work through the SSH tunnel, which forwards the same local ports."""
    shop = f"http://localhost:{os.getenv('SRE_SHOP_PORT', '8090')}"
    links = [f"[Grafana]({shop}/grafana/)", f"[Jaeger]({shop}/jaeger/ui/)", "[Prometheus](http://localhost:9090)",
             f"[Feature flags]({shop}/feature/)", f"[Shop]({shop}/)"]
    if tracing.enabled():
        links.insert(0, f"[Langfuse]({os.getenv('LANGFUSE_BASE_URL', 'https://cloud.langfuse.com')})")
    st.sidebar.subheader("Observability")
    st.sidebar.markdown(" · ".join(links))


# ---------------------------------------------------------------------------- incidents
@st.fragment(run_every=15)
def health_panel() -> None:
    now = time.time()
    try:
        scan = health_scan(triage().backends, Window(now - 120, now, 300))
    except Exception as exc:
        st.error(f"Telemetry unavailable: {exc}")
        return
    rows = [{"service": s, "p95 ms (baseline → now)": f"{h['baseline']['p95_ms']} → {h['incident']['p95_ms']}",
             "errors (baseline → now)": f"{(h['baseline']['error_ratio'] or 0):.1%} → {(h['incident']['error_ratio'] or 0):.1%}",
             "alert": "; ".join(h["anomalies"]) or "–"} for s, h in scan.items()]
    alerts = [r for r in rows if r["alert"] != "–"]
    st.markdown(f"**Live health** (last 2 min vs the healthy baseline): "
                f"{'🔴 ' + str(len(alerts)) + ' service(s) anomalous' if alerts else '🟢 all normal'}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=250)


@st.fragment(run_every=15)
def alert_inbox(model: str, mode: str) -> None:
    """Firing and pending Prometheus alerts by service (earliest first); investigating one opens a case from it."""
    t, now = triage(), time.time()
    try:
        alerts = t.backends.prometheus.alerts()
    except Unavailable:
        st.caption("🔕 Alert inbox: alerts unavailable (Prometheus is unreachable)")
        return
    snoozed = st.session_state.setdefault("snoozed", {})
    alerts = [a for a in alerts if snoozed.get(f"{a['name']}|{a['service']}", 0) < now]
    busy = {c.get("alert_service"): c["id"] for c in t.cases.list() if c.get("source") == "alert" and c["status"] not in FINAL}
    st.markdown(f"**Alert inbox** · {'🔴 ' + str(len(alerts)) + ' alert(s)' if alerts else '🟢 no active alerts'}")
    groups: dict[str, list[dict]] = {}
    for a in sorted(alerts, key=lambda a: a["since"] or now):
        groups.setdefault(a["service"] or "unknown", []).append(a)
    for service, rows in groups.items():
        with st.container(border=True):
            st.markdown(f"**{service}**" + (f" · already being investigated in case `{busy[service]}`" if service in busy else ""))
            for a in rows:
                key = f"{a['name']}|{a['service']}"
                c = st.columns([4, 1.4, 1.2])
                c[0].markdown(f"{'🔴' if a['state'] == 'firing' else '🟡'} **{a['severity']}** `{a['name']}` ({a['state']}) "
                              f"since {ago(a['since'])} · {a['summary']}")
                if service not in busy and c[1].button("Investigate", key=f"inv_{key}"):
                    case_id = t.open_case(a["summary"] or a["name"], start=(a["since"] or now) - 60, mode=mode, model=model,
                                          scenario=active_scenario(), source="alert")
                    t.cases.update(case_id, alert_service=service)
                    start_in_background(t.run, case_id)
                    st.session_state["case"] = case_id
                    st.rerun()
                if c[2].button("Snooze", key=f"snz_{key}", help="Hide this alert for 15 minutes"):
                    snoozed[key] = now + 900
                    st.rerun(scope="fragment")


def incidents_page() -> None:
    model, mode = sidebar()
    st.title("🚨 SRE Incident Triage")
    st.caption("A Google ADK workflow of LlmAgents investigates live OpenTelemetry Demo telemetry, "
               "proposes a fix, and acts only after your approval.")
    health_panel()
    alert_inbox(model, mode)
    with st.expander("Describe a symptom manually"):
        with st.form("new_case"):
            c1, c2 = st.columns([3, 1])
            symptom = c1.text_input("Alert / symptom", "Checkout latency is elevated")
            since = c2.number_input("Incident began (minutes ago)", 1.0, 30.0, 3.0, 0.5)
            if st.form_submit_button("Investigate", type="primary"):
                t = triage()
                case_id = t.open_case(symptom, start=time.time() - since * 60, mode=mode, model=model,
                                      scenario=active_scenario())
                start_in_background(t.run, case_id)
                st.session_state["case"] = case_id
    cases = triage().cases.list()
    if not cases:
        st.info("No cases yet. Investigate an alert above, or inject a fault on the Fault lab page and wait ~90 s.")
        return
    ids = [c["id"] for c in cases]
    current = st.session_state.get("case")
    case_id = st.selectbox("Case", ids, index=ids.index(current) if current in ids else 0,
                           format_func=lambda i: next(f"{STATUS_ICON.get(c['status'], '')} {c['id']} · {c['symptom']} · "
                                                      f"{c['status']} · {c['mode']}" for c in cases if c["id"] == i))
    st.session_state["case"] = case_id
    case_view(case_id)


@st.fragment(run_every=3)
def case_view(case_id: str) -> None:
    t = triage()
    meta = t.cases.get(case_id)
    events = t.cases.events(case_id)
    state = t.state(case_id)
    status = meta["status"]
    report = state.get("report") or meta.get("report") or {}
    st.markdown(f"### {STATUS_ICON.get(status, '')} {status.replace('_', ' ')}  ·  `{case_id}`")
    stats = meta.get("stats") or {}
    cost = f" · ${stats['cost_usd']:.4f} list price" if "cost_usd" in stats else ""
    st.caption(f"model {meta.get('model')} · mode {meta['mode']} · source {meta.get('source', 'operator')} · "
               f"{len(state.get('evidence', []))} evidence items · {stats.get('llm_calls', '…')} LLM calls · "
               f"{stats.get('seconds', '…')} s · {stats.get('tokens', 0)} tokens{cost} · "
               f"{stats.get('guardrail_hits', 0)} guardrail hits"
               + (f" · models used {stats['models_used']}" if stats.get("models_used") else ""))
    if meta.get("traces") and (url := tracing.trace_url(meta["traces"][-1])):
        st.markdown(f"[Open trace in Langfuse]({url})")
    if status == "blocked":
        st.error(f"Input stopped by a guardrail: {meta.get('error') or 'no reason recorded'}")

    if status in WAITING and (request := t.pending(case_id)):
        pending_box(case_id, request, report)

    left, right = st.columns([5, 6])
    with left:
        st.graphviz_chart(graph_dot(events, status), width="stretch")
        graph_legend()
    with right:
        tabs = st.tabs(["Report", "Activity", "Findings", "Evidence", "Memory & feedback"])
        with tabs[0]:
            report_view(report, state)
        with tabs[1]:
            for e in reversed(events[-80:]):
                stamp = time.strftime("%H:%M:%S", time.localtime(e["t"]))
                text = e["text"] if len(e["text"]) < 600 else e["text"][:600] + "…"
                st.markdown(f"{KIND_ICON.get(e['kind'], '•')} `{stamp}` **{e['agent']}** {text}")
        with tabs[2]:
            for f in state.get("findings", []):
                with st.container(border=True):
                    st.markdown(f"**{f['agent']}** (round {f.get('round', 1)}) → suspect **{f.get('suspect_service')}**, "
                                f"{f.get('category')}, confidence {f.get('confidence')}")
                    st.write(f["summary"])
                    if f.get("ruled_out"):
                        st.caption("Ruled out: " + "; ".join(f["ruled_out"]))
                    st.caption("Evidence: " + ", ".join(f.get("evidence_ids") or []))
            for c in state.get("critiques", []):
                st.warning(f"Review board: **{c['verdict']}** (risk {c.get('risk', '?')}) — "
                           + ("; ".join(c["issues"]) or "no issues"))
        with tabs[3]:
            for e in state.get("evidence", []):
                with st.expander(f"{e['id']} · {e['agent']} · {e['tool']}({', '.join(f'{k}={v}' for k, v in e['args'].items())})"):
                    st.json(e["result"])
        with tabs[4]:
            feedback_view(case_id, meta, state)


def pending_box(case_id: str, request: dict, report: dict) -> None:
    """The workflow is waiting for the operator: an answer to the coordinator's question, or an approval."""
    t = triage()
    with st.container(border=True):
        if request.get("kind") == "question":
            st.markdown(f"#### ❓ The coordinator asks: {request['question']}")
            answer = st.text_input("Your answer", key=f"ans_{case_id}")
            if st.button("Answer", type="primary", key=f"answer_{case_id}"):
                t.cases.update(case_id, status="investigating")
                start_in_background(t.answer_case, case_id, answer)
                st.rerun(scope="fragment")
            return
        options = request.get("options") or [request]
        label = lambda i: f"{options[i].get('description', '')} (risk {options[i].get('risk', '?')})"  # noqa: E731
        st.markdown(f"#### ✋ Approval needed: {options[0].get('description', '')}")
        st.write(report.get("summary", ""))
        if len(options) > 1:
            choice = st.radio("Which fix?", range(len(options)), format_func=label, key=f"choice_{case_id}",
                              captions=["primary", "alternative"][:len(options)])
        else:
            choice = 0
            st.markdown(f"Risk of this fix: **{options[0].get('risk', '?')}**")
        if request.get("review_risk"):
            st.markdown(f"Review board risk: **{request['review_risk']}**" + (f" — {request['review_notes']}"
                                                                              if request.get("review_notes") else ""))
        feedback = st.text_input("Feedback if rejecting (sends the team back to investigate)", key=f"fb_{case_id}")
        c1, c2, c3 = st.columns(3)
        clicked = [c1.button("Approve and execute", type="primary", key=f"ok_{case_id}"),
                   c2.button("Reject", key=f"no_{case_id}"),
                   c3.button("Reject with feedback", key=f"fb_btn_{case_id}", disabled=not feedback)]
        decision = next(((approved, text) for hit, approved, text in zip(
            clicked, (True, False, False), ("", "", feedback)) if hit), None)
        if decision:
            t.cases.update(case_id, status="remediating" if decision[0] else "investigating")
            start_in_background(t.decide_case, case_id, decision[0], decision[1], choice)
            st.rerun(scope="fragment")


def report_view(report: dict, state: dict) -> None:
    if not report:
        st.info("The team is investigating…")
        if state.get("hypotheses"):
            st.markdown("**Hypotheses being tested**")
            for h in state["hypotheses"]:
                st.markdown(f"- {h}")
        return
    if report["status"] == "diagnosed":
        st.success(f"Root cause: **{report.get('root_cause_service')}** · {report.get('category')} · "
                   f"confidence {report.get('confidence')}")
    else:
        st.warning("Inconclusive: no cause is supported by the evidence")
    st.write(report.get("summary"))
    if report.get("mechanism"):
        st.markdown(f"**Mechanism:** {report['mechanism']}")
    if report.get("remediation"):
        st.markdown(f"**Proposed fix:** {remediation.describe(report['remediation'])}")
    if report.get("alternative"):
        st.markdown(f"**Alternative:** {remediation.describe(report['alternative'])}")
    for h in report.get("hypotheses") or []:
        icon = {"supported": "✅", "rejected": "❌", "open": "❔"}[h["status"]]
        st.markdown(f"{icon} {h['statement']} `{', '.join(h.get('evidence_ids') or [])}`")
    if rec := state.get("recovery"):
        (st.success if rec["recovered"] else st.error)(
            "Recovery verified" if rec["recovered"] else f"Not recovered: {rec['still_anomalous']}")
    if state.get("checks"):
        st.caption("Automatic checks: " + "; ".join(state["checks"]))


def feedback_view(case_id: str, meta: dict, state: dict) -> None:
    memories = state.get("memories") or []
    st.markdown("**Recalled from ADK memory (SqliteMemoryService) at the start of this case**")
    if not memories:
        st.caption("Nothing similar." if get_settings().memory else "Memory is switched off.")
    for m in memories:
        st.markdown(f"- {m['title']} → {m.get('root_cause_service')}/{m.get('category')}"
                    + (f" · operator said **{m['feedback']}**" if m.get("feedback") else ""))
    if meta["status"] in ACTIVE or meta["status"] in WAITING:
        return
    st.markdown("**Was this diagnosis correct?** Your rating is stored with the incident and changes how "
                "much future investigations trust it.")
    note = st.text_input("Note (optional)", key=f"note_{case_id}")
    c1, c2 = st.columns(2)
    for col, correct, label in ((c1, True, "👍 Correct"), (c2, False, "👎 Incorrect")):
        if col.button(label, key=f"rate_{correct}_{case_id}"):
            if triage().memory.rate(case_id, correct, note):
                triage().cases.update(case_id, rating="correct" if correct else "incorrect")
                tracing.score((meta.get("traces") or [""])[-1], "operator_rating", int(correct), note)
                st.success("Feedback saved to memory")
            else:
                st.info("Only verified fixes are stored in memory")


# ---------------------------------------------------------------------------- fault lab
def lab_inject(scenario: Scenario, blind: bool = False) -> None:
    """Reset to the standard configuration, apply the scenario and remember it as the active injection."""
    t = triage()
    try:
        simulator.reset(t.backends)
        simulator.save_active(t.settings.data_dir, simulator.inject(scenario, t.backends))
    except Exception as exc:
        st.error(f"Injection failed: {exc}")
        return
    st.session_state.setdefault("lab", {})[scenario.id] = scenario  # custom and random scenarios are not in the catalog
    st.session_state["blind"] = {"id": scenario.id, "revealed": False} if blind else None
    st.success(f"Injected {'a hidden fault' if blind else scenario.id}. Wait ~90 s for symptoms.")


def lab_active(model: str, mode: str) -> None:
    """The active injection, reset, and (once a case about it finished) the score against the ground truth."""
    t = triage()
    active = simulator.load_active(t.settings.data_dir)
    if st.button("Reset to the standard configuration"):
        try:
            done = simulator.reset(t.backends)
        except Exception as exc:
            st.error(f"Reset failed: {exc}")
        else:
            simulator.clear_active(t.settings.data_dir)
            st.session_state["blind"] = None
            st.info("; ".join(done) or "Already standard")
            active = None
    if not active:
        st.caption("No active injection.")
        return
    sid = active["scenario"]
    scenario = CATALOG.get(sid) or st.session_state.get("lab", {}).get(sid)
    blind = st.session_state.get("blind")
    hidden = bool(blind and blind["id"] == sid and not blind["revealed"])
    st.markdown(f"**Active injection** · injected {ago(active['at'])}")
    if hidden:
        st.info("Blind injection: the scenario is hidden until you reveal it.")
        if st.button("Reveal"):
            blind["revealed"] = True
            st.rerun()
    else:
        st.markdown(f"{scenario.title if scenario else sid}")
        st.caption("Faults: " + ", ".join(str(Fault(**f)) for f in active["faults"]) +
                   ("; decoys: " + ", ".join(str(Fault(**f)) for f in active["decoys"]) if active["decoys"] else ""))
        if scenario and scenario.truth:
            st.caption(f"Truth: {scenario.truth.service} / {scenario.truth.category}; accepted fixes: "
                       + ", ".join(str(f) for f in scenario.truth.fixes))
    if st.button("Investigate this injection", type="primary"):
        case_id = t.open_case(scenario.symptom if scenario else "Something is wrong with the shop", start=active["at"],
                              mode=mode, model=model, scenario=sid, source="lab")
        start_in_background(t.run, case_id)
        st.session_state["case"] = case_id
        st.info(f"Case {case_id} started; follow it on the Incidents page.")
    done = [c for c in t.cases.list() if c.get("scenario") == sid and c["status"] in FINAL and c.get("report")]
    if done and hidden:
        st.info("A linked case has finished. Reveal the scenario to see its score.")
    elif done and scenario:
        got = simulator.score(scenario, done[0]["report"])
        st.markdown(f"**Score of case `{done[0]['id']}` ({done[0]['status']}) against the truth:** "
                    + " · ".join(f"{'✅' if v else '❌'} {k.replace('_', ' ')}" for k, v in got.items()))


def lab_page() -> None:
    model, mode = sidebar()
    st.title("🧪 Fault lab")
    st.caption("Operator only. Every injection first resets the demo to its standard configuration, then applies real "
               "changes (container limits, stopped containers, feature flags) and harmless decoys.")
    tabs = st.tabs(["Catalog", "Build one", "Random (blind)"])
    with tabs[0]:
        sid = st.selectbox("Scenario", list(CATALOG), format_func=lambda s: f"{CATALOG[s].title} [{CATALOG[s].difficulty}]")
        sc = CATALOG[sid]
        st.caption(f"Symptom: {sc.symptom} · faults: {', '.join(map(str, sc.faults)) or 'none (control)'}"
                   + (f" · decoys: {', '.join(map(str, sc.decoys))}" if sc.decoys else ""))
        if st.button("Inject", key="inject_catalog"):
            lab_inject(sc)
    with tabs[1]:
        kind = st.selectbox("Fault kind", ["cpu_limit", "memory_limit", "stop", "pause", "flag"])
        targets = {"cpu_limit": simulator.CPU_TARGETS, "memory_limit": tuple(sorted(simulator.MEMORY_MB)),
                   "stop": simulator.STOP_TARGETS, "pause": simulator.STOP_TARGETS, "flag": tuple(sorted(simulator.FLAGS))}[kind]
        target = st.selectbox("Target", targets)
        value = {"cpu_limit": lambda: st.selectbox("CPU limit (cores)", simulator.CPU_VALUES),
                 "memory_limit": lambda: st.number_input("Memory limit (MiB)", 16, 512, simulator.MEMORY_MB[target]),
                 "flag": lambda: st.selectbox("Variant", simulator.FLAGS[target][3])}.get(kind, lambda: None)()
        decoy_options = [Fault("flag", simulator.DECOY_FLAG, "on")] + [Fault("cpu_limit", s, 0.75) for s in simulator.CPU_TARGETS]
        decoys = st.multiselect("Decoys (harmless changes)", decoy_options, format_func=str)
        if st.button("Inject", key="inject_custom"):
            fault = Fault(kind, target, value)
            lab_inject(Scenario(id=f"custom-{int(time.time())}", title=str(fault), symptom="Something is wrong with the shop",
                                faults=(fault,), decoys=tuple(decoys), truth=simulator.truth_for(fault)))
    with tabs[2]:
        level = st.selectbox("Difficulty", [*simulator.LEVELS, "mixed"], index=1)
        st.caption("A random fault of a random kind with decoys. Title and truth stay hidden until you reveal them.")
        if st.button("Inject a random fault", key="inject_random"):
            lab_inject(simulator.generate(random.randrange(10 ** 6), 1, level)[0], blind=True)
    st.divider()
    lab_active(model, mode)


# ---------------------------------------------------------------------------- memory, evaluation
def memory_page() -> None:
    st.title("🧠 Incident memory")
    st.caption("ADK SqliteMemoryService (full-text search). Seeded runbook incidents plus verified fixes. "
               "Searched by the recall node and by the coordinator's load_memory tool.")
    if get_settings().memory:
        st.success("Memory is ON. Only fixes that were verified (the system recovered) are stored; rejected or "
                   "unproven diagnoses are not.")
    else:
        st.warning("Memory is OFF (SRE_MEMORY=0): nothing is recalled or stored.")
    query = st.text_input("Search memory like the agents do", "checkout latency cart CPU throttling")
    if query:
        hits = asyncio.run(triage().memory.search(query))
        st.markdown("**Top matches:** " + (", ".join(f"`{h['id']}`" for h in hits) or "none"))
    items = triage().memory.all()
    st.dataframe(pd.DataFrame([{"id": m["id"], "title": m["title"], "root cause": m.get("root_cause_service"),
                                "category": m.get("category"), "outcome": m.get("outcome"),
                                "source": m.get("source"), "feedback": m.get("feedback"),
                                "note": m.get("feedback_note", "")} for m in items]),
                 hide_index=True, width="stretch")


def fmt(key: str, value) -> str:
    """Format a summary number by its name: counts, USD, seconds, other means, else a 0-1 rate as a percentage."""
    if not isinstance(value, (int, float)):
        return "–" if value is None else str(value)
    if key in ("n", "faults", "controls"):
        return str(int(value))
    if "cost" in key:
        return f"${value:.4f}"
    if key.endswith(("_s_mean", "_s_median", "_s")):
        return f"{value:.1f} s"
    return f"{value:.1f}" if key.endswith("_mean") else f"{value:.0%}"


def summary_table(blocks: dict[str, dict], label: str) -> None:
    rows = [{label: name, **{k.removesuffix("_mean").replace("_", " "): fmt(k, v) for k, v in block.items()}}
            for name, block in blocks.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def evaluation_page() -> None:
    st.title("📊 Evaluation")
    st.caption("`uv run sre-triage eval` injects scenarios, investigates with each variant, approves the fix, verifies "
               "recovery and scores the result against the ground truth.")
    from sre_triage import evaluation

    runs = evaluation.load_runs(get_settings().data_dir)
    if not runs:
        st.info("No evaluation runs yet.")
        return
    by_id = {r["id"]: r for r in runs}
    run = by_id[st.selectbox("Run", by_id, format_func=lambda i: f"{i} · {by_id[i]['summary'].get('overall', {}).get('n', '?')} cases")]
    summary = run["summary"]
    st.markdown("**Summary by variant**")
    summary_table({"all": summary.get("overall", {}), **summary.get("by_variant", {})}, "variant")
    for key, label in (("by_difficulty", "difficulty"), ("by_category", "category")):
        if summary.get(key):
            with st.expander(f"By {label}"):
                summary_table(summary[key], label)
    rows = []
    for r in run["results"]:
        score, truth, timings, stats = r.get("score") or {}, r.get("truth"), r.get("timings") or {}, r.get("stats") or {}
        rows.append({"scenario": r["scenario"], "variant": r.get("variant"), "difficulty": r.get("difficulty"),
                     "truth": f"{truth.get('service')}/{truth.get('category')}" if isinstance(truth, dict) else str(truth or "control"),
                     "outcome": r.get("status"), "correct": score.get("correct"), "right service": score.get("service"),
                     "right fix": score.get("remediation"), "false alarm": score.get("false_alarm"),
                     "recovered": r.get("recovered"), "total s": timings.get("total_s"), "LLM calls": stats.get("llm_calls"),
                     "USD": stats.get("cost_usd")})
    st.markdown("**Results per scenario**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Run configuration"):
        st.json(run.get("config") or {})
    folder = Path(run["path"])
    report = (folder if folder.is_dir() else folder.parent) / "report.md"
    if report.exists():
        with st.expander("report.md", expanded=True):
            st.markdown(report.read_text())


# ---------------------------------------------------------------------------- inside ADK
def mcp_tab(events: list[dict]) -> None:
    st.caption("The agents get their tools from two MCP servers (Model Context Protocol, streamable HTTP). "
               "All tools are read-only; the incident window travels in a request header.")
    try:
        urls = triage().mcp_urls
        listing = mcp_servers.tools_listing(urls)
    except Exception as exc:
        st.error(f"MCP servers unreachable: {exc}")
        listing = {}
    for kind, tools in listing.items():
        st.markdown(f"**{kind}** · `{urls[kind]}` · {len(tools)} tools")
        st.dataframe(pd.DataFrame(tools), hide_index=True, width="stretch")
    rows = []
    for e in events:
        name = e["text"].split("(")[0]
        if e["kind"] == "tool" and any(name in tools for tools in SERVER_TOOLS.values()):
            server = next(k for k, tools in SERVER_TOOLS.items() if name in tools)
            call, _, result = e["text"].partition(" -> ")
            rows.append({"time": time.strftime("%H:%M:%S", time.localtime(e["t"])), "agent": e["agent"], "server": server,
                         "call": call, "result": result})
    st.markdown(f"**MCP tool calls made in this case:** {len(rows)}")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def a2a_tab(events: list, log: list[dict], case_id: str) -> None:
    url = get_settings().reviewer_a2a_url
    if not url:
        st.info("The review ran locally (a critic LlmAgent). Start the CrewAI review board "
                "(`cd services/crew_reviewer && uv run python server.py --port 8002`) and set "
                "`SRE_REVIEWER_A2A_URL=http://127.0.0.1:8002` to review reports over A2A.")
    else:
        try:
            st.markdown(f"**Agent card** of the CrewAI review board from `{url}/.well-known/agent-card.json`")
            st.json(agent_card(url), expanded=1)
        except Exception as exc:
            st.error(f"A2A review board unreachable: {exc}")
    st.markdown("**A2A events of this case**")
    for e in (e for e in log if e["kind"] == "a2a"):
        st.markdown(f"🔗 `{time.strftime('%H:%M:%S', time.localtime(e['t']))}` {e['text']}")
    messages = adk_view.a2a_events(events)
    st.markdown(f"**Messages received from the remote review board:** {len(messages)}")
    for m in messages:
        st.code(m["text"], language="json")
    for review in triage().state(case_id).get("critiques") or []:
        st.markdown(f"- review: **{review['verdict']}** (risk {review.get('risk', '?')}) via {review.get('via', 'local LlmAgent')}")


def guardrails_tab(log: list[dict]) -> None:
    st.caption("Guardrails run as an ADK plugin: the operator's input is screened before the workflow starts "
               "(heuristics, then an optional LLM check) and tool outputs are checked for injected instructions.")
    hits = [e for e in log if e["kind"] == "guardrail"]
    st.markdown(f"**Guardrail events in this case:** {len(hits)}")
    for e in hits:
        st.markdown(f"🛡️ `{time.strftime('%H:%M:%S', time.localtime(e['t']))}` {e['text']}")
    st.markdown("**Input screen quality**")
    st.caption(f"Scores the heuristic screen on `{GUARD_PROMPTS.relative_to(PROJECT_ROOT)}` (labelled ok / injection / toxic).")
    if st.button("Evaluate the input screen"):
        st.session_state["guard_eval"] = guardrails.evaluate(GUARD_PROMPTS)
    if result := st.session_state.get("guard_eval"):
        c = st.columns(4)
        c[0].metric("Prompts", result["n"])
        c[1].metric("Precision", f"{result['precision']:.0%}")
        c[2].metric("Recall", f"{result['recall']:.0%}")
        c[3].metric("False positives", result["false_positives"])
        st.dataframe(pd.DataFrame([{"label": k, **v} for k, v in result["per_category"].items()]), hide_index=True, width="stretch")
        with st.expander(f"{len(result['misses'])} misses"):
            st.dataframe(pd.DataFrame(result["misses"]), hide_index=True, width="stretch")


def adk_page() -> None:
    """How the case ran inside Google ADK: Runner events, session state, ReAct traces, validation, MCP, A2A, guardrails."""
    st.title("🧩 Inside ADK")
    st.caption("Everything here is read back from the case's ADK session (DatabaseSessionService, SQLite): "
               "the Runner persists every event the workflow, its agents and their tools produce.")
    cases = triage().cases.list()
    if not cases:
        st.info("No cases yet.")
        return
    ids = [c["id"] for c in cases]
    current = st.session_state.get("case")
    case_id = st.selectbox("Case", ids, index=ids.index(current) if current in ids else 0,
                           format_func=lambda i: next(f"{c['id']} · {c['symptom']} · {c['status']}" for c in cases if c["id"] == i))
    session = triage().session(case_id)
    if session is None:
        st.warning("This case has no ADK session yet.")
        return
    events, log = session.events, triage().cases.events(case_id)
    traces = adk_view.react_traces(events)
    checks = adk_view.validations(events)
    c = st.columns(5)
    c[0].metric("Session id", session.id)
    c[1].metric("Runner events", len(events))
    c[2].metric("State keys", len(session.state))
    c[3].metric("Agent runs (ReAct)", len(traces))
    c[4].metric("Answers rejected → fixed", sum(1 for v in checks if v["status"] == "rejected"))
    tabs = st.tabs(["Runner events", "Session state", "ReAct traces", "Pydantic validation", "MCP", "A2A", "Guardrails", "Graph"])
    with tabs[0]:
        st.caption("`Runner.run_async()` yields these events; the session service stores them. "
                   "`node` is the workflow path, `branch` separates the parallel specialists.")
        rows = adk_view.event_rows(events)
        for r in rows:
            r["time"] = time.strftime("%H:%M:%S", time.localtime(r["time"]))
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=480)
    with tabs[1]:
        left, right = st.columns([2, 3])
        with left:
            st.markdown("**Which node wrote which state key** (`ctx.state[...]` → event `state_delta`)")
            timeline = adk_view.state_timeline(events)
            for r in timeline:
                r["time"] = time.strftime("%H:%M:%S", time.localtime(r["time"]))
            st.dataframe(pd.DataFrame(timeline), hide_index=True, width="stretch")
        with right:
            st.markdown("**Current session state**")
            st.json({k: v for k, v in session.state.items() if k != "evidence"}, expanded=1)
            with st.expander(f"evidence ({len(session.state.get('evidence') or [])} items)"):
                st.json(session.state.get("evidence") or [])
    with tabs[2]:
        st.caption("Each specialist is an ADK LlmAgent in a ReAct loop: it reasons, calls a tool (Action), reads "
                   "the result (Observation) and repeats, then ends by calling its submit tool.")
        for name, steps in traces.items():
            with st.expander(f"{name} — {sum(1 for s in steps if s['step'] == 'Action')} actions"):
                for s in steps:
                    icon = {"Thought": "💭", "Action": "🔧", "Observation": "👁️"}[s["step"]]
                    st.markdown(f"{icon} **{s['step']}** `{s['detail']}`" if s["step"] != "Thought"
                                else f"{icon} **Thought** {s['detail']}")
    with tabs[3]:
        st.caption("Agents answer through submit_* tools. The answer is validated with Pydantic (plus evidence-ID and "
                   "allowlist checks); a rejection is returned to the model, which corrects itself.")
        st.dataframe(pd.DataFrame(checks or [{"agent": "", "tool": "", "status": "", "errors": ""}]),
                     hide_index=True, width="stretch")
        schema = st.selectbox("Schema", list(adk_view.SCHEMAS))
        st.json(adk_view.SCHEMAS[schema].model_json_schema(), expanded=2)
    with tabs[4]:
        mcp_tab(log)
    with tabs[5]:
        a2a_tab(events, log, case_id)
    with tabs[6]:
        guardrails_tab(log)
    with tabs[7]:
        st.caption("The ADK Workflow definition (edges and routes) that produced these events.")
        st.graphviz_chart(graph_dot(log, triage().cases.get(case_id)["status"]), width="stretch")
        graph_legend()


pages = [st.Page(incidents_page, title="Incidents", icon="🚨", default=True),
         st.Page(lab_page, title="Fault lab", icon="🧪"),
         st.Page(adk_page, title="Inside ADK", icon="🧩"),
         st.Page(memory_page, title="Memory", icon="🧠"),
         st.Page(evaluation_page, title="Evaluation", icon="📊")]
st.navigation(pages).run()

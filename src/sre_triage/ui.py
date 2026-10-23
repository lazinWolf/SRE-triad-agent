"""Streamlit dashboard: start investigations, watch the agent graph live, approve fixes, rate results."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `streamlit run` without installing

from sre_triage import remediation  # noqa: E402
from sre_triage.config import get_settings  # noqa: E402
from sre_triage.faults import SCENARIOS, inject, reset  # noqa: E402
from sre_triage.workflow import Triage  # noqa: E402
from sre_triage.llm import SUGGESTED_MODELS, has_key  # noqa: E402
from sre_triage.tools import Window, health_scan  # noqa: E402

ACTIVE = {"investigating", "remediating", "verifying"}
STATUS_ICON = {"investigating": "🔎", "awaiting_approval": "✋", "remediating": "🛠️", "verifying": "⏱️",
               "resolved": "✅", "not_recovered": "⚠️", "inconclusive": "❔", "rejected": "🚫",
               "diagnosed_no_action": "📝", "failed": "❌", "action_failed": "❌"}
KIND_ICON = {"node": "▶", "tool": "🔧", "thought": "💭", "finding": "📌", "plan": "🗺️", "report": "📄",
             "critique": "🧐", "approval": "✋", "action": "🛠️", "result": "📊", "error": "❌", "validation": "↩️"}

st.set_page_config(page_title="SRE Incident Triage", page_icon="🚨", layout="wide")


@st.cache_resource
def triage() -> Triage:
    return Triage(get_settings())


def start_in_background(fn, *args) -> None:
    threading.Thread(target=fn, args=args, daemon=True).start()


# ---------------------------------------------------------------------------- graph drawing
def graph_dot(events: list[dict], status: str) -> str:
    """The ADK workflow graph, with visited nodes green and the current one highlighted."""
    edges = triage().edges()
    visited = [e["agent"] for e in events if e["kind"] == "node"]
    current = visited[-1] if visited and (status in ACTIVE or status == "awaiting_approval") else None
    lines = ["digraph G {", "rankdir=TB; bgcolor=transparent; node [shape=box style=\"rounded,filled\" "
             "fontname=Helvetica fontsize=11 fillcolor=\"#eeeeee\" color=\"#999999\"]; "
             "edge [color=\"#888888\" fontname=Helvetica fontsize=9 fontcolor=\"#666666\"];"]
    names = list(dict.fromkeys([n for e in edges for n in e[:2]]))
    for name in names:
        label = {"__START__": "start", "specialists": "specialists (parallel LlmAgents)"}.get(name, name)
        fill = "#b7e4c7" if name in visited or name == "__START__" else "#eeeeee"
        if name == current:
            fill = "#ffd166" if status == "awaiting_approval" else "#f4a261"
        count = visited.count(name)
        lines.append(f'"{name}" [label="{label}{f" ×{count}" if count > 1 and name != "specialists" else ""}" '
                     f'fillcolor="{fill}" shape={"ellipse" if name == "__START__" else "box"}];')
    for src, dst, route in edges:
        lines.append(f'"{src}" -> "{dst}"' + (f' [label="{route}" style=dashed]' if route else "") + ";")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- sidebar
def sidebar() -> tuple[str, str]:
    settings = get_settings()
    st.sidebar.header("Settings")
    options = [m for m in dict.fromkeys([settings.model, *settings.fallback_models, *SUGGESTED_MODELS]) if has_key(m)]
    model = st.sidebar.selectbox("Model (LiteLLM)", options or [settings.model],
                                 help="Rate-limited models fall back to SRE_FALLBACK_MODELS automatically.")
    mode = st.sidebar.radio("Mode", ["multi", "single"], format_func=lambda m: {
        "multi": "Multi-agent (coordinator + 3 specialists + critic)", "single": "Single agent (baseline)"}[m])
    st.sidebar.divider()
    st.sidebar.caption("Fault lab (operator only, agents never see this)")
    scenario = st.sidebar.selectbox("Scenario", list(SCENARIOS), format_func=lambda s: SCENARIOS[s].title)
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Inject", width="stretch"):
        reset(triage().backends)
        inject(SCENARIOS[scenario], triage().backends)
        st.session_state["injected"] = (scenario, time.time())
        st.sidebar.success(f"Injected {scenario}. Wait ~90 s for symptoms.")
    if c2.button("Reset", width="stretch"):
        done = reset(triage().backends)
        st.session_state.pop("injected", None)
        st.sidebar.info("; ".join(done) or "Already standard")
    if injected := st.session_state.get("injected"):
        st.sidebar.caption(f"{injected[0]} injected {time.time() - injected[1]:.0f}s ago")
    return model, mode


# ---------------------------------------------------------------------------- pages
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
    st.markdown(f"**Live health** (last 2 min vs the 5 min before): "
                f"{'🔴 ' + str(len(alerts)) + ' service(s) anomalous' if alerts else '🟢 all normal'}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=250)


def incidents_page() -> None:
    model, mode = sidebar()
    st.title("🚨 SRE Incident Triage")
    st.caption("A Google ADK workflow of LlmAgents investigates live OpenTelemetry Demo telemetry, "
               "proposes a fix, and acts only after your approval.")
    health_panel()
    with st.form("new_case"):
        st.subheader("New investigation")
        c1, c2 = st.columns([3, 1])
        symptom = c1.text_input("Alert / symptom", "Checkout latency is elevated")
        since = c2.number_input("Incident began (minutes ago)", 1.0, 30.0, 3.0, 0.5)
        if st.form_submit_button("Investigate", type="primary"):
            t = triage()
            case_id = t.open_case(symptom, start=time.time() - since * 60, mode=mode, model=model,
                                  scenario=(st.session_state.get("injected") or [None])[0])
            start_in_background(t.run, case_id)
            st.session_state["case"] = case_id
    cases = triage().cases.list()
    if not cases:
        st.info("No cases yet. Inject a fault in the sidebar, wait ~90 s, then start an investigation.")
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
    st.caption(f"model {meta.get('model')} · mode {meta['mode']} · {len(state.get('evidence', []))} evidence items · "
               f"{stats.get('llm_calls', '…')} LLM calls · {stats.get('seconds', '…')} s"
               + (f" · models used {stats.get('models_used')}" if stats.get("models_used") else ""))

    if status == "awaiting_approval" and (request := t.pending(case_id)):
        approval_box(case_id, request, report)

    left, right = st.columns([2, 3])
    with left:
        st.graphviz_chart(graph_dot(events, status), width="stretch")
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
                st.warning(f"Critic: **{c['verdict']}** — " + ("; ".join(c["issues"]) or "no issues"))
        with tabs[3]:
            for e in state.get("evidence", []):
                with st.expander(f"{e['id']} · {e['agent']} · {e['tool']}({', '.join(f'{k}={v}' for k, v in e['args'].items())})"):
                    st.json(e["result"])
        with tabs[4]:
            feedback_view(case_id, meta, state)


def approval_box(case_id: str, request: dict, report: dict) -> None:
    with st.container(border=True):
        st.markdown(f"#### ✋ Approval needed: {request['description']}")
        st.write(report.get("summary", ""))
        feedback = st.text_input("Feedback if rejecting (sends the team back to investigate)", key=f"fb_{case_id}")
        c1, c2 = st.columns(2)
        if c1.button("Approve and execute", type="primary", key=f"ok_{case_id}"):
            triage().cases.update(case_id, status="remediating")
            start_in_background(triage().decide_case, case_id, True, "")
            st.rerun(scope="fragment")
        if c2.button("Reject", key=f"no_{case_id}"):
            triage().cases.update(case_id, status="investigating")
            start_in_background(triage().decide_case, case_id, False, feedback)
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
    st.markdown("**Recalled from long-term memory at the start of this case**")
    if not memories:
        st.caption("Nothing similar.")
    for m in memories:
        st.markdown(f"- {m['title']} → {m.get('root_cause_service')}/{m.get('category')} (similarity {m['similarity']})")
    if meta["status"] in ACTIVE or meta["status"] == "awaiting_approval":
        return
    st.markdown("**Was this diagnosis correct?** Your rating is stored with the incident and changes how "
                "much future investigations trust it.")
    note = st.text_input("Note (optional)", key=f"note_{case_id}")
    c1, c2 = st.columns(2)
    for col, correct, label in ((c1, True, "👍 Correct"), (c2, False, "👎 Incorrect")):
        if col.button(label, key=f"rate_{correct}_{case_id}"):
            if triage().memory.rate(case_id, correct, note):
                triage().cases.update(case_id, rating="correct" if correct else "incorrect")
                st.success("Feedback saved to memory")
            else:
                st.info("Only diagnosed cases are stored in memory")


def memory_page() -> None:
    st.title("🧠 Incident memory")
    st.caption("Seeded runbook incidents plus every diagnosed case. Recalled by similarity at the start of each investigation.")
    items = triage().memory.all()
    st.dataframe(pd.DataFrame([{"id": m["id"], "title": m["title"], "root cause": m.get("root_cause_service"),
                                "category": m.get("category"), "outcome": m.get("outcome"),
                                "source": m.get("source"), "feedback": m.get("feedback")} for m in items]),
                 hide_index=True, width="stretch")


def evaluation_page() -> None:
    st.title("📊 Evaluation")
    st.caption("`uv run sre-triage eval` injects each scenario, investigates in both modes, approves the fix and scores it.")
    path = get_settings().data_dir / "eval.json"
    if not path.exists():
        st.info("No evaluation results yet.")
        return
    results = json.loads(path.read_text())
    df = pd.DataFrame([{"scenario": r["scenario"], "mode": r["mode"], "correct": r["score"]["correct"],
                        "right service": r["score"]["service"], "right fix": r["score"]["remediation"],
                        "false alarm": r["score"]["false_alarm"], "outcome": r["status"],
                        "seconds": r["stats"].get("seconds"), "LLM calls": r["stats"].get("llm_calls")} for r in results])
    summary = df.groupby("mode")[["correct", "right service", "right fix", "false alarm"]].mean().mul(100).round(0)
    st.markdown("**Accuracy by mode (%)**")
    st.dataframe(summary, width="stretch")
    st.dataframe(df, hide_index=True, width="stretch")


pages = [st.Page(incidents_page, title="Incidents", icon="🚨", default=True),
         st.Page(memory_page, title="Memory", icon="🧠"),
         st.Page(evaluation_page, title="Evaluation", icon="📊")]
st.navigation(pages).run()

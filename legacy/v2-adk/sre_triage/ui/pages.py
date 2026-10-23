"""Dashboard pages: incidents, case detail, memory, evaluation and the operator fault lab."""

from __future__ import annotations

import json
import time

import altair as alt
import pandas as pd
import streamlit as st

from sre_triage.config import APP_SERVICES, PROJECT_ROOT, get_settings
from sre_triage.llm import available_models
from sre_triage.models import CaseInput, ReviewInput
from sre_triage.remediation import describe
from sre_triage.ui.common import (
    CASE_STATUS, EVIDENCE_STATUS, HYPOTHESIS_STATUS, MODE_COLORS, MODE_LABELS, ago, badge, clock, ledger,
    status_text, worker_alive,
)

ACTIVE = ("investigating", "remediating", "verifying")


def _header_status() -> None:
    settings = get_settings()
    cols = st.columns(3)
    cols[0].markdown(("✅ Worker running" if worker_alive() else
                      "⚠️ Worker not running: start it with `uv run sre-triage worker`"))
    cols[1].markdown(f"🧠 Model: `{settings.model}`" if settings.model else "⚠️ No model: set `SRE_MODEL` in `.env`")
    cols[2].markdown(f"🔗 [Jaeger]({settings.jaeger_url}) · [Prometheus]({settings.prometheus_url}) · "
                     f"[Grafana]({settings.grafana_url})")


# Filled by app.py so pages can navigate to each other.
PAGES: dict = {}


def go_to_case(case_id: str) -> None:
    st.switch_page(PAGES["case"], query_params={"id": case_id})


# --------------------------------------------------------------------------- incidents
def incidents() -> None:
    st.title("Incidents")
    _header_status()
    led = ledger()

    with st.expander("Open a case manually", icon=":material/add_alert:"):
        with st.form("open_case"):
            c1, c2 = st.columns([1, 2])
            service = c1.selectbox("Affected service", APP_SERVICES, index=APP_SERVICES.index("checkout"))
            symptom = c2.text_input("Symptom", "Checkout latency is elevated")
            c3, c4, c5 = st.columns(3)
            minutes = c3.number_input("Incident window (minutes back)", 1.0, 120.0, 5.0)
            baseline = c4.number_input("Baseline (minutes before)", 1.0, 120.0, 10.0)
            mode = c5.selectbox("Mode", ["multi_agent", "single_agent"], format_func=MODE_LABELS.get)
            models = [""] + [m.id for m in available_models()]
            labels = {m.id: m.label for m in available_models()}
            model = st.selectbox("Model", models, format_func=lambda m: labels.get(m, f"Default ({get_settings().model or 'unset'})"))
            use_memory = st.checkbox("Use incident memory", True)
            start_now = st.checkbox("Start investigation immediately", True)
            if st.form_submit_button("Open case", type="primary"):
                now = time.time()
                case = led.create_case(CaseInput(service=service, symptom=symptom, incident_start=now - minutes * 60,
                                                 incident_end=now, baseline_seconds=baseline * 60, mode=mode,
                                                 use_memory=use_memory, model=model))
                if start_now:
                    led.enqueue("investigate", case["id"], {"reason": "operator"})
                go_to_case(case["id"])

    cases = led.cases(200)
    if not cases:
        st.info("No cases yet. Alerts open cases automatically while the worker runs, or open one above.")
        return
    rows = []
    for c in cases:
        report = c.get("report") or {}
        root = report.get("root_cause") or {}
        rows.append({
            "Case": c["id"], "Opened": ago(c["created_at"]), "Status": CASE_STATUS.get(c["status"], ("", "", c["status"]))[2],
            "Service": c["service"], "Symptom": c["symptom"], "Mode": MODE_LABELS.get(c["mode"], c["mode"]),
            "Diagnosis": report.get("diagnosis_status", "-"),
            "Root cause": f"{root.get('service')} · {root.get('category')}" if root else "-",
            "Source": c["source"],
        })
    st.subheader("Cases")
    event = st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="single-row", key="cases_table")
    selected = event.selection.rows if event and event.selection else []
    if selected:
        go_to_case(rows[selected[0]]["Case"])

    alerts = led.alerts(20)
    if alerts:
        st.subheader("Recent alerts")
        st.dataframe(pd.DataFrame([{"Fired": ago(a["fired_at"]), "Service": a["service"], "Case": a["case_id"],
                                    "Reasons": "; ".join(a["details"].get("reasons", []))} for a in alerts]),
                     hide_index=True, width="stretch")


# --------------------------------------------------------------------------- case
def case_page() -> None:
    led = ledger()
    case_ids = [c["id"] for c in led.cases(200)]
    if not case_ids:
        st.info("No cases yet.")
        return
    current = st.query_params.get("id", case_ids[0])
    if current not in case_ids:
        current = case_ids[0]
    picked = st.selectbox("Case", case_ids, index=case_ids.index(current),
                          format_func=lambda cid: f"{cid} · {led.case(cid)['service']} · {led.case(cid)['symptom'][:60]}")
    if picked != current:
        st.query_params["id"] = picked
        st.rerun()
    _case_body(picked)


@st.fragment(run_every=3)
def _live(case_id: str) -> None:
    led = ledger()
    case = led.case(case_id)
    if case["status"] in ACTIVE or any(j["status"] in ("queued", "running") for j in led.jobs(case_id, 5)):
        events = led.events(case_id)[-25:]
        with st.status(f"{CASE_STATUS[case['status']][2]}…", expanded=True, state="running"):
            for e in events:
                st.markdown(f"`{clock(e['ts'])}` **{e['actor']}** · {e['kind']} — {e['message'][:300]}")


def _case_body(case_id: str) -> None:
    led = ledger()
    case = led.case(case_id)
    report = case.get("report") or {}
    t0 = case["incident_start"]

    left, right = st.columns([3, 2])
    with left:
        st.title(f"{case['service']}: {case['symptom']}")
        st.markdown(f"{status_text(CASE_STATUS, case['status'])} &nbsp; Case `{case['id']}` · {MODE_LABELS[case['mode']]} · "
                    f"revision {case['revision']} · source {case['source']} · memory {'on' if case['use_memory'] else 'off'}")
        st.caption(f"Incident {clock(case['incident_start'])}–{clock(case['incident_end'])} "
                   f"({(case['incident_end'] - t0) / 60:.1f} min) · baseline from {clock(case['baseline_start'])}")
    with right:
        busy = case["status"] in ACTIVE or any(j["status"] in ("queued", "running") for j in led.jobs(case_id, 5))
        c1, c2 = st.columns(2)
        if c1.button("Investigate" if not report else "Investigate again", type="primary", disabled=busy,
                     icon=":material/play_arrow:", width="stretch"):
            led.enqueue("investigate", case_id, {"reason": "operator"})
            st.rerun()
        new_mode = "single_agent" if case["mode"] == "multi_agent" else "multi_agent"
        if c2.button(f"Switch to {MODE_LABELS[new_mode]}", disabled=busy, width="stretch"):
            led.set_case(case_id, mode=new_mode)
            st.rerun()
        models = [""] + [m.id for m in available_models()]
        current = case.get("model") or ""
        chosen = st.selectbox("Model for the next run", models if current in models else models + [current],
                              index=(models if current in models else models + [current]).index(current),
                              format_func=lambda m: m or f"Default ({get_settings().model or 'unset'})",
                              disabled=busy, key=f"model_{case_id}")
        if chosen != current:
            led.set_case(case_id, model=chosen)
            st.rerun()
        if not worker_alive():
            st.warning("The worker is not running, so queued work will not start.", icon=":material/warning:")
        if case.get("error"):
            st.error(case["error"], icon=":material/error:")

    _live(case_id)

    tabs = st.tabs(["Report", "Agent flow", "Hypotheses", "Activity", "Evidence", "Findings",
                    "Remediation & recovery", "Feedback & review", "Runs"])
    with tabs[0]:
        _report_tab(case, report)
    with tabs[1]:
        _flow_tab(case)
    with tabs[2]:
        _hypotheses_tab(case)
    with tabs[3]:
        _activity_tab(case)
    with tabs[4]:
        _evidence_tab(case)
    with tabs[5]:
        _findings_tab(case)
    with tabs[6]:
        _remediation_tab(case)
    with tabs[7]:
        _feedback_tab(case)
    with tabs[8]:
        _runs_tab(case)


def _report_tab(case: dict, report: dict) -> None:
    if not report:
        st.info("No report yet. Start an investigation.")
        return
    diagnosed = report.get("diagnosis_status") == "diagnosed"
    st.markdown(f"### {'🩺 Diagnosed' if diagnosed else '❔ Insufficient evidence'} · confidence {report.get('confidence')}")
    st.write(report.get("summary"))
    root = report.get("root_cause")
    if root:
        c1, c2, c3 = st.columns(3)
        c1.metric("Root-cause service", root.get("service"))
        c2.metric("Mechanism", root.get("category"))
        c3.metric("Triggering change", root.get("trigger_change_id") or "none")
        st.write(root.get("description"))
        st.caption("Evidence: " + ", ".join(f"`{e}`" for e in root.get("evidence_ids", [])))
    if report.get("impact"):
        st.markdown(f"**Impact.** {report['impact']}")
    cols = st.columns(2)
    with cols[0]:
        if report.get("timeline"):
            st.markdown("**Timeline**")
            for item in report["timeline"]:
                st.markdown(f"- {item}")
        if report.get("next_checks"):
            st.markdown("**Next checks**")
            for item in report["next_checks"]:
                st.markdown(f"- {item}")
    with cols[1]:
        if report.get("remediation"):
            st.markdown("**Proposed remediation** (requires approval)")
            st.info(describe(report["remediation"]) + " — " + report["remediation"].get("rationale", ""))
        if report.get("limitations"):
            st.markdown("**Limitations**")
            for item in report["limitations"]:
                st.markdown(f"- {item}")
    for critique in report.get("critiques") or []:
        icon = "✅" if critique["verdict"] == "accept" else "⚠️"
        with st.expander(f"{icon} Critic review (round {critique['round']}): {critique['verdict']}"):
            st.write(critique.get("summary"))
            for issue in critique.get("issues", []):
                st.markdown(f"- {issue}")
            if critique.get("alternative_hypotheses"):
                st.caption("Alternatives raised: " + "; ".join(critique["alternative_hypotheses"]))


def _dot_label(text: str, width: int = 34) -> str:
    words, lines, line = text.replace('"', "'").split(), [], ""
    for word in words:
        if len(line) + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return "\\n".join(lines[:4]) + ("…" if len(lines) > 4 else "")


def flow_dot(events: list[dict], run_id: str | None) -> str:
    """Build a DOT graph of one run from its events: who did what, in which order, and in parallel."""
    nodes = ['alert [label="Alert / case opened", shape=oval]']
    edges: list[str] = []
    tools: dict[str, int] = {}
    last, joined, delegated = "alert", [], {}
    for n, e in enumerate(events):
        kind, actor, node = e["kind"], e["actor"], f"n{n}"
        if kind == "tool_call":
            tools[actor] = tools.get(actor, 0) + 1
            continue
        message = _dot_label(e["message"])
        if kind == "decision":
            step = (e.get("data") or {}).get("next_step") or "decision"
            nodes.append(f'{node} [label="Coordinator: {step}\\n{message}", shape=box, style="rounded,filled", fillcolor="#cde2fb"]')
            edges += [f"{j} -> {node}" for j in (joined or [last])]
            last, joined, delegated = node, [], {}
        elif kind == "delegation":
            agent = e["message"].split(":", 1)[0]
            nodes.append(f'{node} [label="→ {agent}\\n{_dot_label(e["message"].split(":", 1)[-1])}", shape=note]')
            edges.append(f"{last} -> {node}")
            delegated[agent] = node
        elif kind == "finding" and actor != "critic":
            nodes.append(f'{node} [label="{actor} finding\\n{message}", shape=box, style=rounded]')
            edges.append(f"{delegated.get(actor, last)} -> {node}")
            joined.append(node)
        elif kind == "critique":
            nodes.append(f'{node} [label="Critic\\n{message}", shape=hexagon, style=filled, fillcolor="#fde2cf"]')
            edges += [f"{j} -> {node}" for j in (joined or [last])]
            last, joined = node, []
        elif kind in ("proposal", "action_decision", "action_executed", "action_failed", "recovered", "reopened",
                      "inconclusive", "review", "feedback"):
            shape = {"proposal": "parallelogram", "review": "folder", "feedback": "note",
                     "action_decision": "house"}.get(kind, "box")
            nodes.append(f'{node} [label="{kind.replace("_", " ")}\\n{message}", shape={shape}]')
            edges += [f"{j} -> {node}" for j in (joined or [last])]
            last, joined = node, []
        elif kind == "complete":
            nodes.append(f'{node} [label="{_dot_label(e["message"], 40)}", shape=oval]')
            edges += [f"{j} -> {node}" for j in (joined or [last])]
            last, joined = node, []
    if tools:
        usage = ", ".join(f"{k}: {v}" for k, v in sorted(tools.items()))
        nodes.append(f'tools [label="Tool calls\\n{_dot_label(usage, 40)}", shape=cylinder]')
    return ('digraph flow { rankdir=TB; node [fontname=Helvetica, fontsize=10]; edge [color="#898781"];\n'
            + "\n".join(nodes + edges) + "\n}")


def _flow_tab(case: dict) -> None:
    led = ledger()
    runs = led.runs(case["id"])
    if not runs:
        st.info("No runs yet.")
        return
    run_id = st.selectbox("Run", [r["id"] for r in reversed(runs)], key=f"flow_run_{case['id']}")
    events = [e for e in led.events(case["id"]) if e.get("run_id") in (run_id, None)]
    st.caption("Plan → parallel specialists → reconciliation → critic → proposal → approval → recovery, "
               "reconstructed from the event log.")
    st.graphviz_chart(flow_dot(events, run_id), width="stretch")


def _hypotheses_tab(case: dict) -> None:
    led = ledger()
    history = led.hypothesis_history(case["id"], case.get("run_id"))
    if not history:
        st.info("No hypotheses recorded for the current run.")
        return
    board = history[-1]["body"]
    for h in board:
        with st.container(border=True):
            c1, c2 = st.columns([1, 5])
            with c1:
                st.markdown(f"**{h['id']}**")
                badge(HYPOTHESIS_STATUS, h["status"])
            with c2:
                st.markdown(f"**{h['statement']}**  \n`{h['service']}` · {h['category']}")
                if h.get("rationale"):
                    st.caption(h["rationale"])
                st.caption("Supporting: " + (", ".join(f"`{e}`" for e in h.get("supporting_evidence", [])) or "none") +
                           " · Contradicting: " + (", ".join(f"`{e}`" for e in h.get("contradicting_evidence", [])) or "none"))
    if len(history) > 1:
        st.markdown("**How the board changed across rounds**")
        table: dict[str, dict[str, str]] = {}
        for snapshot in history:
            for h in snapshot["body"]:
                table.setdefault(f"{h['id']}: {h['statement'][:70]}", {})[f"round {snapshot['round']}"] = h["status"]
        st.dataframe(pd.DataFrame(table).T.fillna("—"), width="stretch")


def _activity_tab(case: dict) -> None:
    events = ledger().events(case["id"])
    if not events:
        st.info("No activity yet.")
        return
    kinds = sorted({e["kind"] for e in events})
    default = [k for k in kinds if k not in ("tool_call", "agent_message", "agent_thought")]
    chosen = st.multiselect("Show", kinds, default=default or kinds)
    for e in reversed(events):
        if e["kind"] in chosen:
            st.markdown(f"`{clock(e['ts'])}` **{e['actor']}** · _{e['kind']}_  \n{e['message']}")


def _evidence_tab(case: dict) -> None:
    led = ledger()
    runs = [r["id"] for r in led.runs(case["id"])]
    if not runs:
        st.info("No evidence yet.")
        return
    run_id = st.selectbox("Run", list(reversed(runs)), key=f"ev_run_{case['id']}")
    evidence = led.evidence(case["id"], run_id)
    if not evidence:
        st.info("No evidence in this run.")
        return
    st.dataframe(pd.DataFrame([{"ID": e["id"], "Agent": e["agent"], "Tool": e["tool"],
                                "Args": json.dumps({k: v for k, v in e["args"].items() if v not in ("", 0, None, False)}),
                                "Status": EVIDENCE_STATUS.get(e["status"], e["status"]),
                                "Collected": clock(e["collected_at"])} for e in evidence]),
                 hide_index=True, width="stretch")
    pick = st.selectbox("Inspect evidence", [e["id"] for e in evidence], key=f"ev_pick_{case['id']}")
    row = next(e for e in evidence if e["id"] == pick)
    for link in row.get("links") or []:
        st.markdown(f"🔗 [{link['label']}]({link['url']})")
    st.json(row["summary"], expanded=2)


def _findings_tab(case: dict) -> None:
    findings = ledger().findings(case["id"], case.get("run_id"))
    if not findings:
        st.info("No specialist findings for the current run (single-agent runs have none).")
        return
    for f in findings:
        body = f["body"]
        with st.expander(f"Round {f['round']} · {f['agent']} — {f['task'][:110]}"):
            st.write(body.get("summary"))
            for o in body.get("observations", []):
                st.markdown(f"- _{o.get('kind')}_: {o.get('statement')} " +
                            " ".join(f"`{e}`" for e in o.get("evidence_ids", [])))
            for s in body.get("hypothesis_signals", []):
                st.markdown(f"- **{s.get('hypothesis')}** {s.get('effect')}: {s.get('note')} " +
                            " ".join(f"`{e}`" for e in s.get("evidence_ids", [])))
            if body.get("limitations"):
                st.caption("Limitations: " + "; ".join(body["limitations"]))


def _remediation_tab(case: dict) -> None:
    led = ledger()
    actions = led.actions(case["id"])
    if not actions:
        st.info("No remediation has been proposed. Proposals appear only with a validated diagnosis.")
    for a in reversed(actions):
        p = a["proposal"]
        with st.container(border=True):
            st.markdown(f"#### {describe(p)}")
            st.markdown(f"Status: **{a['status']}** · proposed {ago(a['created_at'])}")
            st.write(p.get("rationale"))
            if p.get("expected_effect"):
                st.caption(f"Expected effect: {p['expected_effect']}")
            st.caption("Evidence: " + ", ".join(f"`{e}`" for e in p.get("evidence_ids", [])))
            rollback = a.get("rollback") or {}
            st.markdown("**Rollback:** " + (rollback.get("description") if rollback.get("available")
                                             else f"not available ({rollback.get('reason', 'unknown')})"))
            if a["status"] == "proposed":
                note = st.text_input("Decision note", key=f"note_{a['id']}")
                c1, c2 = st.columns(2)
                if c1.button("Approve and execute", key=f"ap_{a['id']}", type="primary", icon=":material/check:"):
                    led.decide_action(a["id"], True, "dashboard-operator", note)
                    led.enqueue("execute_action", case["id"], {"action_id": a["id"]})
                    st.rerun()
                if c2.button("Reject", key=f"rj_{a['id']}", icon=":material/close:"):
                    led.decide_action(a["id"], False, "dashboard-operator", note)
                    led.set_case(case["id"], status="needs_review")
                    st.rerun()
            if a.get("decided_at"):
                st.caption(f"Decision by {a['decided_by']} {ago(a['decided_at'])}: {a.get('decision_note') or ''}")
            if a.get("executed_at"):
                st.caption(f"Executed {ago(a['executed_at'])} · before {a.get('before_state')} → after {a.get('after_state')}")
            if a.get("error"):
                st.error(a["error"])
            if a["status"] == "executed" and rollback.get("available"):
                if st.button("Roll back", key=f"rb_{a['id']}", icon=":material/undo:"):
                    led.enqueue("rollback", case["id"], {"action_id": a["id"]})
                    st.rerun()
    checks = led.recovery_checks(case["id"])
    if checks:
        st.markdown("### Recovery checks")
        for c in reversed(checks):
            result = c.get("result") or {}
            icon = {"recovered": "✅", "not_recovered": "❌", "inconclusive": "❔", "pending": "⏳"}.get(c["status"], "•")
            st.markdown(f"{icon} **{c['status']}** · {ago(c['created_at'])}")
            for chk in result.get("checks", []):
                st.markdown(f"- `{chk['signal']}` ok={chk.get('ok')} baseline={chk.get('baseline')} "
                            f"after={chk.get('after')} limit={chk.get('limit')} {chk.get('detail', '')}")


def _feedback_tab(case: dict) -> None:
    led = ledger()
    for f in led.feedback(case["id"]):
        st.markdown(f"🗨️ rev {f['revision']} · {ago(f['ts'])}: {f['text']}")
    with st.form(f"fb_{case['id']}"):
        text = st.text_area("Correction or new context for the agents",
                            placeholder="e.g. The cache was already ruled out by the on-call engineer; focus on CPU.")
        rerun = st.checkbox("Investigate again with this correction", True)
        if st.form_submit_button("Save correction"):
            try:
                led.add_feedback(case["id"], text)
                if rerun:
                    led.enqueue("investigate", case["id"], {"reason": "feedback"})
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    st.divider()
    st.markdown("**Review the outcome** (saved to incident memory and closes the case)")
    report = case.get("report") or {}
    root = report.get("root_cause") or {}
    with st.form(f"rv_{case['id']}"):
        c1, c2, c3 = st.columns(3)
        outcome = c1.selectbox("Outcome", ["resolved", "mitigated", "unresolved", "false_alarm"])
        service = c2.text_input("Confirmed service", root.get("service", ""))
        category = c3.text_input("Confirmed mechanism", root.get("category", ""))
        cause = st.text_input("Confirmed cause (leave blank if unknown)", root.get("description", ""))
        notes = st.text_area("Lessons / notes")
        if st.form_submit_button("Save review"):
            from sre_triage.memory import review_case

            try:
                review_case(led, case["id"], ReviewInput(outcome=outcome, confirmed_service=service,
                                                         confirmed_category=category, confirmed_cause=cause, notes=notes))
                st.success("Saved to incident memory")
            except Exception as exc:
                st.error(str(exc))


def _runs_tab(case: dict) -> None:
    runs = ledger().runs(case["id"])
    if not runs:
        st.info("No runs yet.")
        return
    st.dataframe(pd.DataFrame([{
        "Run": r["id"], "Mode": MODE_LABELS.get(r["mode"], r["mode"]), "Model": r["model"], "Status": r["status"],
        "Started": clock(r["started_at"]), "Duration s": (r.get("stats") or {}).get("duration_s"),
        "Tool calls": (r.get("stats") or {}).get("tool_calls"), "LLM calls": (r.get("stats") or {}).get("llm_calls"),
        "Prompt tokens": (r.get("stats") or {}).get("prompt_tokens"), "Rounds": (r.get("stats") or {}).get("rounds"),
        "Model calls by agent": json.dumps((r.get("stats") or {}).get("llm_calls_by_agent", {})),
        "Result": (r.get("stats") or {}).get("diagnosis_status"), "Error": r.get("error"),
    } for r in runs]), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- memory
def memory_page() -> None:
    st.title("Incident memory")
    st.caption("Only operator-reviewed outcomes and curated seeds become memory. Agents treat them as leads, never proof.")
    led = ledger()
    if st.button("Load seeded historical incidents", icon=":material/upload:"):
        from sre_triage.memory import load_seeds

        n = load_seeds(led, PROJECT_ROOT / "seeds" / "incidents.json")
        st.success(f"Loaded {n} seeds")
    for m in led.memories():
        with st.expander(f"{'🌱 seed' if m['origin'] == 'seed' else '✅ reviewed'} · {m['service']} · {m['category']} — {m['title']}"):
            st.json(m["body"], expanded=1)
            if st.button("Delete", key=f"del_{m['id']}"):
                led.delete_memory(m["id"])
                st.rerun()


# --------------------------------------------------------------------------- evaluation
def _mode_chart(df: pd.DataFrame, metric: str, title: str, fmt: str) -> alt.Chart:
    domain = [MODE_LABELS[m] for m in MODE_COLORS]
    data = df.assign(Mode=df["mode"].map(MODE_LABELS))
    base = alt.Chart(data, title=title).encode(
        x=alt.X("Mode:N", title=None, sort=domain, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{metric}:Q", title=None, axis=alt.Axis(format=fmt, grid=True, gridOpacity=0.4, tickCount=4)),
        color=alt.Color("Mode:N", scale=alt.Scale(domain=domain, range=list(MODE_COLORS.values())),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("Mode:N"), alt.Tooltip(f"{metric}:Q", format=fmt, title=title)],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=42)
    labels = base.mark_text(dy=-8, color="#52514e").encode(text=alt.Text(f"{metric}:Q", format=fmt), color=alt.value("#52514e"))
    return (bars + labels).properties(height=200)


def evaluation_page() -> None:
    from sre_triage.evaluation import load_results, summarize

    st.title("Evaluation")
    st.caption("Multi-agent vs single-agent on the same incidents, tools, model and budget. "
               "Run with `uv run sre-triage eval` (takes several minutes per scenario).")
    rows = load_results(get_settings().data_dir)
    if not rows:
        st.info("No evaluation results yet. Run `uv run sre-triage eval cart_cpu_limit healthy_control` to start.")
        return
    evals = sorted({r["eval_id"] for r in rows})
    chosen = st.multiselect("Evaluation runs", evals, default=evals[-1:])
    rows = [r for r in rows if r["eval_id"] in chosen]
    if not rows:
        return
    summary = summarize(rows)
    df = pd.DataFrame([{"mode": m, **v} for m, v in summary.items()])
    cols = st.columns(4)
    charts = [("diagnosis_accuracy", "Diagnosis accuracy", ".0%"),
              ("unsupported_claims_per_run", "Unsupported claims / run", ".1f"),
              ("mean_duration_s", "Mean investigation time (s)", ".0f"),
              ("mean_tool_calls", "Mean tool calls", ".1f")]
    for col, (metric, title, fmt) in zip(cols, charts):
        if df[metric].notna().any():
            col.altair_chart(_mode_chart(df, metric, title, fmt), width="stretch")
    st.markdown("**Summary table**")
    st.dataframe(df.set_index("mode").rename(index=MODE_LABELS).T, width="stretch")
    st.markdown("**Per scenario**")
    st.dataframe(pd.DataFrame([{
        "Scenario": r["scenario"], "Difficulty": r["difficulty"], "Mode": MODE_LABELS.get(r["mode"], r["mode"]),
        "Correct": "✅" if r["diagnosis_correct"] else "❌", "Reported": f"{r['reported_service']} / {r['reported_category']}",
        "Expected": f"{r['expected_service']} / {r['expected_category']}",
        "Remediation ok": "✅" if r["remediation_correct"] else "❌", "Unsupported": r["unsupported_claims"],
        "Rejected hyps": r["rejected_or_weakened"], "Tools": r["tool_calls"], "LLM": r["llm_calls"],
        "Seconds": r["duration_s"], "Case": r["case_id"],
    } for r in rows]), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- fault lab
def fault_lab() -> None:
    from sre_triage.changes import read_flags
    from sre_triage.faults import SCENARIOS, inject, reset
    from sre_triage.telemetry import DockerInspector

    settings = get_settings()
    st.title("Fault lab")
    st.warning("Operator-only controls. Scenarios and their expected answers are never visible to the agents; "
               "changes made here appear in the configuration audit like any other change.", icon=":material/science:")
    c1, c2 = st.columns([2, 1])
    with c1:
        sid = st.selectbox("Scenario", list(SCENARIOS), format_func=lambda s: f"{s} — {SCENARIOS[s].title} ({SCENARIOS[s].difficulty})")
        scenario = SCENARIOS[sid]
        st.caption(scenario.notes or scenario.symptom)
        if st.button("Inject fault", type="primary", icon=":material/bolt:"):
            with st.spinner("Applying…"):
                st.session_state["last_injection"] = inject(scenario, settings.flagd_file)
        if st.session_state.get("last_injection"):
            st.json(st.session_state["last_injection"], expanded=False)
    with c2:
        if st.button("Reset environment", icon=":material/restart_alt:"):
            with st.spinner("Restoring flags and CPU limits…"):
                st.json(reset(settings.flagd_file), expanded=False)
    st.subheader("Current state")
    try:
        flags = {k: v["defaultVariant"] for k, v in read_flags(settings.flagd_file).items()}
        st.markdown("Non-default flags: " + (", ".join(f"`{k}={v}`" for k, v in flags.items() if v != "off") or "none"))
    except Exception as exc:
        st.error(f"Flags unreadable: {exc}")
    try:
        snaps = DockerInspector(settings.docker_base_url).snapshots(list(APP_SERVICES))
        st.dataframe(pd.DataFrame([{"Container": k, "CPU limit": v.get("cpu_limit_cpus"), "Memory MB": v.get("memory_limit_mb"),
                                    "Status": v.get("status"), "Restarts": v.get("restart_count")} for k, v in snaps.items()]),
                     hide_index=True, width="stretch")
    except Exception as exc:
        st.error(f"Docker unavailable: {exc}")

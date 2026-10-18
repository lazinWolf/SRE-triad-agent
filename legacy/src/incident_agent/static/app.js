"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  selectedId: null,
  currentCase: null,
  cases: [],
  modelConfigured: false,
  runtimeLoaded: false,
  busy: false,
  tab: "brief",
  pollTimer: null,
  lastRender: null,
};

function node(tag, className, content) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (content !== undefined && content !== null) element.textContent = String(content);
  return element;
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function humanize(value) {
  return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function dateValue(value) {
  if (value === undefined || value === null || value === "") return null;
  if (typeof value === "number") return new Date(value < 1e12 ? value * 1000 : value);
  return new Date(value);
}

function dateText(value, compact = false) {
  const date = dateValue(value);
  if (!date || Number.isNaN(date.getTime())) return "Time unavailable";
  return date.toLocaleString(undefined, compact
    ? { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }
    : { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function localInputValue(value) {
  const date = dateValue(value) || new Date();
  const shifted = new Date(date.getTime() - date.getTimezoneOffset() * 60_000);
  return shifted.toISOString().slice(0, 23);
}

function epochFromInput(value) {
  return new Date(value).getTime() / 1000;
}

function activeCase(incident) {
  return Boolean(incident && ["running", "investigating", "queued"].includes(incident.status));
}

function badgeClass(status) {
  if (["running", "investigating", "queued"].includes(status)) return "active";
  if (["completed", "complete", "resolved", "supported", "reviewed"].includes(status)) return "good";
  if (["failed", "error"].includes(status)) return "error";
  if (["insufficient_evidence", "unresolved", "mitigated", "needs_review"].includes(status)) return "warning";
  return "neutral";
}

function badge(status, label) {
  return node("span", `status-badge ${badgeClass(status)}`, label || humanize(status));
}

function notice(message, type = "info") {
  const box = $("global-message");
  box.textContent = message;
  box.className = `notice ${type}`;
  box.hidden = !message;
}

async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000);
  try {
    const response = await fetch(path, {
      ...options,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...options.headers },
    });
    const text = await response.text();
    let data;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      throw new Error(`The API returned an unreadable response (${response.status}).`);
    }
    if (!response.ok) {
      let reason = data && (data.detail || data.error || data.message);
      if (typeof reason !== "string") reason = reason ? JSON.stringify(reason) : `Request failed (${response.status}).`;
      throw new Error(reason);
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The local API did not respond within 30 seconds. Try again.");
    if (error instanceof TypeError) throw new Error("Cannot reach the local API. Check that the server is running.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function action(buttonId, callback) {
  if (state.busy) return;
  state.busy = true;
  const button = $(buttonId);
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Working…";
  notice("");
  try {
    await callback();
  } catch (error) {
    notice(error.message, "error");
  } finally {
    state.busy = false;
    button.textContent = label;
    button.disabled = false;
    updateControls();
  }
}

function updateModeHelp() {
  const replay = $("case-mode").value === "replay";
  $("mode-help").textContent = replay
    ? "Scripted ADK replay / orchestration check. Exercises tools and agent coordination without model reasoning."
    : "The model chooses queries and follow-up investigations using the available tools.";
}

function updateControls() {
  const running = activeCase(state.currentCase);
  const liveUnavailable = state.currentCase && state.currentCase.mode === "live" && !state.modelConfigured;
  $("investigate-case").disabled = state.busy || running || !state.currentCase || Boolean(liveUnavailable);
  $("investigate-case").textContent = running ? "Investigating…" : state.currentCase && state.currentCase.report ? "Investigate again" : "Investigate";
  $("investigate-case").title = liveUnavailable ? "Configure GOOGLE_API_KEY on the server to run a live investigation." : "";
  $("save-feedback").disabled = state.busy || running || !state.currentCase;
  $("save-review").disabled = state.busy || running || !state.currentCase || !state.currentCase.report;
  $("create-case").disabled = state.busy || !state.runtimeLoaded;
  $("load-demo-alert").disabled = state.busy;
}

async function loadRuntime() {
  const runtime = await api("/api/status");
  state.modelConfigured = Boolean(runtime.model_configured);
  state.runtimeLoaded = true;
  $("connection-status").textContent = "API connected";
  $("connection-status").className = "status-badge good";
  $("runtime-label").textContent = state.modelConfigured ? runtime.model || "Model configured" : "Replay available · Model not configured";
  $("runtime-detail").textContent = state.modelConfigured
    ? `Google ADK ${runtime.adk_version || ""} · Live investigations available.`
    : "Set GOOGLE_API_KEY on the server to enable live investigation.";
  const liveOption = $("case-mode").querySelector('option[value="live"]');
  liveOption.disabled = !state.modelConfigured;
  if (!state.modelConfigured) $("case-mode").value = "replay";
  updateModeHelp();
  updateControls();
}

async function loadCases() {
  const response = await api("/api/cases");
  state.cases = asArray(response);
  renderCases();
}

function renderCases() {
  const list = $("case-list");
  list.replaceChildren();
  if (!state.cases.length) {
    list.append(node("p", "empty-note", "No cases yet. Create one from an observed symptom."));
    return;
  }
  for (const incident of state.cases) {
    const item = node("button", `case-item${incident.id === state.selectedId ? " selected" : ""}`);
    item.type = "button";
    item.setAttribute("aria-pressed", String(incident.id === state.selectedId));
    item.append(node("strong", "", incident.summary || `${incident.service || "Service"} incident`));
    const meta = node("div", "case-item-meta");
    meta.append(badge(incident.status), node("span", "", dateText(incident.start, true)));
    if (incident.mode === "replay") meta.append(node("span", "", "Scripted replay"));
    item.append(meta);
    item.addEventListener("click", () => {
      selectCase(incident.id).catch((error) => notice(error.message, "error"));
    });
    list.append(item);
  }
}

async function selectCase(id) {
  clearTimeout(state.pollTimer);
  if (state.selectedId !== id) {
    state.currentCase = null;
    state.lastRender = null;
    $("feedback-form").reset();
    $("review-form").reset();
    setTab("brief");
  }
  state.selectedId = id;
  renderCases();
  $("welcome-view").hidden = true;
  $("case-view").hidden = false;
  if (!state.currentCase) {
    $("case-title").textContent = "Loading case…";
    $("case-context").textContent = "Incident case";
    $("case-window").textContent = "";
    $("case-error").hidden = true;
    $("case-mode-banner").hidden = true;
    $("case-status").textContent = "Loading";
    $("case-status").className = "status-badge neutral";
    for (const id of ["evidence-count", "activity-count", "feedback-count"]) $(id).textContent = "—";
    $("report-view").replaceChildren(node("p", "empty-note", "Loading investigation…"));
    $("evidence-list").replaceChildren();
    $("activity-list").replaceChildren();
    $("feedback-list").replaceChildren();
    $("investigate-case").disabled = true;
    $("save-feedback").disabled = true;
    $("save-review").disabled = true;
  }
  await loadSelectedCase();
}

async function loadSelectedCase() {
  const id = state.selectedId;
  if (!id) return;
  const incident = await api(`/api/cases/${encodeURIComponent(id)}`);
  if (state.selectedId !== id) return;
  state.currentCase = incident;
  renderCase(incident);
  const index = state.cases.findIndex((entry) => entry.id === id);
  if (index >= 0) state.cases[index] = incident;
  else state.cases.unshift(incident);
  renderCases();
  clearTimeout(state.pollTimer);
  if (activeCase(incident)) {
    state.pollTimer = setTimeout(() => {
      loadSelectedCase().catch((error) => {
        notice(error.message, "error");
        state.pollTimer = setTimeout(() => loadSelectedCase().catch((retryError) => notice(retryError.message, "error")), 5000);
      });
    }, 1800);
  }
}

function renderCase(incident) {
  const signature = JSON.stringify([incident.id, incident.revision, incident.status, incident.report, incident.evidence, incident.events, incident.feedback, incident.error]);
  if (signature === state.lastRender) {
    updateControls();
    return;
  }
  state.lastRender = signature;
  $("case-title").textContent = incident.summary;
  $("case-context").textContent = `${incident.service} · ${incident.environment} · Case ${String(incident.id).slice(0, 8)}`;
  $("case-window").textContent = `${dateText(incident.start)} – ${dateText(incident.end)} · Local display time${incident.incident_start ? ` · Symptom onset ${dateText(incident.incident_start)}` : ""}`;
  $("case-status").textContent = humanize(incident.status);
  $("case-status").className = `status-badge ${badgeClass(incident.status)}`;
  const isReplay = incident.mode === "replay" || Boolean(incident.report && incident.report.scripted_mode);
  $("case-mode-banner").hidden = !isReplay;
  $("case-mode-banner").textContent = "Scripted ADK replay / orchestration check. Findings in this mode come from a deterministic script using tool results; this run does not demonstrate model reasoning.";
  $("case-error").hidden = !incident.error;
  $("case-error").textContent = incident.error || "";
  $("evidence-count").textContent = asArray(incident.evidence).length;
  $("activity-count").textContent = asArray(incident.events).length;
  $("feedback-count").textContent = asArray(incident.feedback).length;
  renderReport(incident);
  renderEvidence(incident);
  renderActivity(incident);
  renderFeedback(incident);
  updateControls();
}

function evidenceRefs(ids, label) {
  const row = node("div", "evidence-references");
  row.append(node("span", "reference-label", label));
  if (!asArray(ids).length) row.append(node("span", "none-reference", "None recorded"));
  for (const id of asArray(ids)) {
    const button = node("button", "evidence-ref", id);
    button.type = "button";
    button.setAttribute("aria-label", `Inspect evidence ${id}`);
    button.addEventListener("click", () => {
      setTab("evidence");
      const record = document.getElementById(`evidence-${String(id)}`);
      if (record) {
        record.open = true;
        record.scrollIntoView({ block: "nearest", behavior: "auto" });
        record.querySelector("summary").focus();
      } else {
        notice(`Evidence ${id} is not present in this case's ledger.`, "error");
      }
    });
    row.append(button);
  }
  return row;
}

function textList(title, items, extraClass = "") {
  const section = node("section", `report-section ${extraClass}`);
  section.append(node("h3", "", title));
  const list = node("ul", "plain-list");
  for (const item of asArray(items)) list.append(node("li", "", item));
  section.append(list);
  return section;
}

function renderReport(incident) {
  const container = $("report-view");
  container.replaceChildren();
  if (!incident.report) {
    const empty = node("div", "empty-state");
    let title = "Ready to investigate";
    let explanation = "Start an investigation to gather evidence and develop a supported explanation. You can inspect each tool result and add corrections along the way.";
    if (activeCase(incident)) {
      title = "Investigation in progress";
      explanation = "Agents are gathering and evaluating evidence. Open Evidence or Activity to follow the run; this view updates automatically.";
    } else if (incident.error) {
      title = "Investigation did not complete";
      explanation = "Review the error and agent activity. Evidence already collected remains available; you can retry after addressing the issue.";
    } else if (asArray(incident.feedback).length) {
      title = "Correction saved — ready for another run";
      explanation = "The previous report was cleared. Investigate again to evaluate the evidence with your updated context.";
    }
    empty.append(node("h3", "", title), node("p", "", explanation));
    container.append(empty);
    return;
  }
  const report = incident.report;
  const summary = node("div", "report-summary");
  summary.append(badge(report.diagnosis_status, report.diagnosis_status === "supported" ? "Diagnosis supported by evidence" : "Insufficient evidence"));
  summary.append(node("p", "", report.summary));
  container.append(summary);

  if (asArray(report.hypotheses).length) {
    const section = node("section", "report-section");
    section.append(node("h3", "", "Explanations considered"));
    for (const hypothesis of report.hypotheses) {
      const card = node("article", "hypothesis-card");
      const header = node("div", "hypothesis-header");
      const description = node("div");
      description.append(node("h4", "", hypothesis.cause), node("p", "hypothesis-service", hypothesis.service || "Service not specified"));
      header.append(description, badge(hypothesis.status));
      card.append(header, evidenceRefs(hypothesis.supporting_evidence, "Supports"), evidenceRefs(hypothesis.contradicting_evidence, "Contradicts"));
      section.append(card);
    }
    container.append(section);
  }

  if (report.proposed_action) {
    const proposal = report.proposed_action;
    const section = node("section", "report-section");
    section.append(node("h3", "", "Proposed next action"));
    const card = node("div", "action-card");
    card.append(node("p", "", proposal.description));
    const details = node("dl");
    for (const [label, value] of [["Affected service", proposal.service], ["Reason", proposal.reason], ["Rollback", proposal.rollback]]) {
      details.append(node("dt", "", label), node("dd", "", value || "Not provided"));
    }
    card.append(details, evidenceRefs(proposal.evidence_ids, "Evidence"));
    section.append(card, node("p", "action-note", "Review this proposal before making an operational change. This interface does not execute it."));
    container.append(section);
  }
  if (asArray(report.next_checks).length) container.append(textList("Next checks", report.next_checks));
  if (asArray(report.limitations).length) container.append(textList("Uncertainty and limitations", report.limitations, "limitations"));
}

function pretty(value) {
  if (typeof value === "string") return value;
  return JSON.stringify(value === undefined ? null : value, null, 2);
}

function renderEvidence(incident) {
  const list = $("evidence-list");
  const openIds = new Set(Array.from(list.querySelectorAll("details[open]")).map((element) => element.id));
  list.replaceChildren();
  if (!asArray(incident.evidence).length) {
    list.append(node("p", "empty-note", "No tool evidence has been collected for this case yet."));
    return;
  }
  for (const evidence of incident.evidence) {
    const card = node("details", "evidence-card");
    card.id = `evidence-${String(evidence.id)}`;
    card.open = openIds.has(card.id);
    const heading = node("summary");
    const title = node("div");
    const tool = node("div");
    tool.append(node("span", "evidence-id", evidence.id), node("span", "evidence-tool", evidence.tool));
    title.append(tool, node("p", "evidence-meta", `${evidence.agent || "Agent unavailable"} · ${dateText(evidence.collected_at)}`));
    heading.append(title);
    const body = node("div", "evidence-body");
    body.append(node("h4", "", "Query"), node("pre", "", pretty(evidence.query)), node("h4", "", "Returned evidence"), node("pre", "", pretty(evidence.result)));
    card.append(heading, body);
    list.append(card);
  }
}

function renderActivity(incident) {
  const list = $("activity-list");
  list.replaceChildren();
  if (!asArray(incident.events).length) {
    list.append(node("li", "empty-note", "Agent activity will appear when the investigation starts."));
    return;
  }
  for (const event of incident.events) {
    const item = node("li", "timeline-item");
    const meta = node("div", "timeline-meta");
    meta.append(node("span", "timeline-agent", event.agent || "System"), node("time", "", dateText(event.timestamp)), node("span", "event-kind", humanize(event.kind || "event")));
    item.append(meta, node("p", "timeline-message", event.message));
    list.append(item);
  }
}

function renderFeedback(incident) {
  const list = $("feedback-list");
  list.replaceChildren();
  if (!asArray(incident.feedback).length) return;
  list.append(node("h3", "", "Corrections recorded"));
  for (const feedback of asArray(incident.feedback)) {
    const item = node("article", "feedback-item");
    item.append(node("time", "", dateText(feedback.timestamp)), node("p", "", feedback.text));
    list.append(item);
  }
}

async function loadMemory() {
  const response = await api("/api/memory");
  const list = $("memory-list");
  list.replaceChildren();
  if (!asArray(response).length) {
    list.append(node("p", "empty-note", "No reviewed incidents yet. Use Feedback & review on a case to retain a reviewed finding."));
    return;
  }
  for (const memory of asArray(response)) {
    const card = node("article", "memory-card");
    const meta = node("div", "memory-meta");
    meta.append(badge(memory.outcome), node("span", "", `${memory.service || "Service"} · ${memory.environment || "Environment"}`));
    if (memory.origin_mode === "replay") meta.append(node("span", "subtle-tag", "Scripted replay origin"));
    card.append(meta, node("h3", "", memory.confirmed_cause || "Cause remains unconfirmed"));
    if (memory.summary) card.append(node("p", "", memory.summary));
    if (memory.notes) card.append(node("p", "", memory.notes));
    card.append(node("p", "", `Reviewed ${dateText(memory.reviewed_at || memory.created_at, true)}`));
    if (memory.case_id) {
      const button = node("button", "button text", "Open source case");
      button.type = "button";
      button.addEventListener("click", () => selectCase(memory.case_id).then(() => $("workspace").scrollIntoView({ block: "start" })).catch((error) => notice(error.message, "error")));
      card.append(button);
    }
    list.append(card);
  }
}

function setTab(name, focus = false) {
  state.tab = name;
  document.querySelectorAll("[data-tab]").forEach((button) => {
    const selected = button.dataset.tab === name;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    $(button.getAttribute("aria-controls")).hidden = !selected;
    if (selected && focus) button.focus();
  });
}

document.querySelectorAll("[data-tab]").forEach((button) => {
  button.addEventListener("click", () => setTab(button.dataset.tab));
  button.addEventListener("keydown", (event) => {
    const tabs = Array.from(document.querySelectorAll("[data-tab]"));
    let index = tabs.indexOf(button);
    if (event.key === "ArrowRight") index = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft") index = (index + tabs.length - 1) % tabs.length;
    else if (event.key === "Home") index = 0;
    else if (event.key === "End") index = tabs.length - 1;
    else return;
    event.preventDefault();
    setTab(tabs[index].dataset.tab, true);
  });
});

$("case-mode").addEventListener("change", updateModeHelp);

$("case-form").addEventListener("submit", (event) => {
  event.preventDefault();
  action("create-case", async () => {
    const start = epochFromInput($("case-start").value);
    const end = epochFromInput($("case-end").value);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) throw new Error("Choose a valid time window with the end later than the start.");
    if (end - start > 3600) throw new Error("Keep the investigation window within one hour.");
    const onset = $("case-onset").value ? epochFromInput($("case-onset").value) : null;
    if (onset !== null && (!Number.isFinite(onset) || onset <= start || onset >= end)) throw new Error("Symptom onset must be inside the investigation window.");
    const summary = $("case-summary").value.trim();
    if (!summary) throw new Error("Describe the symptom before creating a case.");
    const incident = await api("/api/cases", {
      method: "POST",
      body: JSON.stringify({ service: "checkout", environment: "local", summary, start, end, incident_start: onset, mode: $("case-mode").value }),
    });
    await loadCases();
    await selectCase(incident.id);
    notice("Case created. Start the investigation when the incident context is ready.", "success");
  });
});

$("load-demo-alert").addEventListener("click", () => {
  action("load-demo-alert", async () => {
    const alert = await api("/api/demo/alert");
    $("case-summary").value = alert.summary || "Checkout requests are slow or timing out.";
    if (alert.start !== undefined) $("case-start").value = localInputValue(alert.start);
    if (alert.end !== undefined) $("case-end").value = localInputValue(alert.end);
    $("case-onset").value = alert.incident_start !== undefined ? localInputValue(alert.incident_start) : "";
    notice("Demo alert loaded. Review the symptom and time window, then create the case.", "success");
  });
});

$("investigate-case").addEventListener("click", () => {
  action("investigate-case", async () => {
    const id = state.selectedId;
    if (!id) return;
    await api(`/api/cases/${encodeURIComponent(id)}/investigate`, { method: "POST", body: "{}" });
    if (id === state.selectedId) {
      setTab("activity");
      await loadSelectedCase();
    }
  });
});

$("feedback-form").addEventListener("submit", (event) => {
  event.preventDefault();
  action("save-feedback", async () => {
    const id = state.selectedId;
    const text = $("feedback-text").value.trim();
    if (!id || !text) throw new Error("Enter a correction or additional context.");
    await api(`/api/cases/${encodeURIComponent(id)}/feedback`, { method: "POST", body: JSON.stringify({ text }) });
    if (id === state.selectedId) {
      $("feedback-text").value = "";
      await loadSelectedCase();
    }
    notice("Correction saved. Run the investigation again to incorporate it.", "success");
  });
});

$("review-form").addEventListener("submit", (event) => {
  event.preventDefault();
  action("save-review", async () => {
    const id = state.selectedId;
    if (!id) return;
    await api(`/api/cases/${encodeURIComponent(id)}/review`, {
      method: "POST",
      body: JSON.stringify({
        confirmed_cause: $("confirmed-cause").value.trim(),
        outcome: $("review-outcome").value,
        notes: $("review-notes").value.trim(),
      }),
    });
    await loadMemory();
    if (id === state.selectedId) await loadSelectedCase();
    notice("Reviewed incident saved to memory. Future investigations can retrieve it as context.", "success");
  });
});

$("refresh-cases").addEventListener("click", () => action("refresh-cases", loadCases));
$("refresh-memory").addEventListener("click", () => action("refresh-memory", loadMemory));

async function initialize() {
  const now = Date.now();
  $("case-start").value = localInputValue(now - 15 * 60_000);
  $("case-end").value = localInputValue(now);
  $("create-case").disabled = true;
  const results = await Promise.allSettled([loadRuntime(), loadCases(), loadMemory()]);
  const errors = results.filter((result) => result.status === "rejected").map((result) => result.reason.message);
  if (errors.length) {
    notice(Array.from(new Set(errors)).join(" "), "error");
    if (!state.runtimeLoaded) {
      $("connection-status").textContent = "API unavailable";
      $("connection-status").className = "status-badge error";
      $("runtime-label").textContent = "Configuration unavailable";
      $("runtime-detail").textContent = "Check the local server, then reload this page.";
    }
    if (results[1].status === "rejected") $("case-list").replaceChildren(node("p", "empty-note", "Could not load cases. Use refresh to retry."));
    if (results[2].status === "rejected") $("memory-list").replaceChildren(node("p", "empty-note", "Could not load reviewed memory. Use refresh to retry."));
  }
  if (state.cases.length) {
    try { await selectCase(state.cases[0].id); }
    catch (error) { notice(error.message, "error"); }
  }
}

initialize();

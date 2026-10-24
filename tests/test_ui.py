"""The dashboard renders (headless) even when telemetry backends are down."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

UI = str(Path(__file__).resolve().parents[1] / "src" / "sre_triage" / "ui.py")


def test_dashboard_renders_without_backends(tmp_path, monkeypatch):
    monkeypatch.setenv("SRE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SRE_PROMETHEUS_URL", "http://127.0.0.1:9")
    from sre_triage.config import get_settings

    get_settings.cache_clear()
    at = AppTest.from_file(UI, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value.startswith("🚨")
    assert any("Telemetry unavailable" in e.value for e in at.error)
    assert any("No cases yet" in i.value for i in at.info)
    get_settings.cache_clear()


def test_inside_adk_page_shows_the_case_session(tmp_path, monkeypatch):
    from fakes import ScriptedModel, World, fake_backends, settings_for, write_flags
    from sre_triage.workflow import Triage

    monkeypatch.setenv("SRE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SRE_PROMETHEUS_URL", "http://127.0.0.1:9")
    from sre_triage.config import get_settings

    get_settings.cache_clear()
    st.cache_resource.clear()  # the dashboard caches its Triage, which may point at another test's data dir
    world = World(write_flags(tmp_path / "flags.json"))
    t = Triage(settings_for(tmp_path), fake_backends(world), model_factory=lambda name: ScriptedModel())
    t.run(t.open_case("Checkout latency is elevated"))
    # load the dashboard without rendering its default page, then render only "Inside ADK"
    script = f"""
import streamlit as st
st.navigation = lambda pages, **_: type("Nav", (), {{"run": lambda self: None}})()
ui = {{"__file__": {UI!r}, "__name__": "__main__"}}
exec(compile(open({UI!r}).read(), {UI!r}, "exec"), ui)
ui["adk_page"]()
"""
    at = AppTest.from_string(script, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "🧩 Inside ADK"
    metrics = {m.label: m.value for m in at.metric}
    assert int(metrics["Runner events"]) > 10 and int(metrics["Agent runs (ReAct)"]) >= 2
    assert any("Critic ran locally" in i.value for i in at.info)
    get_settings.cache_clear()

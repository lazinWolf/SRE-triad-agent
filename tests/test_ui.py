"""The dashboard renders (headless) even when telemetry backends are down."""

from __future__ import annotations

from pathlib import Path

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

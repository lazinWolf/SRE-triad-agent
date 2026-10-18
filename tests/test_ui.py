"""Render every dashboard page against a populated ledger and fail on any exception."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from fakes import fake_backends, model_factory
from sre_triage import config
from sre_triage.ledger import Ledger
from sre_triage.orchestrator import investigate
from sre_triage.ui import common
from test_orchestrator import new_case, team_script

APP = str(config.PROJECT_ROOT / "src" / "sre_triage" / "ui" / "app.py")


@pytest.fixture
def populated(tmp_path, monkeypatch):
    flags = tmp_path / "demo" / "src" / "flagd"
    flags.mkdir(parents=True)
    (flags / "demo.flagd.json").write_text(json.dumps({"flags": {
        "paymentFailure": {"state": "ENABLED", "defaultVariant": "off", "variants": {"off": 0, "50%": 0.5}}}}))
    monkeypatch.setenv("SRE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SRE_DEMO_DIR", str(tmp_path / "demo"))
    monkeypatch.setenv("SRE_DOCKER_URL", "unix:///nonexistent.sock")
    config.get_settings.cache_clear()
    common.ledger.cache_clear()
    settings = config.get_settings()
    ledger = Ledger(settings.db_path)
    ledger.add_change("cart", "container_config", {"cpu_limit_cpus": 1.0}, {"cpu_limit_cpus": 0.05}, "cart cpu", ts=1_800_000_005)
    yield settings, ledger
    config.get_settings.cache_clear()
    common.ledger.cache_clear()


async def test_pages_render(populated):
    settings, ledger = populated
    case = new_case(ledger)
    backends = fake_backends()
    await investigate(ledger, settings, case["id"], backends=backends, model_override=model_factory(team_script))
    await backends.aclose()
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    for page in ("incidents", "case_page", "memory_page", "evaluation_page", "fault_lab"):
        script = f"from sre_triage.ui import pages\npages.{page}()\n"
        at = AppTest.from_string(script, default_timeout=30)
        at.query_params["id"] = case["id"]
        at.run()
        assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
        if page == "case_page":
            text = " ".join(m.value for m in at.markdown)
            assert "Cart CPU limit throttles cart" in text  # hypothesis board
            assert "Set CPU limit of cart to 1.0 CPUs" in text  # remediation awaiting approval
            assert [b.label for b in at.button if b.label == "Approve and execute"]

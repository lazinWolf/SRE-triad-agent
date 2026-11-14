"""Incident simulator: injection and reset against a fake Docker and a temporary flag file, scoring, generation, probing."""

from __future__ import annotations

import copy
import json
import os
import re
from types import SimpleNamespace

import pytest

from sre_triage import simulator as sim
from sre_triage.config import SERVICES
from sre_triage.simulator import CATALOG, Fault, Scenario, generate, inject, probe, reset, score, truth_for
from sre_triage.telemetry import Unavailable

BROKEN_RULE = {"if": [{"==": [{"var": "product_id"}, "OLJCESPC7Z"]}, "off", "off"]}
STANDARD_SERVICES = {**{s: {"cpus": 1.0, "memory_mb": 200} for s in SERVICES},
                     "valkey-cart": {"cpus": None, "memory_mb": 20}, "astronomy-db": {"cpus": None, "memory_mb": None}}


def pristine_flags() -> dict:
    flags = {}
    for name, (_, _, _, values) in sim.FLAGS.items():
        flags[name] = {"state": "ENABLED", "defaultVariant": "off", "variants": {v: True for v in ("off", *values)}}
    flags["productCatalogFailure"]["targeting"] = copy.deepcopy(BROKEN_RULE)
    flags["emitRawPii"] = {"state": "ENABLED", "defaultVariant": "off", "variants": {"off": False, "on": True}}
    return {"$schema": "x", "flags": flags}


class FakeDocker:
    def __init__(self, standard: dict):
        self.state = {n: {"cpus": s["cpus"], "memory": s["memory_mb"], "status": "running"} for n, s in standard.items()}
        self.down: set[str] = set()   # containers whose API calls fail

    def inspect(self, name):
        if name in self.down or name not in self.state:
            raise Unavailable(f"no {name}")
        s = self.state[name]
        return {"cpu_limit": s["cpus"], "memory_limit_mb": s["memory"], "status": s["status"]}

    def _set(self, name, **kw):
        if name not in self.state:
            raise Unavailable(f"no {name}")
        self.state[name].update(kw)

    def set_cpus(self, name, cpus): self._set(name, cpus=cpus)
    def set_memory(self, name, mb): self._set(name, memory=mb)
    def stop(self, name): self._set(name, status="exited")
    def start(self, name): self._set(name, status="running")
    def pause(self, name): self._set(name, status="paused")
    def unpause(self, name): self._set(name, status="running")


class FakePrometheus:
    def __init__(self, script):
        self.script, self.calls = script, 0

    def alerts(self):
        self.calls += 1
        rows = self.script(self.calls)
        return [{"name": "x", "service": s, "kind": "latency", "severity": "warning", "state": st, "since": 1.0, "summary": ""}
                for s, st in rows]


@pytest.fixture
def world(tmp_path):
    flag_file = tmp_path / "demo.flagd.json"
    flag_file.write_text(json.dumps(pristine_flags(), indent=2) + "\n")
    standard = {"services": STANDARD_SERVICES, "flags": pristine_flags()}
    return SimpleNamespace(docker=FakeDocker(STANDARD_SERVICES), flagd_file=flag_file, standard=standard,
                           prometheus=FakePrometheus(lambda n: []))


def at_standard(world) -> bool:
    docker_ok = all(world.docker.state[n] == {"cpus": s["cpus"], "memory": s["memory_mb"], "status": "running"}
                    for n, s in STANDARD_SERVICES.items())
    return docker_ok and json.loads(world.flagd_file.read_text()) == pristine_flags()


def flag(world, name) -> dict:
    return json.loads(world.flagd_file.read_text())["flags"][name]


# --- inject / reset -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", list(CATALOG.values()), ids=list(CATALOG))
def test_inject_reset_round_trip(world, scenario):
    record = inject(scenario, world)
    json.dumps(record)
    assert record["scenario"] == scenario.id and len(record["faults"]) == len(scenario.faults)
    assert at_standard(world) == (not scenario.faults and not scenario.decoys)
    assert bool(reset(world)) == bool(scenario.faults or scenario.decoys)
    assert at_standard(world)
    assert reset(world) == []  # idempotent


def test_every_fault_kind_applies(world):
    inject(Scenario("t", "t", "s", (Fault("cpu_limit", "cart", 0.05), Fault("memory_limit", "payment", 48),
                                    Fault("stop", "shipping"), Fault("pause", "currency"))), world)
    s = world.docker.state
    assert (s["cart"]["cpus"], s["payment"]["memory"], s["shipping"]["status"], s["currency"]["status"]) == (
        0.05, 48, "exited", "paused")
    reset(world)
    assert at_standard(world)


def test_product_catalog_failure_patches_targeting_and_reset_restores_it(world):
    inode = os.stat(world.flagd_file).st_ino
    inject(CATALOG["catalog_failure"], world)
    entry = flag(world, "productCatalogFailure")
    assert entry["defaultVariant"] == "on" and entry["targeting"] == sim.FIRING_TARGETING["productCatalogFailure"]
    lines = reset(world)
    assert flag(world, "productCatalogFailure")["targeting"] == BROKEN_RULE
    assert any("productCatalogFailure targeting restored" in line for line in lines)
    assert os.stat(world.flagd_file).st_ino == inode  # edited in place


def test_plain_flag_leaves_targeting_alone_and_unknown_variant_fails(world):
    inject(Scenario("t", "t", "s", (Fault("flag", "paymentFailure", "50%"),)), world)
    assert flag(world, "paymentFailure")["defaultVariant"] == "50%" and "targeting" not in flag(world, "paymentFailure")
    with pytest.raises(ValueError):
        inject(Scenario("t", "t", "s", (Fault("flag", "paymentFailure", "bogus"),)), world)


def test_reset_describes_changes_and_tolerates_unavailable(world):
    inject(CATALOG["cart_cpu_limit_decoys"], world)
    world.docker.down.add("cart")
    lines = reset(world)
    assert "emitRawPii on -> off" in lines and any(line.startswith("recommendation cpus") for line in lines)
    assert world.docker.state["cart"]["cpus"] == 0.02  # skipped, not fatal
    world.docker.down.clear()
    assert reset(world) == ["cart cpus 0.02 -> 1.0"]


def test_reset_starts_stopped_datastores_and_unpauses(world):
    inject(Scenario("t", "t", "s", (Fault("stop", "valkey-cart"),), (Fault("pause", "astronomy-db"),)), world)
    lines = reset(world)
    assert "valkey-cart started" in lines and "astronomy-db unpaused" in lines and at_standard(world)


def test_reset_without_standard_falls_back_to_flags_off_and_baseline_cpu(world):
    world.standard = {"services": {}, "flags": {"flags": {}}}
    inject(Scenario("t", "t", "s", (Fault("flag", "paymentFailure", "50%"), Fault("cpu_limit", "cart", 0.02))), world)
    assert reset(world) == ["paymentFailure -> off", "cart cpus 0.02 -> 1.0"]
    assert world.docker.state["cart"]["cpus"] == 1.0 and flag(world, "paymentFailure")["defaultVariant"] == "off"


# --- truth and scoring ----------------------------------------------------------------------------------------

def test_truth_for_each_kind():
    t = truth_for(Fault("cpu_limit", "cart", 0.02))
    assert (t.services, t.categories, t.fixes) == (frozenset({"cart"}), frozenset({"cpu_limit"}),
                                                  ({"action": "set_cpu_limit", "service": "cart"},))
    t = truth_for(Fault("memory_limit", "payment", 48))
    assert t.categories == {"memory_limit", "memory_exhaustion"} and t.fixes == ({"action": "set_memory_limit", "service": "payment"},)
    t = truth_for(Fault("stop", "shipping"))
    assert t.categories == {"service_down", "dependency_unavailable"} and t.services == {"shipping"}
    assert [f["action"] for f in t.fixes] == ["start_service", "restart_service"]
    assert [f["action"] for f in truth_for(Fault("pause", "currency")).fixes] == ["unpause_service", "restart_service"]
    t = truth_for(Fault("stop", "valkey-cart"))
    assert t.services == {"valkey-cart", "cart"} and t.service == "valkey-cart"
    assert truth_for(Fault("pause", "astronomy-db")).services == {"astronomy-db", "product-catalog"}


def test_truth_for_flags():
    t = truth_for(Fault("flag", "paymentFailure", "50%"))
    assert (t.service, t.category) == ("payment", "dependency_errors") and "config_change" in t.categories
    assert "checkout" in t.services and t.fixes == ({"action": "set_feature_flag", "flag": "paymentFailure"},)
    assert [f["action"] for f in truth_for(Fault("flag", "emailMemoryLeak", "100x")).fixes] == ["set_feature_flag", "restart_service"]
    assert truth_for(Fault("flag", "productCatalogLockContention", "on")).services == {"product-catalog", "astronomy-db"}
    assert all("config_change" in truth_for(Fault("flag", f, v[3][0])).categories for f, v in sim.FLAGS.items())
    with pytest.raises(ValueError):
        truth_for(Fault("flag", "emitRawPii", "on"))


def report(service="payment", category="dependency_errors", fix=None, status="diagnosed", **extra):
    return {"status": status, "root_cause_service": service, "category": category, "remediation": fix, **extra}


def test_score_accepts_any_listed_service_category_and_fix():
    s = CATALOG["payment_failure"]
    fix = {"action": "set_feature_flag", "flag": "paymentFailure", "value": "off"}
    assert score(s, report(fix=fix)) == {"correct": True, "service": True, "category": True, "remediation": True, "false_alarm": False}
    assert score(s, report("checkout", "config_change", fix))["correct"]


def test_score_wrong_service_is_a_false_alarm():
    got = score(CATALOG["payment_failure"], report("cart", "dependency_errors"))
    assert got["false_alarm"] and not got["correct"] and not got["service"] and got["category"]


def test_score_alternative_fix_counts():
    s = CATALOG["valkey_cart_stopped"]
    wrong, right = {"action": "restart_service", "service": "cart"}, {"action": "start_service", "service": "valkey-cart"}
    assert not score(s, report("valkey-cart", "service_down", wrong))["remediation"]
    assert score(s, report("valkey-cart", "service_down", wrong, alternative=right))["remediation"]
    assert not score(s, report("valkey-cart", "service_down", None))["remediation"]


def test_score_inconclusive_and_none_on_a_fault():
    for r in (None, {"status": "inconclusive"}):
        got = score(CATALOG["cart_cpu_limit"], r)
        assert not any(got.values())


@pytest.mark.parametrize("control", ["healthy_control", "config_change_control"])
def test_score_controls(control):
    s = CATALOG[control]
    assert s.service is None and s.category is None
    assert score(s, {"status": "inconclusive"})["correct"] and not score(s, {"status": "inconclusive"})["false_alarm"]
    bad = score(s, report("cart", "cpu_limit", {"action": "set_cpu_limit", "service": "cart"}))
    assert bad["false_alarm"] and not bad["correct"]
    assert score(s, None)["correct"]


# --- generation -----------------------------------------------------------------------------------------------

def test_generate_is_deterministic_per_seed():
    assert generate(3, 25) == generate(3, 25)
    assert generate(3, 25) != generate(4, 25)
    assert [s.id for s in generate(3, 10)] == [s.id for s in generate(3, 25)][:10]


@pytest.mark.parametrize("seed", range(8))
def test_generated_scenarios_are_valid(seed, world):
    scenarios = generate(seed, 36)
    assert len({s.id for s in scenarios}) == 36
    controls = [s for s in scenarios if s.truth is None]
    assert len(controls) == 6 and all(not s.faults for s in controls)
    containers, flags = set(STANDARD_SERVICES), set(pristine_flags()["flags"])
    for i, s in enumerate(scenarios):
        assert re.fullmatch(rf"gen-{seed}-{i}-[a-z_]+-[\w-]+", s.id)
        assert len(s.faults) <= 1  # never two real faults
        assert s.symptom and s.difficulty in sim.LEVELS
        for f in s.faults:
            assert (f.target in flags) if f.kind == "flag" else (f.target in containers)
            assert f.kind != "cpu_limit" or f.target in SERVICES  # no datastore throttling
        avoid = sim._failing_path(s.truth.services) if s.truth else set()
        for d in s.decoys:
            assert (d.kind, d.target) == ("flag", "emitRawPii") or (d.kind == "cpu_limit" and d.value >= 0.5)
            assert d.target not in avoid
        if s.difficulty == "easy":
            assert not s.decoys
        inject(s, world)  # every combination is applicable, and resettable
        reset(world)
        assert at_standard(world)


def test_generate_difficulty_levels():
    assert all(not s.decoys for s in generate(1, 30, "easy"))
    hard = [s for s in generate(1, 30, "hard") if s.truth]
    assert all(s.decoys and s.difficulty == "hard" for s in hard)
    assert {s.difficulty for s in generate(1, 60)} == set(sim.LEVELS)
    with pytest.raises(ValueError):
        generate(1, 3, "impossible")


def test_generate_covers_kinds_and_flags():
    scenarios = [f for seed in range(5) for s in generate(seed, 40) for f in s.faults]
    assert {f.kind for f in scenarios} == {"cpu_limit", "memory_limit", "stop", "pause", "flag"}
    assert {f.target for f in scenarios if f.kind == "flag"} == set(sim.FLAGS)


# --- probe ----------------------------------------------------------------------------------------------------

# alerts() is called once to see that the system is quiet, once to note what is already there, then once per poll
def test_probe_detects_a_new_alert_on_the_failing_path_and_resets(world):
    world.prometheus = FakePrometheus(lambda n: [] if n <= 2 else [("email", "firing")] if n == 3
                                      else [("email", "firing"), ("checkout", "firing")])
    sleeps = []
    result = probe(CATALOG["payment_failure"], world, wait=100, poll=10, sleep=sleeps.append)
    assert result["detected"] and result["seconds"] == 10 and sleeps == [10] and result["clean_start"]
    assert result["scenario"] == "payment_failure" and {a["service"] for a in result["alerts"]} == {"email", "checkout"}
    assert at_standard(world)


def test_probe_ignores_resolved_off_path_and_leftover_alerts(world):
    # checkout was already firing before the injection: it must not count as a detection of this fault
    world.prometheus = FakePrometheus(lambda n: [("payment", "inactive"), ("checkout", "firing")])
    sleeps = []
    result = probe(CATALOG["payment_failure"], world, wait=30, poll=10, sleep=sleeps.append, settle=0)
    assert (result["detected"], result["seconds"], result["clean_start"]) == (False, None, False)
    assert sleeps == [10, 10, 10] and result["alerts"] == []
    assert at_standard(world)


def test_probe_waits_for_earlier_alerts_to_clear_before_injecting(world):
    world.prometheus = FakePrometheus(lambda n: [("cart", "firing")] if n <= 3 or n >= 7 else [])
    sleeps = []
    result = probe(CATALOG["cart_cpu_limit"], world, wait=100, poll=10, sleep=sleeps.append)
    assert result["detected"] and result["clean_start"] and result["seconds"] == 10
    assert sleeps == [10, 10, 10, 10], "three waits for the old alert to clear, then one poll"


def test_probe_control_detects_false_positives(world):
    quiet = probe(CATALOG["healthy_control"], world, wait=20, poll=10, sleep=lambda s: None)
    assert not quiet["detected"]
    world.prometheus = FakePrometheus(lambda n: [] if n <= 2 else [("ad", "pending")])
    assert not probe(CATALOG["healthy_control"], world, wait=20, poll=10, sleep=lambda s: None)["detected"], \
        "a pending alert may still clear: only a firing one is a detection"
    world.prometheus = FakePrometheus(lambda n: [] if n <= 2 else [("ad", "firing")])
    assert probe(CATALOG["healthy_control"], world, wait=20, poll=10, sleep=lambda s: None)["detected"]


def test_probe_resets_even_when_something_raises(world):
    def boom(n):
        if n > 2:
            raise RuntimeError("prometheus exploded")
        return []
    world.prometheus = FakePrometheus(boom)
    with pytest.raises(RuntimeError):
        probe(CATALOG["cart_cpu_limit"], world, sleep=lambda s: None)
    assert at_standard(world)


def test_probe_survives_unavailable_prometheus(world):
    def flaky(n):
        if n == 3:
            raise Unavailable("down")
        return [("cart", "firing")] if n > 3 else []
    world.prometheus = FakePrometheus(flaky)
    assert probe(CATALOG["cart_cpu_limit"], world, poll=5, sleep=lambda s: None)["seconds"] == 5


# --- catalog and persistence ----------------------------------------------------------------------------------

def test_catalog_sanity():
    assert 12 <= len(CATALOG) <= 18
    assert {"cart_cpu_limit", "cart_cpu_limit_decoys", "payment_failure", "healthy_control"} <= set(CATALOG)
    for key, s in CATALOG.items():
        assert s.id == key and s.title and s.symptom and s.difficulty in sim.LEVELS
        assert (s.service, s.category) == ((s.truth.service, s.truth.category) if s.truth else (None, None))
        assert not any(f.target.lower() in s.symptom.lower() for f in s.faults)  # the symptom never names the cause
        assert len(s.faults) <= 1
    faults = [f for s in CATALOG.values() for f in s.faults]
    assert {f.kind for f in faults} == {"cpu_limit", "memory_limit", "stop", "pause", "flag"}
    assert {f.target for f in faults if f.kind == "flag"} == set(sim.FLAGS)
    controls = [s for s in CATALOG.values() if s.truth is None]
    assert len(controls) == 2 and sorted(bool(s.decoys) for s in controls) == [False, True]
    assert 2 <= sum(bool(s.decoys) for s in CATALOG.values() if s.truth) <= 3


def test_catalog_decoys_are_off_the_failing_path():
    for s in CATALOG.values():
        avoid = sim._failing_path(s.truth.services) if s.truth else set()
        assert all(d.target not in avoid for d in s.decoys), s.id


def test_active_record_round_trip(tmp_path):
    assert sim.load_active(tmp_path) is None
    sim.save_active(tmp_path, {"scenario": "x", "at": 1.5})
    assert (tmp_path / "sim" / "active.json").exists() and sim.load_active(tmp_path) == {"scenario": "x", "at": 1.5}
    sim.clear_active(tmp_path)
    sim.clear_active(tmp_path)  # idempotent
    assert sim.load_active(tmp_path) is None

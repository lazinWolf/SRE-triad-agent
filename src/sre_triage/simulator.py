"""Incident simulator for demos and evaluation (operator only; agents never see this module).

A scenario is a set of real changes (container limits, stopped/paused containers, feature flags) plus harmless
decoys. The ground truth is derived from the faults themselves, so curated and generated scenarios are scored alike.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .config import BASELINE_CPUS, DEPENDENCIES, SERVICES
from .telemetry import Backends, Unavailable, read_flags, write_flag

DATASTORES = ("valkey-cart", "astronomy-db")
LEVELS = ("easy", "medium", "hard")
# Intensities that are visible but not instantly fatal (cpu: cores; memory: MiB, below each service's working set).
CPU_TARGETS = ("frontend", "checkout", "cart", "product-catalog", "currency", "recommendation", "ad")
CPU_VALUES = (0.02, 0.05, 0.1)
MEMORY_MB = {"cart": 32, "payment": 48, "frontend": 64, "ad": 96, "recommendation": 32, "email": 32}
STOP_TARGETS = tuple(s for s in SERVICES if s != "frontend") + DATASTORES
# flag -> (accepted services, accepted categories, also restart the service, variants worth injecting); first of each is primary
FLAGS: dict[str, tuple[tuple[str, ...], tuple[str, ...], bool, tuple[str, ...]]] = {
    "paymentFailure": (("payment", "checkout"), ("dependency_errors", "config_change"), False, ("25%", "50%", "100%")),
    "paymentUnreachable": (("payment", "checkout"), ("dependency_unavailable", "dependency_errors", "config_change"), False, ("on",)),
    "cartFailure": (("cart", "checkout"), ("dependency_errors", "config_change"), False, ("50%", "100%")),
    "adHighCpu": (("ad",), ("cpu_saturation", "config_change"), False, ("on",)),
    "emailMemoryLeak": (("email",), ("memory_exhaustion", "config_change"), True, ("100x", "1000x")),
    "recommendationCacheFailure": (("recommendation",), ("cache_failure", "memory_exhaustion", "config_change"), True, ("on",)),
    "productCatalogLockContention": (("product-catalog", "astronomy-db"), ("lock_contention", "config_change"), False, ("on",)),
    "productCatalogFailure": (("product-catalog",), ("dependency_errors", "config_change"), False, ("on",)),
    "loadGeneratorFloodHomepage": (("frontend",), ("traffic_spike", "config_change"), False, ("on",)),
}
ERROR_FLAGS = ("paymentFailure", "paymentUnreachable", "cartFailure", "productCatalogFailure")
# Upstream ships productCatalogFailure with a rule that returns "off" on both branches; it only fires with this one.
FIRING_TARGETING = {"productCatalogFailure": {"if": [{"==": [{"var": "product_id"}, "OLJCESPC7Z"]}, "on", "off"]}}
DECOY_FLAG = "emitRawPii"  # not a fault: only adds PII to logs


@dataclass(frozen=True)
class Fault:
    kind: str       # "cpu_limit" | "memory_limit" | "stop" | "pause" | "flag"
    target: str     # container name, or flag name for kind "flag"
    value: Any = None   # cpus, memory MiB, or flag variant

    def __str__(self) -> str:
        if self.kind == "flag":
            return f"flag {self.target}={self.value}"
        return f"{self.kind} {self.target}" + ("" if self.value is None else f" {self.value}")


@dataclass(frozen=True)
class Truth:
    services: frozenset[str]     # accepted root-cause services
    categories: frozenset[str]   # accepted categories
    fixes: tuple[dict, ...]      # accepted remediations (match on action + service/flag)
    service: str                 # primary service and category, for display
    category: str


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    symptom: str                 # fallback alert text; never names the cause
    faults: tuple[Fault, ...]
    decoys: tuple[Fault, ...] = ()
    truth: Truth | None = None   # None = healthy control
    difficulty: str = "easy"

    @property
    def service(self) -> str | None:
        return self.truth.service if self.truth else None

    @property
    def category(self) -> str | None:
        return self.truth.category if self.truth else None


def truth_for(fault: Fault) -> Truth:
    """What a correct diagnosis of this fault looks like."""
    t, kind = fault.target, fault.kind
    if kind == "cpu_limit":
        return Truth(frozenset({t}), frozenset({"cpu_limit"}), ({"action": "set_cpu_limit", "service": t},), t, "cpu_limit")
    if kind == "memory_limit":
        return Truth(frozenset({t}), frozenset({"memory_limit", "memory_exhaustion"}),
                     ({"action": "set_memory_limit", "service": t},), t, "memory_limit")
    if kind in ("stop", "pause"):
        dependents = [s for s, deps in DEPENDENCIES.items() if t in deps] if t in DATASTORES else []
        fixes = ({"action": "start_service" if kind == "stop" else "unpause_service", "service": t},
                 {"action": "restart_service", "service": t})
        return Truth(frozenset({t, *dependents}), frozenset({"service_down", "dependency_unavailable"}), fixes, t, "service_down")
    if kind == "flag" and t in FLAGS:
        services, categories, restart, _ = FLAGS[t]
        fixes = ({"action": "set_feature_flag", "flag": t},) + (({"action": "restart_service", "service": services[0]},) if restart else ())
        return Truth(frozenset(services), frozenset(categories), fixes, services[0], categories[0])
    raise ValueError(f"not a fault: {fault}")


def _symptom(fault: Fault) -> str:
    t = fault.target
    failing = fault.kind in ("stop", "pause", "memory_limit") or t in ERROR_FLAGS
    if t in ("frontend", "loadGeneratorFloodHomepage"):
        return "Storefront pages are slow to load"
    if t in ("product-catalog", "recommendation", "ad", "astronomy-db") or t in (
            "productCatalogLockContention", "productCatalogFailure", "recommendationCacheFailure", "adHighCpu"):
        return "Some storefront requests fail" if failing else "Product pages are slow to load"
    return "Checkout requests are failing" if failing else "Checkout latency is elevated"


def _scenario(id: str, title: str, fault: Fault, decoys: tuple[Fault, ...] = (), difficulty: str = "easy",
              symptom: str | None = None) -> Scenario:
    return Scenario(id, title, symptom or _symptom(fault), (fault,), decoys, truth_for(fault), difficulty)


_PII = Fault("flag", DECOY_FLAG, "on")
CATALOG: dict[str, Scenario] = {s.id: s for s in [
    _scenario("cart_cpu_limit", "Cart CPU limit cut to 0.02", Fault("cpu_limit", "cart", 0.02)),
    _scenario("cart_cpu_limit_decoys", "Cart CPU limit cut, plus two unrelated changes", Fault("cpu_limit", "cart", 0.02),
              (_PII, Fault("cpu_limit", "recommendation", 0.75)), "medium"),
    _scenario("payment_failure", "Payment charges failing (flag paymentFailure 50%)", Fault("flag", "paymentFailure", "50%")),
    _scenario("payment_unreachable", "Checkout cannot reach payment (flag paymentUnreachable)",
              Fault("flag", "paymentUnreachable", "on"), difficulty="medium"),
    _scenario("cart_failure", "Cart operations failing (flag cartFailure 100%)", Fault("flag", "cartFailure", "100%"),
              difficulty="medium"),
    _scenario("catalog_failure", "Product catalog errors (flag productCatalogFailure)", Fault("flag", "productCatalogFailure", "on")),
    _scenario("catalog_lock_contention", "Product catalog slow on a locked database (flag productCatalogLockContention)",
              Fault("flag", "productCatalogLockContention", "on"), (_PII,), "hard"),
    _scenario("ad_high_cpu", "Ad service CPU saturated (flag adHighCpu)", Fault("flag", "adHighCpu", "on"), difficulty="medium"),
    _scenario("email_memory_leak", "Email service leaking memory (flag emailMemoryLeak 1000x)", Fault("flag", "emailMemoryLeak", "1000x"),
              (Fault("cpu_limit", "shipping", 0.75),), "hard", symptom="Checkout requests are intermittently failing"),
    _scenario("recommendation_cache", "Recommendation cache growing (flag recommendationCacheFailure)",
              Fault("flag", "recommendationCacheFailure", "on"), difficulty="hard"),
    _scenario("frontend_traffic_spike", "Homepage traffic flood (flag loadGeneratorFloodHomepage)",
              Fault("flag", "loadGeneratorFloodHomepage", "on"), difficulty="medium"),
    _scenario("payment_memory_limit", "Payment memory limit cut to 48 MiB", Fault("memory_limit", "payment", 48), difficulty="medium"),
    _scenario("currency_paused", "Currency service paused (requests hang)", Fault("pause", "currency"), difficulty="medium",
              symptom="Storefront requests are timing out"),
    _scenario("valkey_cart_stopped", "Cart datastore valkey-cart stopped", Fault("stop", "valkey-cart"), difficulty="hard"),
    Scenario("healthy_control", "No fault (control case)", "A user said checkout felt slow", (), difficulty="easy"),
    Scenario("config_change_control", "Only harmless changes (control case)", "A user said the site felt slow", (),
             (_PII, Fault("cpu_limit", "recommendation", 0.75)), difficulty="medium"),
]}


def _reach(start: set[str] | list[str], step: Callable[[str], list[str]]) -> set[str]:
    seen, todo = set(), list(start)
    while todo:
        for nxt in step(todo.pop()):
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return seen


def _callees(s: str) -> list[str]:
    return DEPENDENCIES.get(s, [])


def _callers(s: str) -> list[str]:
    return [c for c, deps in DEPENDENCIES.items() if s in deps]


def _failing_path(services) -> set[str]:
    """The services on the failing path: the faulted ones, everything they call and everything that calls them."""
    return set(services) | _reach(services, _callees) | _reach(services, _callers)


def _decoys(rng: random.Random, level: str, avoid: set[str]) -> tuple[Fault, ...]:
    """Benign changes: a raw-PII flag and a mild CPU limit on a service outside `avoid` (medium: one of them, hard: both, tighter)."""
    candidates = [s for s in SERVICES if s not in avoid]
    cpu = [Fault("cpu_limit", rng.choice(candidates), 0.5 if level == "hard" else 0.75)] if candidates else []
    pool = [_PII] + cpu
    if level == "easy":
        return ()
    return tuple(pool) if level == "hard" else (rng.choice(pool),)


def _control(seed: int, i: int, level: str, rng: random.Random) -> Scenario:
    decoys = _decoys(rng, level, set())
    if not decoys:
        return Scenario(f"gen-{seed}-{i}-control-none", "No fault (control case)", "A user said checkout felt slow", (),
                        difficulty=level)
    return Scenario(f"gen-{seed}-{i}-control-decoys", "Only harmless changes (control case)", "A user said the site felt slow",
                    (), decoys, None, level)


def _fault(rng: random.Random) -> Fault:
    kind = rng.choice(("cpu_limit", "memory_limit", "stop", "pause", "flag", "flag"))
    if kind == "cpu_limit":
        return Fault(kind, rng.choice(CPU_TARGETS), rng.choice(CPU_VALUES))
    if kind == "memory_limit":
        target = rng.choice(sorted(MEMORY_MB))
        return Fault(kind, target, MEMORY_MB[target])
    if kind in ("stop", "pause"):
        return Fault(kind, rng.choice(STOP_TARGETS))
    flag = rng.choice(sorted(FLAGS))
    return Fault(kind, flag, rng.choice(FLAGS[flag][3]))


def generate(seed: int, n: int, difficulty: str = "mixed") -> list[Scenario]:
    """n reproducible scenarios: sampled fault kind x valid target x intensity, decoys by difficulty, ~1 control in 6."""
    if difficulty != "mixed" and difficulty not in LEVELS:
        raise ValueError(f"difficulty must be mixed or one of {LEVELS}")
    rng, out = random.Random(seed), []
    for i in range(n):
        level = rng.choice(LEVELS) if difficulty == "mixed" else difficulty
        if i % 6 == 5:
            out.append(_control(seed, i, level, rng))
            continue
        fault = _fault(rng)
        truth = truth_for(fault)
        out.append(Scenario(f"gen-{seed}-{i}-{fault.kind}-{fault.target}", f"{fault}", _symptom(fault), (fault,),
                            _decoys(rng, level, _failing_path(truth.services)), truth, level))
    return out


def _write_doc(path: Path, data: dict) -> None:
    """Write the flag document in place (flagd watches the inode)."""
    with open(path, "r+") as handle:
        handle.write(json.dumps(data, indent=2) + "\n")
        handle.truncate()


def _apply(fault: Fault, backends: Backends) -> None:
    docker = backends.docker
    if fault.kind == "cpu_limit":
        docker.set_cpus(fault.target, float(fault.value))
    elif fault.kind == "memory_limit":
        docker.set_memory(fault.target, fault.value)
    elif fault.kind in ("stop", "pause"):
        getattr(docker, fault.kind)(fault.target)
    elif fault.target in FIRING_TARGETING and fault.value != "off":
        data = json.loads(backends.flagd_file.read_text())
        entry = data["flags"][fault.target]
        if fault.value not in entry["variants"]:
            raise ValueError(f"unknown flag or variant: {fault.target}={fault.value}")
        entry.update(defaultVariant=fault.value, targeting=FIRING_TARGETING[fault.target])
        _write_doc(backends.flagd_file, data)
    else:
        write_flag(backends.flagd_file, fault.target, str(fault.value))


def inject(scenario: Scenario, backends: Backends) -> dict:
    """Apply decoys, then faults. Returns a JSON-serialisable record of what was done."""
    for fault in scenario.decoys + scenario.faults:
        _apply(fault, backends)
    return {"scenario": scenario.id, "at": time.time(), "faults": [asdict(f) for f in scenario.faults],
            "decoys": [asdict(f) for f in scenario.decoys]}


def _reset_flags(backends: Backends) -> list[str]:
    pristine = backends.standard.get("flags", {}).get("flags", {})
    if not pristine:  # no standard on record: every flag off
        done = []
        for flag, info in read_flags(backends.flagd_file).items():
            if info["variant"] != "off" and "off" in info["variants"]:
                write_flag(backends.flagd_file, flag, "off")
                done.append(f"{flag} -> off")
        return done
    data = json.loads(backends.flagd_file.read_text())
    done: list[str] = []
    for name, want in pristine.items():
        have = data["flags"].get(name)
        if have is None:
            continue
        if have.get("defaultVariant") != want.get("defaultVariant"):
            done.append(f"{name} {have.get('defaultVariant')} -> {want.get('defaultVariant')}")
            have["defaultVariant"] = want.get("defaultVariant")
        if have.get("targeting") != want.get("targeting"):
            done.append(f"{name} targeting restored")
            if "targeting" in want:
                have["targeting"] = want["targeting"]
            else:
                have.pop("targeting", None)
    if done:  # write only when something differs
        _write_doc(backends.flagd_file, data)
    return done


def _reset_container(docker: Any, name: str, want: dict) -> list[str]:
    try:
        info = docker.inspect(name)
        done = []
        cpus, memory = want.get("cpus"), want.get("memory_mb")
        if cpus is not None and abs((info.get("cpu_limit") or 0) - cpus) > 0.005:
            docker.set_cpus(name, cpus)
            done.append(f"{name} cpus {info.get('cpu_limit')} -> {cpus}")
        if memory is not None and info.get("memory_limit_mb") != memory:
            docker.set_memory(name, memory)
            done.append(f"{name} memory {info.get('memory_limit_mb')} -> {memory} MiB")
        if info.get("status") == "paused":
            docker.unpause(name)
            done.append(f"{name} unpaused")
        elif info.get("status") not in ("running", "restarting"):
            docker.start(name)
            done.append(f"{name} started")
        return done
    except Unavailable:
        return []


def reset(backends: Backends) -> list[str]:
    """Back to the standard configuration (flags with their targeting, container state and limits); lists what changed."""
    done = _reset_flags(backends)
    services = backends.standard.get("services") or {}
    if services:
        for name, want in services.items():
            done += _reset_container(backends.docker, name, want)
    else:
        for name in SERVICES:
            done += _reset_container(backends.docker, name, {"cpus": BASELINE_CPUS})
    return done


def _matches(fix: dict | None, accepted: tuple[dict, ...]) -> bool:
    return bool(fix) and any(all(fix.get(k) == v for k, v in a.items()) for a in accepted)


def score(scenario: Scenario, report: dict | None) -> dict[str, bool]:
    report = report or {}
    diagnosed = report.get("status") == "diagnosed"
    truth = scenario.truth
    if truth is None:
        return {"correct": not diagnosed, "service": not diagnosed, "category": not diagnosed,
                "remediation": not (report.get("remediation") or report.get("alternative")), "false_alarm": diagnosed}
    service_ok = diagnosed and report.get("root_cause_service") in truth.services
    category_ok = diagnosed and report.get("category") in truth.categories
    fix_ok = _matches(report.get("remediation"), truth.fixes) or _matches(report.get("alternative"), truth.fixes)
    return {"correct": service_ok and category_ok, "service": service_ok, "category": category_ok,
            "remediation": fix_ok, "false_alarm": diagnosed and not service_ok}


def active_alerts(backends: Backends) -> list[dict]:
    try:
        return [a for a in backends.prometheus.alerts() if a.get("state") in ("pending", "firing")]
    except Unavailable:
        return []


def alert_key(alert: dict) -> tuple:
    return alert.get("name"), alert.get("service")


def wait_clear(backends: Backends, settle: float = 300, poll: float = 10,
               sleep: Callable[[float], None] = time.sleep) -> set[tuple]:
    """Wait (up to `settle` seconds) until no alert is pending or firing, so one fault cannot vouch for the next.
    Returns the alerts that are still there: they were not caused by what is injected afterwards."""
    waited = 0.0
    while active_alerts(backends) and waited < settle:
        sleep(poll)
        waited += poll
    return {alert_key(a) for a in active_alerts(backends)}


def new_firing(backends: Backends, before: set[tuple]) -> list[dict]:
    """Alerts that fire now and were not there before the injection. Pending is not a detection: it may clear."""
    return [a for a in active_alerts(backends) if a.get("state") == "firing" and alert_key(a) not in before]


def probe(scenario: Scenario, backends: Backends, wait: float = 150, poll: float = 10,
          sleep: Callable[[float], None] = time.sleep, settle: float = 300) -> dict:
    """Inject, wait for a NEW firing alert on the faulted path (any new one, for a control), then always reset."""
    truth = scenario.truth
    relevant = None if truth is None else truth.services | _reach(truth.services, _callers)
    before = wait_clear(backends, settle, poll, sleep)
    waited, new = 0.0, []
    try:
        inject(scenario, backends)
        while True:
            new = new_firing(backends, before)
            if any(relevant is None or a.get("service") in relevant for a in new):
                return {"scenario": scenario.id, "detected": True, "seconds": waited, "alerts": new,
                        "clean_start": not before}
            if waited >= wait:
                return {"scenario": scenario.id, "detected": False, "seconds": None, "alerts": new,
                        "clean_start": not before}
            sleep(poll)
            waited += poll
    finally:
        reset(backends)


def _active_file(data_dir: Path) -> Path:
    return Path(data_dir) / "sim" / "active.json"


def save_active(data_dir: Path, record: dict) -> None:
    path = _active_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))


def load_active(data_dir: Path) -> dict | None:
    try:
        return json.loads(_active_file(data_dir).read_text())
    except (OSError, ValueError):
        return None


def clear_active(data_dir: Path) -> None:
    _active_file(data_dir).unlink(missing_ok=True)

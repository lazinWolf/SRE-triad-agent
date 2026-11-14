#!/usr/bin/env python3
"""Write standard.json: the compose-defined resource limits of the demo's app containers and the
pristine flagd document. Usage: compose config --format json | standard.py <demo-repo> <out.json>
Limits come from the compose files (not `docker inspect`, which may carry an injected limit)."""
import json
import subprocess
import sys

CONTAINERS = ["ad", "cart", "checkout", "currency", "email", "frontend", "payment",
              "product-catalog", "quote", "recommendation", "shipping", "valkey-cart", "astronomy-db"]
UNITS = {"": 1, "b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}


def to_bytes(value):
    if value is None:
        return None
    text = str(value).strip().lower().rstrip("b") or "0"
    unit = text[-1] if text[-1] in "kmg" else ""
    return int(float(text[:-1] if unit else text) * UNITS[unit])


def main(demo_repo, out):
    config = json.load(sys.stdin)
    by_container = {s.get("container_name", name): s for name, s in config["services"].items()}
    services = {}
    for name in CONTAINERS:
        limits = (by_container.get(name, {}).get("deploy") or {}).get("resources", {}).get("limits") or {}
        mem = to_bytes(limits.get("memory"))
        cpus = limits.get("cpus")
        services[name] = {"cpus": float(cpus) if cpus is not None else None,
                          "memory_mb": round(mem / 1048576) if mem else None}
    flags = subprocess.run(["git", "-C", demo_repo, "show", "HEAD:src/flagd/demo.flagd.json"],
                           check=True, capture_output=True, text=True).stdout
    with open(out, "w") as f:
        json.dump({"services": services, "flags": json.loads(flags)}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

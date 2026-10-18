#!/usr/bin/env python3
"""Start checkout, inventory, and the operator interface; stop all with Ctrl+C."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def main():
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    services = [("inventory", "demo_app.server:inventory_app", 8102),
                ("checkout", "demo_app.server:checkout_app", 8101),
                ("operator", "incident_agent.api:app", 8080)]
    processes = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        for _, process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for name, module, port in services:
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"], cwd=root, env=env)
            processes.append((name, process))
        print("Operator interface: http://127.0.0.1:8080\nCheckout: http://127.0.0.1:8101\nInventory: http://127.0.0.1:8102", flush=True)
        while not stopping:
            for name, process in processes:
                if process.poll() is not None:
                    if process.returncode:
                        print(f"{name} exited with code {process.returncode}", file=sys.stderr)
                    stop()
                    if process.returncode:
                        raise SystemExit(process.returncode)
                    break
            time.sleep(0.25)
    finally:
        stop()
        for _, process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()

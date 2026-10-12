"""Bounded readiness check used by the notebook after starting its own policy process."""

import argparse
import json
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--timeout", type=float, default=300)
args = parser.parse_args()
deadline = time.monotonic() + args.timeout
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=3) as response:
            report = json.load(response)
        if report["ready"]:
            print(report)
            break
    except (OSError, urllib.error.URLError):
        time.sleep(2)
else:
    raise TimeoutError("Policy server did not become ready. Inspect outputs/policy_server.log.")

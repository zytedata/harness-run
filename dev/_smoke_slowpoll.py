"""Local wrapper: run live_smoke with a slower Cloud Logging tail poll.

The sink tails at 1 Hz (= 60 reads/min); the default per-user logging read quota is
60/min, so the smoke's two parallel mode tails 429 on end-user ADC. 2.5 s per poll keeps
two concurrent tails at ~48 reads/min. Not part of the PR — validation-environment shim.
"""
import runpy
import sys

import remote_agent_toolkit.ports.eventsink as es

es._POLL_INTERVAL_S = 2.5
sys.argv = ["live_smoke"]
runpy.run_path("dev/live_smoke.py", run_name="__main__")

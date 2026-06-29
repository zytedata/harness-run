# Platform notes — Gemini Agent Runtime contracts

> Status: **placeholder (P0).** Expands DESIGN.md §6 at P2/P3 (DESIGN.md §11).

This appendix will expand the hard-won platform contracts the library encodes as defaults:

- **Execution paths** — async `run_query_job` (long, ~2.5 min per-job provisioning), sync `stream_query`
  (warm ~3–9 s, ~600 s ceiling), and the **warm pool** (fast *and* long).
- **Warm pool** — `__POOL_WAIT__` sentinel, competing-consumers atomic claim, heartbeats, pre-warm, refill.
- **Checkpoint / resume** — `SessionStore` keyed by `session_id`, workspace tar, finalize inline at the
  terminal `result` event.
- **Observability** — Cloud Logging (`log_struct`) as the only near-real-time channel.
- **Deploy contracts** — uv on PATH, glibc/Python 3.12, `a2a-sdk` pin, `/tmp`-only cwd, `IS_SANDBOX=1`,
  `min_instances`.

See [`../DESIGN.md`](../DESIGN.md) §6 for the measured facts.

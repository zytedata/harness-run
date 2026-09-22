# Sandbox runtime

The `sandbox` runtime runs each turn in a Google Cloud Agent Sandbox: an engine is deployed once per
agent role, application code looks it up by name, and every turn gets an isolated sandbox that is
deleted when the turn ends. This page covers deploying, managing and versioning engines, and the
latency and cost profile of the platform.

## Prod: deploy once, look up and run

Requires GCP setup — see [GCP setup & required permissions](gcp-setup.md) below.

```python
# Ops / CI deploys once (rare): builds + pushes the agent's image (Docker), creates a template,
# fills a ready pool of two sandboxes.
engine = sandbox.deploy(spec, project="my-project", location="us-central1", warm_pool=True, pool_size=2)
engine.wait_until_warm()                          # block until a ready sandbox is in the pool (pool only)

# App code looks the engine up by name and runs — it never deploys:
engine = sandbox.get_engine("code-agent", project="my-project", location="us-central1")
session = engine.start_session()
result = await session.run("Add type hints to utils.py and run the tests")   # default: wait for the result
```

A turn claims a ready sandbox off the pool (or creates one, ~20 s), hands it the turn over HTTPS and
streams the events straight back from it; the sandbox is deleted at the terminal event. `get_engine`
follows what the deploy recorded (`warm_pool`, the model identity, the baked spec) — pass
`warm_pool=False` to bypass a pool deliberately.

**What `deploy` waits for.** Pass `log=print` to follow it: image build and push, then `creating template …`,
a `template <name>: PROVISIONING for 30s` line every 30 s while the platform provisions, `template <id> is
ACTIVE`, pool fill. The template's provisioning time varies a lot — 12–20 s on most days, ten minutes on
others (observed 2026-09-15 for 4 CPU / 8 GiB templates while 1 CPU / 1 GiB ones took 12 s); `deploy` returns
the moment the template lists as ACTIVE rather than waiting for the platform's long-running operation, which
has been seen to complete nine minutes after that. A template that ends FAILED makes `deploy` raise with the
platform's reason (from the operation); FAILED versions stay visible in `engine.revisions()` with their
`state`, `get_engine` skips them (with a warning) and resolves to the newest ACTIVE version, and the next
successful deploy retires them. The platform itself gives up on a template after about 30 minutes of
PROVISIONING and fails it with a bare `INTERNAL` (seen five times in two days for 4 CPU templates, never for
1 CPU ones, with images and configs identical to templates that came up in 90 s); every such stall followed an
earlier stuck or FAILED template, and one completed 34 s after that FAILED template was deleted, so a failed
template appears to hold its warm-pool capacity until deleted. `deploy` therefore deletes every FAILED template
under the host instance before creating a new one (logging each), deletes its own template when it ends FAILED,
and after 32 minutes of PROVISIONING gives up with the operation name, deleting the stuck template so it does
not hold up the re-run (idempotent). If a create runs past ten minutes, `engine.revisions()` and
`delete_version()` on anything FAILED is the manual version of the same fix.

**The deploy record.** `deploy` writes `deploys/<name>/<template id>.json` under the output bucket
right after the template exists, before the pool is filled: the baked spec, the image, the model
service account, the pool settings. `get_engine` reads it; without it the handle can still address
the template but a Vertex-routed turn fails for lack of a model service account, so `get_engine`
**warns** at lookup when the record is missing. The usual cause is a `deploy` interrupted while it
waited for the platform to create the template (the template came up, the record was never written).
**The fix is to re-run the same `deploy`**: it is idempotent — the image is found in the registry and
not rebuilt, the newest template already serves it so no new version is created, and the record is
written.



```python
sandbox.deploy(spec, project=..., location=...)   # build + push the image, create a template; ops/CI only
sandbox.deploy(spec, ..., warm_pool=True, pool_size=2, pool_max_wait_s=3600)   # + a ready pool, idle life 1 h
sandbox.deploy(spec, ..., resource_limits={"cpu": "8", "memory": "16Gi"})      # sandbox CPU/RAM (default 4 / 4Gi; max 8 vCPU)
sandbox.deploy(spec, ..., image="…-docker.pkg.dev/proj/harness-run/my-agent:tag")     # use an image you pushed; no build
sandbox.get_engine("code-agent", project=..., location=...)   # look up by name (app code; addressing only)
sandbox.list_engines(project=..., location=...)   # discover what's deployed: {name, resource, versions}
engine.name, engine.version, engine.resource     # identity / template id / the template's resource name
engine.wait_until_warm(timeout=300)              # pools: wait for a ready sandbox before dispatching
engine.fill_pool(2)                              # top a pool up (after its sandboxes idled out)
engine.delete()                                  # tear down: every ready sandbox and every version's template
```

Deploying needs the Docker CLI logged into the registry (`docker login -u oauth2accesstoken
--password-stdin <region>-docker.pkg.dev` with an access token, or `gcloud auth configure-docker`) and
the Artifact Registry repo from [GCP setup](gcp-setup.md). Session `fork()` is not
built yet.


## Versions: a template per deploy

Engine identity is `spec.name`. Every deploy whose image or resources differ from the newest version
creates a new immutable **sandbox template** displayed under that name — a template is a version — so app
code's `get_engine("code-agent")` resolves the newest one without re-pointing at anything. A deploy
of an unchanged spec from an unchanged library is a no-op (same image digest, same template). Once the new
version's pool is filled, the previous versions' idle sandboxes and templates are retired.

```python
engine.versions()                        # ['85216549199151104', …] — template ids, newest first
engine.revisions()                       # + image / create_time / state / which one is `current`
engine.version                           # the template this handle's turns run on
engine.delete_version("…")               # delete one version (its template + ready sandboxes)

sandbox.get_engine("code-agent", ..., version="…")   # pin: turns run on that template
```

**`version=` routes.** Any template can be dispatched to, so a pinned handle keeps running its version
after a newer deploy — useful for a canary or a rollback (redeploy the old spec: same digest, same image,
a new template). "Serving" simply means "newest"; there is no traffic configuration.

**A display name is one lineage.** Because `get_engine(name)` resolves the newest template and a deploy
retires the previous versions once its pool is filled, an engine name is a single-revision resource by
design: two deployers on different library revisions (or different specs) addressing the same name
replace each other's version. Consumers that run several library revisions side by side — a `dev`
branch next to `main`, a pinned release next to a candidate — should put the revision in the name
(`code-agent-b040d27`) and tear down the names they stop using with `engine.delete()`. Deleting an
engine touches only sandboxes created from *its* templates, so `x` can be torn down next to `x-<rev>`.


## Latency & cost (the `sandbox` path)

Platform realities the library encodes (measured on Agent Sandbox, 2026-09-10/11; the `local` path has none
of them):

**Deploy** is a rare ops/CI action: an image build on your machine (~30 s with a warm Docker cache, a few
minutes cold), a push, a template (~20–80 s), and the pool fill (~15–35 s per sandbox, in parallel). App
code never deploys — it looks an engine up by name and runs.

**Running a turn** has two latency profiles:

| Path | Start latency | Use for |
|---|---|---|
| **Ready pool** (`warm_pool=True`) | **~1 s** to the first event, ~4–7 s to the result of a one-tool Haiku turn | interactive *and* long |
| No ready sandbox | **~20 s** (2 s to create, 12–30 s until the platform's proxy routes to it) | batch / one-shot |

A ready sandbox's worker is already running and reachable; a turn is one HTTPS round trip to hand it the
prompt, tokens and configs, and the events stream straight back from it (one long-poll per event batch,
~0.2 s proxy round trip). Sandboxes are created from Google's pre-warmed pool per template, so the ~20 s
of the no-pool path is route propagation, not boot. DESIGN.md §13 has the architecture and the measured
numbers.

**How `warm_pool=True` works.** `sandbox.deploy(spec, warm_pool=True, pool_size=N)` creates N sandboxes from
the new template, waits until each answers, and records them in a **roster** (client-owned GCS objects under
`pool/<template>/` in the output bucket). A turn takes the oldest ready sandbox off the roster (claimed
atomically, so several client processes share one pool), hands it the turn, and refills the pool in the
background; the used sandbox is deleted at the terminal event. `engine.wait_until_warm()` blocks until the
roster has a live entry. A turn that finds the roster empty creates a sandbox for itself (the ~20 s path),
never a stranded turn.

> _Keeping the pool full:_ a ready sandbox idles for `pool_max_wait_s` (a `deploy()` parameter; default a
> day), then the platform deletes it — **without replacement** (the only refill is the one after each
> dispatch; `engine.fill_pool(n)` re-warms ahead of time). Its TTL is set at creation and cannot be extended,
> so the library creates pool sandboxes with `pool_max_wait_s + max_turn_s` of life: one claimed at the end
> of its idle life still has the whole `max_turn_s` (default 8 h) for its turn. Size `pool_max_wait_s` to
> your dispatch gaps.

**Event streaming.** The stream you consume with `async for ev in run` comes **directly from the sandbox**:
the client long-polls the worker's `/events` (held until something new exists, so an event is seen within
one proxy round trip of its emission) and the worker keeps writing the **GCS event mirror** as the durable
record — the fallback stream when a sandbox stops answering, and what `session.history()` reads. Nothing
reads Cloud Logging; there is no shared read quota to run into.

**Cost.** A template costs nothing while nothing runs on it; a sandbox bills while it exists
($0.085/vCPU-hour + $0.009/GiB-hour at the time of writing — $0.38/h for the default 4 vCPU / 4 GiB). A
ready pool is therefore idle compute you pay for continuously: `warm_pool=True` trades money for latency —
size the pool to your concurrency and `pool_max_wait_s` to your quiet gaps, and leave it off for batch
agents where a ~20 s start is fine. Google keeps a pool of pre-started containers per template (two
sandboxes created from a 34-hour-old template both had a PID 1 that was 34 hours old), and that pool is
Google's cost, not yours: ten idle 4 CPU / 4 GiB templates with no sandbox ever created from them were
kept for a full billing day (2026-09-17) and the project's Agent Platform Compute / Memory lines went
*down* that day (55 vCPU-h / 145 GiB-h, below the ready-pool sandboxes alone), where billed pools would
have added ~1000–1900 GiB-h. So a cold engine costs only its image storage; still delete engine versions
nobody runs, because each ACTIVE template competes for provisioning capacity. Turn sandboxes are deleted
at the terminal event, so a turn costs its own
duration. Tear a pool down with `engine.delete()`. Model token cost is the same either way and is reported
per run as `result.cost_usd`.

**Limits** (measured 2026-09-11, undocumented by the platform): at most 8 vCPU per sandbox (16 GiB works);
one proxied HTTP call may take up to ~5 min (the library's long-poll holds 20 s); a request body over ~1 MB
fails (a prompt that large would); a response over ~2 MB fails (the worker pages events under 1 MB); sandbox
TTLs up to 30 days are accepted; `/tmp` has ~60 GB.

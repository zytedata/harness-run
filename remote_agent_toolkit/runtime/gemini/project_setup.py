"""One-command GCP project setup for the Gemini Agent Runtime backend.

``ratk-gcp-setup --project <id>`` makes a GCP project ready to run this toolkit's
``gemini`` backend: it audits the project against everything the README's "GCP setup &
required permissions" section requires, prints a report, asks for confirmation, applies
what is missing, and re-audits until the project is ready. It is **additive only** — it
never disables, removes, or narrows anything — so it is safe to run (and re-run: every
step is idempotent) against existing, non-empty projects.

Run it as a human (interactive confirmation) or as an agent/CI (``--check`` audits
without changing anything and exits 0 only when the project is ready; ``--yes`` applies
without a prompt). Auth is plain ADC — no gcloud CLI involved; every API call is made
with the *target* project as the quota project, whatever the ambient ADC default is.

What it manages (the README section explains the *why* of each piece):

* the required APIs (enabled additively);
* the staging + output buckets (created in ``--location`` with uniform bucket-level
  access and public-access prevention; the handoff lifecycle rules on the output bucket);
* the operator service account, its project roles, and **bucket-scoped** storage grants
  (not project-wide ``storage.admin`` — least privilege for shared projects);
* ``roles/iam.serviceAccountTokenCreator`` on the operator SA for the principals that
  will impersonate it (defaults to the ADC principal running this tool);
* the **runtime service account** the engines run as (``gemini.deploy(...,
  service_account=<its email>)``): the account itself, a custom role holding only
  ``aiplatform.endpoints.predict`` (never ``roles/aiplatform.user``), its project roles,
  ``objectViewer`` on the staging bucket, the two *conditional* output-bucket bindings
  (create + read-by-name under ``jobs/`` and ``events/ratk-`` only — nothing under
  ``pool/``, where the warm pool's roster lives), and ``serviceAccountUser`` on it for
  the operator SA. The default Agent Runtime service agent gets **nothing**: its
  Google-managed project role already reads every bucket in the project, which is why
  the engine must run as an account you own (README "The runtime identity is reachable
  by the agent"). Grants that agent still holds from the earlier identity model are
  reported as a NOTE to remove by hand once no engine runs as it;
* a live check that the Claude model is enabled in Vertex Model Garden (accepting the
  Anthropic terms is a one-time console action this tool cannot perform for you).

``--verify`` proves the end state with a real throwaway warm-pool deploy **as the runtime
service account** + one Haiku turn + teardown (a few cents, ~10 min) — it exercises every
grant above end-to-end.

Stdlib-only at import time (repo convention); ``google-auth`` is imported lazily.
"""

from __future__ import annotations

import argparse
import getpass
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .handoff import handoff_lifecycle_rules

DEFAULT_RUNTIME_SA_ID = "ratk-runtime"  # TODO(sandbox): rewritten with the sandbox IAM model
DEFAULT_LOCATION = "us-central1"
DEFAULT_OPERATOR_SA_ID = "agent-runtime"
# The models checked in Vertex Model Garden. Each check costs a handful of input tokens
# + 1 output token. `translate.py` defaults CLOUD_ML_REGION to `global`, so that's the
# location whose enablement actually matters for deployed runs. The FIRST required model
# is also what `--verify`'s throwaway turn runs on — keep the cheapest one first.
DEFAULT_CHECK_MODELS: tuple[str, ...] = ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5")
# Checked and reported, but absence doesn't fail readiness — not every project needs them.
DEFAULT_OPTIONAL_MODELS: tuple[str, ...] = ("claude-fable-5",)
MODEL_CHECK_LOCATION = "global"

# Every API a gemini-backend project needs (README "GCP setup & required permissions").
# Additive: already-enabled extras are never touched.
REQUIRED_SERVICES: tuple[str, ...] = (
    "serviceusage.googleapis.com",  # bootstrap — everything below is managed through it
    "cloudresourcemanager.googleapis.com",  # project IAM policy reads/writes
    "iam.googleapis.com",  # the operator service account
    "iamcredentials.googleapis.com",  # impersonating the operator SA
    "aiplatform.googleapis.com",  # Agent Engine control plane + Vertex Claude
    "storage.googleapis.com",  # staging/output buckets
    "logging.googleapis.com",  # structured step logs + client tailing
    "cloudbuild.googleapis.com",  # deploy builds the engine image...
    "artifactregistry.googleapis.com",  # ...and stores it
    "pubsub.googleapis.com",  # warm-pool dispatch
    "telemetry.googleapis.com",  # OTel trace export (README "Tracing")
    "cloudtrace.googleapis.com",
    "monitoring.googleapis.com",
)

# Project roles for the operator SA (control plane: deploy / get_engine / run / tail).
OPERATOR_PROJECT_ROLES: tuple[str, ...] = (
    "roles/aiplatform.user",
    "roles/logging.viewer",
    "roles/cloudbuild.builds.editor",
    "roles/pubsub.editor",  # create/retire the per-deploy dispatch pair + publish turns
)
# Storage is granted on the two toolkit buckets, NOT project-wide: bucket-scoped
# roles/storage.admin covers staging the deploy bundle, reading job output, and the
# lifecycle-rule update deploy() performs — without touching anyone else's buckets.
OPERATOR_BUCKET_ROLE = "roles/storage.admin"

# The engine's runtime identity: a service account you own, which every deploy names as
# ``service_account=``. ALL runtime resource access (Vertex, GCS, Pub/Sub, Logging)
# authorizes against it — and the agent's shell can obtain its token, so it holds only
# what a turn needs (README "The runtime identity is reachable by the agent").
RUNTIME_PREDICT_ROLE_ID = "ratkRuntimePredict"  # custom role: model calls only
RUNTIME_PREDICT_PERMISSIONS: tuple[str, ...] = ("aiplatform.endpoints.predict",)
RUNTIME_SA_PROJECT_ROLES: tuple[str, ...] = (
    "roles/logging.logWriter",  # the agent emits structured step logs
    # the platform's job runner downloads the job input with a quota project on the
    # request; without this the job dies before any worker event
    "roles/serviceusage.serviceUsageConsumer",
    "roles/telemetry.metricsWriter",  # OTel export (README "Tracing")
    "roles/telemetry.tracesWriter",
    "roles/pubsub.subscriber",  # warm-pool workers pull their own subscription (by name)
)
RUNTIME_SA_STAGING_ROLE = "roles/storage.objectViewer"  # the platform pulls the deploy bundle
# On the output bucket: create the platform's job output + read the job input by exact
# name (jobs/), and write the warm pool's readiness marker (events/ratk-). Conditional, so
# the identity has no read or list right on anything else — everything a turn touches is
# done with its run-scoped token, and pool/ (the roster that decides where turns go) must
# never be reachable by it.
RUNTIME_SA_OUTPUT_ROLES: tuple[str, ...] = (
    "roles/storage.objectCreator",
    "roles/storage.legacyObjectReader",  # objects.get by name, no objects.list
)
RUNTIME_SA_OUTPUT_PREFIXES: tuple[str, ...] = ("jobs/", "events/ratk-")
RUNTIME_SA_OUTPUT_CONDITION_TITLE = "ratk: platform job files and pool readiness markers only"
# The operator deploys AS the runtime identity.
OPERATOR_ON_RUNTIME_SA_ROLE = "roles/iam.serviceAccountUser"

# What the DEFAULT Agent Runtime service agent held under the earlier identity model.
# This tool never removes anything; it reports these as a NOTE (README "Migration").
LEGACY_AGENT_PROJECT_ROLE = "roles/aiplatform.user"
LEGACY_AGENT_BUCKET_ROLE = "roles/storage.objectAdmin"

# Report statuses.
OK = "OK"
FIX = "FIX"  # will be applied by this tool (after confirmation)
BLOCKED = "BLOCKED"  # can't even audit yet (waiting on an earlier fix); re-audited after apply
MANUAL = "MANUAL"  # a human console action this tool cannot perform
NOTE = "NOTE"  # informational — reported, but does not affect readiness or the exit code


class GcpError(RuntimeError):
    """A Google API call failed (carries the HTTP status for callers that branch on it)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def runtime_agent_email(project_number: str | int) -> str:
    """The DEFAULT Agent Runtime service agent for a project (auto-created by Google).

    Only engines deployed before the toolkit named a runtime service account still run as
    it; this tool checks it for leftover grants from that identity model, and grants it
    nothing.
    """
    return f"service-{project_number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"


def predict_role_name(project: str) -> str:
    """Full resource name of the project's ``ratkRuntimePredict`` custom role."""
    return f"projects/{project}/roles/{RUNTIME_PREDICT_ROLE_ID}"


def output_bucket_condition(bucket: str) -> dict:
    """The IAM condition limiting the runtime identity to the platform's own object prefixes."""
    expression = " || ".join(
        f'resource.name.startsWith("projects/_/buckets/{bucket}/objects/{prefix}")'
        for prefix in RUNTIME_SA_OUTPUT_PREFIXES
    )
    return {"title": RUNTIME_SA_OUTPUT_CONDITION_TITLE, "expression": expression}


def _normalized(expression: str) -> str:
    return " ".join(expression.split())


def conditional_binding_present(policy: dict, role: str, member: str, condition: dict) -> bool:
    """Does ``member`` hold ``role`` under exactly ``condition`` (or unconditionally, which is
    broader) in ``policy``? Other conditions on the same role do not count."""
    wanted = _normalized(condition["expression"])
    for b in policy.get("bindings", []) or []:
        if b.get("role") != role or member not in (b.get("members", []) or []):
            continue
        cond = b.get("condition")
        if not cond or _normalized(cond.get("expression", "")) == wanted:
            return True
    return False


def add_conditional_binding(policy: dict, role: str, member: str, condition: dict) -> bool:
    """Merge one conditional (role, member) grant into ``policy`` in place; ``True`` if added.

    Joins an existing binding with the same role and condition, else appends a new one.
    Conditional bindings require policy version 3, which is set here.
    """
    if conditional_binding_present(policy, role, member, condition):
        return False
    wanted = _normalized(condition["expression"])
    bindings = policy.setdefault("bindings", [])
    for b in bindings:
        cond = b.get("condition")
        if b.get("role") == role and cond and _normalized(cond.get("expression", "")) == wanted:
            b.setdefault("members", []).append(member)
            break
    else:
        bindings.append({"role": role, "members": [member], "condition": dict(condition)})
    policy["version"] = 3
    return True


def default_buckets(project: str) -> tuple[str, str]:
    """(staging, output) bucket URIs — must mirror ``backend.deploy``'s defaults."""
    return f"gs://{project}-agent-staging", f"gs://{project}-agent-output"


def normalize_bucket_uri(value: str) -> str:
    """Accept ``gs://name[/prefix]`` or a bare bucket name; return a ``gs://`` URI."""
    return value if value.startswith("gs://") else f"gs://{value}"


def bucket_name_of(uri: str) -> str:
    return uri.removeprefix("gs://").split("/", 1)[0]


def principal(member: str) -> str:
    """Qualify a bare email as an IAM member string (explicit prefixes pass through)."""
    if ":" in member:
        return member
    kind = "serviceAccount" if member.endswith("gserviceaccount.com") else "user"
    return f"{kind}:{member}"


def missing_bindings(
    policy: dict, additions: Iterable[tuple[str, str]]
) -> list[tuple[str, str]]:
    """The (role, member) pairs from ``additions`` not present in ``policy``.

    Only unconditional bindings count as present — a conditional grant of the same role
    is not the standing access the toolkit needs.
    """
    have: set[tuple[str, str]] = set()
    for b in policy.get("bindings", []) or []:
        if not b.get("condition"):
            for m in b.get("members", []) or []:
                have.add((b.get("role", ""), m))
    return [(r, m) for r, m in additions if (r, m) not in have]


def add_bindings(policy: dict, additions: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge (role, member) pairs into an IAM policy in place; return what was added.

    Existing bindings and members are preserved verbatim (additive only); a conditional
    binding for the same role is left alone and a new unconditional one is created.
    """
    added: list[tuple[str, str]] = []
    bindings = policy.setdefault("bindings", [])
    for role, member in additions:
        for b in bindings:
            if b.get("role") == role and not b.get("condition"):
                members = b.setdefault("members", [])
                if member not in members:
                    members.append(member)
                    added.append((role, member))
                break
        else:
            bindings.append({"role": role, "members": [member]})
            added.append((role, member))
    return added


def lifecycle_missing(existing_rules: Sequence[dict], wanted_rules: Sequence[dict]) -> list[dict]:
    """The handoff lifecycle rules whose prefixes no existing Delete rule covers yet.

    Mirrors ``handoff.ensure_handoff_lifecycle``'s coverage logic (kept prefix-based so
    buckets configured by older toolkit revisions with broader rules still count).
    """
    covered: set[str] = set()
    for rule in existing_rules:
        if rule.get("action", {}).get("type") == "Delete":
            covered.update(rule.get("condition", {}).get("matchesPrefix") or [])
    return [
        rule
        for rule in wanted_rules
        if not all(m in covered for m in rule["condition"]["matchesPrefix"])
    ]


class GcpApi:
    """Thin REST facade over the handful of Google APIs the setup touches.

    One seam on purpose: offline tests fake exactly this class. Every call is made with
    the TARGET project as the quota project — google-auth otherwise force-adds the ADC
    file's default quota project header, silently billing (and gating) calls against an
    unrelated project.
    """

    def __init__(self, project: str, credentials: Any | None = None):
        self.project = project
        self._credentials = credentials
        self._session: Any = None
        self._ping_cache: dict[tuple[str, str], tuple[bool, str]] = {}

    # -- plumbing ---------------------------------------------------------------

    def _http(self) -> Any:
        if self._session is None:
            import google.auth  # lazy: keep module import stdlib-only
            from google.auth.transport.requests import AuthorizedSession

            creds = self._credentials
            if creds is None:
                creds, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            if hasattr(creds, "with_quota_project"):
                creds = creds.with_quota_project(self.project)
            self._credentials = creds
            self._session = AuthorizedSession(creds)
        return self._session

    def request(
        self, method: str, url: str, *, json_body: dict | None = None, ok404: bool = False
    ) -> dict | None:
        r = self._http().request(method, url, json=json_body)
        if r.status_code == 404 and ok404:
            return None
        if not r.ok:
            try:
                msg = r.json().get("error", {}).get("message", r.text)
            except Exception:  # noqa: BLE001 — error bodies are best-effort
                msg = r.text
            raise GcpError(f"HTTP {r.status_code}: {msg}", status=r.status_code)
        return r.json() if r.text else {}

    def _wait_operation(self, base: str, op: dict, timeout: float = 600.0) -> None:
        deadline = time.monotonic() + timeout
        while not op.get("done"):
            if time.monotonic() > deadline:
                raise GcpError(f"operation {op.get('name')!r} did not finish in {timeout:.0f}s")
            time.sleep(3)
            op = self.request("GET", f"{base}/{op['name']}") or {}
        if op.get("error"):
            raise GcpError(f"operation failed: {op['error']}")

    # -- Service Usage ----------------------------------------------------------

    def enabled_services(self) -> tuple[set[str], str]:
        """(enabled service names, project number). Also the access/bootstrap probe."""
        names: set[str] = set()
        number: str | None = None
        token: str | None = None
        while True:
            url = (
                f"https://serviceusage.googleapis.com/v1/projects/{self.project}/services"
                f"?filter=state:ENABLED&pageSize=200"
            ) + (f"&pageToken={token}" if token else "")
            data = self.request("GET", url) or {}
            for svc in data.get("services", []):
                names.add(svc["config"]["name"])
                # rows are "projects/<number>/services/<name>" — the number is free here
                number = number or svc["name"].split("/")[1]
            token = data.get("nextPageToken")
            if not token:
                break
        if number is None:  # no enabled services at all — fall back to CRM
            proj = self.request(
                "GET", f"https://cloudresourcemanager.googleapis.com/v3/projects/{self.project}"
            )
            number = (proj or {})["name"].split("/")[1]
        return names, number

    def enable_services(self, services: Sequence[str]) -> None:
        op = self.request(
            "POST",
            f"https://serviceusage.googleapis.com/v1/projects/{self.project}/services:batchEnable",
            json_body={"serviceIds": list(services)},
        )
        self._wait_operation("https://serviceusage.googleapis.com/v1", op or {})

    # -- project IAM (Cloud Resource Manager) ------------------------------------

    def get_project_policy(self) -> dict:
        return self.request(
            "POST",
            f"https://cloudresourcemanager.googleapis.com/v1/projects/{self.project}:getIamPolicy",
            json_body={"options": {"requestedPolicyVersion": 3}},
        ) or {}

    def set_project_policy(self, policy: dict) -> None:
        self.request(
            "POST",
            f"https://cloudresourcemanager.googleapis.com/v1/projects/{self.project}:setIamPolicy",
            json_body={"policy": policy},
        )

    # -- service accounts (IAM) ---------------------------------------------------

    def get_service_account(self, email: str) -> dict | None:
        return self.request(
            "GET",
            f"https://iam.googleapis.com/v1/projects/{self.project}/serviceAccounts/{email}",
            ok404=True,
        )

    def create_service_account(self, account_id: str, display_name: str) -> dict:
        return self.request(
            "POST",
            f"https://iam.googleapis.com/v1/projects/{self.project}/serviceAccounts",
            json_body={"accountId": account_id, "serviceAccount": {"displayName": display_name}},
        ) or {}

    def get_sa_policy(self, email: str) -> dict:
        return self.request(
            "POST",
            f"https://iam.googleapis.com/v1/projects/{self.project}"
            f"/serviceAccounts/{email}:getIamPolicy",
        ) or {}

    def set_sa_policy(self, email: str, policy: dict) -> None:
        self.request(
            "POST",
            f"https://iam.googleapis.com/v1/projects/{self.project}"
            f"/serviceAccounts/{email}:setIamPolicy",
            json_body={"policy": policy},
        )

    # -- custom roles (IAM) -------------------------------------------------------

    def get_role(self, role_id: str) -> dict | None:
        return self.request(
            "GET", f"https://iam.googleapis.com/v1/projects/{self.project}/roles/{role_id}",
            ok404=True,
        )

    def create_role(self, role_id: str, title: str, permissions: Sequence[str]) -> dict:
        return self.request(
            "POST",
            f"https://iam.googleapis.com/v1/projects/{self.project}/roles",
            json_body={
                "roleId": role_id,
                "role": {"title": title, "includedPermissions": list(permissions), "stage": "GA"},
            },
        ) or {}

    def add_role_permissions(self, role_id: str, permissions: Sequence[str]) -> None:
        """Additively extend a custom role's permissions (existing ones are kept)."""
        self.request(
            "PATCH",
            f"https://iam.googleapis.com/v1/projects/{self.project}/roles/{role_id}"
            "?updateMask=includedPermissions",
            json_body={"includedPermissions": list(permissions)},
        )

    # -- Cloud Storage (JSON API) --------------------------------------------------

    def get_bucket(self, name: str) -> dict | None:
        return self.request(
            "GET", f"https://storage.googleapis.com/storage/v1/b/{name}", ok404=True
        )

    def create_bucket(self, name: str, location: str, lifecycle_rules: list[dict]) -> None:
        body: dict[str, Any] = {
            "name": name,
            "location": location,
            "iamConfiguration": {
                "uniformBucketLevelAccess": {"enabled": True},
                "publicAccessPrevention": "enforced",
            },
        }
        if lifecycle_rules:
            body["lifecycle"] = {"rule": lifecycle_rules}
        self.request(
            "POST",
            f"https://storage.googleapis.com/storage/v1/b?project={self.project}",
            json_body=body,
        )

    def set_bucket_lifecycle(self, name: str, rules: list[dict]) -> None:
        self.request(
            "PATCH",
            f"https://storage.googleapis.com/storage/v1/b/{name}",
            json_body={"lifecycle": {"rule": rules}},
        )

    def get_bucket_policy(self, name: str) -> dict:
        # Version 3: the runtime identity's grants are conditional, and GCS refuses to
        # return a policy that has conditions at a lower version.
        return self.request(
            "GET",
            f"https://storage.googleapis.com/storage/v1/b/{name}/iam"
            "?optionsRequestedPolicyVersion=3",
        ) or {}

    def set_bucket_policy(self, name: str, policy: dict) -> None:
        self.request(
            "PUT", f"https://storage.googleapis.com/storage/v1/b/{name}/iam", json_body=policy
        )

    # -- Vertex model access ---------------------------------------------------------

    def model_ping(self, model: str, location: str = MODEL_CHECK_LOCATION) -> tuple[bool, str]:
        """One 1-output-token Claude call through Vertex — the model-enablement probe.

        Cached per (model, location): the audit runs several rounds per invocation and
        enablement can't change from anything this tool does.
        """
        cached = self._ping_cache.get((model, location))
        if cached is None:
            cached = self._model_ping_uncached(model, location)
            self._ping_cache[(model, location)] = cached
        return cached

    def _model_ping_uncached(self, model: str, location: str) -> tuple[bool, str]:
        host = (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        url = (
            f"https://{host}/v1/projects/{self.project}/locations/{location}"
            f"/publishers/anthropic/models/{model}:rawPredict"
        )
        body = {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
        try:
            self.request("POST", url, json_body=body)
            return True, f"{model} responded via Vertex (location {location})"
        except GcpError as e:
            return False, str(e)

    # -- who am I -----------------------------------------------------------------

    def adc_email(self) -> str | None:
        """The ADC principal's email, when discoverable (for the default impersonation grant)."""
        self._http()
        email = getattr(self._credentials, "service_account_email", None)
        if email and email != "default":
            return email
        try:
            r = self._http().get("https://openidconnect.googleapis.com/v1/userinfo")
            if r.ok:
                return r.json().get("email")
        except Exception:  # noqa: BLE001 — absence of an email is handled by the caller
            pass
        return None


@dataclass
class Settings:
    """What the CLI resolves the flags into (also the programmatic entry point's input)."""

    project: str
    location: str = DEFAULT_LOCATION
    operator_sa_id: str | None = DEFAULT_OPERATOR_SA_ID  # None = don't manage an operator SA
    runtime_sa_id: str = DEFAULT_RUNTIME_SA_ID  # what deploy(service_account=) defaults to
    impersonators: tuple[str, ...] = ()  # IAM members granted tokenCreator on the operator SA
    staging_bucket: str | None = None  # gs:// URI; None = the backend.deploy default
    output_bucket: str | None = None
    models: tuple[str, ...] = DEFAULT_CHECK_MODELS  # required; () = skip the model checks
    optional_models: tuple[str, ...] = DEFAULT_OPTIONAL_MODELS  # absence doesn't fail readiness

    def buckets(self) -> tuple[str, str]:
        staging, output = default_buckets(self.project)
        return (self.staging_bucket or staging, self.output_bucket or output)

    def operator_email(self) -> str | None:
        return self._sa_email(self.operator_sa_id)

    def runtime_email(self) -> str:
        """The runtime service account's email — what ``gemini.deploy`` runs engines as."""
        email = self._sa_email(self.runtime_sa_id)
        assert email is not None  # runtime_sa_id is never empty (there is no opt-out)
        return email

    def _sa_email(self, sa_id: str | None) -> str | None:
        if not sa_id:
            return None
        if "@" in sa_id:
            return sa_id
        return f"{sa_id}@{self.project}.iam.gserviceaccount.com"


@dataclass
class Item:
    """One report row: a checked condition, its status, and (for FIX) how to apply it."""

    step: str
    status: str
    detail: str
    fix: Callable[[], None] | None = None
    key: str = ""  # stable id (steps can repeat, e.g. one per bucket)

    def __post_init__(self) -> None:
        self.key = self.key or self.step


def _member_missing(e: GcpError) -> bool:
    """Does this setIamPolicy failure mean the member (service account) doesn't exist?"""
    msg = str(e).lower()
    return e.status == 400 and ("does not exist" in msg or "deleted" in msg)


def _grant(
    get_policy: Callable[[], dict],
    set_policy: Callable[[dict], None],
    additions: Sequence[tuple[str, str]],
    *,
    condition: dict | None = None,
    retry_missing_s: float = 0.0,
) -> None:
    """Read-modify-write an IAM policy to include ``additions`` (idempotent).

    With ``condition`` the grants are conditional bindings (the runtime identity's bucket
    access). Retries etag conflicts, and — for ``retry_missing_s`` — "member does not
    exist" failures, which a *freshly created* service account produces for a short window.
    """
    deadline = time.monotonic() + retry_missing_s
    while True:
        policy = get_policy()
        if condition is None:
            added = bool(add_bindings(policy, additions))
        else:
            added = any(
                add_conditional_binding(policy, role, member, condition)
                for role, member in additions
            )
        if not added:
            return  # someone else (or an earlier round) already granted it
        try:
            set_policy(policy)
            return
        except GcpError as e:
            if e.status == 409:  # etag conflict — re-read and retry
                continue
            if _member_missing(e) and time.monotonic() < deadline:
                time.sleep(5)
                continue
            raise


def _model_items(api: GcpApi, cfg: Settings, enabled: set[str]) -> list[Item]:
    """The Claude-on-Vertex rows: a 1-output-token live probe per model (check only —
    accepting the Anthropic terms in Model Garden is a console action this tool cannot
    perform, so a missing model reports MANUAL with the exact console page).

    Runs as early as the audit can (right after the API check, needing only the
    aiplatform API) so a missing enablement is visible in the FIRST report — not
    discovered by a paid `--verify` deploy minutes in. A missing *optional* model
    reports NOTE: visible, but it neither fails readiness nor blocks a verify.
    """
    checks = [(m, True) for m in cfg.models] + [(m, False) for m in cfg.optional_models]
    if not checks:
        return []
    if "aiplatform.googleapis.com" not in enabled:
        return [Item("Claude on Vertex", BLOCKED, "probed once the aiplatform API is enabled")]
    items: list[Item] = []
    for model, required in checks:
        ok, note = api.model_ping(model)
        step, key = f"Claude on Vertex ({model})", f"model:{model}"
        url = (
            "https://console.cloud.google.com/vertex-ai/publishers/anthropic/"
            f"model-garden/{model}?project={cfg.project}"
        )
        if ok:
            items.append(Item(step, OK, note, key=key))
        elif required:
            items.append(
                Item(
                    step,
                    MANUAL,
                    f"not callable yet ({note.splitlines()[0][:160]}). Enable it "
                    "(accept the Anthropic terms) once in Vertex Model Garden:\n"
                    f"           {url}\n"
                    "           (or deploy with use_vertex=False and pass "
                    "ANTHROPIC_API_KEY per-invocation)",
                    key=key,
                )
            )
        else:
            items.append(
                Item(
                    step,
                    NOTE,
                    "not enabled — optional, so this does not block readiness; if this "
                    f"project needs it, enable it in Vertex Model Garden:\n           {url}",
                    key=key,
                )
            )
    return items


def audit(api: GcpApi, cfg: Settings) -> list[Item]:
    """Audit the project; return report rows whose FIX items carry their own apply()."""
    items: list[Item] = []

    # 1. APIs — also the access probe: if this fails, nothing else is reachable.
    enabled, number = api.enabled_services()
    missing = [s for s in REQUIRED_SERVICES if s not in enabled]
    if missing:
        items.append(
            Item(
                "APIs",
                FIX,
                f"enable {len(missing)} missing of {len(REQUIRED_SERVICES)} required: "
                + ", ".join(missing),
                fix=lambda m=tuple(missing): api.enable_services(m),
            )
        )
    else:
        items.append(Item("APIs", OK, f"all {len(REQUIRED_SERVICES)} required services enabled"))

    # 2. Claude model enablement — checked as early as possible (see _model_items).
    items.extend(_model_items(api, cfg, enabled))

    # Steps below need these APIs to even audit; with any of them missing, report the
    # rest as BLOCKED — the apply loop re-audits right after enabling.
    gate = {"cloudresourcemanager.googleapis.com", "iam.googleapis.com", "storage.googleapis.com"}
    if gate & set(missing):
        items.append(
            Item(
                "everything else",
                BLOCKED,
                "waiting for the APIs above — re-audited automatically after they are enabled",
            )
        )
        return items

    staging_uri, output_uri = cfg.buckets()

    # 3. Buckets (uniform access + public-access prevention; lifecycle on the output one).
    for label, uri in (("staging bucket", staging_uri), ("output bucket", output_uri)):
        name = bucket_name_of(uri)
        wanted_rules: list[dict] = []
        if label == "output bucket":
            _, wanted_rules = handoff_lifecycle_rules(uri)
        bucket = api.get_bucket(name)
        if bucket is None:
            items.append(
                Item(
                    label,
                    FIX,
                    f"create gs://{name} in {cfg.location} "
                    "(uniform bucket-level access, public access prevented)",
                    fix=lambda n=name, r=tuple(wanted_rules): api.create_bucket(
                        n, cfg.location, list(r)
                    ),
                    key=f"bucket:{name}",
                )
            )
            continue
        note = f"gs://{name} exists"
        loc = str(bucket.get("location", "")).lower()
        if loc and loc != cfg.location.lower():
            note += f" (location {loc}, not {cfg.location} — works, but adds cross-region traffic)"
        items.append(Item(label, OK, note, key=f"bucket:{name}"))
        if wanted_rules:
            existing = list((bucket.get("lifecycle") or {}).get("rule") or [])
            gap = lifecycle_missing(existing, wanted_rules)
            if gap:
                items.append(
                    Item(
                        "output lifecycle",
                        FIX,
                        f"append {len(gap)} handoff lifecycle rule(s) to gs://{name} "
                        "(reap staged secrets/configs; never touches other prefixes)",
                        fix=lambda n=name, e=tuple(existing), g=tuple(gap): (
                            api.set_bucket_lifecycle(n, [*e, *g])
                        ),
                    )
                )
            else:
                items.append(Item("output lifecycle", OK, "handoff reaper rules present"))

    # 4. Operator service account + its grants.
    op_email = cfg.operator_email()
    if op_email:
        member = f"serviceAccount:{op_email}"
        sa = api.get_service_account(op_email)
        sa_exists = sa is not None
        if sa_exists:
            items.append(Item("operator SA", OK, f"{op_email} exists"))
        else:
            sa_id = op_email.split("@")[0]
            items.append(
                Item(
                    "operator SA",
                    FIX,
                    f"create {op_email}",
                    fix=lambda i=sa_id: api.create_service_account(
                        i, "remote-agent-toolkit operator (control plane)"
                    ),
                )
            )

        needed = [(role, member) for role in OPERATOR_PROJECT_ROLES]
        gap = missing_bindings(api.get_project_policy(), needed)
        if gap:
            items.append(
                Item(
                    "operator roles",
                    FIX,
                    "grant on the project: " + ", ".join(sorted({r for r, _ in gap})),
                    # A just-created SA can take a moment to be grantable — retry briefly.
                    fix=lambda g=tuple(gap): _grant(
                        api.get_project_policy, api.set_project_policy, g, retry_missing_s=120
                    ),
                )
            )
        else:
            items.append(
                Item("operator roles", OK, f"{len(OPERATOR_PROJECT_ROLES)} project roles present")
            )

        for uri in (staging_uri, output_uri):
            name = bucket_name_of(uri)
            if api.get_bucket(name) is None:
                items.append(
                    Item(
                        f"operator on gs://{name}",
                        BLOCKED,
                        "bucket doesn't exist yet — granted right after it is created",
                        key=f"operator-bucket:{name}",
                    )
                )
                continue
            addition = (OPERATOR_BUCKET_ROLE, member)
            if missing_bindings(api.get_bucket_policy(name), [addition]):
                items.append(
                    Item(
                        f"operator on gs://{name}",
                        FIX,
                        f"grant {OPERATOR_BUCKET_ROLE} (bucket-scoped, not project-wide)",
                        fix=lambda n=name, a=addition: _grant(
                            lambda: api.get_bucket_policy(n),
                            lambda p: api.set_bucket_policy(n, p),
                            [a],
                            retry_missing_s=120,
                        ),
                        key=f"operator-bucket:{name}",
                    )
                )
            else:
                items.append(
                    Item(
                        f"operator on gs://{name}",
                        OK,
                        f"{OPERATOR_BUCKET_ROLE} present",
                        key=f"operator-bucket:{name}",
                    )
                )

        # 5. Who may impersonate the operator SA.
        members = [principal(m) for m in cfg.impersonators]
        if not members:
            adc = api.adc_email()
            if adc:
                members = [principal(adc)]
        if not members:
            items.append(
                Item(
                    "impersonation",
                    MANUAL,
                    "could not discover the ADC principal — pass --impersonator "
                    "user:you@example.com (repeatable) to grant tokenCreator on the operator SA",
                )
            )
        elif not sa_exists:
            items.append(
                Item(
                    "impersonation",
                    BLOCKED,
                    f"granted to {', '.join(members)} right after the operator SA is created",
                )
            )
        else:
            needed = [("roles/iam.serviceAccountTokenCreator", m) for m in members]
            gap = missing_bindings(api.get_sa_policy(op_email), needed)
            if gap:
                items.append(
                    Item(
                        "impersonation",
                        FIX,
                        "grant roles/iam.serviceAccountTokenCreator on the operator SA to "
                        + ", ".join(m for _, m in gap),
                        fix=lambda e=op_email, g=tuple(gap): _grant(
                            lambda: api.get_sa_policy(e),
                            lambda p: api.set_sa_policy(e, p),
                            g,
                            retry_missing_s=120,
                        ),
                    )
                )
            else:
                items.append(
                    Item("impersonation", OK, f"tokenCreator held by {', '.join(members)}")
                )

    # 6. The runtime identity: a service account you own that every deploy names as
    #    service_account=. It holds only what a turn needs (the agent's shell can use it).
    rt_email = cfg.runtime_email()
    if rt_email:
        rt_member = f"serviceAccount:{rt_email}"
        rt_sa = api.get_service_account(rt_email)
        rt_exists = rt_sa is not None
        if rt_exists:
            items.append(Item("runtime SA", OK, f"{rt_email} exists"))
        else:
            rt_id = rt_email.split("@")[0]
            items.append(
                Item(
                    "runtime SA",
                    FIX,
                    f"create {rt_email} (engines run as it: gemini.deploy(service_account=...))",
                    fix=lambda i=rt_id: api.create_service_account(
                        i, "remote-agent-toolkit runtime identity (engines run as this)"
                    ),
                )
            )

        # 6a. The predict-only custom role (never roles/aiplatform.user, which would hand
        #     the agent's shell every run's job id and engine admin).
        role = api.get_role(RUNTIME_PREDICT_ROLE_ID)
        role_name = predict_role_name(cfg.project)
        if role is None:
            items.append(
                Item(
                    "runtime role",
                    FIX,
                    f"create custom role {RUNTIME_PREDICT_ROLE_ID} with "
                    + ", ".join(RUNTIME_PREDICT_PERMISSIONS),
                    fix=lambda: api.create_role(
                        RUNTIME_PREDICT_ROLE_ID,
                        "ratk runtime: model calls only",
                        RUNTIME_PREDICT_PERMISSIONS,
                    ),
                )
            )
        elif role.get("deleted"):
            items.append(
                Item(
                    "runtime role",
                    MANUAL,
                    f"custom role {RUNTIME_PREDICT_ROLE_ID} is soft-deleted; undelete it "
                    f"(gcloud iam roles undelete {RUNTIME_PREDICT_ROLE_ID} "
                    f"--project {cfg.project}) or wait for it to expire and re-run",
                )
            )
        else:
            have = set(role.get("includedPermissions") or [])
            lacking = [p_ for p_ in RUNTIME_PREDICT_PERMISSIONS if p_ not in have]
            if lacking:
                items.append(
                    Item(
                        "runtime role",
                        FIX,
                        f"add {', '.join(lacking)} to custom role {RUNTIME_PREDICT_ROLE_ID}",
                        fix=lambda h=tuple(sorted(have)), l_=tuple(lacking): (
                            api.add_role_permissions(RUNTIME_PREDICT_ROLE_ID, [*h, *l_])
                        ),
                    )
                )
            else:
                items.append(Item("runtime role", OK, f"{RUNTIME_PREDICT_ROLE_ID} present"))

        # 6b. Project roles.
        rt_roles = (role_name, *RUNTIME_SA_PROJECT_ROLES)
        needed = [(r, rt_member) for r in rt_roles]
        if role is None or role.get("deleted"):
            items.append(
                Item(
                    "runtime roles",
                    BLOCKED,
                    "granted right after the custom role exists",
                )
            )
        else:
            gap = missing_bindings(api.get_project_policy(), needed)
            if gap:
                items.append(
                    Item(
                        "runtime roles",
                        FIX,
                        "grant on the project: " + ", ".join(sorted({r for r, _ in gap})),
                        fix=lambda g=tuple(gap): _grant(
                            api.get_project_policy, api.set_project_policy, g,
                            retry_missing_s=120,
                        ),
                    )
                )
            else:
                items.append(
                    Item("runtime roles", OK, f"{len(rt_roles)} project roles present")
                )

        # 6c. Buckets: read the deploy bundle; create + read-by-name under jobs/ and
        #     events/ratk- only. Nothing under pool/ (the roster), nothing to list.
        staging_name, out_name = bucket_name_of(staging_uri), bucket_name_of(output_uri)
        if api.get_bucket(staging_name) is None:
            items.append(
                Item(
                    f"runtime on gs://{staging_name}", BLOCKED,
                    "bucket doesn't exist yet — granted right after it is created",
                    key=f"runtime-bucket:{staging_name}",
                )
            )
        else:
            addition = (RUNTIME_SA_STAGING_ROLE, rt_member)
            if missing_bindings(api.get_bucket_policy(staging_name), [addition]):
                items.append(
                    Item(
                        f"runtime on gs://{staging_name}", FIX,
                        f"grant {RUNTIME_SA_STAGING_ROLE} (the platform pulls the deploy bundle)",
                        fix=lambda n=staging_name, a=addition: _grant(
                            lambda: api.get_bucket_policy(n),
                            lambda p_: api.set_bucket_policy(n, p_),
                            [a], retry_missing_s=120,
                        ),
                        key=f"runtime-bucket:{staging_name}",
                    )
                )
            else:
                items.append(
                    Item(
                        f"runtime on gs://{staging_name}", OK,
                        f"{RUNTIME_SA_STAGING_ROLE} present",
                        key=f"runtime-bucket:{staging_name}",
                    )
                )
        if api.get_bucket(out_name) is None:
            items.append(
                Item(
                    f"runtime on gs://{out_name}", BLOCKED,
                    "bucket doesn't exist yet — granted right after it is created",
                    key=f"runtime-bucket:{out_name}",
                )
            )
        else:
            condition = output_bucket_condition(out_name)
            out_policy = api.get_bucket_policy(out_name)
            gap = [
                r for r in RUNTIME_SA_OUTPUT_ROLES
                if not conditional_binding_present(out_policy, r, rt_member, condition)
            ]
            if gap:
                items.append(
                    Item(
                        f"runtime on gs://{out_name}", FIX,
                        "grant " + ", ".join(gap) + " conditioned on "
                        + " and ".join(f"objects/{p_}" for p_ in RUNTIME_SA_OUTPUT_PREFIXES)
                        + " (platform job files + pool readiness markers; no list, nothing else)",
                        fix=lambda n=out_name, g=tuple(gap), c=condition: _grant(
                            lambda: api.get_bucket_policy(n),
                            lambda p_: api.set_bucket_policy(n, p_),
                            [(r, rt_member) for r in g], condition=c, retry_missing_s=120,
                        ),
                        key=f"runtime-bucket:{out_name}",
                    )
                )
            else:
                items.append(
                    Item(
                        f"runtime on gs://{out_name}", OK,
                        "conditional " + " + ".join(RUNTIME_SA_OUTPUT_ROLES) + " present",
                        key=f"runtime-bucket:{out_name}",
                    )
                )

        # 6d. The operator deploys AS the runtime identity.
        if op_email:
            op_member = f"serviceAccount:{op_email}"
            if not rt_exists or not sa_exists:
                items.append(
                    Item(
                        "operator acts as runtime SA", BLOCKED,
                        "granted right after both service accounts exist",
                    )
                )
            else:
                needed = [(OPERATOR_ON_RUNTIME_SA_ROLE, op_member)]
                if missing_bindings(api.get_sa_policy(rt_email), needed):
                    items.append(
                        Item(
                            "operator acts as runtime SA", FIX,
                            f"grant {OPERATOR_ON_RUNTIME_SA_ROLE} on {rt_email} to {op_email} "
                            "(deploy(service_account=) acts as it)",
                            fix=lambda e=rt_email, g=tuple(needed): _grant(
                                lambda: api.get_sa_policy(e),
                                lambda p_: api.set_sa_policy(e, p_),
                                g, retry_missing_s=120,
                            ),
                        )
                    )
                else:
                    items.append(
                        Item(
                            "operator acts as runtime SA", OK,
                            f"{OPERATOR_ON_RUNTIME_SA_ROLE} held by {op_email}",
                        )
                    )

    # 7. The DEFAULT service agent: nothing is granted to it. Grants left from the earlier
    #    identity model are reported (the agent's shell can use them, README "Migration");
    #    this tool never removes anything, so the removal is a manual step.
    agent_member = f"serviceAccount:{runtime_agent_email(number)}"
    leftovers: list[str] = []
    if not missing_bindings(api.get_project_policy(), [(LEGACY_AGENT_PROJECT_ROLE, agent_member)]):
        leftovers.append(f"{LEGACY_AGENT_PROJECT_ROLE} on the project")
    out_name = bucket_name_of(output_uri)
    if api.get_bucket(out_name) is not None and not missing_bindings(
        api.get_bucket_policy(out_name), [(LEGACY_AGENT_BUCKET_ROLE, agent_member)]
    ):
        leftovers.append(f"{LEGACY_AGENT_BUCKET_ROLE} on gs://{out_name}")
    if leftovers:
        items.append(
            Item(
                "default service agent",
                NOTE,
                f"{runtime_agent_email(number)} still holds " + " and ".join(leftovers)
                + " — from the earlier identity model, where engines ran as it. Once every "
                "engine in this project has been redeployed (the deploy names the runtime "
                "service account by default), remove them by hand (this tool never removes "
                "grants); see README, \"Migration\"",
            )
        )
    else:
        items.append(
            Item(
                "default service agent", OK,
                f"{runtime_agent_email(number)} holds no toolkit grants",
            )
        )

    return items


def print_report(items: Sequence[Item], header: str | None = None) -> None:
    if header:
        print(header, flush=True)
    width = max((len(i.status) for i in items), default=2)
    for i in items:
        print(f"  [{i.status:^{width}}] {i.step}: {i.detail}", flush=True)


def exit_code(items: Sequence[Item]) -> int:
    """0 = ready; 2 = not ready (unapplied fixes, blocked audits, or manual steps left).

    NOTE rows are informational (e.g. an optional model not enabled) — still ready.
    """
    return 0 if all(i.status in (OK, NOTE) for i in items) else 2


def _apply_rounds(
    api: GcpApi, cfg: Settings, first_items: list[Item], *, max_rounds: int = 5
) -> list[Item]:
    """Apply FIX items, re-auditing between rounds until the audit converges.

    Multiple rounds because fixes unlock audits (enabling an API lets IAM be read; a
    created bucket or custom role can then be granted).
    """
    items = first_items
    for _ in range(max_rounds):
        fixes = [i for i in items if i.status == FIX and i.fix]
        if not fixes:
            break
        for item in fixes:
            item.fix()  # type: ignore[misc]
            print(f"  applied: {item.step} — {item.detail.splitlines()[0]}", flush=True)
        items = _audit_with_retry(api, cfg)
    return items


def _audit_with_retry(api: GcpApi, cfg: Settings) -> list[Item]:
    """Audit, riding out the short propagation window after enabling an API."""
    for attempt in range(8):
        try:
            return audit(api, cfg)
        except GcpError as e:
            recently_enabled = "has not been used" in str(e) or "SERVICE_DISABLED" in str(e)
            if not (recently_enabled and attempt < 7):
                raise
            time.sleep(15)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------
# --verify: prove the end state with a real throwaway deploy + one turn + teardown.
# ---------------------------------------------------------------------------------

VERIFY_TASK = (
    'Run `python3 -c "print(6 * 7)"` in the shell and reply with just the number it prints.'
)


def verify_blockers(items: Sequence[Item]) -> list[Item]:
    """The report rows that make a paid verify pointless — deploy anyway and the build
    or the turn fails minutes in (a failed model check is the expensive classic).

    One row may legitimately be non-OK and still verify: the impersonation grant
    (control-plane convenience for *other* principals; it does not affect whether the
    engine deploys and runs).
    """
    allowed = {"impersonation"}
    return [i for i in items if i.status not in (OK, NOTE) and i.step not in allowed]


def verify(api: GcpApi, cfg: Settings, items: Sequence[Item]) -> tuple[bool, list[Item]]:
    """Deploy a throwaway warm-pool engine AS the runtime service account, run one Haiku
    turn, tear down.

    Exercises every grant end-to-end (build, staging, dispatch, logging, the model, the
    runtime identity's conditional bucket access). Refuses to spend on the deploy while
    `verify_blockers` remain (``items`` is the latest audit).
    """
    blockers = verify_blockers(items)
    if blockers:
        print(
            "\nverify: NOT deploying — resolve these first (the deploy or the turn "
            "would fail after minutes of build):",
            flush=True,
        )
        print_report(blockers)
        return False, list(items)

    import asyncio
    import os

    from ...spec import AgentSpec
    from . import backend

    # The deploy path builds its own clients from ambient ADC; point their quota at the
    # target project so a stray ADC default can't misroute (or 403) those calls.
    os.environ.setdefault("GOOGLE_CLOUD_QUOTA_PROJECT", cfg.project)

    user = re.sub(r"[^a-z0-9-]", "-", getpass.getuser().lower()) or "user"
    spec = AgentSpec(
        name=f"ratk-setup-verify-{user}",
        model=cfg.models[0] if cfg.models else DEFAULT_CHECK_MODELS[0],
        max_turns=8,
        max_budget_usd=1.0,
    )
    staging_uri, output_uri = cfg.buckets()
    runtime_sa = cfg.runtime_email()
    print(
        f"\nverify: deploying throwaway engine {spec.name!r} (warm pool of 1"
        + (f", running as {runtime_sa}" if runtime_sa else "")
        + ") — several minutes, a few cents ...",
        flush=True,
    )
    ok = False
    engine = backend.deploy(
        spec,
        cfg.project,
        cfg.location,
        warm_pool=True,
        pool_size=1,
        staging_bucket=staging_uri,
        output_bucket=output_uri,
        service_account=runtime_sa,
    )
    items = list(items)
    try:
        print("verify: waiting for the warm worker ...", flush=True)
        warm = engine.wait_until_warm(timeout=900)
        print(f"verify: warm={warm}; running one turn ...", flush=True)

        async def _turn() -> tuple[bool, str]:
            run = engine.start_session().run(VERIFY_TASK)
            async for ev in run:
                summary = " ".join((ev.summary or "").split())[:80]
                print(f"    {ev.kind:11} {summary}", flush=True)
            r = run.result
            text = " ".join((r.text or "").split())
            return (not r.is_error) and bool(r.num_turns) and "42" in text, text

        turn_ok, text = asyncio.run(_turn())
        print(f"verify: turn {'OK' if turn_ok else 'FAILED'} (reply: {text[:120]!r})", flush=True)
        ok = turn_ok and warm
    finally:
        print("verify: deleting the throwaway engine ...", flush=True)
        engine.delete(delete_pool_resources=True)
    return ok, items


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ratk-gcp-setup",
        description=(
            "Make a GCP project ready for remote-agent-toolkit's gemini backend: audit, "
            "confirm, apply, re-audit. Additive only — safe on existing, non-empty projects."
        ),
    )
    p.add_argument("--project", required=True, help="target GCP project id")
    p.add_argument(
        "--location",
        default=DEFAULT_LOCATION,
        help=f"Agent Engine + bucket location (default {DEFAULT_LOCATION})",
    )
    p.add_argument(
        "--operator-sa",
        default=DEFAULT_OPERATOR_SA_ID,
        help=(
            "operator service account to create/grant — an account id in the target project "
            f"or a full email (default {DEFAULT_OPERATOR_SA_ID!r})"
        ),
    )
    p.add_argument(
        "--no-operator-sa",
        action="store_true",
        help="don't manage an operator SA (you run the control plane as your own identity)",
    )
    p.add_argument(
        "--runtime-sa",
        default=DEFAULT_RUNTIME_SA_ID,
        help=(
            "runtime service account the engines run as — an account id in the target project "
            f"or a full email (default {DEFAULT_RUNTIME_SA_ID!r}, which is also what "
            "gemini.deploy uses when service_account= is omitted; name another one here and "
            "pass it to every deploy)"
        ),
    )
    p.add_argument(
        "--impersonator",
        action="append",
        default=[],
        metavar="MEMBER",
        help=(
            "IAM member (user:..., group:..., serviceAccount:..., or a bare email) granted "
            "tokenCreator on the operator SA; repeatable (default: the ADC principal)"
        ),
    )
    p.add_argument("--staging-bucket", help="override gs://<project>-agent-staging")
    p.add_argument("--output-bucket", help="override gs://<project>-agent-output")
    p.add_argument(
        "--model",
        action="append",
        default=None,
        metavar="MODEL",
        help=(
            "required Claude model(s) to check on Vertex; repeatable, replaces the default "
            f"list {', '.join(DEFAULT_CHECK_MODELS)} — the FIRST one is also what --verify's "
            "turn runs on"
        ),
    )
    p.add_argument(
        "--optional-model",
        action="append",
        default=None,
        metavar="MODEL",
        help=(
            "Claude model(s) checked but whose absence doesn't fail readiness; repeatable, "
            f"replaces the default list {', '.join(DEFAULT_OPTIONAL_MODELS)}"
        ),
    )
    p.add_argument(
        "--skip-model-check", action="store_true", help="skip the live Vertex model probes"
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="audit only, change nothing; exit 0 iff the project is ready",
    )
    p.add_argument("--yes", action="store_true", help="apply without an interactive prompt")
    p.add_argument(
        "--verify",
        action="store_true",
        help=(
            "after setup, prove the end state: deploy a throwaway warm-pool engine as the "
            "runtime service account, run one Haiku turn, tear down (COSTS a few cents and "
            "~10 min)"
        ),
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Settings(
        project=args.project,
        location=args.location,
        operator_sa_id=None if args.no_operator_sa else args.operator_sa,
        runtime_sa_id=args.runtime_sa,
        impersonators=tuple(args.impersonator),
        staging_bucket=normalize_bucket_uri(args.staging_bucket) if args.staging_bucket else None,
        output_bucket=normalize_bucket_uri(args.output_bucket) if args.output_bucket else None,
        models=()
        if args.skip_model_check
        else tuple(args.model) if args.model else DEFAULT_CHECK_MODELS,
        optional_models=()
        if args.skip_model_check
        else tuple(args.optional_model) if args.optional_model else DEFAULT_OPTIONAL_MODELS,
    )
    api = GcpApi(cfg.project)

    try:
        items = audit(api, cfg)
    except GcpError as e:
        print(f"cannot audit project {cfg.project!r}: {e}", file=sys.stderr, flush=True)
        if "serviceusage" in str(e).lower() or (e.status == 403 and "not been used" in str(e)):
            print(
                "bootstrap: the Service Usage API must be enabled once by hand — "
                f"https://console.developers.google.com/apis/api/serviceusage.googleapis.com/"
                f"overview?project={cfg.project}",
                file=sys.stderr,
                flush=True,
            )
        return 1

    print_report(items, f"remote-agent-toolkit GCP setup — {cfg.project} ({cfg.location})")
    fixes = [i for i in items if i.status == FIX]
    blocked = [i for i in items if i.status == BLOCKED]

    if args.check:
        return exit_code(items)

    if fixes or blocked:
        n = len(fixes) + len(blocked)
        if not args.yes:
            if not sys.stdin.isatty():
                print(
                    f"\n{n} change(s) needed — re-run with --yes to apply "
                    "(or --check for audit-only).",
                    flush=True,
                )
                return 2
            answer = input(
                f"\nApply the {n} change(s) above to {cfg.project} "
                "(and whatever they unblock)? [y/N] "
            )
            if answer.strip().lower() not in ("y", "yes"):
                print("nothing changed.", flush=True)
                return 2
        print(flush=True)
        items = _apply_rounds(api, cfg, items)

    if args.verify:
        try:
            ok, items = verify(api, cfg, items)
        except Exception as e:  # noqa: BLE001 — verify failure is a verdict, not a crash
            print(f"verify FAILED: {e}", file=sys.stderr, flush=True)
            ok = False
        print_report(items, f"\nfinal state — {cfg.project}")
        code = exit_code(items)
        print(f"\nverify: {'PASS' if ok else 'FAIL'}", flush=True)
        return code if ok else (code or 1)

    print_report(items, f"\nfinal state — {cfg.project}")
    code = exit_code(items)
    if code == 0:
        print(f"\n{cfg.project} is ready.", flush=True)
        if cfg.runtime_email():
            print(
                f"deploy with gemini.deploy(..., service_account={cfg.runtime_email()!r}).",
                flush=True,
            )
    else:
        left = [i for i in items if i.status not in (OK, NOTE)]
        print(
            f"\nnot fully ready yet — {len(left)} item(s) above remain "
            f"({', '.join(sorted({i.status for i in left}))}).",
            flush=True,
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())

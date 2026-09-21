"""Offline tests for the GCP project-setup tool (agent-run-gcp-setup), sandbox edition.

Everything runs against FakeGcp — a stand-in for the one REST seam (`GcpApi`) — so the
audit->plan->apply->re-audit loop, the IAM policy merging, and the CLI wiring are all
exercised without a network. The tool's real effect on a live project is validated by
running it against a fresh project (and `--verify` proves the end state with a real
deploy)."""

from __future__ import annotations

import copy

import pytest

from agent_run.runtime.gemini import project_setup as ps
from agent_run.runtime.gemini.handoff import handoff_lifecycle_rules

# ---------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------


def test_sandbox_agent_email_and_repo_helpers():
    assert ps.sandbox_agent_email("123456789012") == (
        "service-123456789012@gcp-sa-vertex-sandbox.iam.gserviceaccount.com"
    )
    assert ps.repo_resource("p", "us-central1", "agent-run") == "projects/p/locations/us-central1/repositories/agent-run"
    assert ps.image_repo_uri("p", "us-central1", "agent-run") == "us-central1-docker.pkg.dev/p/agent-run"


def test_defaults_mirror_backend_deploy_defaults():
    # backend.deploy defaults to gs://<project>-agent-output and the `agent-run` repo; the setup
    # tool must create exactly those or a plain deploy() won't find them.
    from agent_run.runtime.gemini import _image, model_token

    assert ps.default_output_bucket("proj") == "gs://proj-agent-output"
    assert ps.DEFAULT_REPO_ID == _image.DEFAULT_REPO_ID
    assert ps.Settings(project="p").image_repo() == _image.default_image_repo("p", ps.DEFAULT_LOCATION)
    assert ps.Settings(project="p").model_email() == model_token.default_model_service_account("p")


def test_bucket_uri_helpers():
    assert ps.normalize_bucket_uri("b") == "gs://b"
    assert ps.normalize_bucket_uri("gs://b/prefix") == "gs://b/prefix"
    assert ps.bucket_name_of("gs://b/prefix/deep") == "b"
    assert ps.bucket_name_of("gs://b") == "b"


def test_principal_qualifies_bare_emails_only():
    assert ps.principal("a@example.com") == "user:a@example.com"
    assert ps.principal("x@proj.iam.gserviceaccount.com") == "serviceAccount:x@proj.iam.gserviceaccount.com"
    assert ps.principal("group:team@example.com") == "group:team@example.com"
    assert ps.principal("serviceAccount:a@b.c") == "serviceAccount:a@b.c"


def test_missing_and_add_bindings_are_additive_and_idempotent():
    policy = {
        "etag": "abc",
        "bindings": [
            {"role": "roles/viewer", "members": ["user:old@x.com"]},
            {"role": "roles/editor", "members": ["user:cond@x.com"],
             "condition": {"expression": "request.time < timestamp('2020-01-01T00:00:00Z')"}},
        ],
    }
    additions = [
        ("roles/viewer", "user:new@x.com"),
        ("roles/viewer", "user:old@x.com"),
        ("roles/editor", "user:cond@x.com"),  # only conditionally present -> still missing
        ("roles/owner", "user:new@x.com"),
    ]
    assert ps.missing_bindings(policy, additions) == [
        ("roles/viewer", "user:new@x.com"), ("roles/editor", "user:cond@x.com"), ("roles/owner", "user:new@x.com"),
    ]
    ps.add_bindings(policy, additions)
    assert policy["bindings"][0]["members"] == ["user:old@x.com", "user:new@x.com"]
    assert policy["bindings"][1]["condition"]
    assert [b for b in policy["bindings"] if b["role"] == "roles/editor" and not b.get("condition")] == [
        {"role": "roles/editor", "members": ["user:cond@x.com"]}]
    assert ps.add_bindings(policy, additions) == []
    assert ps.missing_bindings(policy, additions) == []


def test_lifecycle_missing_matches_handoff_coverage_semantics():
    _, wanted = handoff_lifecycle_rules("gs://b")
    assert ps.lifecycle_missing([], wanted) == wanted
    broad = [{"action": {"type": "Delete"},
              "condition": {"age": 99, "matchesPrefix": [m for rule in wanted for m in rule["condition"]["matchesPrefix"]]}}]
    assert ps.lifecycle_missing(broad, wanted) == []
    setclass = [{"action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
                 "condition": {"age": 1, "matchesPrefix": ["invocation-secrets/"]}}]
    assert ps.lifecycle_missing(setclass, wanted) == wanted


def test_settings_emails_and_overrides():
    cfg = ps.Settings(project="p")
    assert cfg.operator_email() == "agent-runtime@p.iam.gserviceaccount.com"
    assert cfg.model_email() == "agent-run-model@p.iam.gserviceaccount.com"
    assert cfg.bucket() == "gs://p-agent-output"
    cfg = ps.Settings(project="p", operator_sa_id="ops@other.iam.gserviceaccount.com",
                      model_sa_id="m@o.iam.gserviceaccount.com", output_bucket="gs://o/pre", repo_id="imgs")
    assert cfg.operator_email() == "ops@other.iam.gserviceaccount.com"
    assert cfg.model_email() == "m@o.iam.gserviceaccount.com"
    assert cfg.bucket() == "gs://o/pre" and cfg.image_repo() == "us-central1-docker.pkg.dev/p/imgs"
    assert ps.Settings(project="p", operator_sa_id=None).operator_email() is None


def test_exit_code():
    ok = ps.Item("a", ps.OK, "d")
    assert ps.exit_code([ok]) == 0
    assert ps.exit_code([ok, ps.Item("b", ps.NOTE, "d")]) == 0
    for status in (ps.FIX, ps.BLOCKED, ps.MANUAL):
        assert ps.exit_code([ok, ps.Item("b", status, "d")]) == 2


# ---------------------------------------------------------------------------------
# FakeGcp: a controllable stand-in for the GcpApi seam
# ---------------------------------------------------------------------------------

PROJECT = "test-proj"
NUMBER = "12345"
LOCATION = ps.DEFAULT_LOCATION
AGENT = f"serviceAccount:service-{NUMBER}@gcp-sa-vertex-sandbox.iam.gserviceaccount.com"
OPERATOR = f"serviceAccount:agent-runtime@{PROJECT}.iam.gserviceaccount.com"
MODEL = f"serviceAccount:agent-run-model@{PROJECT}.iam.gserviceaccount.com"
PREDICT_ROLE = f"projects/{PROJECT}/roles/{ps.MODEL_PREDICT_ROLE_ID}"
BUCKET = f"{PROJECT}-agent-output"


class FakeGcp:
    """In-memory project state implementing the GcpApi surface audit/apply use."""

    def __init__(self, *, enabled=None, model_ok=True, disabled_models=()):
        self.project = PROJECT
        self.enabled = set(enabled if enabled is not None else ps.REQUIRED_SERVICES)
        self.disabled_models = set(disabled_models)
        self.buckets: dict[str, dict] = {}
        self.bucket_policies: dict[str, dict] = {}
        self.repos: dict[str, dict] = {}
        self.repo_policies: dict[str, dict] = {}
        self.service_accounts: dict[str, dict] = {}
        self.sa_policies: dict[str, dict] = {}
        self.roles: dict[str, dict] = {}
        self.project_policy: dict = {"etag": "e", "bindings": []}
        self.model_ok = model_ok
        self.calls: list[str] = []

    def enabled_services(self):
        return set(self.enabled), NUMBER

    def enable_services(self, services):
        self.calls.append(f"enable:{','.join(services)}")
        self.enabled.update(services)

    def get_project_policy(self):
        return copy.deepcopy(self.project_policy)

    def set_project_policy(self, policy):
        self.project_policy = policy

    def get_service_account(self, email):
        return self.service_accounts.get(email)

    def create_service_account(self, account_id, display_name):
        email = f"{account_id}@{PROJECT}.iam.gserviceaccount.com"
        self.service_accounts[email] = {"email": email}
        self.sa_policies[email] = {"etag": "e", "bindings": []}
        self.calls.append(f"create-sa:{email}")
        return self.service_accounts[email]

    def get_sa_policy(self, email):
        return copy.deepcopy(self.sa_policies[email])

    def set_sa_policy(self, email, policy):
        self.sa_policies[email] = policy

    def get_role(self, role_id):
        return self.roles.get(role_id)

    def create_role(self, role_id, title, permissions):
        self.roles[role_id] = {"name": f"projects/{PROJECT}/roles/{role_id}", "title": title,
                               "includedPermissions": list(permissions), "stage": "GA"}
        self.calls.append(f"create-role:{role_id}")
        return self.roles[role_id]

    def add_role_permissions(self, role_id, permissions):
        self.roles[role_id]["includedPermissions"] = list(permissions)
        self.calls.append(f"patch-role:{role_id}")

    def get_bucket(self, name):
        return self.buckets.get(name)

    def create_bucket(self, name, location, lifecycle_rules):
        b = {"name": name, "location": location.upper()}
        if lifecycle_rules:
            b["lifecycle"] = {"rule": list(lifecycle_rules)}
        self.buckets[name] = b
        self.bucket_policies[name] = {"etag": "e", "bindings": []}
        self.calls.append(f"create-bucket:{name}")

    def set_bucket_lifecycle(self, name, rules):
        self.buckets[name]["lifecycle"] = {"rule": list(rules)}
        self.calls.append(f"lifecycle:{name}")

    def get_bucket_policy(self, name):
        return copy.deepcopy(self.bucket_policies[name])

    def set_bucket_policy(self, name, policy):
        self.bucket_policies[name] = policy

    def get_repository(self, location, repo_id):
        return self.repos.get((location, repo_id))

    def create_repository(self, location, repo_id):
        self.repos[(location, repo_id)] = {"format": "DOCKER"}
        self.repo_policies[(location, repo_id)] = {"etag": "e", "bindings": []}
        self.calls.append(f"create-repo:{location}/{repo_id}")

    def get_repository_policy(self, location, repo_id):
        return copy.deepcopy(self.repo_policies[(location, repo_id)])

    def set_repository_policy(self, location, repo_id, policy):
        self.repo_policies[(location, repo_id)] = policy

    def model_ping(self, model, location="global"):
        if self.model_ok and model not in self.disabled_models:
            return (True, f"{model} ok")
        return (False, "HTTP 403: no access")

    def adc_email(self):
        return "dev@example.com"


def _by_key(items):
    return {i.key: i for i in items}


# ---------------------------------------------------------------------------------
# audit / apply
# ---------------------------------------------------------------------------------


def test_audit_ready_project_is_all_ok():
    api = FakeGcp()
    cfg = ps.Settings(project=PROJECT)
    _, rules = handoff_lifecycle_rules(cfg.bucket())
    api.create_bucket(BUCKET, LOCATION, rules)
    api.create_repository(LOCATION, ps.DEFAULT_REPO_ID)
    api.create_service_account("agent-runtime", "x")
    api.create_service_account("agent-run-model", "x")
    api.create_role(ps.MODEL_PREDICT_ROLE_ID, "x", ps.MODEL_PREDICT_PERMISSIONS)
    ps.add_bindings(api.project_policy, [(r, OPERATOR) for r in ps.OPERATOR_PROJECT_ROLES] + [(PREDICT_ROLE, MODEL)])
    ps.add_bindings(api.bucket_policies[BUCKET], [(ps.OPERATOR_BUCKET_ROLE, OPERATOR)])
    ps.add_bindings(api.repo_policies[(LOCATION, ps.DEFAULT_REPO_ID)],
                    [(ps.OPERATOR_REPO_ROLE, OPERATOR), (ps.SANDBOX_AGENT_REPO_ROLE, AGENT)])
    ps.add_bindings(api.sa_policies[OPERATOR.split(":", 1)[1]], [(ps.TOKEN_CREATOR_ROLE, "user:dev@example.com")])
    ps.add_bindings(api.sa_policies[MODEL.split(":", 1)[1]],
                    [(ps.TOKEN_CREATOR_ROLE, OPERATOR), (ps.TOKEN_CREATOR_ROLE, "user:dev@example.com")])
    items = ps.audit(api, cfg)
    assert all(i.status == ps.OK for i in items), [(i.step, i.status, i.detail) for i in items]
    assert ps.exit_code(items) == 0


def test_audit_blocks_everything_behind_missing_gate_apis():
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    items = ps.audit(api, ps.Settings(project=PROJECT))
    assert [(i.step, i.status) for i in items] == [
        ("APIs", ps.FIX), ("Claude on Vertex", ps.BLOCKED), ("everything else", ps.BLOCKED),
    ]
    assert "iam.googleapis.com" in items[0].detail and "artifactregistry.googleapis.com" in items[0].detail


def test_model_check_runs_before_the_gate_when_aiplatform_is_enabled():
    api = FakeGcp(enabled={"serviceusage.googleapis.com", "aiplatform.googleapis.com"}, model_ok=False)
    items = ps.audit(api, ps.Settings(project=PROJECT))
    by = _by_key(items)
    assert by["model:claude-haiku-4-5"].status == ps.MANUAL
    assert "model-garden/claude-haiku-4-5" in by["model:claude-haiku-4-5"].detail
    assert by["model:claude-fable-5"].status == ps.NOTE
    assert by["everything else"].status == ps.BLOCKED


def test_verify_blockers_allow_only_the_token_grants():
    ok = ps.Item("APIs", ps.OK, "d")
    manual_imp = ps.Item("impersonation", ps.MANUAL, "d")
    manual_tokens = ps.Item("model tokens", ps.MANUAL, "d")
    model_manual = ps.Item("Claude on Vertex (claude-haiku-4-5)", ps.MANUAL, "d")
    model_note = ps.Item("Claude on Vertex (claude-fable-5)", ps.NOTE, "d")
    repo_fix = ps.Item("image repo", ps.FIX, "d")
    assert ps.verify_blockers([ok, manual_imp, manual_tokens, model_note]) == []
    assert ps.verify_blockers([ok, model_manual]) == [model_manual]
    assert ps.verify_blockers([ok, repo_fix]) == [repo_fix]


def test_verify_refuses_to_deploy_with_blockers(capsys):
    api = FakeGcp()
    items = [ps.Item("Claude on Vertex", ps.MANUAL, "not enabled")]
    ok, out_items = ps.verify(api, ps.Settings(project=PROJECT), items)
    assert ok is False and out_items == items
    assert "NOT deploying" in capsys.readouterr().out


def test_apply_rounds_converge_on_an_empty_project():
    """Empty project -> everything applied: bucket + lifecycle, repo + the sandbox agent's
    read, the model SA with only the predict role, the operator with its scoped grants, the
    token grants — and the project is ready."""
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    cfg = ps.Settings(project=PROJECT)
    items = ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    by = _by_key(items)
    for key in ("APIs", f"bucket:{BUCKET}", "output lifecycle", "image repo", "sandbox agent reads images",
                "model SA", "model role", "model roles", "operator SA", "operator roles",
                f"operator-bucket:{BUCKET}", "operator pushes images", "impersonation", "model tokens"):
        assert by[key].status == ps.OK, (key, by[key].detail)
    for model in (*ps.DEFAULT_CHECK_MODELS, *ps.DEFAULT_OPTIONAL_MODELS):
        assert by[f"model:{model}"].status == ps.OK
    assert ps.exit_code(items) == 0

    # The model identity holds exactly the predict-only custom role, project-wide, nothing else.
    assert api.roles[ps.MODEL_PREDICT_ROLE_ID]["includedPermissions"] == ["aiplatform.endpoints.predict"]
    model_roles = {b["role"] for b in api.project_policy["bindings"] if MODEL in b["members"]}
    assert model_roles == {PREDICT_ROLE}
    # The operator: aiplatform.user on the project, storage.admin on the bucket, writer on the repo.
    operator_roles = {b["role"] for b in api.project_policy["bindings"] if OPERATOR in b["members"]}
    assert operator_roles == set(ps.OPERATOR_PROJECT_ROLES)
    assert [b["role"] for b in api.bucket_policies[BUCKET]["bindings"] if OPERATOR in b["members"]] == [
        ps.OPERATOR_BUCKET_ROLE]
    repo_policy = api.repo_policies[(LOCATION, ps.DEFAULT_REPO_ID)]
    assert {(b["role"], m) for b in repo_policy["bindings"] for m in b["members"]} == {
        (ps.OPERATOR_REPO_ROLE, OPERATOR), (ps.SANDBOX_AGENT_REPO_ROLE, AGENT)}
    # The sandbox service agent gets the repo read and NOTHING on the project or the bucket.
    assert not any(AGENT in b["members"] for b in api.project_policy["bindings"])
    assert not any(AGENT in b["members"] for b in api.bucket_policies[BUCKET]["bindings"])
    # Token minting: the operator SA and the ADC principal on the model SA; the principal on the operator SA.
    model_sa_policy = api.sa_policies[MODEL.split(":", 1)[1]]
    assert sorted(m for b in model_sa_policy["bindings"] for m in b["members"]) == sorted([OPERATOR, "user:dev@example.com"])
    assert all(b["role"] == ps.TOKEN_CREATOR_ROLE for b in model_sa_policy["bindings"])
    op_sa_policy = api.sa_policies[OPERATOR.split(":", 1)[1]]
    assert op_sa_policy["bindings"] == [{"role": ps.TOKEN_CREATOR_ROLE, "members": ["user:dev@example.com"]}]


def test_existing_custom_role_is_extended_additively():
    api = FakeGcp()
    api.create_role(ps.MODEL_PREDICT_ROLE_ID, "old", ["aiplatform.endpoints.explain"])
    cfg = ps.Settings(project=PROJECT)
    ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    perms = api.roles[ps.MODEL_PREDICT_ROLE_ID]["includedPermissions"]
    assert sorted(perms) == ["aiplatform.endpoints.explain", "aiplatform.endpoints.predict"]
    api.roles[ps.MODEL_PREDICT_ROLE_ID]["deleted"] = True
    by = _by_key(ps.audit(api, cfg))
    assert by["model role"].status == ps.MANUAL and "undelete" in by["model role"].detail
    assert by["model roles"].status == ps.BLOCKED


def test_model_sa_default_matches_what_deploy_uses():
    assert ps.Settings(project="p").model_email() == f"{ps.DEFAULT_MODEL_SA_ID}@p.iam.gserviceaccount.com"
    help_text = ps.build_parser().format_help()
    assert "--model-sa" in help_text and "--runtime-sa" not in help_text and "--staging-bucket" not in help_text


def test_verify_deploys_with_the_projects_repo_and_model_sa(monkeypatch):
    from agent_run.runtime.gemini import backend

    seen = {}

    class FakeRun:
        def __init__(self):
            from types import SimpleNamespace as NS

            self.result = NS(is_error=False, num_turns=1, text="42")

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class FakeEngine:
        def wait_until_warm(self, timeout=None):
            return True

        def start_session(self):
            from types import SimpleNamespace as NS

            return NS(run=lambda task: FakeRun())

        def delete(self):
            seen["deleted"] = True

    def fake_deploy(spec, project, location, **kw):
        seen.update(kw)
        return FakeEngine()

    monkeypatch.setattr(backend, "deploy", fake_deploy)
    ok, _ = ps.verify(FakeGcp(), ps.Settings(project=PROJECT), [ps.Item("APIs", ps.OK, "d")])
    assert ok is True and seen["deleted"] is True
    assert seen["model_service_account"] == MODEL.split(":", 1)[1]
    assert seen["image_repo"] == f"{LOCATION}-docker.pkg.dev/{PROJECT}/agent-run"
    assert seen["output_bucket"] == f"gs://{BUCKET}" and seen["warm_pool"] is True and seen["pool_size"] == 1


def test_apply_is_idempotent_and_additive():
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    api.project_policy["bindings"].append({"role": "roles/owner", "members": ["user:boss@x.com"]})
    cfg = ps.Settings(project=PROJECT)
    ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    calls_after_first = list(api.calls)
    policy_after_first = api.get_project_policy()
    items = ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    assert api.calls == calls_after_first
    assert api.get_project_policy() == policy_after_first
    assert all(i.status == ps.OK for i in items)
    assert {"role": "roles/owner", "members": ["user:boss@x.com"]} in api.project_policy["bindings"]


def test_existing_foreign_bucket_is_reported_not_recreated():
    api = FakeGcp()
    cfg = ps.Settings(project=PROJECT)
    api.create_bucket(BUCKET, "EU", [{"action": {"type": "Delete"}, "condition": {"age": 7, "matchesPrefix": ["tmp/"]}}])
    api.calls.clear()
    by = _by_key(ps.audit(api, cfg))
    assert by[f"bucket:{BUCKET}"].status == ps.OK and "location eu" in by[f"bucket:{BUCKET}"].detail
    assert by["output lifecycle"].status == ps.FIX
    by["output lifecycle"].fix()
    rules = api.buckets[BUCKET]["lifecycle"]["rule"]
    assert rules[0]["condition"]["matchesPrefix"] == ["tmp/"]
    _, wanted = handoff_lifecycle_rules(cfg.bucket())
    assert len(rules) == 1 + len(wanted)


def test_required_model_failure_is_manual_with_console_pointer():
    api = FakeGcp(disabled_models={"claude-opus-5"})
    cfg = ps.Settings(project=PROJECT)
    items = ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    by = _by_key(items)
    assert by["model:claude-opus-5"].status == ps.MANUAL
    assert "model-garden/claude-opus-5" in by["model:claude-opus-5"].detail
    assert by["model:claude-haiku-4-5"].status == ps.OK
    assert ps.exit_code(items) == 2


def test_optional_model_failure_is_note_and_still_ready():
    api = FakeGcp(enabled={"serviceusage.googleapis.com"}, disabled_models={"claude-fable-5"})
    cfg = ps.Settings(project=PROJECT)
    items = ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    by = _by_key(items)
    assert by["model:claude-fable-5"].status == ps.NOTE
    assert ps.exit_code(items) == 0 and ps.verify_blockers(items) == []


def test_model_flags_wire_into_settings(monkeypatch):
    seen = {}

    def fake_audit(api, cfg):
        seen["cfg"] = cfg
        return [ps.Item("APIs", ps.OK, "d")]

    monkeypatch.setattr(ps, "GcpApi", lambda project: object())
    monkeypatch.setattr(ps, "audit", fake_audit)
    ps.main(["--project", PROJECT, "--check", "--model", "claude-sonnet-5", "--optional-model", "claude-opus-5",
             "--model-sa", "m@o.iam.gserviceaccount.com", "--repo", "imgs", "--output-bucket", "b"])
    assert seen["cfg"].models == ("claude-sonnet-5",) and seen["cfg"].optional_models == ("claude-opus-5",)
    assert seen["cfg"].model_email() == "m@o.iam.gserviceaccount.com"
    assert seen["cfg"].repo_id == "imgs" and seen["cfg"].bucket() == "gs://b"
    ps.main(["--project", PROJECT, "--check"])
    assert seen["cfg"].models == ps.DEFAULT_CHECK_MODELS
    ps.main(["--project", PROJECT, "--check", "--skip-model-check"])
    assert seen["cfg"].models == () and seen["cfg"].optional_models == ()


def test_no_operator_sa_still_sets_up_the_model_identity_for_the_principal():
    api = FakeGcp()
    cfg = ps.Settings(project=PROJECT, operator_sa_id=None)
    items = ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    steps = {i.step for i in items}
    assert not any("operator" in s or s == "impersonation" for s in steps)
    assert set(api.service_accounts) == {MODEL.split(":", 1)[1]}
    # The ADC principal drives turns as itself: it mints the model tokens.
    assert api.sa_policies[MODEL.split(":", 1)[1]]["bindings"] == [
        {"role": ps.TOKEN_CREATOR_ROLE, "members": ["user:dev@example.com"]}]
    assert all(i.status == ps.OK for i in items)


def test_explicit_impersonators_override_adc_principal():
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    cfg = ps.Settings(project=PROJECT, impersonators=("cto@x.com", "group:eng@x.com"))
    ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    pol = api.sa_policies[OPERATOR.split(":", 1)[1]]
    members = [m for b in pol["bindings"] if b["role"] == ps.TOKEN_CREATOR_ROLE for m in b["members"]]
    assert sorted(members) == ["group:eng@x.com", "user:cto@x.com"]
    model_members = [m for b in api.sa_policies[MODEL.split(":", 1)[1]]["bindings"] for m in b["members"]]
    assert sorted(model_members) == sorted([OPERATOR, "group:eng@x.com", "user:cto@x.com"])


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


def test_cli_check_mode_reports_and_exits_without_changes(monkeypatch, capsys):
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    monkeypatch.setattr(ps, "GcpApi", lambda project: api)
    rc = ps.main(["--project", PROJECT, "--check"])
    assert rc == 2 and not api.calls
    out = capsys.readouterr().out
    assert "FIX" in out and PROJECT in out


def test_cli_yes_applies_and_reports_ready(monkeypatch, capsys):
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    monkeypatch.setattr(ps, "GcpApi", lambda project: api)
    rc = ps.main(["--project", PROJECT, "--yes"])
    out = capsys.readouterr().out
    assert rc == 0 and f"{PROJECT} is ready" in out
    assert "model_service_account=" in out and "image_repo=" in out


def test_cli_non_tty_without_yes_applies_nothing(monkeypatch, capsys):
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    monkeypatch.setattr(ps, "GcpApi", lambda project: api)
    monkeypatch.setattr(ps.sys.stdin, "isatty", lambda: False)
    rc = ps.main(["--project", PROJECT])
    assert rc == 2 and not api.calls and "--yes" in capsys.readouterr().out


def test_cli_prompt_decline_applies_nothing(monkeypatch, capsys):
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    monkeypatch.setattr(ps, "GcpApi", lambda project: api)
    monkeypatch.setattr(ps.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert ps.main(["--project", PROJECT]) == 2 and not api.calls


def test_cli_bootstrap_error_points_at_serviceusage(monkeypatch, capsys):
    class Boom:
        def __init__(self, project):
            pass

        def enabled_services(self):
            raise ps.GcpError("HTTP 403: Service Usage API has not been used in project x before or it is disabled.",
                              status=403)

    monkeypatch.setattr(ps, "GcpApi", Boom)
    assert ps.main(["--project", PROJECT, "--check"]) == 1
    assert "serviceusage.googleapis.com" in capsys.readouterr().err


def test_grant_retries_etag_conflict_then_succeeds():
    state = {"policy": {"etag": "1", "bindings": []}, "sets": 0}

    def set_policy(policy):
        state["sets"] += 1
        if state["sets"] == 1:
            raise ps.GcpError("HTTP 409: etag mismatch", status=409)
        state["policy"] = policy

    ps._grant(lambda: copy.deepcopy(state["policy"]), set_policy, [("roles/x", "user:a@b.c")])
    assert state["sets"] == 2
    assert state["policy"]["bindings"] == [{"role": "roles/x", "members": ["user:a@b.c"]}]


def test_grant_gives_up_on_missing_member_without_retry_window():
    def set_policy(policy):
        raise ps.GcpError("HTTP 400: Service account sa@x does not exist.", status=400)

    with pytest.raises(ps.GcpError):
        ps._grant(lambda: {"bindings": []}, set_policy, [("roles/x", "serviceAccount:sa@x")])


def test_adc_email_asks_userinfo_with_a_bare_bearer_token_and_falls_back_to_tokeninfo(monkeypatch):
    # Field report on #84: on a plain ``gcloud auth application-default login`` the principal
    # was not discovered — the authorized session's quota-project header made userinfo 403.
    import requests

    class Creds:
        valid = True
        token = "tok"

    calls = []

    class Resp:
        def __init__(self, ok, payload):
            self.ok, self._payload = ok, payload

        def json(self):
            return self._payload

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append((url, headers, params))
        if "userinfo" in url:
            return Resp(False, {"error": "403"})
        return Resp(True, {"email": "dev@example.com", "scope": "openid email"})

    monkeypatch.setattr(requests, "get", fake_get)
    api = ps.GcpApi(project=PROJECT, credentials=Creds())
    api._session = object()  # _http() already ran
    assert api.adc_email() == "dev@example.com"
    assert calls[0][1] == {"Authorization": "Bearer tok"} and "x-goog-user-project" not in str(calls[0])
    assert "tokeninfo" in calls[1][0]

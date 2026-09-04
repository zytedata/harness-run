"""Offline tests for the GCP project-setup tool (ratk-gcp-setup).

Everything runs against FakeGcp — a stand-in for the one REST seam (`GcpApi`) — so the
audit->plan->apply->re-audit loop, the IAM policy merging, and the CLI wiring are all
exercised without a network. The tool's real effect on a live project is validated by
running it against a fresh project (and `--verify` proves the end state with a real
deploy)."""

from __future__ import annotations

import pytest

from remote_agent_toolkit.runtime.gemini import project_setup as ps
from remote_agent_toolkit.runtime.gemini.handoff import handoff_lifecycle_rules


# ---------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------


def test_runtime_agent_email():
    assert (
        ps.runtime_agent_email("123456789012")
        == "service-123456789012@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
    )


def test_default_buckets_mirror_backend_deploy_defaults():
    # backend.deploy defaults to gs://<project>-agent-staging / -agent-output; the setup
    # tool must create exactly those names or a plain deploy() won't find its buckets.
    assert ps.default_buckets("proj") == ("gs://proj-agent-staging", "gs://proj-agent-output")


def test_bucket_uri_helpers():
    assert ps.normalize_bucket_uri("b") == "gs://b"
    assert ps.normalize_bucket_uri("gs://b/prefix") == "gs://b/prefix"
    assert ps.bucket_name_of("gs://b/prefix/deep") == "b"
    assert ps.bucket_name_of("gs://b") == "b"


def test_principal_qualifies_bare_emails_only():
    assert ps.principal("a@example.com") == "user:a@example.com"
    assert ps.principal("x@proj.iam.gserviceaccount.com") == (
        "serviceAccount:x@proj.iam.gserviceaccount.com"
    )
    # explicit prefixes (incl. group:) pass through untouched
    assert ps.principal("group:team@example.com") == "group:team@example.com"
    assert ps.principal("serviceAccount:a@b.c") == "serviceAccount:a@b.c"


def test_missing_and_add_bindings_are_additive_and_idempotent():
    policy = {
        "etag": "abc",
        "bindings": [
            {"role": "roles/viewer", "members": ["user:old@x.com"]},
            # a conditional grant of the SAME role must not count as present
            {
                "role": "roles/editor",
                "members": ["user:cond@x.com"],
                "condition": {"expression": "request.time < timestamp('2020-01-01T00:00:00Z')"},
            },
        ],
    }
    additions = [
        ("roles/viewer", "user:new@x.com"),  # existing binding, new member
        ("roles/viewer", "user:old@x.com"),  # already present
        ("roles/editor", "user:cond@x.com"),  # only conditionally present -> still missing
        ("roles/owner", "user:new@x.com"),  # brand new role
    ]
    assert ps.missing_bindings(policy, additions) == [
        ("roles/viewer", "user:new@x.com"),
        ("roles/editor", "user:cond@x.com"),
        ("roles/owner", "user:new@x.com"),
    ]

    added = ps.add_bindings(policy, additions)
    assert added == ps.missing_bindings({"etag": "abc", "bindings": [
        {"role": "roles/viewer", "members": ["user:old@x.com"]},
        {"role": "roles/editor", "members": ["user:cond@x.com"],
         "condition": {"expression": "request.time < timestamp('2020-01-01T00:00:00Z')"}},
    ]}, additions)
    # existing content untouched (additive only): old member still there, condition intact
    assert policy["bindings"][0]["members"] == ["user:old@x.com", "user:new@x.com"]
    assert policy["bindings"][1]["condition"]  # the conditional binding was not modified
    # the conditional role got a NEW unconditional binding rather than a merged member
    unconditional_editor = [
        b for b in policy["bindings"] if b["role"] == "roles/editor" and not b.get("condition")
    ]
    assert unconditional_editor == [{"role": "roles/editor", "members": ["user:cond@x.com"]}]
    # idempotent: a second merge adds nothing
    assert ps.add_bindings(policy, additions) == []
    assert ps.missing_bindings(policy, additions) == []


def test_lifecycle_missing_matches_handoff_coverage_semantics():
    _, wanted = handoff_lifecycle_rules("gs://b")
    # nothing there -> everything missing
    assert ps.lifecycle_missing([], wanted) == wanted
    # a broader pre-existing Delete rule covering the same prefixes counts
    broad = [{
        "action": {"type": "Delete"},
        "condition": {"age": 99, "matchesPrefix": [
            m for rule in wanted for m in rule["condition"]["matchesPrefix"]
        ]},
    }]
    assert ps.lifecycle_missing(broad, wanted) == []
    # a non-Delete rule does not count as coverage
    setclass = [{
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 1, "matchesPrefix": ["invocation-secrets/"]},
    }]
    assert ps.lifecycle_missing(setclass, wanted) == wanted


def test_settings_operator_email_and_bucket_overrides():
    cfg = ps.Settings(project="p")
    assert cfg.operator_email() == "agent-runtime@p.iam.gserviceaccount.com"
    assert cfg.buckets() == ("gs://p-agent-staging", "gs://p-agent-output")
    cfg = ps.Settings(project="p", operator_sa_id="ops@other.iam.gserviceaccount.com",
                      staging_bucket="gs://s", output_bucket="gs://o/pre")
    assert cfg.operator_email() == "ops@other.iam.gserviceaccount.com"
    assert cfg.buckets() == ("gs://s", "gs://o/pre")
    assert ps.Settings(project="p", operator_sa_id=None).operator_email() is None


def test_exit_code():
    ok = ps.Item("a", ps.OK, "d")
    assert ps.exit_code([ok]) == 0
    assert ps.exit_code([ok, ps.Item("b", ps.NOTE, "d")]) == 0  # informational only
    for status in (ps.FIX, ps.BLOCKED, ps.PENDING, ps.MANUAL):
        assert ps.exit_code([ok, ps.Item("b", status, "d")]) == 2


# ---------------------------------------------------------------------------------
# FakeGcp: a controllable stand-in for the GcpApi seam
# ---------------------------------------------------------------------------------

PROJECT = "test-proj"
NUMBER = "12345"
AGENT = f"serviceAccount:service-{NUMBER}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
OPERATOR = f"serviceAccount:agent-runtime@{PROJECT}.iam.gserviceaccount.com"


class FakeGcp:
    """In-memory project state implementing the GcpApi surface audit/apply use."""

    def __init__(self, *, enabled=None, runtime_agent_exists=True, model_ok=True,
                 disabled_models=()):
        self.project = PROJECT
        self.enabled = set(enabled if enabled is not None else ps.REQUIRED_SERVICES)
        self.disabled_models = set(disabled_models)  # model_ok=False disables all
        self.buckets: dict[str, dict] = {}
        self.bucket_policies: dict[str, dict] = {}
        self.service_accounts: dict[str, dict] = {}
        self.sa_policies: dict[str, dict] = {}
        self.project_policy: dict = {"etag": "e", "bindings": []}
        self.runtime_agent_exists = runtime_agent_exists
        self.model_ok = model_ok
        self.identity_generated = 0
        self.calls: list[str] = []

    # service usage
    def enabled_services(self):
        return set(self.enabled), NUMBER

    def enable_services(self, services):
        self.calls.append(f"enable:{','.join(services)}")
        self.enabled.update(services)

    def generate_service_identity(self, service="aiplatform.googleapis.com"):
        self.identity_generated += 1

    # project IAM
    def get_project_policy(self):
        import copy

        return copy.deepcopy(self.project_policy)

    def set_project_policy(self, policy):
        self._reject_missing_members(policy)
        self.project_policy = policy

    # service accounts
    def get_service_account(self, email):
        return self.service_accounts.get(email)

    def create_service_account(self, account_id, display_name):
        email = f"{account_id}@{PROJECT}.iam.gserviceaccount.com"
        self.service_accounts[email] = {"email": email}
        self.sa_policies[email] = {"etag": "e", "bindings": []}
        self.calls.append(f"create-sa:{email}")
        return self.service_accounts[email]

    def get_sa_policy(self, email):
        import copy

        return copy.deepcopy(self.sa_policies[email])

    def set_sa_policy(self, email, policy):
        self.sa_policies[email] = policy

    # storage
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
        import copy

        return copy.deepcopy(self.bucket_policies[name])

    def set_bucket_policy(self, name, policy):
        self._reject_missing_members(policy)
        self.bucket_policies[name] = policy

    # model + identity
    def model_ping(self, model, location="global"):
        if self.model_ok and model not in self.disabled_models:
            return (True, f"{model} ok")
        return (False, "HTTP 403: no access")

    def adc_email(self):
        return "dev@example.com"

    # helpers
    def _reject_missing_members(self, policy):
        """Like GCP: binding a nonexistent service agent is a 400."""
        if self.runtime_agent_exists:
            return
        for b in policy.get("bindings", []):
            if AGENT in b.get("members", []):
                raise ps.GcpError(
                    f"Service account {AGENT.split(':', 1)[1]} does not exist.", status=400
                )


def _by_key(items):
    return {i.key: i for i in items}


# ---------------------------------------------------------------------------------
# audit / apply
# ---------------------------------------------------------------------------------


def test_audit_ready_project_is_all_ok():
    api = FakeGcp()
    cfg = ps.Settings(project=PROJECT)
    staging, output = (ps.bucket_name_of(u) for u in cfg.buckets())
    _, rules = handoff_lifecycle_rules(cfg.buckets()[1])
    api.create_bucket(staging, "us-central1", [])
    api.create_bucket(output, "us-central1", rules)
    api.create_service_account("agent-runtime", "x")
    ps.add_bindings(
        api.project_policy,
        [(r, OPERATOR) for r in ps.OPERATOR_PROJECT_ROLES]
        + [(r, AGENT) for r in ps.RUNTIME_AGENT_PROJECT_ROLES],
    )
    for name in (staging, output):
        ps.add_bindings(api.bucket_policies[name], [(ps.OPERATOR_BUCKET_ROLE, OPERATOR)])
    ps.add_bindings(api.bucket_policies[output], [(ps.RUNTIME_AGENT_BUCKET_ROLE, AGENT)])
    ps.add_bindings(
        api.sa_policies[OPERATOR.split(":", 1)[1]],
        [("roles/iam.serviceAccountTokenCreator", "user:dev@example.com")],
    )

    items = ps.audit(api, cfg)
    assert all(i.status == ps.OK for i in items), [(i.step, i.status, i.detail) for i in items]
    assert ps.exit_code(items) == 0


def test_audit_blocks_everything_behind_missing_gate_apis():
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    items = ps.audit(api, ps.Settings(project=PROJECT))
    assert [(i.step, i.status) for i in items] == [
        ("APIs", ps.FIX),
        ("Claude on Vertex", ps.BLOCKED),  # aiplatform API itself not enabled yet
        ("everything else", ps.BLOCKED),
    ]
    assert "iam.googleapis.com" in items[0].detail


def test_model_check_runs_before_the_gate_when_aiplatform_is_enabled():
    """The expensive-mistake guard: with aiplatform already on (as on a fresh Vertex
    project), the model row must be a REAL probe result in the very first report, even
    while everything else is still blocked behind the other APIs."""
    api = FakeGcp(enabled={"serviceusage.googleapis.com", "aiplatform.googleapis.com"},
                  model_ok=False)
    items = ps.audit(api, ps.Settings(project=PROJECT))
    by = _by_key(items)
    assert by["model:claude-haiku-4-5"].status == ps.MANUAL
    assert "model-garden/claude-haiku-4-5" in by["model:claude-haiku-4-5"].detail
    # an unavailable OPTIONAL model is a NOTE, not a readiness failure
    assert by["model:claude-fable-5"].status == ps.NOTE
    assert by["everything else"].status == ps.BLOCKED


def test_verify_blockers_allow_only_runtime_agent_and_impersonation():
    ok = ps.Item("APIs", ps.OK, "d")
    pending_agent = ps.Item("runtime agent", ps.PENDING, "d")
    manual_imp = ps.Item("impersonation", ps.MANUAL, "d")
    model_manual = ps.Item("Claude on Vertex (claude-haiku-4-5)", ps.MANUAL, "d")
    model_note = ps.Item("Claude on Vertex (claude-fable-5)", ps.NOTE, "d")
    bucket_fix = ps.Item("staging bucket", ps.FIX, "d")
    # the two legitimate leftovers do not block a verify (it resolves/ignores them),
    # and neither does an informational NOTE (an optional model being unavailable) ...
    assert ps.verify_blockers([ok, pending_agent, manual_imp, model_note]) == []
    # ... anything else does — the deploy or the turn would fail after minutes of build
    assert ps.verify_blockers([ok, model_manual]) == [model_manual]
    assert ps.verify_blockers([ok, bucket_fix]) == [bucket_fix]


def test_verify_refuses_to_deploy_with_blockers(capsys):
    """No backend import, no deploy, no spend — just the printed refusal."""
    api = FakeGcp()  # would explode if verify() touched real deploy paths anyway
    items = [ps.Item("Claude on Vertex", ps.MANUAL, "not enabled")]
    ok, out_items = ps.verify(api, ps.Settings(project=PROJECT), items)
    assert ok is False
    assert out_items == items
    assert "NOT deploying" in capsys.readouterr().out


def test_apply_rounds_converge_on_an_empty_project():
    """The signature end-to-end path: empty project -> everything except the two
    irreducible leftovers (runtime agent pending until first deploy) gets applied."""
    api = FakeGcp(enabled={"serviceusage.googleapis.com"}, runtime_agent_exists=False)
    cfg = ps.Settings(project=PROJECT)
    items = ps._apply_rounds(api, cfg, ps.audit(api, cfg))

    by = _by_key(items)
    assert by["APIs"].status == ps.OK
    assert by[f"bucket:{PROJECT}-agent-staging"].status == ps.OK
    assert by[f"bucket:{PROJECT}-agent-output"].status == ps.OK
    assert by["output lifecycle"].status == ps.OK
    assert by["operator SA"].status == ps.OK
    assert by["operator roles"].status == ps.OK
    assert by[f"operator-bucket:{PROJECT}-agent-staging"].status == ps.OK
    assert by[f"operator-bucket:{PROJECT}-agent-output"].status == ps.OK
    assert by["impersonation"].status == ps.OK
    for model in (*ps.DEFAULT_CHECK_MODELS, *ps.DEFAULT_OPTIONAL_MODELS):
        assert by[f"model:{model}"].status == ps.OK
    # the one legitimate leftover: the runtime service agent is created on first deploy
    assert by["runtime agent"].status == ps.PENDING
    assert api.identity_generated  # we did try to provision it ahead of time
    assert ps.exit_code(items) == 2

    # after the "first deploy" creates the agent, a re-run converges to fully OK
    api.runtime_agent_exists = True
    items = ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    assert all(i.status == ps.OK for i in items)
    assert ps.exit_code(items) == 0


def test_apply_is_idempotent_and_additive():
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    # pre-existing unrelated state that must survive untouched
    api.project_policy["bindings"].append({"role": "roles/owner", "members": ["user:boss@x.com"]})
    cfg = ps.Settings(project=PROJECT)
    ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    calls_after_first = list(api.calls)
    policy_after_first = api.get_project_policy()

    items = ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    assert api.calls == calls_after_first  # second run changed nothing
    assert api.get_project_policy() == policy_after_first
    assert all(i.status == ps.OK for i in items)
    assert {"role": "roles/owner", "members": ["user:boss@x.com"]} in api.project_policy[
        "bindings"
    ]


def test_existing_foreign_bucket_is_reported_not_recreated():
    api = FakeGcp()
    cfg = ps.Settings(project=PROJECT)
    # an existing output bucket in another region with its own lifecycle rules
    api.create_bucket(f"{PROJECT}-agent-output", "EU", [
        {"action": {"type": "Delete"}, "condition": {"age": 7, "matchesPrefix": ["tmp/"]}},
    ])
    api.calls.clear()
    items = ps.audit(api, cfg)
    by = _by_key(items)
    assert by[f"bucket:{PROJECT}-agent-output"].status == ps.OK
    assert "location eu" in by[f"bucket:{PROJECT}-agent-output"].detail
    assert by["output lifecycle"].status == ps.FIX
    by["output lifecycle"].fix()
    # existing rule kept, handoff rules appended
    rules = api.buckets[f"{PROJECT}-agent-output"]["lifecycle"]["rule"]
    assert rules[0]["condition"]["matchesPrefix"] == ["tmp/"]
    _, wanted = handoff_lifecycle_rules(cfg.buckets()[1])
    assert len(rules) == 1 + len(wanted)  # counted from handoff.py, not pinned here


def test_required_model_failure_is_manual_with_console_pointer():
    api = FakeGcp(disabled_models={"claude-opus-5"})
    cfg = ps.Settings(project=PROJECT)
    items = ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    by = _by_key(items)
    assert by["model:claude-opus-5"].status == ps.MANUAL
    assert "model-garden/claude-opus-5" in by["model:claude-opus-5"].detail
    assert by["model:claude-haiku-4-5"].status == ps.OK
    assert ps.exit_code(items) == 2  # a REQUIRED model missing -> not ready


def test_optional_model_failure_is_note_and_still_ready():
    api = FakeGcp(enabled={"serviceusage.googleapis.com"},
                  disabled_models={"claude-fable-5"})
    cfg = ps.Settings(project=PROJECT)
    items = ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    by = _by_key(items)
    assert by["model:claude-fable-5"].status == ps.NOTE
    assert "model-garden/claude-fable-5" in by["model:claude-fable-5"].detail
    assert ps.exit_code(items) == 0  # optional -> project still counts as ready
    assert ps.verify_blockers(items) == []  # ... and a --verify would proceed


def test_model_flags_wire_into_settings(monkeypatch):
    seen = {}

    def fake_audit(api, cfg, pending=frozenset()):
        seen["cfg"] = cfg
        return [ps.Item("APIs", ps.OK, "d")]

    monkeypatch.setattr(ps, "GcpApi", lambda project: object())
    monkeypatch.setattr(ps, "audit", fake_audit)
    ps.main(["--project", PROJECT, "--check",
             "--model", "claude-sonnet-5", "--optional-model", "claude-opus-5"])
    assert seen["cfg"].models == ("claude-sonnet-5",)
    assert seen["cfg"].optional_models == ("claude-opus-5",)
    ps.main(["--project", PROJECT, "--check"])
    assert seen["cfg"].models == ps.DEFAULT_CHECK_MODELS
    assert seen["cfg"].optional_models == ps.DEFAULT_OPTIONAL_MODELS
    ps.main(["--project", PROJECT, "--check", "--skip-model-check"])
    assert seen["cfg"].models == ()
    assert seen["cfg"].optional_models == ()


def test_no_operator_sa_skips_sa_steps():
    api = FakeGcp()
    cfg = ps.Settings(project=PROJECT, operator_sa_id=None)
    items = ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    steps = {i.step for i in items}
    assert not any("operator" in s or s == "impersonation" for s in steps)
    assert not api.service_accounts


def test_explicit_impersonators_override_adc_principal():
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    cfg = ps.Settings(project=PROJECT, impersonators=("cto@x.com", "group:eng@x.com"))
    ps._apply_rounds(api, cfg, ps.audit(api, cfg))
    pol = api.sa_policies[OPERATOR.split(":", 1)[1]]
    members = [m for b in pol["bindings"]
               if b["role"] == "roles/iam.serviceAccountTokenCreator" for m in b["members"]]
    assert sorted(members) == ["group:eng@x.com", "user:cto@x.com"]
    assert "user:dev@example.com" not in members


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


def test_cli_check_mode_reports_and_exits_without_changes(monkeypatch, capsys):
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    monkeypatch.setattr(ps, "GcpApi", lambda project: api)
    rc = ps.main(["--project", PROJECT, "--check"])
    assert rc == 2
    assert not api.calls  # audit only — nothing applied
    out = capsys.readouterr().out
    assert "FIX" in out and PROJECT in out


def test_cli_yes_applies_and_reports_ready(monkeypatch, capsys):
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    monkeypatch.setattr(ps, "GcpApi", lambda project: api)
    rc = ps.main(["--project", PROJECT, "--yes"])
    assert rc == 0
    assert f"{PROJECT} is ready" in capsys.readouterr().out


def test_cli_non_tty_without_yes_applies_nothing(monkeypatch, capsys):
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    monkeypatch.setattr(ps, "GcpApi", lambda project: api)
    monkeypatch.setattr(ps.sys.stdin, "isatty", lambda: False)
    rc = ps.main(["--project", PROJECT])
    assert rc == 2
    assert not api.calls
    assert "--yes" in capsys.readouterr().out


def test_cli_prompt_decline_applies_nothing(monkeypatch, capsys):
    api = FakeGcp(enabled={"serviceusage.googleapis.com"})
    monkeypatch.setattr(ps, "GcpApi", lambda project: api)
    monkeypatch.setattr(ps.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    rc = ps.main(["--project", PROJECT])
    assert rc == 2
    assert not api.calls


def test_cli_bootstrap_error_points_at_serviceusage(monkeypatch, capsys):
    class Boom:
        def __init__(self, project):
            pass

        def enabled_services(self):
            raise ps.GcpError(
                "HTTP 403: Service Usage API has not been used in project x before "
                "or it is disabled.",
                status=403,
            )

    monkeypatch.setattr(ps, "GcpApi", Boom)
    rc = ps.main(["--project", PROJECT, "--check"])
    assert rc == 1
    assert "serviceusage.googleapis.com" in capsys.readouterr().err


def test_grant_retries_etag_conflict_then_succeeds():
    state = {"policy": {"etag": "1", "bindings": []}, "sets": 0}

    def get_policy():
        import copy

        return copy.deepcopy(state["policy"])

    def set_policy(policy):
        state["sets"] += 1
        if state["sets"] == 1:
            raise ps.GcpError("HTTP 409: etag mismatch", status=409)
        state["policy"] = policy

    ps._grant(get_policy, set_policy, [("roles/x", "user:a@b.c")])
    assert state["sets"] == 2
    assert state["policy"]["bindings"] == [{"role": "roles/x", "members": ["user:a@b.c"]}]


def test_grant_gives_up_on_missing_member_without_retry_window():
    def set_policy(policy):
        raise ps.GcpError("HTTP 400: Service account sa@x does not exist.", status=400)

    with pytest.raises(ps.GcpError):
        ps._grant(lambda: {"bindings": []}, set_policy, [("roles/x", "serviceAccount:sa@x")])

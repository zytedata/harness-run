# GCP setup and required permissions

Everything below is created in your own GCP project.

> **One command sets all of this up:** `agent-run-gcp-setup --project <your-project>` (installed with the
> library; plain ADC, no gcloud needed) audits a project against everything in this section, shows
> what's missing, asks for confirmation, applies it, and re-audits. It is **additive only** and
> idempotent — safe to run, and re-run, against existing non-empty projects. `--check` audits without
> changing anything (exit 0 iff ready); `--yes` skips the prompt (CI/agents); `--verify` proves the
> end state with a real throwaway deploy (image build + push with your Docker, a pool of one) + one Haiku
> turn (a few cents, ~5 min), and refuses to spend on the deploy while any check it depends on is still
> failing (the model check runs as a 1-token live probe in the first report, so a missing Model Garden
> enablement surfaces before any money is spent). Two things stay manual: enabling Claude in Vertex Model
> Garden (the tool live-probes each model — by default Haiku 4.5, Sonnet 5 and Opus 5 as required plus
> Fable 5 as optional/non-blocking; tune with `--model`/`--optional-model` — and links the exact console
> page for a missing one) and, on a project with APIs fully disabled, the Service Usage API bootstrap.
> The tables below remain the reference for what it grants and why.

Two identities take part; the sandbox itself has none:

**1. The operator identity** — you (a human or CI) *impersonate* the operator service account, or run as
your own identity, to drive the whole control plane: `sandbox.deploy`, `get_engine`, `list_engines`,
running turns. Grants (tighten to your policy):

| Role | Scope | Why |
|---|---|---|
| `roles/aiplatform.user` | project | create/list/delete sandbox templates and sandboxes, execute into them (`aiplatform.sandboxEnvironments.*`, `aiplatform.sandboxEnvironmentTemplates.*`), and the host `reasoningEngine` they hang off |
| `roles/storage.admin` | the output bucket | the records (event mirror, configs, checkpoints), the ready-pool roster, the lifecycle-rule update `deploy` performs, and minting the per-turn run-scoped GCS tokens (a downscoped token can only carry rights its source already has) |
| `roles/artifactregistry.writer` | the image repo | `deploy` pushes the agent image |
| `roles/iam.serviceAccountTokenCreator` | **on the model service account** | every turn mints a one-hour Vertex token for the sandbox by impersonating it |

Why `roles/aiplatform.user` and not a narrower custom role: the platform authorizes template calls with
`aiplatform.sandboxEnvironmentTemplates.{list,get,create,delete}`, permissions that (as of 2026-09-15) are
absent from IAM's public catalog — not testable on the project or on the host reasoning engine, listed by no
predefined role, so a custom role cannot carry them. A custom role with the published
`aiplatform.sandboxEnvironments.*` + `aiplatform.reasoningEngines.{create,get,list}` was tried live and
failed on the first template listing. `roles/aiplatform.user` evidently includes them through an unpublished
grant. Revisit when Agent Sandbox reaches GA.

The principal that impersonates the operator SA needs `roles/iam.serviceAccountTokenCreator` **on that SA**
(and on the model SA, if it drives turns as itself). Impersonation is the pattern for callers that already
have a Google identity (humans, GCP-hosted services, CI with workload identity). A production app running
*outside* GCP instead authenticates **as** the operator SA directly with a service-account **key** stored
in the app's secret store (`gcloud iam service-accounts keys create key.json --iam-account=<operator SA>`,
then point `GOOGLE_APPLICATION_CREDENTIALS` at it). A key is a long-lived credential, so prefer
impersonation or workload identity federation where they're available, and rotate keys you do hand out.

**2. The model service account** — `agent-run-model@<project>.iam.gserviceaccount.com` (created by
`agent-run-gcp-setup`; `sandbox.deploy(model_service_account=)` names another one). The sandbox runs the model
on a token minted for this account, and the agent's shell can read that token, so it holds **only** a custom
role with `aiplatform.endpoints.predict` (`agentRunPredict`) — model calls and nothing else. Never
`roles/aiplatform.user` here: it would hand the shell every sandbox and template in the project.

**Platform side**: the Google-managed **Agent Sandbox service agent**,
`service-<PROJECT_NUMBER>@gcp-sa-vertex-sandbox.iam.gserviceaccount.com`, pulls the agent image when a
sandbox starts and needs `roles/artifactregistry.reader` on the image repo. It gets nothing else; the
sandbox it starts runs as a zero-permission tenant identity.

**Prerequisites** (`agent-run-gcp-setup` creates them; `deploy` ensures the lifecycle rules):

- An output bucket `gs://<project>-agent-output` with uniform bucket-level access.
- An Artifact Registry Docker repo `agent-run` in the location (`sandbox.deploy(image_repo=)` names another).
- **Claude model access** — see the note below.
- The Docker CLI on the deploying machine, logged into the registry.

**Claude model access.** By default the library routes Claude through **Vertex AI** with the model service
account's token. For that to work:

1. **Enable the Claude models you use in Vertex Model Garden** (accept the Anthropic terms once per project).
   Until a model is enabled, a deployed run fails with *"model … may not exist or you may not have access to
   it."*
2. Set `spec.model` to the id shown in Model Garden — for current Claude models that's the family alias (e.g.
   `claude-opus-4-8`), the same form you use locally. The library defaults `CLOUD_ML_REGION` to `global`,
   where these models are served; override `vertex_region` at deploy if you need a specific location.
3. **Long turns need no setup.** IAM mints the model token for one hour, but the sandbox worker serves it to
   Claude Code from a loopback **metadata server** (the CLI's Google auth fetches it as on a VM and refreshes
   it itself when it nears expiry), and the client re-mints a token every 25 minutes and pushes it to the
   worker for as long as the turn runs. The one condition: a client process must hold the session while the
   turn runs — the one that started it, or one that re-attached with `get_session` (see
   [From another process](runs-and-sessions.md#talking-to-a-running-turn-steer-interrupt-stop-exec)) — because the sandbox itself
   cannot mint tokens; a turn whose last client exits keeps running on its current tokens and loses model
   access within the hour. `max_turn_s` (a `deploy` knob, default 8 h) bounds the sandbox's lifetime, not
   the token.

Prefer an API key (e.g. for models you haven't enabled on Vertex)? Deploy with `use_vertex=False` and pass
`ANTHROPIC_API_KEY` as a per-invocation secret — the library then uses the key (no Vertex routing), so any
model alias the key supports works. Note the security trade-off: in API-key mode the agent's process can read
a long-lived key, so prefer Vertex for anything exposed to untrusted input (see
[Secrets & security](secrets-and-security.md)).

The setup tool discovers the ADC principal itself (a plain `gcloud auth application-default login`
included); `--impersonator user:...` only names *other* people to grant. Doing the same by hand, for a
fresh project:

```bash
PROJECT=your-project; REGION=us-central1; NUMBER=$(gcloud projects describe $PROJECT --format='value(projectNumber)')
OP="agent-runtime@$PROJECT.iam.gserviceaccount.com"        # operator SA you create
MODEL="agent-run-model@$PROJECT.iam.gserviceaccount.com"        # model identity you create
OUT="gs://$PROJECT-agent-output"

gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com storage.googleapis.com \
  iamcredentials.googleapis.com --project $PROJECT
gcloud iam service-accounts create agent-runtime --project $PROJECT
gcloud iam service-accounts create agent-run-model --project $PROJECT
gcloud projects add-iam-policy-binding $PROJECT --member "serviceAccount:$OP" --role roles/aiplatform.user
## the model identity reaches Vertex only for model calls: a custom role, never roles/aiplatform.user
gcloud iam roles create agentRunPredict --project $PROJECT --stage GA \
  --title "agent-run model identity: model calls only" --permissions aiplatform.endpoints.predict
gcloud projects add-iam-policy-binding $PROJECT --member "serviceAccount:$MODEL" \
  --role projects/$PROJECT/roles/agentRunPredict
gcloud storage buckets create $OUT --project $PROJECT --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding $OUT --member "serviceAccount:$OP" --role roles/storage.admin
gcloud artifacts repositories create agent-run --repository-format=docker --location=$REGION --project $PROJECT
gcloud artifacts repositories add-iam-policy-binding agent-run --location=$REGION --project $PROJECT \
  --member "serviceAccount:$OP" --role roles/artifactregistry.writer
gcloud artifacts repositories add-iam-policy-binding agent-run --location=$REGION --project $PROJECT \
  --member "serviceAccount:service-$NUMBER@gcp-sa-vertex-sandbox.iam.gserviceaccount.com" \
  --role roles/artifactregistry.reader
## who mints tokens: the operator SA mints model tokens; you impersonate the operator SA
gcloud iam service-accounts add-iam-policy-binding $MODEL --member "serviceAccount:$OP" \
  --role roles/iam.serviceAccountTokenCreator
gcloud iam service-accounts add-iam-policy-binding $OP --member "user:you@org.com" \
  --role roles/iam.serviceAccountTokenCreator
gcloud auth configure-docker $REGION-docker.pkg.dev
```

Then authenticate impersonating the operator SA (`gcloud auth application-default login
--impersonate-service-account=$OP`); `sandbox.deploy` pushes to the `agent-run` repo and mints model tokens
from `$MODEL` by default (pass `image_repo=` / `model_service_account=` for other names).

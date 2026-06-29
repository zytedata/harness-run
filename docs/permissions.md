# Permissions runbook (the two-identity IAM model)

> Status: **placeholder (P0).** Filled in at P2/P3 (DESIGN.md §11). This is where the per-port IAM grants
> live.

There are **two identities** (DESIGN.md §6):

- **Deploy / operator** — the impersonated service account (publish, read logs, submit jobs).
- **Runtime identity** — the RE service agent
  `service-<n>@gcp-sa-aiplatform-re.iam.gserviceaccount.com`. *All* runtime resource access (Secret
  Manager, GCS, Pub/Sub, Logging) authorizes against **it**, not the operator SA.

This runbook will list every grant each port needs and on which identity:

- `BlobStore` (`GcsBlobStore`) → GCS object read/write on the runtime identity.
- `EventSink` (`CloudLoggingSink`) → Logging write (runtime) + read (operator/client).
- `DispatchTransport` (`PubSubDispatch`) → Pub/Sub publish/subscribe on the runtime identity.
- `SecretResolver` (`GcpSecretResolver`) → Secret Manager accessor on the runtime identity.

See [`../DESIGN.md`](../DESIGN.md) §6 ("Identity / IAM").

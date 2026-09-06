# Session submission and cancellation ownership

A setup failure before a dispatch call stops token refresh, attempts staged-secret
deletion and retires any worker claimed by that submission. Config objects remain
as the existing post-mortem record. Cleanup failures do not replace the original
exception or log credential-bearing exception details; lifecycle/pool TTLs remain
backstops, not proof of immediate deletion.

The boundary is deliberately conservative: once `publish` or `run_query_job` is
entered, an exception can mean the server accepted the request but the response
was lost. The client retains handoff credentials and the refresh lease in that
case and refuses another submission through the same handle. Operators must
reconcile the remote job before retrying or terminating the owning client.
Automatic reconciliation, a bounded orphan lease and a durable cross-process
session lock remain separate work; this change does not claim to solve uncertain
remote acceptance. A new handle is not evidence that the old job has stopped.

One handle also refuses overlapping active/pending submissions, preventing local
rollback from deleting another turn's credentials. This is not multi-process
exclusivity or the steering feature proposed in PR #45.

Warm interruption captures the dedicated worker job before local completion
clears its handle. It sends a cancellation request to that exact job. A successful
local cancellation is not confirmation of remote termination; legacy shared
subscription workers still lack a per-turn cancellation target.

The regression tests use fake dispatch/storage, dummy values, and local waiting
tasks only. They do not create or cancel real jobs.

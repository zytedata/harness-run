# Codex checkpoint failure reporting

PR #72, merged in `48d043d88e232c79313be00f908de4d70d09e343`, added the
Codex conversation prefix and validated checkpoint destinations. Those changes
and their existing regression tests remain on main; this follow-up covers the
remaining reliability work in issue #53 and audit B03.

A completed turn can save its workspace while failing to save its conversation.
The harness reports `checkpoint_error` with `conversation_saved=false` and a
separate `workspace_saved` flag when the rollout is absent or a conversation
object cannot be written. This status precedes the terminal result, so callers
that stop reading at that result still see it. The completed turn's result is
preserved; a checkpoint failure does not retroactively fail the work.

On resume, absent metadata can still start a fresh conversation. If metadata
exists but cannot be read or decoded, its thread ID is invalid, or the rollout is
missing or unreadable, the harness raises an explicit error before starting a
model turn. Storage exception text and checkpoint contents are not echoed in
these errors. Local and Gemini runtimes surface harness exceptions through their
normal error handling.

Destination checks continue to use #72's existing validator. An unsafe or
missing rollout path is refused with the existing warning and fresh-conversation
fallback. This PR does not change that behavior, add token permissions, or repeat
the path-containment implementation.

The conversation metadata, rollout and workspace archive are separate writes,
not an atomic checkpoint. This change reports failures; it does not repair
partial or stale checkpoints. The tests use fake Codex transport, dummy
credentials and temporary local storage. Live cloud validation is separate.

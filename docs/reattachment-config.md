# Reattachment configuration failures

Only the BlobStore's known-missing-object result means "no session config".
Permission errors, timeouts and malformed persisted JSON prevent dispatch and do
not mark config resolution complete. A subsequent attempt can retry the read.
Service error details are not included in the new read/decode error messages.

This preserves sessions intentionally created without an overlay. It cannot tell
a never-configured session from one whose previously bound config was deleted;
that requires a separate durable binding/integrity mechanism. A worker-writable
config also remains untrusted policy evidence. No stored configuration is
rewritten, migrated or deleted by this fix.

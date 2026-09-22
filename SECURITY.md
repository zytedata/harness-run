# Security policy

## Reporting a vulnerability

Please do not report security issues through public GitHub issues.

Email konstantin@zyte.com with a description of the issue, the version or commit affected, and steps
to reproduce if you have them. You will get an acknowledgement within a few business days, and we
will keep you informed as the report is investigated and fixed.

## Scope

`agent-run` runs coding agents that execute model-chosen commands, locally or in a remote sandbox.
Reports that are especially welcome:

- a credential reaching a place the documentation says it does not (the agent's environment, an
  event, a checkpoint, a transcript, a log);
- one run being able to read or affect another run's records or sandbox;
- a way for a sandboxed agent to reach resources in the hosting project beyond the model endpoint.

The threat model these claims rest on is written down in
[Secrets and security](docs/secrets-and-security.md).

## Supported versions

Security fixes go into the latest release. Until 1.0, minor releases may contain breaking changes;
see the changelog for upgrade notes.

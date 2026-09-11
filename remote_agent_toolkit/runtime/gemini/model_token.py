"""Model credentials for a sandbox turn: a short-lived OAuth token for Vertex AI.

The sandbox has no usable Google identity (DESIGN.md §13.1), so the client mints the model
token: an access token for a **predict-only service account** the client may impersonate
(``roles/iam.serviceAccountTokenCreator`` on it), handed to the worker with the turn and
consumed by Claude Code as ``ANTHROPIC_AUTH_TOKEN`` with ``CLAUDE_CODE_USE_VERTEX=1`` /
``CLAUDE_CODE_SKIP_VERTEX_AUTH=1`` (verified live 2026-09-10).

Claude Code reads that token once, at start, and does not consult ``apiKeyHelper`` in
this mode (checked 2026-09-11: with no token in the env the CLI hangs on its own auth), so
the token's **lifetime** is what bounds a turn's model access. IAM mints impersonated
tokens for up to an hour by default and up to 12 hours when the organization policy
``constraints/iam.allowServiceAccountCredentialLifetimeExtension`` lists the account:
:func:`mint_model_token` asks for the turn's ceiling and falls back to an hour when IAM
refuses (:func:`mint_turn_model_token`).

SECURITY: the token is worth ``aiplatform.endpoints.predict`` in the project for its
lifetime and nothing else; treat it as a bearer, never log it.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

# The service account ``gemini.deploy`` uses when ``model_service_account=`` is omitted.
# ``ratk-gcp-setup`` creates it with the predict-only custom role.
DEFAULT_MODEL_SA_ID = "ratk-model"

_CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_LIFETIME_S = 3600
MAX_LIFETIME_S = 12 * 3600  # IAM's ceiling with the lifetime-extension org policy


def default_model_service_account(project: str) -> str:
    return f"{DEFAULT_MODEL_SA_ID}@{project}.iam.gserviceaccount.com"


def mint_model_token(
    source_credentials: Any | None, service_account: str, lifetime_s: int = DEFAULT_LIFETIME_S
) -> tuple[str, _dt.datetime | None]:
    """Impersonate ``service_account`` and return ``(access_token, expiry)``.

    ``expiry`` is a naive UTC datetime (google-auth's convention) or ``None``. Raises when
    the source credentials cannot be resolved or IAM refuses the impersonation.
    """
    import google.auth
    from google.auth import impersonated_credentials
    from google.auth.transport.requests import Request

    if source_credentials is None:
        source_credentials, _ = google.auth.default(scopes=[_CLOUD_PLATFORM])
    creds = impersonated_credentials.Credentials(
        source_credentials=source_credentials,
        target_principal=service_account,
        target_scopes=[_CLOUD_PLATFORM],
        lifetime=int(lifetime_s),
    )
    creds.refresh(Request())
    if not creds.token:
        raise RuntimeError("IAM returned no access token for the model service account")
    return creds.token, creds.expiry


def vertex_model_env(token: str, project: str, region: str) -> dict[str, str]:
    """The agent env that routes Claude Code to Vertex with ``token`` as the bearer."""
    return {
        "CLAUDE_CODE_USE_VERTEX": "1",
        "CLAUDE_CODE_SKIP_VERTEX_AUTH": "1",
        "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_VERTEX_PROJECT_ID": project,
        "CLOUD_ML_REGION": region,
    }


def mint_turn_model_token(
    source_credentials: Any | None, service_account: str, max_turn_s: float
) -> tuple[str, _dt.datetime | None, bool]:
    """A model token that outlives a ``max_turn_s`` turn when IAM allows; else an hour.

    Returns ``(token, expiry, extended)``: ``extended`` is ``False`` when the account's
    lifetime cannot be extended beyond the default hour (the org policy is not set for
    it) and the turn's model access therefore ends after an hour — callers surface that.
    """
    wanted = int(min(MAX_LIFETIME_S, max(DEFAULT_LIFETIME_S, max_turn_s + 600)))
    if wanted > DEFAULT_LIFETIME_S:
        try:
            token, expiry = mint_model_token(source_credentials, service_account, wanted)
            return token, expiry, True
        except Exception:  # noqa: BLE001 — the policy is not set: fall back to an hour
            pass
    token, expiry = mint_model_token(source_credentials, service_account, DEFAULT_LIFETIME_S)
    return token, expiry, wanted <= DEFAULT_LIFETIME_S

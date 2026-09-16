"""Model credentials for a sandbox turn: a short-lived OAuth token for Vertex AI.

The sandbox has no usable Google identity (DESIGN.md §13.1), so the client mints the model
token: an access token for a **predict-only service account** the client may impersonate
(``roles/iam.serviceAccountTokenCreator`` on it). The worker does not put the token in the
agent's environment: it serves it from a loopback **metadata server** (``worker.py``), and
Claude Code — pointed at it with ``GCE_METADATA_HOST`` and ``CLAUDE_CODE_USE_VERTEX=1`` —
fetches it the way it would on a Compute Engine VM and fetches it *again* when it nears
expiry (google-auth refreshes a token with less than five minutes left; verified with the
CLI 2026-09-16). So the client keeps minting hourly tokens for as long as the turn runs and
pushes each one to the worker (``/token``, ``backend.GeminiSession._start_refresh``) — no
organization policy, no long-lived credential anywhere.

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
DEFAULT_LIFETIME_S = 3600  # IAM's default ceiling for an impersonated token


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


def token_record(token: str, expiry: _dt.datetime | None) -> dict[str, Any]:
    """The ``model_token`` body the worker takes on ``/turn`` and ``/token``.

    ``expires_at`` is the naive-UTC ISO timestamp (or ``None`` when IAM gave none); the
    worker turns it into the ``expires_in`` the metadata protocol reports.
    """
    return {"access_token": token, "expires_at": expiry.isoformat() if expiry else None}


def vertex_model_env(project: str, region: str) -> dict[str, str]:
    """The agent env that routes Claude Code to Vertex.

    No bearer here: the worker adds ``GCE_METADATA_HOST`` (its metadata server) when the
    turn carries a model token, and the CLI's Google auth fetches the token from it.
    """
    return {
        "CLAUDE_CODE_USE_VERTEX": "1",
        "ANTHROPIC_VERTEX_PROJECT_ID": project,
        "CLOUD_ML_REGION": region,
    }

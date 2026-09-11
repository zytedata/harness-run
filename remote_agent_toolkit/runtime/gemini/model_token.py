"""Model credentials for a sandbox turn: a short-lived OAuth token for Vertex AI.

The sandbox has no usable Google identity (DESIGN.md §13.1), so the client mints the model
token: an access token for a **predict-only service account** the client may impersonate
(``roles/iam.serviceAccountTokenCreator`` on it), handed to the worker with the turn and
consumed by Claude Code as ``ANTHROPIC_AUTH_TOKEN`` with ``CLAUDE_CODE_USE_VERTEX=1`` /
``CLAUDE_CODE_SKIP_VERTEX_AUTH=1`` (verified live 2026-09-10). Its lifetime is an hour
(the IAM default ceiling); a turn that outlives it is refreshed through the run's token
object, see ``scoped_gcs.py`` and the worker.

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

"""Best-effort structured-output parsing (DESIGN.md §5).

When an ``AgentSpec`` sets ``output_schema``, the run result's final text is parsed into
a structured object. The agent is steered (by the harness) to emit a JSON object as its
final message; we extract it from a fenced ```json block, a bare object, or the whole
string, then validate against the schema:

* a **pydantic model** (anything with ``model_validate``) → a validated instance;
* a **JSON-schema dict** → validated with ``jsonschema`` if installed, else passed through;
* anything else → the parsed JSON value.

Parsing is forgiving: on any failure it returns ``None`` (the caller still has the raw
``text``) rather than raising, so a malformed final message never crashes a run.
Stdlib-only at import; ``jsonschema`` is imported lazily and is optional.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def extract_json(text: str) -> Any | None:
    """Pull the most likely JSON value out of an assistant message.

    Tries, in order: a fenced ```json block, then the first balanced ``{...}``/``[...]``
    span, then the whole stripped string. Returns the parsed value or ``None``.
    """
    if not text:
        return None
    m = _FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # First balanced object/array span (greedy from first opener to last matching closer).
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError):
        return None


def parse_structured_output(text: str | None, schema: Any) -> Any | None:
    """Parse ``text`` into a value matching ``schema`` (best effort, never raises).

    ``schema`` may be a pydantic model class, a JSON-schema dict, or ``None`` (in which
    case the parsed JSON is returned as-is). Returns ``None`` if nothing parseable is
    found or validation fails.
    """
    if not text:
        return None
    parsed = extract_json(text)
    if parsed is None:
        return None

    # pydantic model → validated instance.
    model_validate = getattr(schema, "model_validate", None)
    if callable(model_validate):
        try:
            return model_validate(parsed)
        except Exception:  # noqa: BLE001 — validation failure is non-fatal
            return None

    # JSON-schema dict → validate with jsonschema if available, else pass through.
    if isinstance(schema, dict):
        try:
            import jsonschema  # lazy, optional

            jsonschema.validate(parsed, schema)
        except ImportError:
            pass
        except Exception:  # noqa: BLE001 — invalid against schema
            return None
        return parsed

    # No schema (or an unrecognized one): return the parsed JSON.
    return parsed

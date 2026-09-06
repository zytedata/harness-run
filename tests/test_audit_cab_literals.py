"""CAB construction must keep caller-supplied names inside string literals (S07)."""
import json

import pytest

from remote_agent_toolkit.runtime.gemini.scoped_gcs import access_boundary, run_object_prefixes


@pytest.mark.parametrize("session_id", [
    "ordinary-session", 'audit") || true || resource.name.startsWith("',
    'quote"and\\escape', "line\nbreak", "unicode-\u2603-\U0001f600",
])
def test_cab_names_are_encoded_as_literals(session_id):
    bucket, _, prefixes = run_object_prefixes("gs://audit-bucket", session_id)
    for prefix, rule in zip(prefixes, access_boundary(bucket, prefixes).rules):
        obj = f"projects/_/buckets/{bucket}/objects/{prefix}"
        expected = (
            f"resource.name.startsWith({json.dumps(obj, ensure_ascii=False)}) || "
            'api.getAttribute("storage.googleapis.com/objectListPrefix", "")'
            f".startsWith({json.dumps(prefix, ensure_ascii=False)})"
        )
        assert rule.availability_condition.expression == expected


def test_neighboring_session_prefixes_remain_disjoint():
    _, _, own = run_object_prefixes("gs://audit-bucket/prefix", "session-a")
    _, _, other = run_object_prefixes("gs://audit-bucket/prefix", "session-b")
    assert not any(a.startswith(b) or b.startswith(a) for a in own for b in other)

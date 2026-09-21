"""CAB construction must keep caller-supplied names inside string literals (S07)."""
import json

import pytest

from agent_run.runtime.sandbox.scoped_gcs import access_boundary, run_object_prefixes


@pytest.mark.parametrize("session_id", [
    "ordinary-session", 'audit") || true || resource.name.startsWith("',
    'quote"and\\escape', "line\nbreak", "unicode-\u2603-\U0001f600",
])
@pytest.mark.parametrize("output_bucket", ["gs://audit-bucket", 'gs://audit-bucket/prefix"\\quoted'])
def test_cab_names_are_encoded_as_literals(session_id, output_bucket):
    bucket, base, prefixes = run_object_prefixes(output_bucket, session_id)
    listed = {f"{base}checkpoints/sessions/"}
    rules = access_boundary(bucket, prefixes, base).rules
    list_count = 0
    for prefix, rule in zip(prefixes, rules):
        obj = f"projects/_/buckets/{bucket}/objects/{prefix}"
        expected = f"resource.name.startsWith({json.dumps(obj, ensure_ascii=False)})"
        if any(prefix.startswith(start) for start in listed):
            expected += (
                ' || api.getAttribute("storage.googleapis.com/objectListPrefix", "")'
                f".startsWith({json.dumps(prefix, ensure_ascii=False)})"
            )
            list_count += 1
        assert rule.availability_condition.expression == expected
    assert list_count == 1


def test_neighboring_session_prefixes_remain_disjoint():
    _, _, own = run_object_prefixes("gs://audit-bucket/prefix", "session-a")
    _, _, other = run_object_prefixes("gs://audit-bucket/prefix", "session-b")
    assert not any(a.startswith(b) or b.startswith(a) for a in own for b in other)

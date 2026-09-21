"""Prefix mentions are not proof of an effective secret-retention backstop (S15)."""
from copy import deepcopy

import pytest

from remote_agent_toolkit.runtime.gemini.handoff import (
    ensure_handoff_lifecycle,
    handoff_lifecycle_rules,
)


class FakeBucket:
    def __init__(self, rules):
        self.lifecycle_rules = deepcopy(rules)
        self.patches = 0

    def patch(self):
        self.patches += 1


@pytest.mark.parametrize("condition", [
    {"age": 3650}, {"age": 1, "isLive": False},
    {"age": 1, "numNewerVersions": 10}, {"age": 1, "matchesSuffix": [".not-secrets"]},
    {"age": 1, "createdBefore": "2020-01-01"}, {}, {"age": True}, {"age": -1},
])
def test_ineffective_rules_do_not_count_as_coverage(condition):
    _, wanted = handoff_lifecycle_rules("gs://audit-bucket")
    rule = {"action": {"type": "Delete"}, "condition": {
        **condition, "matchesPrefix": ["invocation-secrets/", "session-config/", "turn-config/"],
    }}
    bucket = FakeBucket([rule])
    assert ensure_handoff_lifecycle("gs://audit-bucket", bucket_obj=bucket)
    assert bucket.patches == 1
    assert all(item in bucket.lifecycle_rules for item in wanted)
    assert rule in bucket.lifecycle_rules  # preserve existing policy; append only our backstops
    assert ensure_handoff_lifecycle("gs://audit-bucket", bucket_obj=bucket)
    assert bucket.patches == 1


def test_equal_or_shorter_unconditional_rules_are_preserved():
    _, wanted = handoff_lifecycle_rules("gs://audit-bucket")
    wanted[1]["condition"]["age"] = 7
    bucket = FakeBucket(wanted)
    assert ensure_handoff_lifecycle("gs://audit-bucket", bucket_obj=bucket)
    assert bucket.patches == 0

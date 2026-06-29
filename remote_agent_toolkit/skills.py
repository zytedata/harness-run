"""Discover + provision skills from ``SkillSource[]`` (DESIGN.md §8, §12) — P1.

Resolves git/path/builtin sources, stages them into the per-job cwd, and merges
multiple sources (precedence on name collision / per-source ref pinning is the open
P1 merge-policy decision — DESIGN.md §12). Lifted from the PoC ``skills.py``. Uses git
via ``integrations.git`` lazily. P0: stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spec import SkillSource


def provision(sources: tuple[SkillSource, ...], dest: str) -> list[str]:
    """Resolve & stage ``sources`` into ``dest``, returning staged skill dirs (P1).

    Merge policy for name collisions across sources is the open question in DESIGN.md §12.
    """
    raise NotImplementedError(
        "P1: resolve each SkillSource (git clone / local path / builtin), stage skills "
        "under dest, and merge per the §12 precedence policy; return the staged dirs."
    )

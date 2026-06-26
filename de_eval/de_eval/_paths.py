"""Make the thesis shared libraries importable regardless of cwd.

``de_eval`` reuses ``thesis_lib_shared`` (io + metrics). When run from an
arbitrary directory the package may not already be on ``sys.path``, so we
locate the repo root (``.../TESI/Codice``) relative to this file and insert
the shared-lib locations once, at import time.
"""
from __future__ import annotations

import sys
from pathlib import Path

# .../Codice/MetricDeProof/de_eval/de_eval/_paths.py  ->  .../Codice
REPO_ROOT = Path(__file__).resolve().parents[3]

_SHARED = REPO_ROOT / "thesis_lib_shared"


def ensure_shared_on_path() -> None:
    """Insert thesis_lib_shared on sys.path if it isn't importable yet."""
    try:
        import thesis_lib_shared  # noqa: F401
        return
    except ImportError:
        pass
    for p in (_SHARED, REPO_ROOT):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


ensure_shared_on_path()

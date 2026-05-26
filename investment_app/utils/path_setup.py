"""
Dodaje katalog główny repozytorium do sys.path (Streamlit Cloud / uruchomienie z podfolderu).

Użycie w app.py (po dodaniu repo root lub z poziomu pakietu):
    import investment_app.utils.path_setup  # noqa: F401

Użycie na stronach multipage — najpierw krótki bootstrap, potem import pakietu:
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    import investment_app.utils.path_setup  # noqa: F401
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root() -> Path:
    """Idempotentnie wstawia katalog główny repo na początek sys.path."""
    root = Path(__file__).resolve().parent.parent.parent
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


ensure_repo_root()

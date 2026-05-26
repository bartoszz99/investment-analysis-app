"""
Bootstrap sys.path dla stron multipage (każda strona startuje osobno).

Importuj przed investment_app.*:
    import _path_setup
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ROOT_STR = str(_REPO_ROOT)
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

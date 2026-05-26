"""

Dodaje katalog główny repozytorium do sys.path (Streamlit Cloud / uruchomienie z podfolderu).



Importuj jako pierwszy moduł w app.py:

    import path_setup

"""



from __future__ import annotations



import sys

from pathlib import Path



_REPO_ROOT = Path(__file__).resolve().parent.parent

_ROOT_STR = str(_REPO_ROOT)

if _ROOT_STR not in sys.path:

    sys.path.insert(0, _ROOT_STR)



"""Configuration pytest : src/ importable + accès au bloc de fonctions ajouté dans src/app.py.

app.py est un script Streamlit (il affiche les pages dès qu'on l'importe) : on ne l'importe donc pas.
Les fonctions de calcul se trouvent dans un bloc délimité par deux marqueurs ; la fixture `app_block`
l'exécute isolément, avec les mêmes imports que app.py, et rend ses fonctions testables.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BEGIN, END = "# >>> DÉBUT DU BLOC AJOUTÉ", "# <<< FIN DU BLOC AJOUTÉ"


def extract_block(source: str) -> str:
    lines = source.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(BEGIN))
    stop = next(i for i, l in enumerate(lines) if l.startswith(END))
    return "\n".join(lines[start:stop + 1])


@pytest.fixture(scope="session")
def app_block():
    """Espace de noms contenant les fonctions du bloc ajouté (eda_*, confusion_at, ...)."""
    pytest.importorskip("streamlit")
    import altair as alt
    import numpy as np
    import pandas as pd
    import streamlit as st
    from scipy import stats as scipy_stats
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import roc_auc_score, roc_curve

    ns = dict(alt=alt, np=np, pd=pd, st=st, Path=Path, scipy_stats=scipy_stats, CountVectorizer=CountVectorizer,
              roc_auc_score=roc_auc_score, roc_curve=roc_curve, PROJECT_ROOT=ROOT)
    block = extract_block((ROOT / "src" / "app.py").read_text(encoding="utf-8"))
    exec(compile(block, "src/app.py (bloc ajouté)", "exec"), ns)
    return ns

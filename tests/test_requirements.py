"""requirements.txt doit s'installer partout (Windows, Linux, macOS) : pas de faute de frappe, pas de paquet propre à un système."""
import re

from ml_utils import ROOT

WINDOWS_ONLY = ("colorama", "pywin", "pywinpty", "pypiwin", "win32", "pefile")
REQUIRED = ["streamlit", "pandas", "numpy", "scipy", "scikit-learn", "joblib", "matplotlib", "seaborn", "altair", "xgboost"]


def _names():
    names = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            names.append(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower())
    return names


def test_no_windows_only_packages():
    assert not [n for n in _names() if n.startswith(WINDOWS_ONLY)]


def test_required_packages_are_listed_with_the_right_spelling():
    names = _names()
    missing = [r for r in REQUIRED if r not in names]
    assert not missing, f"paquets manquants ou mal orthographiés : {missing}"


def test_no_deep_learning_library():
    forbidden = {"torch", "tensorflow", "keras", "transformers", "sentence-transformers"}
    assert not forbidden.intersection(_names()), "Deep Learning / LLM interdits comme modèles ML (cahier des charges)"

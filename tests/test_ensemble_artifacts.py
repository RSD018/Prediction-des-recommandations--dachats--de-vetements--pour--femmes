"""Cohérence des fichiers produits par les notebooks 04 et 05 (modèles d'ensemble, analyse d'erreurs)."""
import pandas as pd
import pytest

from ml_utils import ROOT, load_clean_data, split_data

MODELS = ROOT / "models"
SCORES = MODELS / "ensemble_test_scores.csv"
pytestmark = pytest.mark.skipif(not SCORES.exists(), reason="exécuter les notebooks 04 et 05")


def test_ensemble_scores_cover_exactly_the_test_set():
    """Les scores des modèles d'ensemble doivent porter sur le même jeu de test (split 80/20, seed 42)."""
    _, X_test, _, y_test = split_data(load_clean_data())
    scores = pd.read_csv(SCORES).set_index("row_id")
    assert set(scores.index) == set(X_test.index)
    assert (scores.loc[y_test.index, "y_true"].values == y_test.values).all()


@pytest.mark.parametrize("model", ["AdaBoost", "XGBoost", "Soft Voting (NB + RF)"])
def test_ensemble_scores_are_probabilities(model):
    scores = pd.read_csv(SCORES)[model]
    assert scores.between(0, 1).all()


def test_saved_ensemble_models_load():
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("xgboost")
    for name in ["adaboost_model.pkl", "xgboost_model.pkl"]:
        assert hasattr(joblib.load(MODELS / name), "predict")

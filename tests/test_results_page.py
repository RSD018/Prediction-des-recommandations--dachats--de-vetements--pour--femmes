"""Fonctions de calcul des analyses détaillées (bloc ajouté dans src/app.py)."""
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT


def test_confusion_counts_add_up_and_match_hand_computation(app_block):
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([0.9, 0.6, 0.2, 0.7, 0.4, 0.1])
    m = app_block["confusion_at"](y, s, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 2)
    assert m["tp"] + m["fn"] + m["fp"] + m["tn"] == len(y)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["recall_1"] == pytest.approx(2 / 3) and m["precision_1"] == pytest.approx(2 / 3)
    assert m["recall_0"] == pytest.approx(2 / 3)


def test_extreme_thresholds_do_not_divide_by_zero(app_block):
    y = np.array([1, 0, 1, 0])
    s = np.array([0.8, 0.3, 0.6, 0.2])
    all_yes = app_block["confusion_at"](y, s, -1.0)
    all_no = app_block["confusion_at"](y, s, 2.0)
    assert all_yes["recall_1"] == 1.0 and all_yes["recall_0"] == 0.0 and all_yes["f1_0"] == 0.0
    assert all_no["recall_1"] == 0.0 and all_no["precision_1"] == 0.0 and all_no["f1_1"] == 0.0


def test_natural_threshold_is_zero_for_margins_and_half_for_probabilities(app_block):
    assert app_block["natural_threshold"]("Linear SVC") == 0.0
    assert app_block["natural_threshold"]("XGBoost") == 0.5


def test_scores_file_matches_the_team_numbers(app_block):
    """Le seuil naturel doit redonner la matrice de confusion publiée par l'équipe (models/ensemble_class_metrics.csv)."""
    data = app_block["load_test_scores"](ROOT)
    assert data is not None and {"y_true", "Linear SVC", "AdaBoost"} <= set(data.columns)
    ref = pd.read_csv(ROOT / "models" / "ensemble_class_metrics.csv").set_index("Model")
    for model in ("Linear SVC", "AdaBoost", "XGBoost"):
        m = app_block["confusion_at"](data["y_true"], data[model], app_block["natural_threshold"](model))
        assert (m["tn"], m["fp"], m["fn"], m["tp"]) == tuple(int(ref.loc[model, c]) for c in ("TN", "FP", "FN", "TP")), model

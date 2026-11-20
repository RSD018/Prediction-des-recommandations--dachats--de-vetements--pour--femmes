"""Tests du prétraitement et de l'évaluation (src/ml_utils.py)."""
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from ml_utils import (
    CLEAN_PATH, N_SPLITS, TARGET, build_preprocessor, evaluate_model, get_cv, load_clean_data, split_data,
)

pytestmark = pytest.mark.skipif(not CLEAN_PATH.exists(), reason="reviews_clean.csv absent")


@pytest.fixture(scope="module")
def small_split():
    df = load_clean_data().sample(2500, random_state=0)
    return split_data(df)


def test_split_is_stratified_and_80_20():
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)
    assert abs(len(X_test) / len(df) - 0.20) < 0.01
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert "Clothing ID" not in X_train.columns and TARGET not in X_train.columns


def test_split_is_reproducible():
    df = load_clean_data()
    a = split_data(df)[0].index
    b = split_data(df)[0].index
    assert list(a) == list(b)


def test_cross_validation_uses_at_least_5_folds():
    assert N_SPLITS >= 5
    assert get_cv().get_n_splits() >= 5


def test_preprocessor_has_same_columns_on_train_and_test(small_split):
    X_train, X_test, _, _ = small_split
    prep = build_preprocessor()
    A = prep.fit_transform(X_train)
    B = prep.transform(X_test)
    assert A.shape[1] == B.shape[1] == len(prep.get_feature_names_out())


def test_minmax_preprocessor_is_non_negative(small_split):
    X_train, X_test, _, _ = small_split
    prep = build_preprocessor("minmax")
    A = prep.fit_transform(X_train)
    assert A.min() >= 0  # exigé par MultinomialNB


def test_ablation_without_rating_removes_the_feature(small_split):
    X_train, _, _, _ = small_split
    prep = build_preprocessor(drop_columns=("Rating",))
    prep.fit(X_train)
    assert not any("Rating" in n for n in prep.get_feature_names_out())


def test_text_only_and_meta_only_variants(small_split):
    X_train, _, _, _ = small_split
    text_only = build_preprocessor(use_meta=False).fit(X_train).get_feature_names_out()
    meta_only = build_preprocessor(use_text=False).fit(X_train).get_feature_names_out()
    assert all("tfidf" in n for n in text_only)
    assert not any("tfidf" in n for n in meta_only)


def test_evaluate_model_returns_all_metrics(small_split):
    X_train, X_test, y_train, y_test = small_split
    prep = build_preprocessor()
    A, B = prep.fit_transform(X_train), prep.transform(X_test)
    metrics = evaluate_model(DummyClassifier(strategy="most_frequent").fit(A, y_train), B, y_test)
    for key in ["Accuracy", "Precision (1)", "Recall (1)", "F1 (1)", "F1 (0)", "F1 macro", "ROC-AUC", "PR-AUC"]:
        assert key in metrics and not np.isnan(metrics[key])
    assert metrics["F1 (0)"] == 0.0  # Zero-R ne détecte jamais la classe minoritaire

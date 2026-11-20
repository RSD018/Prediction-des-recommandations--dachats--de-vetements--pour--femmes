"""Utilitaires ML partagés par les notebooks, les tests et l'application.

Emplacement : src/ml_utils.py

Ce module reproduit EXACTEMENT le découpage et le prétraitement du notebook 02
(split 80/20 stratifié, random_state=42, mêmes transformateurs) afin que les
notebooks 06 et 07 n'aient plus besoin des fichiers .npz ignorés par Git.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "reviews.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "reviews_clean.csv"

TARGET = "Recommended IND"
NUMERIC = ["Age", "Rating", "Positive Feedback Count"]
CATEGORICAL = ["Division Name", "Department Name", "Class Name"]
TITLE = "Title"
REVIEW = "Review Text"

RANDOM_STATE = 42
N_SPLITS = 6  # exigence du cahier des charges : K >= 5 ; 6 plis = même valeur que les notebooks 03 et 04


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
def load_raw_data(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Dataset brut (23 486 lignes, 11 colonnes dont l'index 'Unnamed: 0')."""
    return pd.read_csv(path)


def load_clean_data(path: Path | str = CLEAN_PATH) -> pd.DataFrame:
    """Dataset nettoyé par le notebook 01 (mêmes options que le notebook 02)."""
    return pd.read_csv(path, keep_default_na=False)


def split_data(df: pd.DataFrame, test_size: float = 0.20):
    """Split stratifié identique au notebook 02 → X_train, X_test, y_train, y_test."""
    X = df.drop(columns=[TARGET, "Clothing ID"], errors="ignore")
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def get_cv(n_splits: int = N_SPLITS) -> StratifiedKFold:
    """Validation croisée stratifiée K-Fold (K >= 5) reproductible."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------- #
# Prétraitement
# --------------------------------------------------------------------------- #
def build_preprocessor(
    scaler: str = "standard",
    drop_columns: tuple[str, ...] = (),
    use_meta: bool = True,
    use_text: bool = True,
) -> ColumnTransformer:
    """ColumnTransformer du notebook 02, avec options pour l'ablation.

    scaler       : "standard" (Z-score, notebook 02) ou "minmax" (valeurs >= 0,
                   nécessaire pour MultinomialNB sans « clipper » les négatifs).
    drop_columns : colonnes à retirer (ex. ("Rating",) pour tester la fuite).
    use_meta     : garde les variables numériques + catégorielles.
    use_text     : garde les deux TF-IDF (Title, Review Text).
    """
    scaler_obj = StandardScaler() if scaler == "standard" else MinMaxScaler()
    numeric = [c for c in NUMERIC if c not in drop_columns]
    categorical = [c for c in CATEGORICAL if c not in drop_columns]

    transformers = []
    if use_meta and numeric:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scaler_obj)]),
                numeric,
            )
        )
    if use_meta and categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    if use_text and TITLE not in drop_columns:
        transformers.append(
            (
                "title_tfidf",
                TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=3000),
                TITLE,
            )
        )
    if use_text and REVIEW not in drop_columns:
        transformers.append(
            (
                "review_tfidf",
                TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=10000),
                REVIEW,
            )
        )
    if not transformers:
        raise ValueError("Aucune variable sélectionnée pour le prétraitement.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #
def get_scores(model, X):
    """Score continu de la classe 1 : probabilité si disponible, sinon marge SVM."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate_scores(y_true, y_pred, scores=None) -> dict:
    """Métriques multicritères (critère 1.5 du barème) à partir de prédictions et d'un score continu.

    Le F1 « classe 1 » seul est trompeur sur un jeu 82/18 : on rapporte donc aussi
    le F1 de la classe 0 (minoritaire) et le F1 macro.
    """
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (1)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (1)": recall_score(y_true, y_pred, zero_division=0),
        "F1 (1)": f1_score(y_true, y_pred),
        "F1 (0)": f1_score(y_true, y_pred, pos_label=0),
        "F1 macro": f1_score(y_true, y_pred, average="macro"),
        "ROC-AUC": roc_auc_score(y_true, scores) if scores is not None else np.nan,
        "PR-AUC": average_precision_score(y_true, scores) if scores is not None else np.nan,
    }
    return {k: float(v) for k, v in out.items()}


def evaluate_model(model, X_test, y_test) -> dict:
    """Même chose, à partir d'un modèle scikit-learn déjà entraîné."""
    return evaluate_scores(y_test, model.predict(X_test), get_scores(model, X_test))

"""Utilitaires ML partagés par les notebooks, les tests et l'application.

Emplacement : src/ml_utils.py

Ce module reproduit EXACTEMENT le découpage et le prétraitement du notebook 02
(split 80/20 stratifié, random_state=42, mêmes transformateurs) afin que les
notebooks 06 et 07 n'aient plus besoin des fichiers .npz ignorés par Git.
"""
from __future__ import annotations
import re
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
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



# =========================================================================== #
# NLP des avis (sentiment + taille) et recommandation hybride  -> notebook 09
# =========================================================================== #
RATING = "Rating"
RECO_PATH = ROOT / "models" / "hybrid_reco.pkl"
NLP_PIPE_PATH = ROOT / "models" / "nlp_hybrid_pipeline.pkl"

# Expressions régulières (les avis sont en anglais)
_HEIGHT = re.compile(r"\b([456])\s*(?:'|’|ft\.?|feet|foot)\s*(\d{1,2})?(?!\d)")
_SIZE_LETTER = re.compile(
    r"\b(?:size|sz)\s+(xxs|xs|s|m|l|xl|xxl|small|medium|large)\b|\b(xxs|xs|xl|xxl)\b"
)
_SIZE_NUM = re.compile(r"\b(?:size|sz)\s+(\d{1,2})\s*p?\b")
_PETITE = re.compile(r"\bpetites?\b")
_RUNS_SMALL = re.compile(
    r"run[s]?\s+(?:a\s+)?(?:bit\s+|little\s+|very\s+|quite\s+)?small|too small|too tight|size up|sized up|went up a size"
)
_RUNS_LARGE = re.compile(
    r"run[s]?\s+(?:a\s+)?(?:bit\s+|little\s+|very\s+|quite\s+)?(?:big|large)|too (?:big|large|loose)|size down|sized down|went down a size"
)
_TTS = re.compile(r"true to size|\btts\b|fits? perfectly")

SIZE_ORDER = {"xxs": 0, "xs": 1, "s": 2, "small": 2, "m": 3, "medium": 3, "l": 4, "large": 4, "xl": 5, "xxl": 6}
SIZE_LABELS = ["XXS", "XS", "S", "M", "L", "XL", "XXL"]
SIZE_FEATURE_NAMES = ["height_cm", "size_ord", "size_num", "petite", "runs_small", "runs_large", "true_to_size"]


def join_text(df: pd.DataFrame) -> pd.Series:
    """Titre + corps de l'avis, en minuscules."""
    return (df[TITLE].fillna("").astype(str) + " " + df[REVIEW].fillna("").astype(str)).str.lower()


def extract_size_features(text: str) -> dict:
    """Attributs de taille d'UN avis (NaN si l'information n'est pas mentionnée)."""
    t = text.lower()
    out = {k: np.nan for k in SIZE_FEATURE_NAMES}
    m = _HEIGHT.search(t)
    if m:
        feet, inches = int(m.group(1)), int(m.group(2) or 0)
        if inches <= 11:
            out["height_cm"] = round((feet * 12 + inches) * 2.54, 1)
    m = _SIZE_LETTER.search(t)
    if m:
        out["size_ord"] = SIZE_ORDER[(m.group(1) or m.group(2))]
    m = _SIZE_NUM.search(t)
    if m and int(m.group(1)) <= 24:
        out["size_num"] = int(m.group(1))
    out["petite"] = float(bool(_PETITE.search(t)))
    out["runs_small"] = float(bool(_RUNS_SMALL.search(t)))
    out["runs_large"] = float(bool(_RUNS_LARGE.search(t)))
    out["true_to_size"] = float(bool(_TTS.search(t)))
    return out


def size_frame(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame (une ligne par avis) des attributs de taille."""
    return pd.DataFrame([extract_size_features(t) for t in join_text(df)], index=df.index, columns=SIZE_FEATURE_NAMES)


class SizeFeatures(BaseEstimator, TransformerMixin):
    """Transformateur sans apprentissage : texte -> 7 attributs de taille (NaN gardés pour l'imputer du Pipeline)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return size_frame(X).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(SIZE_FEATURE_NAMES)


class OOFSentimentScorer(BaseEstimator, TransformerMixin):
    """Score de sentiment issu du texte seul (TF-IDF n-grammes + régression logistique).

    Polarité : Rating >= 4 -> positif (lue dans X, donc uniquement sur les lignes d'apprentissage lors d'un fit).
    Sortie : 1 colonne = logit du score. En fit_transform le score est calculé HORS-PLI (anti-fuite).
    """

    def __init__(self, ngram_range=(1, 3), min_df=3, max_features=60000, C=2.0, n_splits=5, random_state=42):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.C = C
        self.n_splits = n_splits
        self.random_state = random_state

    def _make_model(self) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=self.ngram_range, min_df=self.min_df,
                                      max_features=self.max_features, sublinear_tf=True)),
            ("lr", LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced")),
        ])

    @staticmethod
    def polarity(X) -> np.ndarray:
        return (X[RATING].to_numpy() >= 4).astype(int)

    def fit(self, X, y=None):
        self.model_ = self._make_model().fit(join_text(X), self.polarity(X))
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Score HORS-PLI sur le train (anti-fuite), puis modèle final ajusté sur tout le train."""
        text, pol = join_text(X), self.polarity(X)
        cv = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        oof = cross_val_predict(self._make_model(), text, pol, cv=cv, method="decision_function")
        self.model_ = self._make_model().fit(text, pol)
        return oof.reshape(-1, 1)

    def transform(self, X):
        return self.model_.decision_function(join_text(X)).reshape(-1, 1)

    def score_texts(self, texts) -> np.ndarray:
        """P(positif) pour une liste de textes (utile dans l'application)."""
        return self.model_.predict_proba(pd.Series(texts).str.lower())[:, 1]

    def top_terms(self, k: int = 15):
        """Termes (n-grammes) les plus positifs / négatifs."""
        vec, lr = self.model_.named_steps["tfidf"], self.model_.named_steps["lr"]
        names = np.array(vec.get_feature_names_out())
        order = np.argsort(lr.coef_[0])
        return names[order[::-1][:k]].tolist(), names[order[:k]].tolist()

    def get_feature_names_out(self, input_features=None):
        return np.array(["sentiment_logit"])


def sentiment_label(logit: float) -> str:
    p = 1 / (1 + np.exp(-logit))
    return "positif" if p >= 0.65 else "négatif" if p <= 0.35 else "mitigé"


# --- Recommandation hybride : caractéristiques « profil vs avis du produit » ---
K = 5.0  # force de l'a priori (équivalent à 5 avis « moyens »)
BANDWIDTH = {"age": 8.0, "height_cm": 6.0, "size_ord": 0.75}
FEATURES = ["log_n", "prior_logit", "rating_mean", "sent_mean", "runs_small", "runs_large",
            "age_delta", "age_w", "height_delta", "height_w", "size_delta", "size_w", "has_height", "has_size"]

MORPHO_KEYWORDS = {  # profil morphologique (FR) -> mots-clés anglais présents dans les avis
    "Silhouette sablier": "hourglass curvy waist",
    "Silhouette poire (hanches marquées)": "pear hips wide hips curvy hips",
    "Silhouette pomme (ventre)": "tummy belly stomach forgiving flowy",
    "Silhouette rectangle / athlétique": "straight athletic boyish rectangle",
    "Poitrine généreuse": "bust busty large bust chest",
    "Poitrine menue": "small bust flat chested small chested",
    "Épaules larges": "broad shoulders wide shoulders",
    "Buste court": "short torso short waisted",
    "Buste long": "long torso long waisted",
    "Bras forts": "arms upper arms sleeves",
    "Petite taille (jambes courtes)": "petite short length hem",
    "Grande taille (jambes longues)": "tall long length leggy",
}
STYLE_KEYWORDS = {
    "Coupe ajustée": "fitted tailored slim form fitting",
    "Coupe ample / fluide": "loose flowy relaxed oversized comfortable",
    "Élégant / soirée": "elegant dressy formal wedding party",
    "Quotidien / décontracté": "casual everyday comfortable soft",
    "Chaud / hiver": "warm cozy winter thick",
    "Léger / été": "lightweight summer breezy airy",
}


def _kernel_rate(vals, y, v, bw, prior):
    ok = ~np.isnan(vals)
    if np.isnan(v) or not ok.any():
        return prior, 0.0
    w = np.exp(-0.5 * ((vals[ok] - v) / bw) ** 2)
    return (float(np.sum(w * y[ok])) + K * prior) / (float(w.sum()) + K), float(w.sum())


def profile_features(P: dict, profile: dict, p0: float, rating0: float, exclude: int | None = None) -> np.ndarray:
    """Caractéristiques pour (produit P, profil). `exclude` = avis écarté (leave-one-out à l'apprentissage)."""
    n = len(P["y"])
    mask = np.ones(n, bool)
    if exclude is not None:
        mask[exclude] = False
    a = {k: v[mask] for k, v in P.items()}
    m = int(mask.sum())
    prior = (a["y"].sum() + K * p0) / (m + K)
    f = {"log_n": np.log1p(m), "prior_logit": np.log(prior / (1 - prior)),
         "rating_mean": (a["rating"].sum() + K * rating0) / (m + K), "sent_mean": a["sent"].sum() / (m + K),
         "runs_small": a["rs"].sum() / (m + K), "runs_large": a["rl"].sum() / (m + K)}
    for dim, short in (("age", "age"), ("height_cm", "height"), ("size_ord", "size")):
        rate, w = _kernel_rate(a[dim], a["y"], profile.get(dim, np.nan), BANDWIDTH[dim], prior)
        f[f"{short}_delta"], f[f"{short}_w"] = rate - prior, np.log1p(w)
    f["has_height"] = float(not np.isnan(profile.get("height_cm", np.nan)))
    f["has_size"] = float(not np.isnan(profile.get("size_ord", np.nan)))
    return np.array([f[c] for c in FEATURES], dtype=float)


def product_arrays(rev: pd.DataFrame) -> dict:
    """{Clothing ID -> tableaux numpy} pour un accès rapide."""
    out = {}
    for pid, g in rev.groupby("Clothing ID", sort=False):
        out[pid] = {"y": g["y"].to_numpy(float), "age": g["Age"].to_numpy(float), "height_cm": g["height_cm"].to_numpy(float),
                    "size_ord": g["size_ord"].to_numpy(float), "sent": g["sent"].to_numpy(float),
                    "rating": g["Rating"].to_numpy(float), "rs": g["runs_small"].to_numpy(float),
                    "rl": g["runs_large"].to_numpy(float), "idx": g.index.to_numpy()}
    return out


def ece(y, p, bins=10) -> float:
    """Expected Calibration Error."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def keywords_from_profile(silhouette: list[str], styles: list[str], free_text: str = "") -> str:
    parts = [MORPHO_KEYWORDS[s] for s in silhouette if s in MORPHO_KEYWORDS] + [STYLE_KEYWORDS[s] for s in styles if s in STYLE_KEYWORDS]
    return " ".join(parts + [free_text])


def fit_hint(row) -> str:
    rs, rl = row["tombe_petit"], row["tombe_grand"]
    if rs >= 0.10 and rs > 1.5 * rl:
        return "Tend à tailler petit : envisager la taille au-dessus"
    if rl >= 0.10 and rl > 1.5 * rs:
        return "Tend à tailler grand : envisager la taille en dessous"
    return "Taille plutôt standard d'après les avis"


def recommend(art: dict, profile: dict, query: str = "", alpha: float = 0.7, top_n: int = 6, min_reviews: int = 5,
              departments: list[str] | None = None, classes: list[str] | None = None) -> pd.DataFrame:
    """profile = {'age', 'height_cm' | nan, 'size_ord' | nan}. Top_n produits triés par score hybride."""
    cat = art["catalog"].copy()
    keep = cat["n_avis"] >= min_reviews
    if departments:
        keep &= cat["Departement"].isin(departments)
    if classes:
        keep &= cat["Classe"].isin(classes)
    cat = cat[keep]
    if cat.empty:
        return cat
    X = np.vstack([profile_features(art["arrays"][pid], profile, art["p0"], art["rating0"]) for pid in cat.index])
    cat["P_calibree"] = art["model"].predict_proba(X)[:, 1]
    n = cat["n_avis"].to_numpy()
    se = np.sqrt(cat["P_calibree"] * (1 - cat["P_calibree"]) / (n + K))
    cat["P_bas"], cat["P_haut"] = (cat["P_calibree"] - 1.64 * se).clip(0, 1), (cat["P_calibree"] + 1.64 * se).clip(0, 1)
    if query.strip():
        pos = {p: i for i, p in enumerate(art["pids"])}
        sims = (art["centroids"][[pos[p] for p in cat.index]] @ art["vectorizer"].transform([query.lower()]).T).toarray().ravel()
        cat["sim_texte"] = sims / sims.max() if sims.max() > 0 else 0.0
    else:
        cat["sim_texte"] = 0.0
        alpha = 1.0
    cat["score_hybride"] = alpha * cat["P_calibree"] + (1 - alpha) * cat["sim_texte"]
    cat["conseil_taille"] = cat.apply(fit_hint, axis=1)
    return cat.sort_values("score_hybride", ascending=False).head(top_n)


def best_snippets(art: dict, df_clean: pd.DataFrame, pid, query: str, k: int = 2, max_chars: int = 220) -> list[str]:
    """Avis positifs du produit les plus proches des mots-clés de la cliente."""
    idx = art["arrays"][pid]["idx"]
    sub = df_clean.iloc[idx]
    sub = sub[(sub["Rating"] >= 4) & (sub["Review Text"].str.len() > 40)]
    if sub.empty:
        return []
    if query.strip():
        M = art["vectorizer"].transform(join_text(sub))
        s = (M @ art["vectorizer"].transform([query.lower()]).T).toarray().ravel()
        sub = sub.iloc[np.argsort(-s)]
    return [(t[:max_chars] + "…") if len(t) > max_chars else t for t in sub["Review Text"].head(k)]


def load_reco_artifacts():
    """Artefacts du recommandeur (créés par le notebook 09) ou None."""
    return joblib.load(RECO_PATH) if RECO_PATH.exists() else None
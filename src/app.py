import html as _html
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from scipy import stats as scipy_stats
from sklearn.feature_extraction.text import CountVectorizer


try:
    for _k, _v in {
        "theme.base": "light",
        "theme.primaryColor": "#D81B60",
        "theme.backgroundColor": "#FFF7FA",
        "theme.secondaryBackgroundColor": "#FCE4EC",
        "theme.textColor": "#2D1420",
    }.items():
        st._config.set_option(_k, _v)
except Exception:
    pass

st.set_page_config(
    page_title="Prédiction des recommandations d'achats",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def h(s: str) -> str:
    """Aplatit le HTML : évite que Markdown le prenne pour un bloc de code."""
    return " ".join(line.strip() for line in s.strip().splitlines() if line.strip())


def esc(s: str) -> str:
    """Échappe le texte saisi par l'utilisateur."""
    return _html.escape(str(s))


DEPARTMENTS = {
    "Dresses": ["Dresses"],
    "Tops": ["Knits", "Blouses", "Sweaters", "Fine gauge"],
    "Bottoms": ["Pants", "Jeans", "Skirts", "Shorts"],
    "Jackets": ["Jackets", "Outerwear"],
    "Intimate": ["Lounge", "Sleep", "Swim", "Intimates", "Legwear"],
    "Trend": ["Trend"],
}

DIVISIONS = {
    "Dresses": ["General", "General Petite"],
    "Tops": ["General", "General Petite"],
    "Bottoms": ["General", "General Petite"],
    "Jackets": ["General", "General Petite"],
    "Intimate": ["Initmates", "General"],
    "Trend": ["General", "General Petite"],
}

DEPT_LABELS = {
    "Dresses": "Robes",
    "Tops": "Hauts",
    "Bottoms": "Bas",
    "Jackets": "Vestes & Manteaux",
    "Intimate": "Lingerie & Détente",
    "Trend": "Tendances",
}

CLASS_LABELS = {
    "Dresses": "Robes",
    "Knits": "Mailles",
    "Blouses": "Blouses",
    "Sweaters": "Pulls",
    "Fine gauge": "Maille fine",
    "Pants": "Pantalons",
    "Jeans": "Jeans",
    "Skirts": "Jupes",
    "Shorts": "Shorts",
    "Jackets": "Vestes",
    "Outerwear": "Manteaux",
    "Lounge": "Loungewear",
    "Sleep": "Nuit",
    "Swim": "Maillots de bain",
    "Intimates": "Lingerie",
    "Legwear": "Collants & Chaussettes",
    "Trend": "Pièces tendance",
}

DIV_LABELS = {
    "General": "Collection Classique",
    "General Petite": "Collection Petite",
    "Initmates": "Collection Intime",
}

DEPT_EMOJI = {
    "Dresses": "👗",
    "Tops": "👚",
    "Bottoms": "👖",
    "Jackets": "🧥",
    "Intimate": "🩱",
    "Trend": "✨",
}


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #FFF7FA; }
.block-container { padding-top: 2.5rem; max-width: 1200px; }

header[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], [data-testid="stToolbarActions"],
[data-testid="stDecoration"], #MainMenu, footer { display: none !important; }

.ticker {
    background: linear-gradient(90deg, #880E4F 0%, #E91E63 50%, #880E4F 100%);
    color: #FFFFFF; overflow: hidden; white-space: nowrap;
    border-radius: 10px; padding: 9px 0; font-size: 0.75rem; letter-spacing: 2.5px;
    box-shadow: 0 4px 14px rgba(216, 27, 96, 0.25);
}
.ticker-track { display: inline-block; animation: scroll 32s linear infinite; }
.ticker-track span { margin: 0 34px; }
@keyframes scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }

.navbar { text-align: center; padding: 18px 8px 12px 8px; border-bottom: 1px solid #F2C9D8; }
.brand {
    font-family: 'Playfair Display', serif; font-size: 1.85rem; color: #2D1420;
    letter-spacing: 3px; line-height: 1.25;
}
.brand small { display: block; font-family: 'Inter', sans-serif; font-size: 0.75rem;
    letter-spacing: 5px; color: #D81B60; margin-top: 8px; font-weight: 600; }
.tagline { text-align: center; color: #8A6B78; font-size: 0.95rem; margin: 12px 0 18px 0; }

.section-title {
    font-family: 'Playfair Display', serif; font-size: 1.35rem; color: #2D1420;
    border-bottom: 2px solid #F06292; padding-bottom: 6px; margin: 4px 0 14px 0;
}
div[data-testid="stForm"] {
    background: #FFFFFF; border: 1px solid #F2C9D8; border-radius: 16px;
    padding: 1.5rem; box-shadow: 0 6px 22px rgba(216, 27, 96, 0.08);
}
.product-chip {
    display: inline-block; background: #FCE4EC; border: 1px solid #F4A8C0;
    color: #4A1330; border-radius: 30px; padding: 7px 16px; font-size: 0.86rem;
    margin: 2px 0 18px 0;
}
.hint { font-size: 0.8rem; color: #8A6B78; margin: -6px 0 10px 0; }

.metric-card {
    background: #FFFFFF; border: 1px solid #F2C9D8; border-radius: 14px;
    padding: 1.2rem; text-align: center; box-shadow: 0 4px 12px rgba(216, 27, 96, 0.05);
}
.metric-val { font-family: 'Playfair Display', serif; font-size: 1.8rem; color: #D81B60; font-weight: 700; }
.metric-lbl { font-size: 0.82rem; color: #6B4A58; font-weight: 600; margin-top: 4px; }

div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #D81B60 0%, #F06292 100%); color: #FFFFFF;
    border: none; border-radius: 10px; padding: 0.75rem 1.5rem; width: 100%;
    letter-spacing: 1.5px; font-weight: 600; transition: all 0.2s ease;
    box-shadow: 0 6px 16px rgba(216, 27, 96, 0.28);
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(90deg, #AD1457 0%, #D81B60 100%); color: #FFFFFF;
    transform: translateY(-1px); box-shadow: 0 8px 20px rgba(216, 27, 96, 0.38);
}

.result-card { border-radius: 16px; padding: 1.6rem; text-align: center; border: 1px solid; }
.result-yes { background: #EEF6EE; border-color: #8FBF8F; color: #1F5A2B; }
.result-no  { background: #FBEDED; border-color: #D69A9A; color: #7A2222; }
.result-icon { font-size: 2.4rem; }
.result-title { font-family: 'Playfair Display', serif; font-size: 1.6rem; margin: 0.3rem 0; }
.result-sub { font-size: 0.95rem; opacity: 0.85; }
.conf-track { background: rgba(0,0,0,0.08); border-radius: 20px; height: 10px; margin-top: 14px; }
.conf-fill { height: 10px; border-radius: 20px; }
.fill-yes { background: #4C9A5B; }
.fill-no { background: #C25555; }
.pills { display: flex; gap: 10px; margin-top: 12px; }
.pill { flex: 1; background: #FFFFFF; border: 1px solid #F2C9D8; border-radius: 12px;
    padding: 10px; text-align: center; font-size: 0.8rem; color: #6B4A58; }
.pill b { display: block; font-size: 1.15rem; color: #2D1420; font-family: 'Playfair Display', serif; }

.review-preview { background: #FFFFFF; border: 1px dashed #F06292; border-radius: 14px;
    padding: 1rem 1.2rem; margin-top: 14px; }
.review-stars { color: #F2A900; font-size: 1.2rem; letter-spacing: 2px; }
.review-title { font-weight: 600; color: #2D1420; margin: 4px 0; }
.review-body { color: #5A4250; font-size: 0.92rem; font-style: italic; }
.review-meta { color: #8A6B78; font-size: 0.8rem; margin-top: 6px; }
.info-card { background:#FFFFFF; border:1px solid #F2C9D8; border-left:5px solid #D81B60;
    border-radius:12px; padding:1rem 1.2rem; margin-bottom:12px; color:#2D1420; font-size:0.93rem; }
.info-card b { color:#880E4F; }
.badge { display:inline-block; background:#FCE4EC; color:#880E4F; border:1px solid #F4A8C0;
    border-radius:20px; padding:3px 12px; font-size:0.78rem; margin:2px 4px 2px 0; font-weight:600; }
.disclaimer { background:#FFF8E1; border:1px solid #F2C94C; border-radius:12px; padding:0.9rem 1.2rem;
    color:#6B5200; font-size:0.88rem; margin-top:10px; }
.placeholder-box { background: #FFFFFF; border: 1px solid #F2C9D8; border-radius: 16px;
    padding: 2.5rem 1.5rem; text-align: center; color: #8A6B78; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FDE7EF 0%, #F9C6D8 100%) !important;
    border-right: 1px solid #F4A8C0;
}
section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] { background: transparent !important; }
section[data-testid="stSidebar"] * { color: #3B1A2A; }
.side-brand { font-family: 'Playfair Display', serif; font-size: 1.35rem; line-height: 1.25;
    color: #880E4F !important; margin-bottom: 4px; }
.side-sub { font-size: 0.72rem; letter-spacing: 3px; color: #D81B60 !important;
    margin-bottom: 18px; font-weight: 600; }
.side-title { font-family: 'Playfair Display', serif; font-size: 1.05rem; margin: 20px 0 10px 0;
    color: #880E4F !important; border-bottom: 1px solid #F06292; padding-bottom: 5px; }
.side-card { background: #FFFFFF; border: 1px solid #F4A8C0; border-radius: 12px;
    padding: 12px 14px; font-size: 0.84rem; line-height: 1.5; }
.step { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; font-size: 0.85rem; }
.step-num { min-width: 24px; height: 24px; border-radius: 50%; background: #D81B60;
    color: #FFFFFF !important; font-size: 0.75rem; display: flex; align-items: center;
    justify-content: center; margin-top: 1px; }
.side-foot { font-size: 0.72rem; color: #880E4F !important; margin-top: 22px; opacity: 0.8; }

/* ===== STYLE FÉMININ RAFFINÉ ===== */
:root { --rose:#D81B60; --rose-soft:#F06292; --blush:#FCE4EC; --border:#F2C9D8; --ink:#2D1420; --gold:#C9A227; }
::selection { background:#F8BBD0; color:#2D1420; }
::-webkit-scrollbar { width:10px; } ::-webkit-scrollbar-thumb { background:#F4A8C0; border-radius:10px; }
::-webkit-scrollbar-track { background:#FFF0F5; }

.stApp { background: radial-gradient(circle at 10% 6%, #FFE3EE 0, transparent 36%),
    radial-gradient(circle at 93% 92%, #FBD5E4 0, transparent 38%), #FFF7FA; }
.block-container { animation: fadeUp 0.6s ease both; }
@keyframes fadeUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }

.navbar { background: linear-gradient(135deg, #FFFFFF 0%, #FDE7EF 100%); border:1px solid var(--border);
    border-radius:20px; margin-top:14px; padding:26px 12px 20px 12px; box-shadow:0 10px 30px rgba(216,27,96,0.10); }
.orn { color:var(--gold); letter-spacing:12px; font-size:0.95rem; margin-bottom:6px; }
.brand { background: linear-gradient(90deg, #880E4F, #D81B60, #880E4F); -webkit-background-clip:text;
    background-clip:text; -webkit-text-fill-color:transparent; font-weight:700; }
.brand small { -webkit-text-fill-color:#C9A227; }
.tagline { font-style:italic; }

.section-title { border-bottom:none; padding-bottom:10px; position:relative;
    background: linear-gradient(90deg, #F06292, rgba(240,98,146,0)) bottom left / 100% 2px no-repeat; }

.metric-card { border-top:4px solid var(--rose-soft); transition: transform .25s ease, box-shadow .25s ease; }
.metric-card:hover { transform:translateY(-4px); box-shadow:0 14px 28px rgba(216,27,96,0.16); }
.metric-val { background: linear-gradient(90deg, #AD1457, #F06292); -webkit-background-clip:text;
    background-clip:text; -webkit-text-fill-color:transparent; }
.info-card { transition: transform .25s ease, box-shadow .25s ease; box-shadow:0 4px 14px rgba(216,27,96,0.06); }
.info-card:hover { transform:translateY(-2px); box-shadow:0 10px 22px rgba(216,27,96,0.13); }

div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="textarea"] {
    border-radius:12px !important; border-color:#F4A8C0 !important; background:#FFFFFF !important; }
div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"]:focus-within,
div[data-baseweb="textarea"]:focus-within { border-color:var(--rose) !important; box-shadow:0 0 0 3px rgba(216,27,96,0.15) !important; }
label p, .stSelectbox label, .stNumberInput label { color:#4A1330 !important; font-weight:600 !important; }
div[data-testid="stSlider"] [role="slider"] { background:var(--rose) !important; box-shadow:0 0 0 4px rgba(216,27,96,0.18); }

div[data-testid="stForm"] { background: linear-gradient(180deg, #FFFFFF 0%, #FFF9FC 100%); border-radius:20px; }
div[data-testid="stFormSubmitButton"] > button { border-radius:999px; text-transform:uppercase; font-size:0.85rem; }
div[data-testid="stFormSubmitButton"] > button:active { transform:scale(0.98); }

.product-chip { background: linear-gradient(90deg, #FCE4EC, #FFFFFF); box-shadow:0 2px 8px rgba(216,27,96,0.08); }
.result-card { box-shadow:0 12px 30px rgba(0,0,0,0.07); animation: pop .5s ease both; }
.result-yes { background: linear-gradient(135deg, #F4FAF4, #DFF0E2); }
.result-no  { background: linear-gradient(135deg, #FEF3F3, #F8DEDE); }
@keyframes pop { from { opacity:0; transform:scale(.96); } to { opacity:1; transform:scale(1); } }
.conf-fill { transition: width 1s ease; }
.pill { box-shadow:0 3px 10px rgba(216,27,96,0.06); }
.review-preview { background: linear-gradient(180deg, #FFFFFF, #FFF9FC); box-shadow:0 4px 14px rgba(216,27,96,0.07); }
.placeholder-box { background: linear-gradient(180deg, #FFFFFF, #FFF3F8); border-style:dashed; }

div[data-testid="stExpander"] { background:#FFFFFF; border:1px solid var(--border) !important; border-radius:16px;
    box-shadow:0 4px 14px rgba(216,27,96,0.06); }
div[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:16px; overflow:hidden;
    box-shadow:0 4px 14px rgba(216,27,96,0.06); }
div[data-testid="stAlert"] { border-radius:14px; }

section[data-testid="stSidebar"] div[role="radiogroup"] { gap:6px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label { background:rgba(255,255,255,0.75);
    border:1px solid #F4A8C0; border-radius:14px; padding:9px 14px; margin:0; width:100%;
    transition: all .2s ease; cursor:pointer; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background:#FFFFFF;
    transform:translateX(4px); box-shadow:0 4px 12px rgba(216,27,96,0.18); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display:none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, #D81B60, #F06292); border-color:#D81B60;
    box-shadow:0 6px 16px rgba(216,27,96,0.32); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * { color:#FFFFFF !important; font-weight:600; }
.side-card { background: rgba(255,255,255,0.85); box-shadow:0 3px 10px rgba(136,14,79,0.10); }

.site-foot { text-align:center; color:#8A6B78; font-size:0.78rem; letter-spacing:3px; margin:48px 0 8px 0;
    padding-top:16px; border-top:1px solid var(--border); }
.site-foot span { color:var(--gold); }
</style>
"""

st.markdown(h(CSS), unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = Path(__file__).resolve().parent / "models"


@st.cache_resource
def load_resources():
    model_path = MODELS_DIR / "best_model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    if model_path.exists() and preprocessor_path.exists():
        return joblib.load(model_path), joblib.load(preprocessor_path)
    return None, None


model, preprocessor = load_resources()

# ==============================================================================
# >>> DÉBUT DU BLOC AJOUTÉ : tableau de bord statistique + analyses détaillées
# Uniquement des définitions (aucune page n'est affichée ici) : les pages « Statistiques du Dataset » et
# « Performance des Modèles » les appellent plus bas. Les noms sont préfixés eda_ / _res_ / DASH_ pour ne
# jamais entrer en collision avec les variables des autres pages (ex. `cramers_v` et `fmt_p` existent déjà
# dans la page « Tests d'hypothèses »). Les données lues : data/, models/ensemble_test_scores.csv, reports/, docs/figures/.
# ==============================================================================

# --- Tableau de bord statistique interactif (page « Statistiques du Dataset ») ---
EDA_TARGET = "Recommended IND"
EDA_LABEL = "Recommandation"
EDA_YES, EDA_NO = "Recommandé", "Non recommandé"
EDA_NUMERIC = ["Age", "Rating", "Positive Feedback Count", "Review Length"]
EDA_CATEGORICAL = ["Department Name", "Division Name", "Class Name"]
DASH_PINK, DASH_PURPLE, DASH_GREY = "#D81B60", "#6A1B9A", "#8A6B78"
EDA_COLOR = alt.Scale(domain=[EDA_YES, EDA_NO], range=[DASH_PINK, DASH_PURPLE])


# --------------------------------------------------------------------------- #
# Fonctions pures (testables sans Streamlit)
# --------------------------------------------------------------------------- #
def eda_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la longueur des avis (en mots) et une étiquette lisible de la cible."""
    d = df.copy()
    words = d["Review Text"].astype(str).str.split().str.len().fillna(0)
    d["Review Length"] = words.astype(int)
    d[EDA_LABEL] = d[EDA_TARGET].map({1: EDA_YES, 0: EDA_NO})
    return d


def eda_wilson_interval(successes, n, z: float = 1.96):
    """Intervalle de confiance de Wilson pour une proportion (robuste aux petits effectifs)."""
    successes, n = np.asarray(successes, dtype=float), np.asarray(n, dtype=float)
    n_safe = np.where(n == 0, 1, n)
    p = successes / n_safe
    denom = 1 + z**2 / n_safe
    centre = (p + z**2 / (2 * n_safe)) / denom
    half = z * np.sqrt(p * (1 - p) / n_safe + z**2 / (4 * n_safe**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def eda_tukey_summary(series: pd.Series) -> dict:
    """Résumé de Tukey : quartiles, moustaches (dernières valeurs dans les clôtures) et valeurs atypiques."""
    s = series.dropna()
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = s[(s >= low_fence) & (s <= high_fence)]
    return {
        "q1": float(q1), "median": float(med), "q3": float(q3),
        "low": float(inside.min()), "high": float(inside.max()),
        "low_fence": float(low_fence), "high_fence": float(high_fence),
        "n_outliers": int(len(s) - len(inside)), "n": int(len(s)),
    }


def eda_histogram_frame(d: pd.DataFrame, var: str, bins: int, by_target: bool, clip_p99: bool) -> pd.DataFrame:
    """Histogramme pré-calculé : une ligne par (classe, intervalle).

    Variable discrète à peu de valeurs (ex. Rating = 1..5) : une barre par valeur, centrée dessus.
    """
    values = d[var]
    groups = [(EDA_YES, d[d[EDA_TARGET] == 1]), (EDA_NO, d[d[EDA_TARGET] == 0])] if by_target else [("Tous les avis", d)]
    discrete = values.nunique() <= 12 and bool(np.all(values.dropna() == np.round(values.dropna())))
    if discrete:
        levels = np.sort(values.dropna().unique())
        edges = np.concatenate([levels - 0.45, [levels[-1] + 0.45]])
        rows = []
        for name, sub in groups:
            counts = sub[var].value_counts().reindex(levels, fill_value=0).to_numpy()
            rows.append(pd.DataFrame({"start": levels - 0.45, "end": levels + 0.45, "Effectif": counts, EDA_LABEL: name}))
        return pd.concat(rows, ignore_index=True)

    upper = values.quantile(0.99) if clip_p99 else values.max()
    lo = float(values.min())
    hi = float(upper) if float(upper) > lo else lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    rows = []
    for name, sub in groups:
        counts, _ = np.histogram(sub[var][sub[var] <= upper], bins=edges)
        rows.append(pd.DataFrame({"start": edges[:-1], "end": edges[1:], "Effectif": counts, EDA_LABEL: name}))
    return pd.concat(rows, ignore_index=True)


def eda_category_table(d: pd.DataFrame, var: str) -> pd.DataFrame:
    """Effectif, taux de recommandation et intervalle de Wilson à 95 % par modalité."""
    g = d.groupby(var)[EDA_TARGET].agg(["sum", "count"]).reset_index()
    g.columns = [var, "Recommandés", "Effectif"]
    low, high = eda_wilson_interval(g["Recommandés"], g["Effectif"])
    g["Taux (%)"] = (g["Recommandés"] / g["Effectif"] * 100).round(1)
    g["IC95 bas (%)"] = (low * 100).round(1)
    g["IC95 haut (%)"] = (high * 100).round(1)
    return g.sort_values("Effectif", ascending=False).reset_index(drop=True)


def eda_cramers_v(table: pd.DataFrame) -> tuple[float, float, float, int]:
    """χ² d'indépendance + V de Cramér (taille d'effet). Retourne (χ², p, V, ddl)."""
    chi2, p, dof, _ = scipy_stats.chi2_contingency(table.to_numpy())
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if n and k > 0 else float("nan")
    return float(chi2), float(p), v, int(dof)


def eda_mann_whitney(d: pd.DataFrame, var: str) -> dict | None:
    """Mann-Whitney U (recommandé vs non recommandé) + corrélation bisériale par rangs."""
    a = d.loc[d[EDA_TARGET] == 1, var].dropna()
    b = d.loc[d[EDA_TARGET] == 0, var].dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    u, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 2 * u / (len(a) * len(b)) - 1  # >0 : les « recommandé » ont des valeurs plus élevées
    return {"U": float(u), "p": float(p), "r": float(r), "n1": len(a), "n0": len(b),
            "median1": float(a.median()), "median0": float(b.median())}


def eda_fmt_p(p: float) -> str:
    """Affichage d'une p-value sans « 0.00e+00 » trompeur quand elle sous-dépasse la précision machine."""
    return "< 1e-300" if p < 1e-300 else f"{p:.2e}"


def eda_effect_label(value: float, kind: str = "r") -> str:
    """Seuils usuels de Cohen (0,1 / 0,3 / 0,5) pour r de corrélation par rangs et le V de Cramér."""
    v = abs(value)
    return "négligeable" if v < 0.1 else "faible" if v < 0.3 else "moyen" if v < 0.5 else "fort"


@st.cache_data(show_spinner="Analyse du vocabulaire…")
def eda_discriminating_words(texts: pd.Series, y: pd.Series, ngram: int, min_count: int, top: int):
    """Mots/expressions les plus associés à « recommandé » ou « non recommandé » (log-odds lissé, classés par z).

    Présence/absence du mot dans l'avis (document frequency), pas de fréquence brute :
    un mot répété 10 fois dans un seul avis ne domine pas le classement.
    """
    vec = CountVectorizer(stop_words="english", ngram_range=(1, ngram), min_df=min_count, binary=True, max_features=20000)
    X = vec.fit_transform(texts.astype(str))
    yes = np.asarray(y == 1)
    n1, n0 = int(yes.sum()), int((~yes).sum())
    if n1 == 0 or n0 == 0:
        return None
    df1 = np.asarray(X[yes].sum(axis=0)).ravel()
    df0 = np.asarray(X[~yes].sum(axis=0)).ravel()
    a = 0.5
    log_odds = np.log((df1 + a) / (n1 - df1 + a)) - np.log((df0 + a) / (n0 - df0 + a))
    # écart-type du log-odds ratio : on classe par z = log-odds / écart-type pour privilégier les mots à la fois
    # discriminants ET assez fréquents (un mot vu 30 fois ne doit pas dominer un mot vu 3 000 fois)
    se = np.sqrt(1 / (df1 + a) + 1 / (n1 - df1 + a) + 1 / (df0 + a) + 1 / (n0 - df0 + a))
    out = pd.DataFrame({"Mot": vec.get_feature_names_out(), "Avis « recommandé »": df1, "Avis « non recommandé »": df0,
                        "Log-odds": log_odds.round(3), "z": (log_odds / se).round(2)})
    out[EDA_LABEL] = np.where(out["Log-odds"] > 0, EDA_YES, EDA_NO)
    pos = out.nlargest(top, "z")
    neg = out.nsmallest(top, "z")
    return pd.concat([pos, neg]).reset_index(drop=True), out


# --------------------------------------------------------------------------- #
# Filtres
# --------------------------------------------------------------------------- #
def _eda_filters(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    st.markdown("##### 🎛️ Filtres — ils pilotent tous les onglets")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        depts = sorted(d["Department Name"].unique())
        chosen = st.multiselect("Département", depts, default=depts, key="eda_dept")
    with c2:
        lo, hi = int(d["Age"].min()), int(d["Age"].max())
        age = st.slider("Âge", lo, hi, (lo, hi), key="eda_age")
    with c3:
        rating = st.slider("Note", 1, 5, (1, 5), key="eda_rating")
    with c4:
        reco = st.selectbox("Recommandation", ["Toutes", EDA_YES, EDA_NO], key="eda_reco")

    out = d[d["Department Name"].isin(chosen) & d["Age"].between(*age) & d["Rating"].between(*rating)]
    if reco != "Toutes":
        out = out[out[EDA_LABEL] == reco]
    active = []
    if len(chosen) != len(depts):
        active.append(f"départements : {', '.join(chosen) if chosen else 'aucun'}")
    if age != (lo, hi):
        active.append(f"âge {age[0]}–{age[1]}")
    if rating != (1, 5):
        active.append(f"note {rating[0]}–{rating[1]}")
    if reco != "Toutes":
        active.append(reco.lower())
    return out, active


# --------------------------------------------------------------------------- #
# Onglets
# --------------------------------------------------------------------------- #
def _eda_tab_overview(d: pd.DataFrame) -> None:
    st.markdown("#### Statistiques descriptives")
    desc = d[EDA_NUMERIC].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["asymétrie"] = d[EDA_NUMERIC].skew()
    desc["aplatissement"] = d[EDA_NUMERIC].kurt()
    desc["% atypiques (Tukey)"] = [
        round(eda_tukey_summary(d[c])["n_outliers"] / len(d) * 100, 1) for c in EDA_NUMERIC
    ]
    desc = desc.rename(columns={"count": "n", "mean": "moyenne", "std": "écart-type", "25%": "Q1", "50%": "médiane", "75%": "Q3"})
    st.dataframe(desc.round(2))
    skewed = desc["asymétrie"].abs().idxmax()
    st.caption(
        f"Lecture : la variable la plus asymétrique est **{skewed}** (asymétrie = {desc.loc[skewed, 'asymétrie']:.2f}). "
        "Une forte asymétrie et des valeurs atypiques justifient les tests non paramétriques et la normalisation avant les modèles."
    )
    with st.expander("Variables catégorielles"):
        st.dataframe(d[EDA_CATEGORICAL].describe().T.rename(columns={"count": "n", "unique": "modalités", "top": "modalité la plus fréquente", "freq": "effectif"}))
    with st.expander("Aperçu et export des données filtrées"):
        cols = ["Clothing ID", "Age", "Rating", EDA_LABEL, "Positive Feedback Count", "Department Name", "Class Name", "Title", "Review Text"]
        st.dataframe(d[[c for c in cols if c in d.columns]].head(200), hide_index=True)
        st.download_button("⬇️ Télécharger la sélection (CSV)", d.drop(columns=[EDA_LABEL]).to_csv(index=False).encode("utf-8"),
                           file_name="avis_filtres.csv", mime="text/csv")


def _eda_tab_quality(raw: pd.DataFrame | None, clean: pd.DataFrame) -> None:
    st.markdown("#### Qualité des données (avant nettoyage)")
    src = raw if raw is not None else clean.replace("", np.nan)
    src = src.drop(columns=[c for c in ["Unnamed: 0"] if c in src.columns])
    miss = src.isna().sum().rename("Manquantes").to_frame()
    miss["% du dataset"] = (miss["Manquantes"] / len(src) * 100).round(2)
    miss = miss.reset_index().rename(columns={"index": "Variable"})

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.dataframe(miss, hide_index=True)
        n_dup = int(len(src) - len(clean))
        st.metric("Lignes supprimées au nettoyage", f"{n_dup:,}".replace(",", " "),
                  help="Doublons exacts et lignes à cible manquante retirés lors du nettoyage.")
    with c2:
        chart = (
            alt.Chart(miss).mark_bar(color=DASH_PINK)
            .encode(x=alt.X("% du dataset:Q", title="% de valeurs manquantes"),
                    y=alt.Y("Variable:N", sort="-x", title=None),
                    tooltip=["Variable", "Manquantes", "% du dataset"])
            .properties(height=280)
        )
        st.altair_chart(chart)
    worst = miss.sort_values("Manquantes", ascending=False).iloc[0]
    st.caption(
        f"Calculé sur le dataset brut ({len(src):,} lignes). Variable la plus touchée : **{worst['Variable']}** "
        f"({worst['% du dataset']:.1f} %). Les textes manquants sont remplacés par une chaîne vide, les catégories par « Unknown »."
        .replace(",", " ")
    )


def _eda_tab_distributions(d: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    with c1:
        var = st.selectbox("Variable numérique", EDA_NUMERIC, key="eda_num_var")
    with c2:
        bins = st.slider("Classes (bins)", 5, 60, 25, key="eda_bins", disabled=(var == "Rating"), help="Sans effet pour Rating : une barre par note.")
    with c3:
        by_target = st.checkbox("Séparer par recommandation", value=True, key="eda_by_target")
    with c4:
        clip = st.checkbox("Couper au 99ᵉ centile", value=var in ("Positive Feedback Count", "Review Length"), key="eda_clip")
    log_y = st.checkbox("Axe vertical logarithmique", value=False, key="eda_logy")

    hist = eda_histogram_frame(d, var, bins, by_target, clip)
    y_scale = alt.Scale(type="symlog") if log_y else alt.Scale(zero=True)
    hist["base"] = 0
    base = alt.Chart(hist).mark_bar(opacity=0.8 if by_target else 1)
    enc = dict(
        x=alt.X("start:Q", title=var), x2="end:Q",
        y=alt.Y("Effectif:Q", stack=None, scale=y_scale, title="Effectif"), y2="base:Q",
        tooltip=[EDA_LABEL, alt.Tooltip("start:Q", title="de", format=".1f"), alt.Tooltip("end:Q", title="à", format=".1f"), "Effectif"],
    )
    enc["color"] = alt.Color(f"{EDA_LABEL}:N", scale=EDA_COLOR) if by_target else alt.value(DASH_PINK)
    st.altair_chart(base.encode(**enc).properties(height=340, title=f"Distribution de « {var} »"))

    # Boîte à moustaches de Tukey par classe, calculée à la main (aucune donnée brute envoyée)
    rows = []
    for name, code in ((EDA_YES, 1), (EDA_NO, 0)):
        sub = d.loc[d[EDA_TARGET] == code, var]
        if len(sub):
            rows.append({EDA_LABEL: name, **eda_tukey_summary(sub)})
    if rows:
        box = pd.DataFrame(rows)
        y = alt.Y(f"{EDA_LABEL}:N", title=None, axis=alt.Axis(labelLimit=220))
        whisk = alt.Chart(box).mark_rule(strokeWidth=2).encode(y=y, x=alt.X("low:Q", title=var), x2="high:Q", color=alt.Color(f"{EDA_LABEL}:N", scale=EDA_COLOR, legend=None))
        body = alt.Chart(box).mark_bar(size=26, opacity=0.7).encode(y=y, x="q1:Q", x2="q3:Q", color=alt.Color(f"{EDA_LABEL}:N", scale=EDA_COLOR, legend=None),
                                                                      tooltip=[EDA_LABEL, "q1", "median", "q3", "n_outliers"])
        mid = alt.Chart(box).mark_tick(color="white", thickness=3, size=26).encode(y=y, x="median:Q")
        st.altair_chart((whisk + body + mid).properties(height=230, title="Boîte à moustaches de Tukey (clôtures à 1,5 × IQR)"))
        out_txt = " · ".join(f"{r[EDA_LABEL]} : {r['n_outliers']} valeur(s) atypique(s) sur {r['n']}" for r in rows)
        st.caption(f"Médianes — " + " ; ".join(f"{r[EDA_LABEL]} = {r['median']:g}" for r in rows) + f". {out_txt}.")


def _eda_tab_categories(d: pd.DataFrame) -> None:
    var = st.selectbox("Variable catégorielle", EDA_CATEGORICAL, key="eda_cat_var")
    tab = eda_category_table(d, var)
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(
            alt.Chart(tab).mark_bar(color=DASH_PINK)
            .encode(x=alt.X("Effectif:Q"), y=alt.Y(f"{var}:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=200)),
                    tooltip=[var, "Effectif", "Taux (%)"])
            .properties(height=max(200, 44 * len(tab) + 60), title="Fréquence des modalités")
        )
    with c2:
        y = alt.Y(f"{var}:N", sort=alt.EncodingSortField("Effectif", order="descending"), title=None,
                  axis=alt.Axis(labelOverlap=False, labelLimit=200))
        bars = alt.Chart(tab).mark_bar(color=DASH_PURPLE, opacity=0.8).encode(
            x=alt.X("Taux (%):Q", scale=alt.Scale(domain=[0, 100]), title="Taux de recommandation (%)"), y=y,
            tooltip=[var, "Effectif", "Taux (%)", "IC95 bas (%)", "IC95 haut (%)"])
        ci = alt.Chart(tab).mark_rule(color="#2D1420", strokeWidth=2).encode(y=y, x="IC95 bas (%):Q", x2="IC95 haut (%):Q")
        st.altair_chart((bars + ci).properties(height=max(200, 44 * len(tab) + 60), title="Taux de recommandation et IC à 95 % (Wilson)"))
    st.dataframe(tab, hide_index=True)
    wide = tab[tab["Effectif"] >= 30]
    if len(wide) >= 2:
        best, worst = wide.loc[wide["Taux (%)"].idxmax()], wide.loc[wide["Taux (%)"].idxmin()]
        st.caption(
            f"Lecture : parmi les modalités ayant au moins 30 avis, **{best[var]}** est la plus recommandée ({best['Taux (%)']:.1f} %) "
            f"et **{worst[var]}** la moins recommandée ({worst['Taux (%)']:.1f} %). Une barre d'erreur très large signale un échantillon trop petit pour conclure."
        )

    st.markdown("#### Note donnée × recommandation")
    ct = pd.crosstab(d["Rating"], d[EDA_LABEL]).reindex(columns=[EDA_YES, EDA_NO], fill_value=0)
    long = ct.reset_index().melt(id_vars="Rating", var_name=EDA_LABEL, value_name="Effectif")
    long["% de la note"] = (long["Effectif"] / long.groupby("Rating")["Effectif"].transform("sum").replace(0, np.nan) * 100).round(1)
    st.altair_chart(
        alt.Chart(long).mark_bar().encode(
            x=alt.X("Rating:O", title="Note", axis=alt.Axis(labelAngle=0)), y=alt.Y("Effectif:Q", stack="normalize", axis=alt.Axis(format="%"), title="Part des avis"),
            color=alt.Color(f"{EDA_LABEL}:N", scale=EDA_COLOR), tooltip=["Rating", EDA_LABEL, "Effectif", "% de la note"],
        ).properties(height=260)
    )


def _eda_tab_correlations(d: pd.DataFrame) -> None:
    method = st.radio("Méthode", ["pearson", "spearman"], horizontal=True, key="eda_corr_method",
                      format_func=lambda m: {"pearson": "Pearson (linéaire)", "spearman": "Spearman (rangs)"}[m])
    cols = ["Age", "Rating", "Positive Feedback Count", "Review Length", EDA_TARGET]
    if len(d) < 3 or d[cols].nunique().min() < 2:
        st.info("Pas assez de variabilité dans la sélection pour calculer des corrélations (élargissez les filtres).")
        return
    corr = d[cols].corr(method=method)
    long = corr.reset_index().melt(id_vars="index")
    long.columns = ["x", "y", "corr"]
    base = alt.Chart(long).encode(x=alt.X("x:N", title=None, axis=alt.Axis(labelLimit=200)), y=alt.Y("y:N", title=None, axis=alt.Axis(labelLimit=200)))
    heat = base.mark_rect().encode(color=alt.Color("corr:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue"), title="r"),
                                   tooltip=["x", "y", alt.Tooltip("corr:Q", format=".3f")])
    text = base.mark_text(fontSize=13).encode(text=alt.Text("corr:Q", format=".2f"),
                                              color=alt.condition(alt.datum.corr > 0.6, alt.value("white"), alt.value("black")))
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.altair_chart((heat + text).properties(height=360))
    with c2:
        tgt = corr[EDA_TARGET].drop(EDA_TARGET).rename("r").reset_index().rename(columns={"index": "Variable"})
        tgt["|r|"] = tgt["r"].abs()
        st.altair_chart(
            alt.Chart(tgt).mark_bar().encode(
                x=alt.X("r:Q", scale=alt.Scale(domain=[-1, 1]), title=f"Corrélation avec « {EDA_TARGET} »"),
                y=alt.Y("Variable:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
                color=alt.condition(alt.datum.r > 0, alt.value(DASH_PINK), alt.value(DASH_PURPLE)),
                tooltip=["Variable", alt.Tooltip("r:Q", format=".3f")],
            ).properties(height=220)
        )
        top = tgt.sort_values("|r|", ascending=False).iloc[0]
        st.caption(f"**{top['Variable']}** est la variable la plus liée à la recommandation (r = {top['r']:.2f}). "
                   "Une corrélation forte avec la cible est un signal prédictif… mais aussi un risque de fuite si la variable est connue trop tard.")


def _eda_tab_text(d: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        ngram = st.radio("Unité", [1, 2], horizontal=True, format_func=lambda n: "mots" if n == 1 else "mots + bigrammes", key="eda_ngram")
    with c2:
        min_count = st.slider("Présent dans au moins … avis", 5, 300, 50, key="eda_min_count")
    with c3:
        top = st.slider("Nombre de mots par côté", 5, 25, 12, key="eda_top_words")

    has_text = d["Review Text"].astype(str).str.strip().ne("")
    sub = d[has_text]
    if len(sub) < 50 or sub[EDA_TARGET].nunique() < 2:
        st.info("Il faut au moins 50 avis contenant du texte et les deux classes pour comparer le vocabulaire (élargissez les filtres).")
        return
    texts = (sub["Title"].astype(str) + " " + sub["Review Text"].astype(str))
    res = eda_discriminating_words(texts, sub[EDA_TARGET], ngram, min_count, top)
    if res is None:
        return
    top_df, full = res
    st.altair_chart(
        alt.Chart(top_df).mark_bar().encode(
            x=alt.X("Log-odds:Q", title="← plus associé à « non recommandé »      plus associé à « recommandé » →"),
            y=alt.Y("Mot:N", sort=alt.EncodingSortField("Log-odds", order="descending"), title=None, axis=alt.Axis(labelOverlap=False, labelLimit=200)),
            color=alt.Color(f"{EDA_LABEL}:N", scale=EDA_COLOR, legend=None),
            tooltip=["Mot", "Avis « recommandé »", "Avis « non recommandé »", "Log-odds", "z"],
        ).properties(height=max(360, 26 * len(top_df) + 60), title="Vocabulaire le plus discriminant (classé par z du log-odds)")
    )
    st.caption("Un log-odds de +1 signifie que le mot apparaît environ 2,7 fois plus (en proportion d'avis) chez les recommandées. "
               "C'est exactement le signal que TF-IDF + SVM linéaire exploitent.")

    st.markdown("#### Longueur des avis")
    hist = eda_histogram_frame(d, "Review Length", 30, True, True)
    hist["base"] = 0
    st.altair_chart(
        alt.Chart(hist).mark_bar(opacity=0.8).encode(
            x=alt.X("start:Q", title="Longueur de l'avis (mots, coupé au 99ᵉ centile)"), x2="end:Q",
            y=alt.Y("Effectif:Q", stack=None, title="Effectif"), y2="base:Q", color=alt.Color(f"{EDA_LABEL}:N", scale=EDA_COLOR), tooltip=[EDA_LABEL, "Effectif"],
        ).properties(height=300)
    )
    with st.expander("Tableau complet des mots"):
        st.dataframe(full.sort_values("z", ascending=False), hide_index=True)


def _eda_tab_tests(d: pd.DataFrame) -> None:
    st.caption("Les tests sont recalculés sur la **sélection courante** (voir les filtres). "
               "Pour les résultats officiels du projet, voir la page « Tests d'hypothèses ».")
    alpha = st.select_slider("Seuil α", options=[0.001, 0.01, 0.05, 0.10], value=0.05, key="eda_alpha")

    st.markdown("#### 1. Variable numérique × recommandation — Mann-Whitney U")
    var = st.selectbox("Variable numérique", EDA_NUMERIC, key="eda_test_var")
    mw = eda_mann_whitney(d, var)
    if mw is None:
        st.info("Les deux classes doivent être présentes (au moins 2 avis chacune) : changez le filtre « Recommandation ».")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("U", f"{mw['U']:,.0f}".replace(",", " "))
        c2.metric("p-value", eda_fmt_p(mw["p"]))
        c3.metric(f"Taille d'effet r ({eda_effect_label(mw['r'], 'r')})", f"{mw['r']:+.3f}")
        c4.metric("Médianes (rec. / non rec.)", f"{mw['median1']:g} / {mw['median0']:g}")
        verdict = "rejetée" if mw["p"] < alpha else "non rejetée"
        st.markdown(f"**H₀ : les deux distributions sont identiques — {verdict} au seuil α = {alpha}.** "
                    f"Effet {eda_effect_label(mw['r'], 'r')} : un avis « recommandé » pris au hasard a une valeur plus "
                    f"{'élevée' if mw['r'] > 0 else 'faible'} qu'un « non recommandé » dans {(mw['r'] + 1) / 2 * 100:.0f} % des paires.")

    st.markdown("#### 2. Variable catégorielle × recommandation — χ² d'indépendance")
    cat = st.selectbox("Variable catégorielle", EDA_CATEGORICAL, key="eda_test_cat")
    table = pd.crosstab(d[cat], d[EDA_TARGET])
    if table.shape[0] < 2 or table.shape[1] < 2:
        st.info("Il faut au moins 2 modalités et les deux classes dans la sélection.")
    else:
        chi2, p, v, dof = eda_cramers_v(table)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"χ² (ddl = {dof})", f"{chi2:,.1f}".replace(",", " "))
        c2.metric("p-value", eda_fmt_p(p))
        c3.metric(f"V de Cramér ({eda_effect_label(v, 'v')})", f"{v:.3f}")
        st.markdown(f"**H₀ : {cat} et la recommandation sont indépendantes — {'rejetée' if p < alpha else 'non rejetée'} au seuil α = {alpha}.**")
        if (table.to_numpy() < 5).any():
            st.warning("Certains effectifs sont inférieurs à 5 : l'approximation du χ² est fragile pour cette sélection.")

    st.markdown("#### 3. Pourquoi des tests non paramétriques ? — normalité (Shapiro-Wilk)")
    sample = d[var].dropna()
    sample = sample.sample(min(len(sample), 3000), random_state=42) if len(sample) > 3 else sample
    if len(sample) >= 8 and sample.nunique() > 1:
        w, p_sw = scipy_stats.shapiro(sample)
        st.markdown(f"Sur un échantillon de {len(sample)} valeurs de **{var}** : W = {w:.3f}, p = {eda_fmt_p(p_sw)} "
                    f"→ {'distribution non normale' if p_sw < alpha else 'normalité non rejetée'}.")
        qq = scipy_stats.probplot(sample, dist="norm")
        theo, ordered = qq[0]
        qq_df = pd.DataFrame({"Quantiles théoriques (loi normale)": theo, f"Quantiles observés ({var})": ordered})
        ref = pd.DataFrame({"x": [theo.min(), theo.max()], "y": [qq[1][1] + qq[1][0] * theo.min(), qq[1][1] + qq[1][0] * theo.max()]})
        pts = alt.Chart(qq_df).mark_circle(size=14, opacity=0.5, color=DASH_PINK).encode(x=qq_df.columns[0], y=qq_df.columns[1])
        line = alt.Chart(ref).mark_line(color="#2D1420").encode(x="x:Q", y="y:Q")
        st.altair_chart((pts + line).properties(height=280, title="Diagramme Q-Q : plus les points s'écartent de la droite, moins la variable est normale"))


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #
def render_eda_dashboard(clean: pd.DataFrame, raw: pd.DataFrame | None = None) -> None:
    """Affiche le dashboard complet. `clean` = reviews_clean.csv, `raw` = reviews.csv (optionnel)."""
    st.markdown('<div class="section-title">📈 Exploration interactive du Dataset (Women\'s E-Commerce Clothing Reviews)</div>',
                unsafe_allow_html=True)
    full = eda_prepare(clean)
    d, active = _eda_filters(full)

    if d.empty:
        st.warning("Aucun avis ne correspond aux filtres choisis : élargissez la sélection.")
        return

    share = len(d) / len(full) * 100
    st.caption(f"Sélection : **{len(d):,}** avis sur {len(full):,} ({share:.1f} %)".replace(",", " ")
               + (" — filtres actifs : " + " · ".join(active) if active else " — aucun filtre actif"))
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Avis sélectionnés", f"{len(d):,}".replace(",", " "))
    k2.metric("Recommandés", f"{d[EDA_TARGET].mean() * 100:.1f} %", help=f"Taux sur l'ensemble du dataset : {full[EDA_TARGET].mean() * 100:.1f} %")
    k3.metric("Âge moyen", f"{d['Age'].mean():.1f} ans")
    k4.metric("Note moyenne", f"{d['Rating'].mean():.2f} / 5")
    k5.metric("Longueur moyenne", f"{d['Review Length'].mean():.0f} mots")

    tabs = st.tabs(["📋 Aperçu", "🕳️ Qualité", "📊 Distributions", "🏷️ Catégories", "🔗 Corrélations", "📝 Texte", "🧪 Tests rapides"])
    with tabs[0]:
        _eda_tab_overview(d)
    with tabs[1]:
        _eda_tab_quality(raw, clean)
    with tabs[2]:
        _eda_tab_distributions(d)
    with tabs[3]:
        _eda_tab_categories(d)
    with tabs[4]:
        _eda_tab_correlations(d)
    with tabs[5]:
        _eda_tab_text(d)
    with tabs[6]:
        _eda_tab_tests(d)


# --------------------------------------------------------------------------- #
# Données du tableau de bord (nettoyées + brutes pour la qualité)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Chargement des données…")
def load_datasets():
    """Dataset nettoyé et dataset brut, pour le tableau de bord statistique."""
    clean = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "reviews_clean.csv", keep_default_na=False)
    raw_path = PROJECT_ROOT / "data" / "raw" / "reviews.csv"
    return clean, (pd.read_csv(raw_path) if raw_path.exists() else None)


# --- Analyses détaillées des modèles ---

def _res_read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _res_figure(figs: Path, name: str, caption: str) -> bool:
    if (figs / name).exists():
        st.image(str(figs / name), caption=caption)
        return True
    return False


def _res_missing() -> None:
    st.info("Résultats non disponibles. Vérifiez que les fichiers d'analyse requis sont présents.")


def _res_ranked_bars(df: pd.DataFrame, label: str, title: str, key: str, default: str | None = None) -> None:
    """Diagramme à barres triable : l'utilisateur choisit la métrique à classer."""
    metrics = [c for c in df.select_dtypes("number").columns if not c.lower().startswith(("dimensions", "temps", "écart"))]
    if not metrics:
        return
    idx = metrics.index(default) if default in metrics else 0
    metric = st.selectbox("Classer selon", metrics, index=idx, key=key)
    data = df[[label, metric]].dropna().sort_values(metric, ascending=False)
    lo = max(0.0, float(data[metric].min()) - 0.05) if data[metric].max() <= 1 else 0
    chart = (
        alt.Chart(data)
        .mark_bar(color=DASH_PINK, clip=True)  # clip : l'axe est tronqué, les barres ne doivent pas déborder sur les noms de modèles
        .encode(
            x=alt.X(f"{metric}:Q", scale=alt.Scale(domain=[lo, min(1.0, float(data[metric].max()) + 0.01)] if data[metric].max() <= 1 else alt.Undefined), title=metric),
            y=alt.Y(f"{label}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=[label, alt.Tooltip(f"{metric}:Q", format=".4f")],
        )
        .properties(height=max(180, 34 * len(data)), title=title)
    )
    st.altair_chart(chart)
    st.caption("L'axe horizontal est tronqué pour rendre les écarts lisibles : lire les valeurs dans l'info-bulle ou le tableau.")


def _res_table(df: pd.DataFrame, name: str) -> None:
    num = df.select_dtypes("number").columns
    fmt = {c: "{:.4f}" for c in num if not c.lower().startswith(("dimensions", "temps"))}
    st.dataframe(df.style.format(fmt, na_rep="n/d"), hide_index=True)
    st.download_button("⬇️ Télécharger (CSV)", df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv", key=f"dl_{name}")


# --------------------------------------------------------------------------- #
# Matrice de confusion et courbes ROC interactives (curseur de seuil)
# --------------------------------------------------------------------------- #
def confusion_at(y_true, score, threshold: float) -> dict:
    """Matrice de confusion et métriques par classe pour un seuil donné (score >= seuil -> « recommandé »)."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(score, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def _f(num, den):
        return float(num / den) if den else 0.0

    p1, r1 = _f(tp, tp + fp), _f(tp, tp + fn)
    p0, r0 = _f(tn, tn + fn), _f(tn, tn + fp)
    f1_1, f1_0 = _f(2 * p1 * r1, p1 + r1), _f(2 * p0 * r0, p0 + r0)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": _f(tp + tn, len(y)),
        "precision_1": p1, "recall_1": r1, "f1_1": f1_1,
        "precision_0": p0, "recall_0": r0, "f1_0": f1_0,
        "f1_macro": (f1_0 + f1_1) / 2,
    }


def natural_threshold(model_name: str) -> float:
    """Seuil « naturel » : 0 pour une marge de SVM, 0,5 pour une probabilité."""
    return 0.0 if "svc" in model_name.lower() or "svm" in model_name.lower() else 0.5


def load_test_scores(project_root: Path) -> pd.DataFrame | None:
    """Scores des modèles sur le même jeu de test : ensembles de l'équipe + Gradient Boosting."""
    team = _res_read(project_root / "models" / "ensemble_test_scores.csv")
    if team is None:
        return None
    gb = _res_read(project_root / "reports" / "gradient_boosting_test_scores.csv")
    if gb is not None and "row_id" in gb.columns:
        gb = gb.drop(columns=["y_true"], errors="ignore")
        team = team.merge(gb, on="row_id", how="left")
    return team


def _interactive_threshold_tab(project_root: Path) -> None:
    data = load_test_scores(project_root)
    if data is None or "y_true" not in data.columns:
        _res_missing()
        return
    models = [c for c in data.columns if c not in ("row_id", "y_true") and data[c].notna().all()]
    y = data["y_true"].to_numpy()

    c1, c2 = st.columns([1, 2])
    with c1:
        name = st.selectbox("Modèle", models, index=models.index("Gradient Boosting") if "Gradient Boosting" in models else 0, key="thr_model")
    score = data[name].to_numpy(dtype=float)
    lo, hi = float(score.min()), float(score.max())
    default = min(max(natural_threshold(name), lo), hi)
    with c2:
        thr = st.slider("Seuil de décision (≥ seuil → « recommandé »)", lo, hi, default, step=(hi - lo) / 400 or 0.01,
                        key=f"thr_value_{name}", help="Baisser le seuil détecte plus d'avis recommandés mais laisse passer plus de faux positifs ; "
                                                    "le relever protège la classe minoritaire (avis non recommandés).")
    m = confusion_at(y, score, thr)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{m['accuracy']:.3f}")
    k2.metric("F1 macro", f"{m['f1_macro']:.3f}")
    k3.metric("F1 classe 0 (non recommandé)", f"{m['f1_0']:.3f}")
    k4.metric("F1 classe 1 (recommandé)", f"{m['f1_1']:.3f}")

    left, right = st.columns(2)
    with left:
        cells = pd.DataFrame({
            "Réel": ["Non recommandé", "Non recommandé", "Recommandé", "Recommandé"],
            "Prédit": ["Non recommandé", "Recommandé", "Non recommandé", "Recommandé"],
            "Avis": [m["tn"], m["fp"], m["fn"], m["tp"]],
        })
        cells["Part de la classe réelle"] = cells["Avis"] / cells.groupby("Réel")["Avis"].transform("sum")
        order = ["Non recommandé", "Recommandé"]
        base = alt.Chart(cells).encode(x=alt.X("Prédit:N", sort=order, title="Classe prédite", axis=alt.Axis(labelAngle=0)), y=alt.Y("Réel:N", sort=order, title="Classe réelle"))
        heat = base.mark_rect().encode(
            color=alt.Color("Part de la classe réelle:Q", scale=alt.Scale(domain=[0, 1], range=["#FCE4EC", DASH_PINK]), legend=None),
            tooltip=["Réel", "Prédit", "Avis", alt.Tooltip("Part de la classe réelle:Q", format=".1%")],
        )
        text = base.mark_text(fontSize=15, fontWeight="bold").encode(
            text=alt.Text("Avis:Q"),
            color=alt.condition(alt.datum["Part de la classe réelle"] > 0.5, alt.value("white"), alt.value("#3E1030")),
        )
        st.altair_chart((heat + text).properties(height=300, title=f"Matrice de confusion — {name}"))
    with right:
        curves = []
        for col in models:
            fpr, tpr, _ = roc_curve(y, data[col].to_numpy(dtype=float))
            curves.append(pd.DataFrame({"Taux de faux positifs": fpr, "Taux de vrais positifs": tpr, "Modèle": col,
                                        "Sélection": "Modèle choisi" if col == name else "Autres modèles"}))
        roc = pd.concat(curves, ignore_index=True)
        lines = alt.Chart(roc).mark_line().encode(
            x=alt.X("Taux de faux positifs:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Taux de vrais positifs:Q", scale=alt.Scale(domain=[0, 1])),
            detail="Modèle:N",
            color=alt.Color("Sélection:N", scale=alt.Scale(domain=["Modèle choisi", "Autres modèles"], range=[DASH_PINK, "#A78BBF"]), legend=alt.Legend(title=None)),
            size=alt.condition(alt.datum["Sélection"] == "Modèle choisi", alt.value(3.5), alt.value(1.3)),
            tooltip=["Modèle", alt.Tooltip("Taux de faux positifs:Q", format=".3f"), alt.Tooltip("Taux de vrais positifs:Q", format=".3f")],
        )
        point = pd.DataFrame({"Taux de faux positifs": [m["fp"] / max(m["fp"] + m["tn"], 1)],
                              "Taux de vrais positifs": [m["recall_1"]], "Point de fonctionnement": ["Seuil choisi"]})
        dot = alt.Chart(point).mark_point(size=160, filled=True, color="#1B1B1B").encode(
            x="Taux de faux positifs:Q", y="Taux de vrais positifs:Q", tooltip=["Point de fonctionnement"])
        diag = alt.Chart(pd.DataFrame({"Taux de faux positifs": [0, 1], "Taux de vrais positifs": [0, 1]})).mark_line(
            strokeDash=[4, 4], color="#999").encode(x="Taux de faux positifs:Q", y="Taux de vrais positifs:Q")
        st.altair_chart((diag + lines + dot).properties(height=300, title=f"Courbes ROC — AUC {name} = {roc_auc_score(y, score):.3f}"))

    st.dataframe(pd.DataFrame({
        "Classe": ["0 — non recommandé (minoritaire)", "1 — recommandé"],
        "Précision": [m["precision_0"], m["precision_1"]],
        "Rappel": [m["recall_0"], m["recall_1"]],
        "F1": [m["f1_0"], m["f1_1"]],
    }).style.format({"Précision": "{:.3f}", "Rappel": "{:.3f}", "F1": "{:.3f}"}), hide_index=True)
    st.caption("Le point noir de la courbe ROC se déplace avec le curseur : c'est le compromis réellement choisi. "
               "Sur un jeu 82/18, regarder le F1 de la classe 0 évite de se contenter d'une accuracy flatteuse.")


def _render_model_analysis_tabs(project_root: Path) -> None:
    reports, figs = project_root / "reports", project_root / "docs" / "figures"
    st.caption(
        "Comparaison approfondie des modèles, représentations du texte, réglages et ablation. "
        "F1 (0) mesure la détection des avis **non recommandés** (classe minoritaire, 18 %)."
    )
    tabs_all = st.tabs(["🏁 Tous les modèles", "🎚️ Seuil interactif", "🚀 Gradient Boosting", "📝 Représentations du texte", "🔧 Réglages & ablation", "🧠 Importance des variables"])
    tabs = [tabs_all[0], tabs_all[2], tabs_all[3], tabs_all[4], tabs_all[5]]  # numérotation historique 0..4 conservée
    with tabs_all[1]:
        _interactive_threshold_tab(project_root)

    with tabs[0]:
        res = _res_read(reports / "model_results.csv")
        if res is None:
            _res_missing()
        else:
            label = res.columns[0]
            _res_ranked_bars(res.drop(columns=["Temps (s)"], errors="ignore"), label, "Comparaison des modèles (jeu de test)", "res_metric", "F1 macro")
            _res_table(res.drop(columns=["Temps (s)"], errors="ignore"), "model_results.csv")
        for name, cap in [("confusion_matrices.png", "Matrices de confusion normalisées par classe réelle (jeu de test)"),
                          ("roc_pr_curves.png", "Courbes ROC et précision-rappel"),
                          ("model_comparison.png", "F1 macro et ROC-AUC par modèle")]:
            _res_figure(figs, name, cap)

    with tabs[1]:
        comp = _res_read(reports / "ensemble_comparison_with_gb.csv")
        if comp is None:
            _res_missing()
        else:
            params = _res_read(reports / "gradient_boosting_best_params.csv")
            if params is not None:
                st.markdown("**Hyperparamètres retenus** : " + " · ".join(f"`{r['hyperparamètre']}` = {r['valeur']}" for _, r in params.iterrows()))
            _res_ranked_bars(comp, "Modèle", "Gradient Boosting face aux autres ensembles (jeu de test)", "gb_metric", "F1 (0)")
            _res_table(comp, "ensemble_comparison_with_gb.csv")
            cv = _res_read(reports / "gradient_boosting_cv.csv")
            if cv is not None:
                with st.expander("Grille de validation croisée (6 plis)"):
                    _res_table(cv, "gradient_boosting_cv.csv")
            for name, cap in [("gradient_boosting_vs_ensembles.png", "Courbes ROC et F1 de la classe minoritaire"),
                              ("gradient_boosting_confusion.png", "Matrice de confusion du Gradient Boosting"),
                              ("gradient_boosting_stages.png", "Apprentissage au fil des arbres (sans utiliser le jeu de test)"),
                              ("gradient_boosting_cv.png", "Effet de la profondeur et du nombre d'arbres"),
                              ("gradient_boosting_importance.png", "Variables les plus importantes")]:
                _res_figure(figs, name, cap)

    with tabs[2]:
        txt = _res_read(reports / "text_representations.csv")
        if txt is None:
            _res_missing()
        else:
            st.markdown(
                "Même classifieur (SVM linéaire), même validation croisée à 6 plis, **texte seul** : seule la représentation change. "
                "LSA et Word2Vec sont des représentations denses de Machine Learning classique (aucun réseau profond, aucun modèle pré-entraîné)."
            )
            _res_ranked_bars(txt, txt.columns[0], "Représentations du texte", "txt_metric", "F1 macro (test)")
            _res_table(txt, "text_representations.csv")
            _res_figure(figs, "text_representations.png", "Comparaison des représentations textuelles")

    with tabs[3]:
        shown = False
        abl = _res_read(reports / "ablation.csv")
        if abl is not None:
            st.markdown("**Ablation** : que perd-on sans la note, sans le texte ou sans les métadonnées ? (modèle ré-optimisé à chaque fois)")
            _res_table(abl, "ablation.csv")
            shown = True
        for name, cap in [("svm_kernels.png", "SVM : noyaux linéaire, RBF et polynomial"),
                          ("knn_k_curve.png", "K-NN : impact de K et de la distance"),
                          ("tree_pruning_curve.png", "Arbre de décision : élagage par coût-complexité"),
                          ("ablation.png", "Ablation : sans la note, sans le texte")]:
            shown = _res_figure(figs, name, cap) or shown
        if not shown:
            _res_missing()

    with tabs[4]:
        cons = _res_read(reports / "feature_importance_consensus.csv")
        shown = False
        if cons is not None:
            _res_table(cons.head(30), "feature_importance_consensus.csv")
            shown = True
        for name, cap in [("importance_consensus.png", "Importance « consensus » de plusieurs méthodes"),
                          ("feature_importance.png", "Coefficients du SVM linéaire")]:
            shown = _res_figure(figs, name, cap) or shown
        if not shown:
            _res_missing()


def render_model_analysis(project_root: Path) -> None:
    with st.expander("Analyses approfondies des modèles"):
        _render_model_analysis_tabs(project_root)


# <<< FIN DU BLOC AJOUTÉ
# ==============================================================================

PAGES = [
    "🏠 Accueil",
    "🔮 Prédiction en Direct",
    "📊 Performance des Modèles",
    "📈 Statistiques du Dataset",
    "💡 Découvertes Clés",
    "🧪 Tests d'hypothèses",
    "🌲 Ensemble Learning",
    "ℹ️ À propos",
    "🔍 Analyse des erreurs",
]


ticker_items = [
    "✦ NOUVELLE COLLECTION AUTOMNE-HIVER",
    "✦ MODÈLES DE MACHINE LEARNING CLASSIQUES",
    "✦ VALIDATION CROISÉE STRATIFIÉE 6-FOLD",
    "✦ SÉLECTION D'HYPERPARAMÈTRES PAR GRIDSEARCHCV",
    "✦ ENSEMBLE LEARNING : ADABOOST · XGBOOST · SOFT VOTING",
]
ticker_html = "".join(f"<span>{t}</span>" for t in ticker_items)

st.markdown(
    h(f"""
    <div class="ticker"><div class="ticker-track">{ticker_html}{ticker_html}</div></div>
    <div class="navbar">
        <div class="orn">✿ ✦ ✿</div><div class="brand">PRÉDICTION DES RECOMMANDATIONS D'ACHATS
            <small>VÊTEMENTS POUR FEMME</small></div>
    </div>
    <div class="tagline">Analyse des avis clients par algorithmes de Machine Learning de classification supervisée.</div>
    """),
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown('<div class="side-title" style="margin-top:0;">Navigation</div>', unsafe_allow_html=True)
    st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
    st.markdown(
        h("""
        <div class="side-brand">Recommandations d'achats</div>
        <div class="side-sub">MODE FEMME · ML CLASSIC</div>

        <div class="side-card">
            Les modèles de Machine Learning (k-NN, Naïve Bayes, SVM, Random Forest...) sont entraînés sur 
            <b>23 465 avis réels</b> pour prédire si l'article sera <b>recommandé</b>.
        </div>

        <div class="side-title">Comment ça marche ?</div>
        <div class="step"><div class="step-num">1</div><div>Choisissez l'article : département, catégorie et collection.</div></div>
        <div class="step"><div class="step-num">2</div><div>Renseignez le profil de la cliente.</div></div>
        <div class="step"><div class="step-num">3</div><div>Rédigez son avis et sa note.</div></div>
        <div class="step"><div class="step-num">4</div><div>Lancez la classification et visualisez le résultat.</div></div>

        <div class="side-title">Cadre Académique</div>
        <div class="side-card">
            Projet Data Mining & Machine Learning Supervisé · Prétraitement TF-IDF & One-Hot Encoding.
        </div>

        <div class="side-foot">École Militaire Polytechnique · Informatique</div>
        """),
        unsafe_allow_html=True,
    )


page = st.session_state["page"]

# ==============================================================================
# PAGE 1 : ACCUEIL
# ==============================================================================
if page == PAGES[0]:
    st.markdown('<div class="section-title">🏠 Vue d\'ensemble du projet</div>', unsafe_allow_html=True)
    st.markdown(
        h("""
        <div class="info-card"><b>Question de recherche :</b> à partir de l'avis d'une cliente (texte, note,
        profil) et des caractéristiques de l'article, peut-on prédire si elle le <b>recommandera</b> ?</div>
        """),
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl in [
        (k1, "23 465", "Avis clients"),
        (k2, "9", "Algorithmes comparés"),
        (k3, "0.9622", "Meilleur F1-Score"),
        (k4, "Linear SVC", "Modèle champion"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:2rem;">🧭 Pipeline de Data Mining</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    for col, ico, ttl, txt in [
        (p1, "🗂️", "1. Données", "Women's E-Commerce Clothing Reviews, cible binaire <i>Recommended IND</i>."),
        (p2, "🧹", "2. Prétraitement", "TF-IDF sur le texte, One-Hot Encoding des catégories : 13 035 variables."),
        (p3, "🤖", "3. Modélisation", "9 algorithmes (dont 3 d'Ensemble Learning), GridSearchCV, validation croisée stratifiée 6-Fold."),
        (p4, "🚀", "4. Déploiement", "Prédiction en direct via ce tableau de bord Streamlit."),
    ]:
        with col:
            st.markdown(f'<div class="info-card"><div style="font-size:1.6rem;">{ico}</div><b>{ttl}</b><br>{txt}</div>', unsafe_allow_html=True)

    st.info("👈 Utilisez le menu de gauche pour naviguer : lancez une prédiction, comparez les modèles ou explorez le dataset.")


# ==============================================================================
# ONGLET 1 : PRÉDICTION EN DIRECT
# ==============================================================================
if page == PAGES[1]:
    if model is None or preprocessor is None:
        st.error(
            "⚠️ Fichiers introuvables ! Assurez-vous d'avoir sauvegardé "
            "`models/best_model.pkl` et `models/preprocessor.pkl`."
        )
    else:
        left, right = st.columns([1.15, 1], gap="large")

        with left:
            st.markdown('<div class="section-title">👗 Fiche produit</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                department_name = st.selectbox(
                    "Département",
                    list(DEPARTMENTS),
                    format_func=lambda d: DEPT_LABELS[d],
                    key="department",
                )
            with c2:
                class_name = st.selectbox(
                    "Catégorie",
                    DEPARTMENTS[department_name],
                    format_func=lambda c: CLASS_LABELS[c],
                    key=f"class_{department_name}",
                )
            with c3:
                division_name = st.selectbox(
                    "Collection",
                    DIVISIONS[department_name],
                    format_func=lambda d: DIV_LABELS[d],
                    key=f"division_{department_name}",
                )

            st.markdown(
                f'<div class="product-chip">{DEPT_EMOJI[department_name]} '
                f'{DEPT_LABELS[department_name]} › {CLASS_LABELS[class_name]} · '
                f'{DIV_LABELS[division_name]}</div>',
                unsafe_allow_html=True,
            )

            with st.form("review_form"):
                st.markdown('<div class="section-title">👩 Profil de la cliente</div>', unsafe_allow_html=True)
                c4, c5 = st.columns(2)
                with c4:
                    age = st.number_input("Âge", min_value=18, max_value=100, value=33)
                with c5:
                    positive_feedback_count = st.number_input(
                        "👍 Avis jugé utile par",
                        min_value=0,
                        value=3,
                        help="Nombre de personnes ayant trouvé cet avis utile.",
                    )

                st.markdown(
                    '<div class="section-title" style="margin-top:1.2rem;">⭐ Avis client</div>',
                    unsafe_allow_html=True,
                )
                rating = st.select_slider(
                    "Note attribuée",
                    options=[1, 2, 3, 4, 5],
                    value=5,
                    format_func=lambda x: "★" * x + "☆" * (5 - x),
                )
                title_text = st.text_input("Titre de l'avis", value="I love it")
                review_text = st.text_area("Texte de l'avis", value="Great quality and comfortable fit!", height=120)
                st.markdown(
                    '<div class="hint">💡 Les avis du dataset sont en anglais : rédigez l\'avis en anglais '
                    'pour de meilleurs résultats.</div>',
                    unsafe_allow_html=True,
                )

                submitted = st.form_submit_button("✦ PRÉDIRE LA RECOMMANDATION")

        with right:
            st.markdown('<div class="section-title">📊 Résultat du Classifieur</div>', unsafe_allow_html=True)

            if not submitted:
                st.markdown(
                    h("""
                    <div class="placeholder-box">
                        <div style="font-size:2.5rem;">🧵</div>
                        <p>Renseignez la fiche et lancez l'évaluation pour découvrir
                        si l'article sera recommandé par le modèle.</p>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )
            elif not review_text.strip():
                st.warning("Veuillez saisir au moins le texte de l'avis client.")
            else:
                input_data = pd.DataFrame([{
                    "Age": age,
                    "Rating": rating,
                    "Positive Feedback Count": positive_feedback_count,
                    "Division Name": division_name,
                    "Department Name": department_name,
                    "Class Name": class_name,
                    "Title": title_text,
                    "Review Text": review_text,
                }])

                try:
                    X_input_processed = preprocessor.transform(input_data)
                    prediction = model.predict(X_input_processed)[0]
                    proba = (
                        model.predict_proba(X_input_processed)[0]
                        if hasattr(model, "predict_proba") else None
                    )

                    if prediction == 1:
                        css, fill, icon = "result-yes", "fill-yes", "💚"
                        title, sub = "Article recommandé", "La cliente recommandera ce produit (Classe 1)."
                    else:
                        css, fill, icon = "result-no", "fill-no", "💔"
                        title, sub = "Article non recommandé", "La cliente ne recommandera pas ce produit (Classe 0)."

                    extra_html = ""
                    pills_html = ""
                    if proba is not None:
                        confidence = proba[prediction] * 100
                        p_no, p_yes = proba[0] * 100, proba[1] * 100
                        extra_html = f"""
                            <div class="result-sub" style="margin-top:12px;">
                                Probabilité de prédiction du modèle : <b>{confidence:.2f}%</b>
                            </div>
                            <div class="conf-track">
                                <div class="conf-fill {fill}" style="width:{confidence:.1f}%;"></div>
                            </div>
                        """
                        pills_html = f"""
                            <div class="pills">
                                <div class="pill"><b>{p_yes:.1f}%</b>Probabilité (Recommandé)</div>
                                <div class="pill"><b>{p_no:.1f}%</b>Probabilité (Non Recommandé)</div>
                            </div>
                        """

                    st.markdown(
                        h(f"""
                        <div class="result-card {css}">
                            <div class="result-icon">{icon}</div>
                            <div class="result-title">{title}</div>
                            <div class="result-sub">{sub}</div>
                            {extra_html}
                        </div>
                        {pills_html}
                        """),
                        unsafe_allow_html=True,
                    )

                    stars = "★" * rating + "☆" * (5 - rating)
                    st.markdown(
                        h(f"""
                        <div class="review-preview">
                            <div class="review-stars">{stars}</div>
                            <div class="review-title">{esc(title_text) or "Sans titre"}</div>
                            <div class="review-body">« {esc(review_text)} »</div>
                            <div class="review-meta">
                                Cliente, {age} ans · {DEPT_LABELS[department_name]} /
                                {CLASS_LABELS[class_name]} · {DIV_LABELS[division_name]}
                                · 👍 {positive_feedback_count}
                            </div>
                        </div>
                        """),
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.error(f"Erreur lors de l'exécution du modèle : {e}")

# ==============================================================================
# ONGLET 2 : PERFORMANCE ET COMPARATIF DES MODÈLES
# ==============================================================================
if page == PAGES[2]:
    st.markdown('<div class="section-title">📊 Comparatif des Modèles Classiques</div>', unsafe_allow_html=True)
    st.markdown(
        "Comparaison des modèles à partir des métriques d'évaluation disponibles."
    )
    render_model_analysis(PROJECT_ROOT)

    # Recherche des métriques dans les emplacements du projet.
    metrics_path = PROJECT_ROOT / "notebooks" / "evaluation_metrics.csv"
    if not metrics_path.exists():
        metrics_path = Path(__file__).resolve().parent / "notebooks" / "evaluation_metrics.csv"

    if not metrics_path.exists():
        st.error(
            "Les métriques d'évaluation sont introuvables dans les fichiers du projet."
        )
        st.stop()

    try:
        benchmark_data = pd.read_csv(metrics_path)
    except Exception as e:
        st.error(f"Impossible de lire evaluation_metrics.csv : {e}")
        st.stop()

    if benchmark_data.empty:
        st.warning("Le fichier evaluation_metrics.csv ne contient aucune ligne.")
        st.stop()

    # Harmonisation des noms de colonnes pour accepter les variantes du CSV.
    def _normalize_metric_name(name):
        return re.sub(r"[^a-z0-9]", "", str(name).lower())

    normalized_columns = {
        _normalize_metric_name(col): col for col in benchmark_data.columns
    }

    def _find_metric_column(candidates):
        for candidate in candidates:
            key = _normalize_metric_name(candidate)
            if key in normalized_columns:
                return normalized_columns[key]
        return None

    model_col = _find_metric_column(["Modèle", "Model", "Classifier", "Estimator", "Algorithm"])
    if model_col is None:
        # Certains exports CSV enregistrent le nom du modèle dans la première colonne.
        model_col = benchmark_data.columns[0]
    if model_col != "Modèle":
        benchmark_data = benchmark_data.rename(columns={model_col: "Modèle"})

    metric_aliases = {
        "Accuracy": ["Accuracy", "Acc"],
        "F1-macro": ["F1-macro", "F1_macro", "Macro F1", "F1 Macro", "Macro-F1"],
        "F1-weighted": ["F1-weighted", "F1_weighted", "Weighted F1", "F1 Weighted"],
        "Balanced Accuracy": ["Balanced Accuracy", "Balanced_Accuracy", "Balanced-Accuracy"],
        "ROC-AUC": ["ROC-AUC", "ROC_AUC", "ROC AUC", "AUC"],
        "Precision": ["Precision", "Precision weighted"],
        "Recall": ["Recall", "Recall weighted"],
        "F1-Score": ["F1-Score", "F1 Score", "F1", "F1-score"],
    }

    available_metrics = {}
    for display_name, aliases in metric_aliases.items():
        column = _find_metric_column(aliases)
        if column is not None and column != "Modèle":
            available_metrics[display_name] = column

    # Convertit les métriques en valeurs numériques pour les tableaux et graphiques.
    for display_name, source_column in available_metrics.items():
        benchmark_data[display_name] = pd.to_numeric(
            benchmark_data[source_column], errors="coerce"
        )

    preferred_metrics = [
        "Accuracy", "Precision", "Recall", "F1-Score", "F1-macro",
        "F1-weighted", "Balanced Accuracy", "ROC-AUC"
    ]
    display_metrics = [m for m in preferred_metrics if m in available_metrics]

    if not display_metrics:
        st.error(
            "Aucune colonne de métrique reconnue dans le CSV. "
            f"Colonnes trouvées : {', '.join(map(str, benchmark_data.columns))}"
        )
        st.stop()

    # Cartes calculées à partir des résultats du CSV, jamais à partir de valeurs fixes.
    primary_metric = "F1-macro" if "F1-macro" in display_metrics else (
        "F1-Score" if "F1-Score" in display_metrics else display_metrics[0]
    )
    best_row = benchmark_data.loc[benchmark_data[primary_metric].idxmax()]
    best_model_name = str(best_row["Modèle"])
    best_primary_value = best_row[primary_metric]

    metric_cards = st.columns(4)
    with metric_cards[0]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{len(benchmark_data)}</div>'
            '<div class="metric-lbl">Modèles évalués</div></div>',
            unsafe_allow_html=True
        )
    with metric_cards[1]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{best_primary_value:.4f}</div>'
            f'<div class="metric-lbl">Meilleur {primary_metric}</div></div>',
            unsafe_allow_html=True
        )
    with metric_cards[2]:
        if "ROC-AUC" in display_metrics and benchmark_data["ROC-AUC"].notna().any():
            best_auc = benchmark_data["ROC-AUC"].max()
            auc_value = f"{best_auc:.4f}"
        else:
            auc_value = "N/D"
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{auc_value}</div>'
            '<div class="metric-lbl">Meilleur ROC-AUC</div></div>',
            unsafe_allow_html=True
        )
    with metric_cards[3]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val" style="font-size:1.25rem;">{best_model_name}</div>'
            f'<div class="metric-lbl">Meilleur modèle ({primary_metric})</div></div>',
            unsafe_allow_html=True
        )

    st.caption(f"Source : {metrics_path.name} · Critère de classement principal : {primary_metric}")

    # Tableau comparatif : conserve aussi les éventuelles colonnes d'hyperparamètres.
    columns_to_show = ["Modèle"] + display_metrics
    extra_columns = [
        col for col in benchmark_data.columns
        if any(term in _normalize_metric_name(col) for term in ["param", "hyper"])
        and col not in columns_to_show
    ]
    columns_to_show += extra_columns
    table_data = benchmark_data[columns_to_show].copy()

    st.markdown("### Tableau comparatif des performances")
    st.dataframe(
        table_data.style.format(
            {col: "{:.4f}" for col in display_metrics},
            na_rep="N/A"
        ).highlight_max(
            subset=[col for col in display_metrics if table_data[col].notna().any()],
            color="#F8BBD0"
        ),
        use_container_width=True,
        hide_index=True
    )

    # Comparaison graphique des métriques réellement disponibles.
    chart_metrics = [
        metric for metric in ["F1-macro", "Balanced Accuracy", "ROC-AUC", "Accuracy"]
        if metric in display_metrics and benchmark_data[metric].notna().any()
    ]

    if chart_metrics:
        st.markdown(
            '<div class="section-title" style="margin-top:2rem;">📈 Comparaison visuelle des modèles</div>',
            unsafe_allow_html=True
        )
        chart_df = benchmark_data.set_index("Modèle")[chart_metrics]
        st.bar_chart(chart_df, use_container_width=True)

    st.info(
        f"Lecture : le modèle mis en avant est **{best_model_name}**, "
        f"car il obtient le meilleur {primary_metric} ({best_primary_value:.4f}). "
        "Le F1-macro et la Balanced Accuracy sont particulièrement utiles lorsque "
        "les classes sont déséquilibrées."
    )

# ==============================================================================
# ONGLET 3 : STATISTIQUES DU DATASET
# ==============================================================================
if page == PAGES[3]:
    st.markdown('<div class="section-title">📈 Exploration du Dataset (Women\'s E-Commerce Clothing Reviews)</div>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown('<div class="metric-card"><div class="metric-val">23,465</div><div class="metric-lbl">Avis Totaux</div></div>', unsafe_allow_html=True)
    with d2:
        st.markdown('<div class="metric-card"><div class="metric-val">82.2%</div><div class="metric-lbl">Recommandations (1)</div></div>', unsafe_allow_html=True)
    with d3:
        st.markdown('<div class="metric-card"><div class="metric-val">17.8%</div><div class="metric-lbl">Non-Recommandations (0)</div></div>', unsafe_allow_html=True)
    with d4:
        st.markdown('<div class="metric-card"><div class="metric-val">13,035</div><div class="metric-lbl">Features TF-IDF & OHE</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Répartition de la cible (Recommended IND)")
        target_dist = pd.DataFrame({
            "Classe": ["Recommandé (1)", "Non Recommandé (0)"],
            "Proportion (%)": [82.22, 17.78]
        }).set_index("Classe")
        st.bar_chart(target_dist, color="#D81B60")

    with col_b:
        st.markdown("#### Répartition par Département")
        dept_dist = pd.DataFrame({
            "Département": ["Tops", "Dresses", "Bottoms", "Intimate", "Jackets", "Trend"],
            "Nombre d'avis": [10468, 6319, 3799, 1735, 1032, 112]
        }).set_index("Département")
        st.bar_chart(dept_dist, color="#F06292")

    st.markdown("---")
    render_eda_dashboard(*load_datasets())

# ==============================================================================
# PAGE 5 : DÉCOUVERTES CLÉS
# ==============================================================================
if page == PAGES[4]:
    st.markdown('<div class="section-title">💡 Découvertes Clés</div>', unsafe_allow_html=True)
    findings = [
        ("Déséquilibre des classes",
         "Quelle est la répartition de la cible ?",
         "82,2 % d'avis recommandés contre 17,8 % de non-recommandés.",
         "Le baseline Zero-R atteint déjà 82,23 % d'accuracy : l'accuracy seule est trompeuse, "
         "d'où le choix du F1-Score et du ROC-AUC."),
        ("Champion : Linear SVC",
         "Quel algorithme obtient le meilleur F1-Score sur le jeu de test ?",
         "Linear SVC : F1 = 0.9622, devant Random Forest (F1 = 0.9604). Multinomial Naïve Bayes obtient le ROC-AUC le plus élevé (0.9784).",
         "Les modèles linéaires conviennent bien aux données textuelles creuses et de grande dimension (13 035 variables)."),
        ("Limite de l'arbre de décision",
         "Un modèle simple suffit-il ?",
         "Decision Tree : ROC-AUC = 0.9684. Parmi les modèles disposant d'un ROC-AUC calculé, il présente la valeur la plus faible dans les résultats exportés.",
         "Un arbre unique sépare mal les classes sur des données TF-IDF ; les méthodes d'ensemble ou linéaires sont préférables."),
        ("Apport du tuning",
         "Que gagne-t-on avec GridSearchCV ?",
         "Les cinq modèles entraînés dépassent le baseline Zero-R en F1-Score (de 0.9425 à 0.9622, contre 0.9025 pour Zero-R).",
         "Le réglage des hyperparamètres avec validation croisée stratifiée stabilise les performances."),
    ]
    for i, (ttl, q, res, interp) in enumerate(findings, 1):
        with st.expander(f"Découverte {i} · {ttl}", expanded=(i == 1)):
            st.markdown(
                h(f"""
                <div class="info-card"><b>❓ Question :</b> {q}</div>
                <div class="info-card"><b>📌 Résultat :</b> {res}</div>
                <div class="info-card"><b>🧠 Interprétation :</b> {interp}</div>
                """),
                unsafe_allow_html=True,
            )

# ==============================================================================
# PAGE 6 : TESTS D'HYPOTHÈSES
# ==============================================================================
if page == PAGES[5]:
    ALPHA = 0.05

   
    N_REC, N_NOT_REC = 19314, 4172          
    U_STAT, P_MW = 78344145.0, 0.0         
    CHI2, DOF, P_CHI2 = 50.24343, 5, 1.235608e-09
    N_CHI2 = 23472                          

    rank_biserial = 2 * U_STAT / (N_REC * N_NOT_REC) - 1
    cramers_v = float(np.sqrt(CHI2 / (N_CHI2 * (2 - 1))))   

    HYPOTHESES = [
        {
            "id": "H1", "test": "Mann-Whitney U Test",
            "vars": "Rating vs Recommended IND",
            "stat": U_STAT, "p": P_MW,
            "effect": f"Rank-Biserial r = {rank_biserial:.4f}",
            "h0": "Les distributions de Rating sont similaires chez les clientes qui recommandent et celles qui ne recommandent pas.",
            "note": f"Médiane du Rating : 5 (recommandé, n = {N_REC:,}) contre 2 (non recommandé, n = {N_NOT_REC:,}).".replace(",", " "),
        },
        {
            "id": "H2", "test": "Chi-Square Test of Independence",
            "vars": "Department Name vs Recommended IND",
            "stat": CHI2, "p": P_CHI2,
            "effect": f"Cramér's V = {cramers_v:.4f}",
            "h0": "Department Name et Recommended IND sont indépendantes.",
            "note": f"{DOF} degrés de liberté. Association significative mais très faible (V < 0.1) : le département explique peu la recommandation.",
        },
    ]

    def bh_adjust(pvals):
        """Correction FDR de Benjamini-Hochberg."""
        p = np.asarray(pvals, dtype=float)
        m = len(p)
        order = np.argsort(p)
        ranked = p[order] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(m)
        out[order] = np.clip(ranked, 0, 1)
        return out

    def fmt_p(p):
        return "< 0.0001" if p < 1e-4 else f"{p:.4f}"

    fdr = bh_adjust([x["p"] for x in HYPOTHESES])

    st.markdown('<div class="section-title">🧪 Tests d\'hypothèses statistiques</div>', unsafe_allow_html=True)
    st.markdown(
        "Tests réalisés lors de l'analyse exploratoire sur la relation "
        "entre les variables du dataset et la recommandation (`Recommended IND`)."
    )

    st.markdown(
        h("""
        <style>
        .hyp-wrap { overflow-x:auto; border:1px solid #F2C9D8; border-radius:16px; background:#FFFFFF;
            box-shadow:0 4px 14px rgba(216,27,96,0.06); margin:10px 0 18px 0; }
        table.hyp { width:100%; border-collapse:collapse; font-size:0.9rem; color:#2D1420; }
        table.hyp th { background:#FCE4EC; color:#880E4F; text-align:left; padding:12px 14px;
            font-weight:600; border-bottom:2px solid #F06292; white-space:nowrap; }
        table.hyp td { padding:12px 14px; border-bottom:1px solid #F8DCE7; vertical-align:top; }
        table.hyp tr:last-child td { border-bottom:none; }
        table.hyp td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
        table.hyp .sub { display:block; font-size:0.76rem; color:#8A6B78; margin-top:2px; }
        .dec-rej { display:inline-block; background:#FCE4EC; color:#AD1457; border:1px solid #F4A8C0;
            border-radius:20px; padding:2px 12px; font-weight:600; font-size:0.8rem; white-space:nowrap; }
        .dec-keep { display:inline-block; background:#EEF6EE; color:#1F5A2B; border:1px solid #8FBF8F;
            border-radius:20px; padding:2px 12px; font-weight:600; font-size:0.8rem; white-space:nowrap; }
        </style>
        """),
        unsafe_allow_html=True,
    )

    rows_html = ""
    for hyp, p_adj in zip(HYPOTHESES, fdr):
        reject = p_adj < ALPHA
        dec = ('<span class="dec-rej">Reject H0</span>' if reject
               else '<span class="dec-keep">Fail to Reject H0</span>')
        rows_html += (
            f'<tr><td><b>{hyp["id"]}</b></td>'
            f'<td>{hyp["test"]}<span class="sub">{esc(hyp["vars"])}</span></td>'
            f'<td class="num">{hyp["stat"]:,.4f}</td>'
            f'<td class="num">{fmt_p(hyp["p"])}</td>'
            f'<td class="num">{fmt_p(p_adj)}</td>'
            f'<td>{hyp["effect"]}</td>'
            f'<td>{dec}</td></tr>'
        )

    st.markdown(f'<div class="section-title" style="font-size:1.1rem;">Summary Table of Statistical Hypotheses (α = {ALPHA})</div>', unsafe_allow_html=True)
    st.markdown(
        h(f"""
        <div class="hyp-wrap"><table class="hyp">
        <thead><tr><th>ID</th><th>Test</th><th>Statistic</th><th>Raw p-value</th>
        <th>FDR p-value</th><th>Effect Size</th><th>Decision</th></tr></thead>
        <tbody>{rows_html}</tbody></table></div>
        """),
        unsafe_allow_html=True,
    )
   

    st.markdown('<div class="section-title" style="margin-top:1.5rem;">📝 Hypothèses et interprétation</div>', unsafe_allow_html=True)
    for hyp in HYPOTHESES:
        st.markdown(
            h(f"""
            <div class="info-card"><b>{hyp["id"]} · {esc(hyp["vars"])}</b><br>
            <b>H0 :</b> {esc(hyp["h0"])}<br>
            <b>🧠 Lecture :</b> {esc(hyp["note"])}</div>
            """),
            unsafe_allow_html=True,
        )

# ==============================================================================
# PAGE 7 : ENSEMBLE LEARNING (AdaBoost, XGBoost, Soft Voting)
# ==============================================================================
ENS_ORDER = ["AdaBoost", "XGBoost", "Soft Voting (NB + RF)", "Random Forest", "Linear SVC"]
ENS_PALETTE = {
    "AdaBoost": "#D81B60",
    "XGBoost": "#880E4F",
    "Soft Voting (NB + RF)": "#C9A227",
    "Random Forest": "#F06292",
    "Linear SVC": "#7B5E7B",
}
ENS_LENGTH_LABELS = ["≤ 20 mots", "21-40", "41-60", "61-80", "> 80 mots"]


@st.cache_data
def load_ensemble_data():
    """Charge et normalise les données nécessaires aux analyses d'ensemble."""
    needed = ["error_analysis.csv", "ensemble_feature_importance.csv", "ensemble_block_importance.csv"]
    if not all((MODELS_DIR / f).exists() for f in needed):
        return None

    errors_df = pd.read_csv(MODELS_DIR / "error_analysis.csv", keep_default_na=False)

    features_df = pd.read_csv(MODELS_DIR / "ensemble_feature_importance.csv")
    for column in ("Model", "Feature", "Importance"):
        if column not in features_df.columns:
     u       features_df[column] = "" if column != "Importance" else 0.0
    if "Block" not in features_df.columns:
        features_df["Block"] = "Autres"

    features_df["Model"] = features_df["Model"].fillna("Inconnu").astype(str).str.strip()
    features_df["Feature"] = features_df["Feature"].fillna("Inconnue").astype(str).str.strip()
    features_df["Block"] = features_df["Block"].fillna("Autres").astype(str).str.strip().replace("", "Autres")
    features_df["Importance"] = pd.to_numeric(features_df["Importance"], errors="coerce").fillna(0.0)
    features_df = (
        features_df[["Model", "Feature", "Block", "Importance"]]
        .query("Model != '' and Feature != ''")
        .sort_values(["Model", "Importance", "Feature"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    blocks_df = pd.read_csv(MODELS_DIR / "ensemble_block_importance.csv", index_col="Bloc")
    for column in blocks_df.columns:
        blocks_df[column] = pd.to_numeric(blocks_df[column], errors="coerce").fillna(0.0)
    blocks_df = blocks_df.fillna(0.0)
    if blocks_df.index.name is None:
        blocks_df.index.name = "Bloc"

    return errors_df, features_df, blocks_df


def ens_confusion(y_true, score, threshold):
    """Renvoie (TN, FP, FN, TP) ; la classe 1 (recommandé) est la classe positive."""
    pred = (score >= threshold).astype(int)
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    tp = int(((y_true == 1) & (pred == 1)).sum())
    return tn, fp, fn, tp


def ens_metrics(y_true, score, threshold):
    """Métriques globales et par classe pour un seuil de décision donné."""
    tn, fp, fn, tp = ens_confusion(y_true, score, threshold)

    def div(a, b):
        return a / b if b else 0.0

    p0, r0 = div(tn, tn + fn), div(tn, tn + fp)
    p1, r1 = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": div(tn + tp, tn + fp + fn + tp),
        "precision_0": p0, "recall_0": r0, "f1_0": div(2 * p0 * r0, p0 + r0),
        "precision_1": p1, "recall_1": r1, "f1_1": div(2 * p1 * r1, p1 + r1),
    }


def ens_downsample(frame, n_points=400):
    """Réduit le nombre de points d'une courbe pour garder un graphique fluide."""
    if len(frame) <= n_points:
        return frame
    idx = np.unique(np.linspace(0, len(frame) - 1, n_points).astype(int))
    return frame.iloc[idx]


def ens_colors(labels, models):
    """Couleur de chaque courbe : celle de son modèle (labels et models dans le même ordre)."""
    return [ENS_PALETTE.get(m, "#999999") for m in models]


def ens_curves(y_true, errors_df, models):
    """Courbes ROC et précision-rappel de la classe minoritaire (non recommandé)."""
    roc_frames, pr_frames, labels = [], [], []
    for m in models:
        score = errors_df[m].to_numpy()
        auc = roc_auc_score(y_true, score)
        label = f"{m} (AUC {auc:.3f})"
        labels.append(label)

        fpr, tpr, _ = roc_curve(y_true, score)
        roc = ens_downsample(pd.DataFrame({"FPR": fpr, "TPR": tpr}))
        roc["Modele"] = label
        roc_frames.append(roc)

        precision, recall, _ = precision_recall_curve(1 - y_true, 1 - score)
        pr = ens_downsample(pd.DataFrame({"Rappel": recall, "Precision": precision}))
        pr["Modele"] = label
        pr_frames.append(pr)
    return pd.concat(roc_frames, ignore_index=True), pd.concat(pr_frames, ignore_index=True), labels


def ens_threshold_frame(y_true, score):
    """Précision, rappel et F1 de la classe non recommandé en fonction du seuil."""
    rows = []
    for t in np.arange(0.05, 0.96, 0.01):
        m = ens_metrics(y_true, score, t)
        rows.append({"Seuil": round(float(t), 2), "Mesure": "Précision", "Valeur": m["precision_0"]})
        rows.append({"Seuil": round(float(t), 2), "Mesure": "Rappel", "Valeur": m["recall_0"]})
        rows.append({"Seuil": round(float(t), 2), "Mesure": "F1", "Valeur": m["f1_0"]})
    return pd.DataFrame(rows)


def ens_error_rates(group, is_error):
    """Taux d'erreur (en %) et effectif pour chaque groupe."""
    frame = pd.DataFrame({"Groupe": group.astype(str).to_numpy(), "Erreur": is_error.astype(int)})
    out = frame.groupby("Groupe", sort=False)["Erreur"].agg(["mean", "size"]).reset_index()
    out.columns = ["Groupe", "Taux", "Effectif"]
    out["Taux"] = (out["Taux"] * 100).round(1)
    return out


def ens_confusion_chart(tn, fp, fn, tp):
    labels = ["Non recommandé", "Recommandé"]
    cm = pd.DataFrame(
        [
            ("Non recommandé", "Non recommandé", tn),
            ("Non recommandé", "Recommandé", fp),
            ("Recommandé", "Non recommandé", fn),
            ("Recommandé", "Recommandé", tp),
        ],
        columns=["Reel", "Predit", "Effectif"],
    )
    totals = cm.groupby("Reel")["Effectif"].transform("sum")
    cm["Part"] = np.where(totals > 0, cm["Effectif"] / totals.replace(0, 1), 0.0)
    cm["Texte"] = [f"{int(e)} ({p:.0%})" for e, p in zip(cm["Effectif"], cm["Part"])]

    base = alt.Chart(cm).encode(
        x=alt.X("Predit:N", sort=labels, title="Classe prédite", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Reel:N", sort=labels, title="Classe réelle"),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("Part:Q", scale=alt.Scale(domain=[0, 1], range=["#FFF0F5", "#D81B60"]), legend=None),
        tooltip=[
            alt.Tooltip("Reel:N", title="Classe réelle"),
            alt.Tooltip("Predit:N", title="Classe prédite"),
            alt.Tooltip("Effectif:Q", title="Nombre d'avis"),
            alt.Tooltip("Part:Q", title="Part de la classe réelle", format=".1%"),
        ],
    )
    text = base.mark_text(fontSize=15, fontWeight="bold").encode(
        text="Texte:N",
        color=alt.condition(alt.datum.Part > 0.5, alt.value("white"), alt.value("#2D1420")),
    )
    return (heat + text).properties(height=300, width="container")


def ens_bar_chart(frame, x_title, order):
    """Barres du taux d'erreur par groupe, avec infobulle."""
    return (
        alt.Chart(frame)
        .mark_bar(color="#D81B60")
        .encode(
            x=alt.X("Groupe:N", sort=order, title=x_title, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Taux:Q", title="Taux d'erreur (%)"),
            tooltip=[
                alt.Tooltip("Groupe:N", title=x_title),
                alt.Tooltip("Taux:Q", title="Taux d'erreur (%)", format=".1f"),
                alt.Tooltip("Effectif:Q", title="Nombre d'avis"),
            ],
        )
        .properties(height=260, width="container")
    )


if page == PAGES[6]:
    st.markdown('<div class="section-title">🌲 Ensemble Learning : AdaBoost, XGBoost et Soft Voting</div>', unsafe_allow_html=True)
    st.markdown(
        h("""
        <div class="info-card"><b>Idée :</b> au lieu de s'appuyer sur un seul modèle, les méthodes d'ensemble en
        combinent plusieurs. <b>AdaBoost</b> et <b>XGBoost</b> sont des méthodes de <i>boosting</i> : les arbres sont
        construits l'un après l'autre, chacun corrigeant les erreurs du précédent. Le <b>Soft Voting</b> (bonus)
        fait voter Naïve Bayes et Random Forest sur leurs probabilités. Le Random Forest (<i>bagging</i>) et le
        champion Linear SVC servent de références.</div>
        <div class="info-card"><b>Classe 0 = « non recommandé »</b> (la classe minoritaire, environ 18 % des avis).
        L'accuracy seule étant trompeuse sur des classes déséquilibrées, l'analyse porte surtout sur la précision,
        le rappel et le F1 de cette classe. Le modèle utilisé pour la prédiction en direct reste Linear SVC.</div>
        """),
        unsafe_allow_html=True,
    )

    ens_data = load_ensemble_data()
    if ens_data is None:
        st.warning(
            "Fichiers d'analyse introuvables dans `models/`. Vérifiez que les analyses d'ensemble ont été générées."
        )
    else:
        errors_df, features_df, blocks_df = ens_data
        models = [m for m in ENS_ORDER if m in errors_df.columns]
        y_true = errors_df["y_true"].to_numpy()

        ctrl_a, ctrl_b = st.columns([1, 2])
        with ctrl_a:
            selected = st.selectbox("Modèle analysé", models, key="ens_model")
        with ctrl_b:
            threshold = st.slider(
                "Seuil de décision (score minimal pour prédire « recommandé »)",
                min_value=0.05, max_value=0.95, value=0.50, step=0.01, key="ens_threshold",
            )
        st.markdown(
            '<div class="hint">Un seuil plus élevé rend le modèle plus méfiant : il détecte plus d\'avis négatifs '
            '(meilleur rappel de la classe 0) mais se trompe plus souvent sur les avis positifs.</div>',
            unsafe_allow_html=True,
        )

        rows = []
        for m in models:
            score_m = errors_df[m].to_numpy()
            rows.append({"model": m, **ens_metrics(y_true, score_m, threshold), "auc": roc_auc_score(y_true, score_m)})
        all_metrics = pd.DataFrame(rows)
        best_f1 = all_metrics.loc[all_metrics["f1_0"].idxmax()]
        best_auc = all_metrics.loc[all_metrics["auc"].idxmax()]

        k1, k2, k3, k4 = st.columns(4)
        for col, val, lbl in [
            (k1, f"{len(errors_df):,}".replace(",", " "), "Avis de test"),
            (k2, f"{(y_true == 0).mean():.1%}", "Avis non recommandés"),
            (k3, f"{best_f1['f1_0']:.3f}", f"Meilleur F1 classe 0 · {esc(best_f1['model'])}"),
            (k4, f"{best_auc['auc']:.4f}", f"Meilleur ROC-AUC · {esc(best_auc['model'])}"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)

        tab_cmp, tab_cm, tab_roc, tab_err, tab_feat = st.tabs(
            ["📊 Comparatif", "🧮 Matrice de confusion", "📈 Courbes ROC et PR", "🔍 Analyse des erreurs", "🧠 Variables"]
        )

        # ---------------------------------------------------------------- Comparatif
        with tab_cmp:
            table = all_metrics[
                ["model", "accuracy", "auc", "precision_0", "recall_0", "f1_0", "precision_1", "recall_1", "f1_1"]
            ].rename(columns={
                "model": "Modèle", "accuracy": "Accuracy", "auc": "ROC-AUC",
                "precision_0": "Précision (0)", "recall_0": "Rappel (0)", "f1_0": "F1 (0)",
                "precision_1": "Précision (1)", "recall_1": "Rappel (1)", "f1_1": "F1 (1)",
            })
            metric_cols = [c for c in table.columns if c != "Modèle"]
            st.dataframe(
                table.style.highlight_max(subset=metric_cols, color="#F8BBD0").format({c: "{:.3f}" for c in metric_cols}),
                hide_index=True,
            )
            st.caption("(0) = non recommandé, (1) = recommandé. Valeurs calculées sur les avis de test au seuil choisi ci-dessus.")

            long_df = table.melt(
                id_vars="Modèle", value_vars=["Précision (0)", "Rappel (0)", "F1 (0)"],
                var_name="Mesure", value_name="Valeur",
            )
            bars = (
                alt.Chart(long_df)
                .mark_bar()
                .encode(
                    x=alt.X("Modèle:N", sort=models, title=None, axis=alt.Axis(labelAngle=0)),
                    xOffset="Mesure:N",
                    y=alt.Y("Valeur:Q", title="Valeur", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color(
                        "Mesure:N", title="Classe non recommandé",
                        scale=alt.Scale(domain=["Précision (0)", "Rappel (0)", "F1 (0)"],
                                        range=["#F8BBD0", "#F06292", "#880E4F"]),
                    ),
                    tooltip=["Modèle:N", "Mesure:N", alt.Tooltip("Valeur:Q", format=".3f")],
                )
                .properties(height=320, width="container")
            )
            st.altair_chart(bars, width="stretch")

        # ----------------------------------------------------- Matrice de confusion
        with tab_cm:
            sel = all_metrics[all_metrics["model"] == selected].iloc[0]
            left, right = st.columns([1.1, 1], gap="large")
            with left:
                st.markdown(f'<div class="section-title" style="font-size:1.1rem;">{esc(selected)} · seuil {threshold:.2f}</div>', unsafe_allow_html=True)
                st.altair_chart(ens_confusion_chart(sel["tn"], sel["fp"], sel["fn"], sel["tp"]), width="stretch")
                st.caption("Chaque case indique le nombre d'avis et la part de la classe réelle (ligne). Survolez une case pour le détail.")
            with right:
                st.markdown(
                    h(f"""
                    <div class="info-card"><b>Classe 0 · non recommandé</b><br>
                    Précision : <b>{sel['precision_0']:.3f}</b> · Rappel : <b>{sel['recall_0']:.3f}</b> · F1 : <b>{sel['f1_0']:.3f}</b></div>
                    <div class="info-card"><b>Classe 1 · recommandé</b><br>
                    Précision : <b>{sel['precision_1']:.3f}</b> · Rappel : <b>{sel['recall_1']:.3f}</b> · F1 : <b>{sel['f1_1']:.3f}</b></div>
                    <div class="info-card"><b>Erreurs :</b> {int(sel['fp'])} avis négatifs manqués (faux positifs) et
                    {int(sel['fn'])} fausses alertes (faux négatifs), soit {(sel['fp'] + sel['fn']) / len(errors_df):.1%} des avis de test.</div>
                    """),
                    unsafe_allow_html=True,
                )

            sweep = ens_threshold_frame(y_true, errors_df[selected].to_numpy())
            lines = (
                alt.Chart(sweep)
                .mark_line(strokeWidth=2.5)
                .encode(
                    x=alt.X("Seuil:Q", title="Seuil de décision", scale=alt.Scale(domain=[0.05, 0.95])),
                    y=alt.Y("Valeur:Q", title="Classe non recommandé", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color(
                        "Mesure:N", title=None,
                        scale=alt.Scale(domain=["Précision", "Rappel", "F1"], range=["#F8BBD0", "#F06292", "#880E4F"]),
                    ),
                    tooltip=["Mesure:N", "Seuil:Q", alt.Tooltip("Valeur:Q", format=".3f")],
                )
            )
            rule = (
                alt.Chart(pd.DataFrame({"Seuil": [float(threshold)]}))
                .mark_rule(strokeDash=[5, 4], color="#2D1420")
                .encode(x="Seuil:Q")
            )
            st.markdown('<div class="section-title" style="margin-top:1rem;font-size:1.1rem;">Effet du seuil sur la classe non recommandé</div>', unsafe_allow_html=True)
            st.altair_chart(alt.layer(lines, rule).properties(height=280, width="container"), width="stretch")
            st.markdown(
                '<div class="disclaimer">Le seuil est ici ajusté sur les avis de test : il sert à comprendre le compromis '
                'précision / rappel, pas à annoncer une performance finale.</div>',
                unsafe_allow_html=True,
            )

        with tab_roc:
            chosen = st.multiselect("Modèles à comparer", models, default=models, key="ens_curve_models")
            if not chosen:
                st.info("Sélectionnez au moins un modèle.")
            else:
                roc_df, pr_df, curve_labels = ens_curves(y_true, errors_df, chosen)
                color_scale = alt.Scale(domain=curve_labels, range=ens_colors(curve_labels, chosen))

                roc_chart = (
                    alt.Chart(roc_df)
                    .mark_line(strokeWidth=2.2)
                    .encode(
                        x=alt.X("FPR:Q", title="Taux de faux positifs", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("TPR:Q", title="Taux de vrais positifs", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("Modele:N", title=None, scale=color_scale),
                        tooltip=["Modele:N", alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")],
                    )
                )
                diagonal = (
                    alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
                    .mark_line(strokeDash=[4, 4], color="#999999")
                    .encode(x="x:Q", y="y:Q")
                )
                pr_chart = (
                    alt.Chart(pr_df)
                    .mark_line(strokeWidth=2.2)
                    .encode(
                        x=alt.X("Rappel:Q", title="Rappel (non recommandé)", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("Precision:Q", title="Précision (non recommandé)", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("Modele:N", title=None, scale=color_scale),
                        tooltip=["Modele:N", alt.Tooltip("Rappel:Q", format=".3f"), alt.Tooltip("Precision:Q", format=".3f")],
                    )
                )

                c_roc, c_pr = st.columns(2, gap="large")
                with c_roc:
                    st.markdown('<div class="section-title" style="font-size:1.1rem;">Courbes ROC</div>', unsafe_allow_html=True)
                    st.altair_chart(alt.layer(roc_chart, diagonal).properties(height=360, width="container"), width="stretch")
                with c_pr:
                    st.markdown('<div class="section-title" style="font-size:1.1rem;">Précision-rappel (classe non recommandé)</div>', unsafe_allow_html=True)
                    st.altair_chart(pr_chart.properties(height=360, width="container"), width="stretch")
                st.caption(
                    "Plus une courbe ROC est proche du coin supérieur gauche, meilleur est le modèle. La courbe "
                    "précision-rappel montre le prix à payer en précision pour détecter davantage d'avis négatifs."
                )

        with tab_err:
            pred_sel = (errors_df[selected].to_numpy() >= threshold).astype(int)
            is_error = pred_sel != y_true
            length_group = pd.cut(
                errors_df["n_words"], [-1, 20, 40, 60, 80, 10000], labels=ENS_LENGTH_LABELS
            )

            st.markdown(
                f'<div class="section-title" style="font-size:1.1rem;">Où {esc(selected)} se trompe-t-il ? (seuil {threshold:.2f})</div>',
                unsafe_allow_html=True,
            )
            e1, e2, e3 = st.columns(3, gap="medium")
            with e1:
                st.markdown("**Par note (Rating)**")
                by_rating = ens_error_rates(errors_df["Rating"], is_error)
                st.altair_chart(ens_bar_chart(by_rating, "Note", ["1", "2", "3", "4", "5"]), width="stretch")
            with e2:
                st.markdown("**Par département**")
                by_dept = ens_error_rates(errors_df["Department Name"], is_error)
                by_dept = by_dept[by_dept["Effectif"] >= 30]
                dept_order = by_dept.sort_values("Effectif", ascending=False)["Groupe"].tolist()
                st.altair_chart(ens_bar_chart(by_dept, "Département", dept_order), width="stretch")
            with e3:
                st.markdown("**Par longueur de l'avis**")
                by_len = ens_error_rates(length_group, is_error)
                st.altair_chart(ens_bar_chart(by_len, "Longueur", ENS_LENGTH_LABELS), width="stretch")
            st.caption("Les départements de moins de 30 avis de test ne sont pas affichés.")

            st.markdown('<div class="section-title" style="margin-top:1rem;font-size:1.1rem;">Avis mal classés</div>', unsafe_allow_html=True)
            f1c, f2c, f3c = st.columns([1.3, 1, 1])
            with f1c:
                kind = st.radio(
                    "Type d'erreur",
                    ["Toutes les erreurs", "Avis négatifs manqués (FP)", "Fausses alertes (FN)"],
                    key="ens_kind",
                )
            with f2c:
                ratings = st.multiselect("Notes", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5], key="ens_ratings")
            with f3c:
                n_rows = st.slider("Nombre d'avis affichés", min_value=5, max_value=100, value=20, step=5, key="ens_nrows")

            wrong = errors_df.loc[is_error].copy()
            wrong["Type d'erreur"] = np.where(
                y_true[is_error] == 0, "Avis négatif manqué (FP)", "Fausse alerte (FN)"
            )
            wrong["Confiance"] = np.where(pred_sel[is_error] == 1, wrong[selected], 1 - wrong[selected])
            if kind == "Avis négatifs manqués (FP)":
                wrong = wrong[wrong["Type d'erreur"].str.contains("FP")]
            elif kind == "Fausses alertes (FN)":
                wrong = wrong[wrong["Type d'erreur"].str.contains("FN")]
            wrong = wrong[wrong["Rating"].isin(ratings)]
            wrong = wrong.sort_values("Confiance", ascending=False).head(n_rows)

            st.caption("Triés par confiance décroissante : les erreurs les plus sûres d'elles-mêmes sont les plus instructives.")
            if wrong.empty:
                st.info("Aucun avis mal classé pour ces filtres.")
            else:
                shown = wrong[["Type d'erreur", "Rating", "Department Name", "Confiance", "Title", "Review Text"]].rename(
                    columns={"Rating": "Note", "Department Name": "Département", "Title": "Titre", "Review Text": "Avis"}
                )
                shown["Confiance"] = shown["Confiance"].round(2)
                st.dataframe(shown, hide_index=True)

        with tab_feat:
            feat_models = [m for m in ["XGBoost", "AdaBoost"] if m in features_df["Model"].unique()]
            if not feat_models:
                st.info("Aucune importance de variables disponible.")
            else:
                f_left, f_right = st.columns([1.3, 1], gap="large")
                with f_left:
                    feat_model = st.radio("Modèle", feat_models, horizontal=True, key="ens_feat_model")
                    top = features_df[features_df["Model"] == feat_model].sort_values("Importance", ascending=False).head(15)
                    top_chart = (
                        alt.Chart(top)
                        .mark_bar(color="#D81B60")
                        .encode(
                            y=alt.Y("Feature:N", sort="-x", title=None),
                            x=alt.X("Importance:Q", title="Importance"),
                            tooltip=["Feature:N", "Block:N", alt.Tooltip("Importance:Q", format=".4f")],
                        )
                        .properties(height=420, width="container")
                    )
                    st.altair_chart(top_chart, width="stretch")
                with f_right:
                    st.markdown('<div class="section-title" style="font-size:1.1rem;">Importance par bloc de variables</div>', unsafe_allow_html=True)
                    blocks_long = (blocks_df * 100).reset_index().melt(id_vars="Bloc", var_name="Modele", value_name="Part")
                    block_chart = (
                        alt.Chart(blocks_long)
                        .mark_bar()
                        .encode(
                            y=alt.Y("Bloc:N", title=None, axis=alt.Axis(labelLimit=220)),
                            yOffset="Modele:N",
                            x=alt.X("Part:Q", title="Part de l'importance (%)"),
                            color=alt.Color("Modele:N", title=None,
                                            scale=alt.Scale(domain=["XGBoost", "AdaBoost"], range=["#880E4F", "#F06292"])),
                            tooltip=["Bloc:N", "Modele:N", alt.Tooltip("Part:Q", format=".1f")],
                        )
                        .properties(height=300, width="container")
                    )
                    st.altair_chart(block_chart, width="stretch")
                st.caption(
                    "L'importance mesure la contribution de chaque variable aux décisions des arbres. Les mots du texte "
                    "(TF-IDF) sont très nombreux : leur part cumulée est à comparer à celle de la note (Rating)."
                )


# ==============================================================================
# PAGE 8 : À PROPOS
# ==============================================================================
if page == PAGES[7]:
    st.markdown('<div class="section-title">ℹ️ À propos</div>', unsafe_allow_html=True)
    st.markdown(
        h("""
        <div class="info-card"><b>Projet :</b> Prédiction des recommandations d'achats · Data Mining &
        Machine Learning supervisé<br>
        """),
        unsafe_allow_html=True,
    )
    st.markdown("**Stack technique**")
    st.markdown(
        "".join(f'<span class="badge">{t}</span>' for t in
                ["Python", "Streamlit", "scikit-learn", "XGBoost", "Altair", "pandas", "NumPy", "joblib", "TF-IDF", "GridSearchCV"]),
        unsafe_allow_html=True,
    )
    st.markdown(
        h("""
        <div class="disclaimer">⚠️ Application développée à des fins académiques. Les prédictions sont
        indicatives et dépendent des données d'entraînement (avis en anglais).</div>
        """),
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="site-foot"><span>✿</span> PRÉDICTION DES RECOMMANDATIONS D\'ACHATS · ÉCOLE MILITAIRE POLYTECHNIQUE <span>✿</span></div>',
    unsafe_allow_html=True,
)
# ==============================================================================
# PAGE : ANALYSE DES ERREURS - LINEAR SVC
# ==============================================================================

if page == PAGES[8]:

    st.markdown(
        '<div class="section-title">🔍 Analyse des erreurs — Linear SVC</div>',
        unsafe_allow_html=True
    )

    # Charge la matrice de confusion enregistrée.
    cm_path = MODELS_DIR / "linear_svc_confusion_matrix.csv"

    if not cm_path.exists():
        st.error(
            "Matrice de confusion introuvable. "
            "Les données d'évaluation sont introuvables."
        )

    else:
        cm_df = pd.read_csv(cm_path, index_col=0)
        cm = cm_df.values

        tn, fp = cm[0]
        fn, tp = cm[1]

        # Calculate error rates
        fp_rate = fp / (tn + fp)
        fn_rate = fn / (tp + fn)

        # Main metrics
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("True Negatives", f"{tn:,}")
        c2.metric("False Positives", f"{fp:,}")
        c3.metric("False Negatives", f"{fn:,}")
        c4.metric("True Positives", f"{tp:,}")

        st.markdown("### Matrice de confusion")

        st.dataframe(
            cm_df.style.background_gradient(cmap="RdPu"),
            use_container_width=True
        )

        st.markdown("### Comparaison des erreurs")

        error_df = pd.DataFrame({
            "Type d'erreur": ["False Positives", "False Negatives"],
            "Nombre": [fp, fn]
        }).set_index("Type d'erreur")

        st.bar_chart(error_df)

        st.markdown("### Taux d'erreur par classe")

        col1, col2 = st.columns(2)

        col1.metric(
            "False Positive Rate",
            f"{fp_rate:.2%}"
        )

        col2.metric(
            "False Negative Rate",
            f"{fn_rate:.2%}"
        )

        st.markdown("### Interprétation des résultats")

        st.info(
            f"Le modèle a correctement identifié {tn:,} avis non recommandés "
            f"et {tp:,} avis recommandés."
        )

        st.warning(
            f"Il a classé à tort {fp:,} avis non recommandés comme recommandés "
            f"(False Positives), et {fn:,} avis recommandés comme non recommandés "
            f"(False Negatives)."
        )

        st.markdown(
            "**Conclusion :** Linear SVC présente des erreurs dans les deux classes. "
            "Les taux d'erreur permettent d'évaluer séparément sa capacité "
            "à identifier les avis non recommandés et recommandés."
        )


# ==============================================================================

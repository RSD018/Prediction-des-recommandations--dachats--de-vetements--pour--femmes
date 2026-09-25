import html as _html
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st


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

PAGES = [
    "🏠 Accueil",
    "🔮 Prédiction en Direct",
    "📊 Performance des Modèles",
    "📈 Statistiques du Dataset",
    "💡 Découvertes Clés",
    "ℹ️ À propos",
]


ticker_items = [
    "✦ NOUVELLE COLLECTION AUTOMNE-HIVER",
    "✦ MODÈLES DE MACHINE LEARNING CLASSIQUES",
    "✦ VALIDATION CROISÉE STRATIFIÉE 5-FOLD",
    "✦ SÉLECTION D'HYPERPARAMÈTRES PAR GRIDSEARCHCV",
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
        (k2, "6", "Algorithmes comparés"),
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
        (p3, "🤖", "3. Modélisation", "6 algorithmes, GridSearchCV, validation croisée stratifiée 5-Fold."),
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
# ONGLET 2 : PERFORMANCE ET COMPARATIF DES MODÈLES (Exigence Dr. Hosni)
# ==============================================================================
if page == PAGES[2]:
    st.markdown('<div class="section-title">📊 Comparatif des Modèles Classiques</div>', unsafe_allow_html=True)
    st.markdown(
        "Évaluation comparative des algorithmes de classification supervisée entraînés sur le dataset "
        "avec **GridSearchCV (Stratified 5-Fold CV)** et scoring optimisé sur le **F1-Score**."
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="metric-card"><div class="metric-val">6</div><div class="metric-lbl">Modèles Évalués</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><div class="metric-val">0.9550</div><div class="metric-lbl">Meilleur F1-Score</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><div class="metric-val">0.9650</div><div class="metric-lbl">Meilleur ROC-AUC</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-card"><div class="metric-val">Linear SVC</div><div class="metric-lbl">Modèle Champion</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    benchmark_data = pd.DataFrame([
    {
        "Modèle": "Linear SVC / SVM",
        "Accuracy": 0.9378,
        "F1-Score": 0.9622,
        "ROC-AUC": 0.9775,
        "Meilleurs Hyperparamètres": "{'C': 0.1, 'loss': 'squared_hinge'}"
    },
    {
        "Modèle": "Random Forest",
        "Accuracy": 0.9356,
        "F1-Score": 0.9613,
        "ROC-AUC": 0.9730,
        "Meilleurs Hyperparamètres": "{'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}"
    },
    {
        "Modèle": "Decision Tree",
        "Accuracy": 0.9301,
        "F1-Score": 0.9563,
        "ROC-AUC": 0.9684,
        "Meilleurs Hyperparamètres": "{'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 5}"
    },
    {
        "Modèle": "K-Nearest Neighbors (k-NN)",
        "Accuracy": 0.9186,
        "F1-Score": 0.9519,
        "ROC-AUC": 0.9642,
        "Meilleurs Hyperparamètres": "{'algorithm': 'brute', 'n_neighbors': 21, 'weights': 'uniform'}"
    },
    {
        "Modèle": "Multinomial Naïve Bayes",
        "Accuracy": 0.9052,
        "F1-Score": 0.9425,
        "ROC-AUC": 0.9531,
        "Meilleurs Hyperparamètres": "{'alpha': 0.1}"
    },
    {
        "Modèle": "Zero-R (Baseline)",
        "Accuracy": 0.8223,
        "F1-Score": 0.9025,
        "ROC-AUC": "N/A",
        "Meilleurs Hyperparamètres": "{'strategy': 'most_frequent'}"
    }
])

    st.dataframe(
        benchmark_data.style.highlight_max(subset=["Accuracy", "F1-Score"], color="#F8BBD0"),
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div class="section-title" style="margin-top:2rem;">📈 Graphique Comparatif (F1-Score)</div>', unsafe_allow_html=True)
    chart_df = benchmark_data.set_index("Modèle")[["F1-Score", "Accuracy"]]
    st.bar_chart(chart_df, color=["#D81B60", "#F8BBD0"])

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
         "Quel algorithme généralise le mieux ?",
         "Linear SVC : F1 = 0.9622, ROC-AUC = 0.9775, devant Random Forest (F1 = 0.9613).",
         "Les modèles linéaires conviennent bien aux données textuelles creuses et de grande dimension (13 035 variables)."),
        ("Limite de l'arbre de décision",
         "Un modèle simple suffit-il ?",
         "Decision Tree : ROC-AUC = 0.85, le plus faible des modèles évalués.",
         "Un arbre unique sépare mal les classes sur des données TF-IDF ; les méthodes d'ensemble ou linéaires sont préférables."),
        ("Apport du tuning",
         "Que gagne-t-on avec GridSearchCV ?",
         "Tous les modèles battent nettement le baseline en F1 (0.93 à 0.955 contre 0.9025).",
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
# PAGE 6 : À PROPOS
# ==============================================================================
if page == PAGES[5]:
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
                ["Python", "Streamlit", "scikit-learn", "pandas", "NumPy", "joblib", "TF-IDF", "GridSearchCV"]),
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
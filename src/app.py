import html as _html

import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

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
    """Aplatit le HTML : évite que Markdown le prenne pour un bloc de code
    à cause de l'indentation."""
    return " ".join(line.strip() for line in s.strip().splitlines() if line.strip())


def esc(s: str) -> str:
    """Échappe le texte saisi par l'utilisateur avant de l'injecter dans le HTML."""
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
    "Dresses": "Robes", "Tops": "Hauts", "Bottoms": "Bas",
    "Jackets": "Vestes & Manteaux", "Intimate": "Lingerie & Détente",
    "Trend": "Tendances",
}
CLASS_LABELS = {
    "Dresses": "Robes", "Knits": "Mailles", "Blouses": "Blouses",
    "Sweaters": "Pulls", "Fine gauge": "Maille fine", "Pants": "Pantalons",
    "Jeans": "Jeans", "Skirts": "Jupes", "Shorts": "Shorts",
    "Jackets": "Vestes", "Outerwear": "Manteaux", "Lounge": "Loungewear",
    "Sleep": "Nuit", "Swim": "Maillots de bain", "Intimates": "Lingerie",
    "Legwear": "Collants & Chaussettes", "Trend": "Pièces tendance",
}
DIV_LABELS = {
    "General": "Collection Classique",
    "General Petite": "Collection Petite",
    "Initmates": "Collection Intime",
}
DEPT_EMOJI = {
    "Dresses": "👗", "Tops": "👚", "Bottoms": "👖",
    "Jackets": "🧥", "Intimate": "🩱", "Trend": "✨",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #FFF7FA; }
.block-container { padding-top: 3.2rem; max-width: 1150px; }

header[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], [data-testid="stToolbarActions"],
[data-testid="stDecoration"], #MainMenu, footer { display: none !important; }

[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { visibility: visible !important; opacity: 1 !important; }
[data-testid="stExpandSidebarButton"] button, [data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
    background: #D81B60 !important; color: #FFFFFF !important; border-radius: 50% !important;
}
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] *,
[data-testid="collapsedControl"] * { color: #880E4F !important; fill: #FFFFFF !important; }
[data-testid="stSidebarCollapseButton"] * { color: #880E4F !important; }

.ticker {
    background: linear-gradient(90deg, #880E4F 0%, #E91E63 50%, #880E4F 100%);
    color: #FFFFFF; overflow: hidden; white-space: nowrap;
    border-radius: 10px; padding: 9px 0; font-size: 0.75rem; letter-spacing: 2.5px;
    box-shadow: 0 4px 14px rgba(216, 27, 96, 0.25);
}
.ticker-track { display: inline-block; animation: scroll 32s linear infinite; }
.ticker-track span { margin: 0 34px; }
@keyframes scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }

.navbar { text-align: center; padding: 22px 8px 16px 8px; border-bottom: 1px solid #F2C9D8; }
.brand {
    font-family: 'Playfair Display', serif; font-size: 1.85rem; color: #2D1420;
    letter-spacing: 3px; line-height: 1.25;
}
.brand small { display: block; font-family: 'Inter', sans-serif; font-size: 0.75rem;
    letter-spacing: 5px; color: #D81B60; margin-top: 8px; font-weight: 600; }
.tagline { text-align: center; color: #8A6B78; font-size: 0.95rem; margin: 14px 0 22px 0; }

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

[data-testid="stWidgetLabel"] p, label p, label {
    color: #2D1420 !important; font-weight: 600 !important; font-size: 0.9rem !important;
}

[data-testid="stSlider"] [role="slider"], [data-testid="stSelectSlider"] [role="slider"] {
    background-color: #D81B60 !important; border-color: #FFFFFF !important;
    box-shadow: 0 0 0 5px rgba(216, 27, 96, 0.18) !important;
}
[data-testid="stThumbValue"] {
    color: #D81B60 !important; font-size: 1.1rem !important; letter-spacing: 2px !important;
    font-weight: 600 !important;
}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: #8A6B78 !important; }

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
</style>
"""

_W = ["stSelectbox", "stNumberInput", "stTextInput", "stTextArea"]
_NL = ':not([data-testid="stWidgetLabel"])'


def _box(suffix=""):
    return ", ".join(f'[data-testid="{w}"] > div{_NL}{suffix}' for w in _W)


def _inner(suffix=""):
    return ", ".join(f'[data-testid="{w}"] > div{_NL} *{suffix}' for w in _W)


def _btn(suffix=""):
    return f'[data-testid="stNumberInput"] > div{_NL} button{suffix}'


FIELD_CSS = f"""
{_box()} {{
    border: 1.5px solid #EBB9CB !important; border-radius: 10px !important;
    box-shadow: 0 2px 6px rgba(216, 27, 96, 0.06) !important;
}}
{_box(":hover")} {{ border-color: #F06292 !important; }}
{_box(":focus-within")} {{
    border-color: #D81B60 !important; box-shadow: 0 0 0 3px rgba(216, 27, 96, 0.15) !important;
}}
{_inner()} {{
    background-color: #FFF9FB !important; color: #2D1420 !important;
    -webkit-text-fill-color: #2D1420 !important;
}}
{_inner(" svg")} {{ fill: #D81B60 !important; }}
{_inner("::placeholder")} {{ color: #A98896 !important; -webkit-text-fill-color: #A98896 !important; }}
{_btn()}, {_btn(" *")} {{
    background-color: #FCE4EC !important; color: #AD1457 !important;
    fill: #AD1457 !important; -webkit-text-fill-color: #AD1457 !important;
}}
{_btn(":hover")} {{ background-color: #F8BBD0 !important; }}
[data-baseweb="popover"], [data-baseweb="popover"] *, [role="listbox"], [role="listbox"] *,
ul[role="listbox"] li {{
    background-color: #FFFFFF !important; color: #2D1420 !important;
    -webkit-text-fill-color: #2D1420 !important;
}}
[role="option"]:hover, [role="option"]:hover *,
li[role="option"][aria-selected="true"], li[role="option"][aria-selected="true"] * {{
    background-color: #FCE4EC !important;
}}
"""
CSS = CSS.replace("</style>", FIELD_CSS + "</style>")
st.markdown(h(CSS), unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


@st.cache_resource
def load_resources():
    model_path = MODELS_DIR / "best_model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    if model_path.exists() and preprocessor_path.exists():
        return joblib.load(model_path), joblib.load(preprocessor_path)
    return None, None


model, preprocessor = load_resources()

ticker_items = [
    "✦ NOUVELLE COLLECTION AUTOMNE-HIVER",
    "✦ LIVRAISON OFFERTE DÈS 60 €",
    "✦ RETOURS GRATUITS SOUS 30 JOURS",
    "✦ L'AVIS DE NOS CLIENTES ANALYSÉ PAR IA",
]
ticker_html = "".join(f"<span>{t}</span>" for t in ticker_items)

st.markdown(
    h(f"""
    <div class="ticker"><div class="ticker-track">{ticker_html}{ticker_html}</div></div>
    <div class="navbar">
        <div class="brand">PRÉDICTION DES RECOMMANDATIONS D'ACHATS
            <small>VÊTEMENTS POUR FEMME</small></div>
    </div>
    <div class="tagline">Une cliente recommandera-t-elle cet article ? Laissez l'IA analyser son avis.</div>
    """),
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        h("""
        <div class="side-brand">Recommandations d'achats</div>
        <div class="side-sub">MODE FEMME</div>

        <div class="side-card">
            Un modèle de Machine Learning, entraîné sur des avis réels de clientes
            d'une boutique de mode en ligne, prédit si l'article sera <b>recommandé</b>.
        </div>

        <div class="side-title">Comment ça marche ?</div>
        <div class="step"><div class="step-num">1</div><div>Choisissez l'article : département, catégorie et collection.</div></div>
        <div class="step"><div class="step-num">2</div><div>Renseignez le profil de la cliente.</div></div>
        <div class="step"><div class="step-num">3</div><div>Rédigez son avis et sa note.</div></div>
        <div class="step"><div class="step-num">4</div><div>Lancez l'analyse et lisez la recommandation.</div></div>

        <div class="side-title">Conseil</div>
        <div class="side-card">
            Plus l'avis est détaillé, plus la prédiction est fiable. Un titre et un texte
            expressifs aident le modèle à saisir le ressenti de la cliente.
        </div>

        <div class="side-foot">Projet Data Mining · Prédiction des recommandations d'achat</div>
        """),
        unsafe_allow_html=True,
    )

if model is None or preprocessor is None:
    st.error(
        "⚠️ Fichiers introuvables ! Assurez-vous d'avoir exécuté la sauvegarde de "
        "`models/best_model.pkl` et `models/preprocessor.pkl`."
    )
else:
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        # Fiche produit (hors formulaire pour que les listes soient liées entre elles)
        st.markdown('<div class="section-title">👗 Fiche produit</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            department_name = st.selectbox(
                "Département", list(DEPARTMENTS),
                format_func=lambda d: DEPT_LABELS[d], key="department",
            )
        with c2:
            class_name = st.selectbox(
                "Catégorie", DEPARTMENTS[department_name],
                format_func=lambda c: CLASS_LABELS[c], key=f"class_{department_name}",
            )
        with c3:
            division_name = st.selectbox(
                "Collection", DIVISIONS[department_name],
                format_func=lambda d: DIV_LABELS[d], key=f"division_{department_name}",
            )

        st.markdown(
            f'<div class="product-chip">{DEPT_EMOJI[department_name]} '
            f'{DEPT_LABELS[department_name]} › {CLASS_LABELS[class_name]} · '
            f'{DIV_LABELS[division_name]}</div>',
            unsafe_allow_html=True,
        )

        # Profil + avis
        with st.form("review_form"):
            st.markdown('<div class="section-title">👩 Profil de la cliente</div>', unsafe_allow_html=True)
            c4, c5 = st.columns(2)
            with c4:
                age = st.number_input("Âge", min_value=18, max_value=100, value=33)
            with c5:
                positive_feedback_count = st.number_input(
                    "👍 Avis jugé utile par", min_value=0, value=3,
                    help="Nombre de personnes ayant trouvé cet avis utile.",
                )

            st.markdown('<div class="section-title" style="margin-top:1.2rem;">⭐ Avis client</div>',
                        unsafe_allow_html=True)
            rating = st.select_slider(
                "Note attribuée", options=[1, 2, 3, 4, 5], value=5,
                format_func=lambda x: "★" * x + "☆" * (5 - x),
            )
            title_text = st.text_input("Titre de l'avis", value="I love it")
            review_text = st.text_area("Texte de l'avis", value="Thank you", height=120)
            st.markdown(
                '<div class="hint">💡 Les avis du dataset sont en anglais : rédigez l\'avis en anglais '
                'pour de meilleurs résultats.</div>',
                unsafe_allow_html=True,
            )

            submitted = st.form_submit_button("✦ ANALYSER L'AVIS")

    with right:
        st.markdown('<div class="section-title">📊 Résultat de l\'analyse</div>', unsafe_allow_html=True)

        if not submitted:
            st.markdown(
                h("""
                <div class="placeholder-box">
                    <div style="font-size:2.5rem;">🧵</div>
                    <p>Renseignez la fiche et lancez l'analyse pour découvrir
                    si l'article sera recommandé.</p>
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
                    title, sub = "Article recommandé", "La cliente recommanderait ce produit."
                else:
                    css, fill, icon = "result-no", "fill-no", "💔"
                    title, sub = "Article non recommandé", "La cliente ne recommanderait pas ce produit."

                extra_html = ""
                pills_html = ""
                if proba is not None:
                    confidence = proba[prediction] * 100
                    p_no, p_yes = proba[0] * 100, proba[1] * 100
                    extra_html = f"""
                        <div class="result-sub" style="margin-top:12px;">
                            Niveau de confiance du modèle : <b>{confidence:.2f}%</b>
                        </div>
                        <div class="conf-track">
                            <div class="conf-fill {fill}" style="width:{confidence:.1f}%;"></div>
                        </div>
                    """
                    pills_html = f"""
                        <div class="pills">
                            <div class="pill"><b>{p_yes:.1f}%</b>Probabilité de recommandation</div>
                            <div class="pill"><b>{p_no:.1f}%</b>Probabilité de non-recommandation</div>
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
                st.error(f"Erreur lors du prétraitement des données : {e}")

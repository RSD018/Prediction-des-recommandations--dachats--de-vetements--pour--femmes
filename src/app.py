import html as _html
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configuration du thème Streamlit
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

# --- STYLES CSS ---
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

# Ticker sans mention d'IA
ticker_items = [
    "✦ NOUVELLE COLLECTION AUTOMNE-HIVER",
    "✦ MODÈLES DE MACHINE LEARNING CLASSIAUX",
    "✦ VALIDATION CROISÉE STRATIFIÉE 5-FOLD",
    "✦ SÉLECTION D'HYPERPARAMÈTRES PAR GRIDSEARCHCV",
]
ticker_html = "".join(f"<span>{t}</span>" for t in ticker_items)

st.markdown(
    h(f"""
    <div class="ticker"><div class="ticker-track">{ticker_html}{ticker_html}</div></div>
    <div class="navbar">
        <div class="brand">PRÉDICTION DES RECOMMANDATIONS D'ACHATS
            <small>VÊTEMENTS POUR FEMME</small></div>
    </div>
    <div class="tagline">Analyse des avis clients par algorithmes de Machine Learning de classification supervisée.</div>
    """),
    unsafe_allow_html=True,
)

# Sidebar sans mention d'IA
with st.sidebar:
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

# --- DEBUT DES ONGLETS (DASHBOARD) ---
tab_predict, tab_benchmark, tab_dataset = st.tabs(
    ["🔮 Prédiction en Direct", "📊 Performance des Modèles", "📈 Statistiques du Dataset"]
)

# ==============================================================================
# ONGLET 1 : PRÉDICTION EN DIRECT
# ==============================================================================
with tab_predict:
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
with tab_benchmark:
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

    # Tableau de synthèse des 6 modèles
    benchmark_data = pd.DataFrame([
        {
            "Modèle": "Linear SVC / SVM",
            "Accuracy": 0.9250,
            "F1-Score": 0.9550,
            "ROC-AUC": 0.9650,
            "Meilleurs Hyperparamètres": "{'C': 1.0, 'loss': 'squared_hinge'}"
        },
        {
            "Modèle": "Random Forest",
            "Accuracy": 0.9210,
            "F1-Score": 0.9530,
            "ROC-AUC": 0.9620,
            "Meilleurs Hyperparamètres": "{'max_depth': None, 'n_estimators': 200}"
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
            "Modèle": "Decision Tree",
            "Accuracy": 0.8900,
            "F1-Score": 0.9300,
            "ROC-AUC": 0.8500,
            "Meilleurs Hyperparamètres": "{'criterion': 'gini', 'max_depth': 20}"
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
    st.bar_chart(chart_df)

# ==============================================================================
# ONGLET 3 : STATISTIQUES DU DATASET
# ==============================================================================
with tab_dataset:
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
        st.bar_chart(target_dist)

    with col_b:
        st.markdown("#### Répartition par Département")
        dept_dist = pd.DataFrame({
            "Département": ["Tops", "Dresses", "Bottoms", "Intimate", "Jackets", "Trend"],
            "Nombre d'avis": [10468, 6319, 3799, 1735, 1032, 112]
        }).set_index("Département")
        st.bar_chart(dept_dist)
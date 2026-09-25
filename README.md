# 🛍️ Prédiction des Recommandations d'Achats — Women's E-Commerce Clothing Reviews

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-red)
![License](https://img.shields.io/badge/Deep%20Learning-Aucun-lightgrey)

Projet de **Data Mining / Machine Learning classique** (sans Deep Learning) visant à prédire si une cliente **recommande** ou **ne recommande pas** un article de mode à partir de son avis (texte + métadonnées produit).

Le pipeline complet — nettoyage, prétraitement, entraînement, évaluation — aboutit à une **application web interactive** (Streamlit) permettant de tester le modèle en direct.

📄 Le rapport détaillé du mini-projet est disponible dans [`docs/Rapport_MiniProject.pdf`](docs/Rapport_MiniProject.pdf).

---

## 📑 Sommaire

- [Objectif du projet](#-objectif-du-projet)
- [Structure du projet](#️-structure-du-projet)
- [Dataset](#-dataset)
- [Démarche Data Mining](#-démarche-data-mining)
- [Application Streamlit](#️-application-streamlit-srcapppy)
- [Stack technique](#️-stack-technique)
- [Installation](#️-installation)
- [Reproductibilité](#-reproductibilité)
- [Limites](#️-limites)
- [Pistes d'amélioration](#-pistes-damélioration)

---

## 📌 Objectif du projet

À partir du dataset **Women's E-Commerce Clothing Reviews** (23 486 avis clients), prédire la variable cible binaire :

- `Recommended IND = 1` → la cliente recommande le produit
- `Recommended IND = 0` → la cliente ne recommande pas le produit

en utilisant uniquement des algorithmes de **Machine Learning classique** (aucun réseau de neurones / deep learning).

---

## 🗂️ Structure du projet

```
Prediction-des-recommandations-dachat/
│
├── data/
│   ├── raw/
│   │   └── reviews.csv                  # Dataset brut (23 486 lignes, 11 colonnes)
│   └── processed/
│       └── reviews_clean.csv            # Dataset nettoyé (notebook 01)
│
├── docs/
│   └── Rapport_MiniProject.pdf          # Rapport écrit du mini-projet
│
├── models/
│   ├── best_model.pkl                   # Modèle final (Linear SVC)
│   ├── preprocessor.pkl                 # ColumnTransformer (imputation, scaling, OHE, TF-IDF)
│   ├── review_tfidf.pkl                 # Vectoriseur TF-IDF (Review Text)
│   └── title_tfidf.pkl                  # Vectoriseur TF-IDF (Title)
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb        # Exploration, nettoyage, tests statistiques
│   ├── 02_preprocessing_encoding.ipynb  # Split train/test, encodage, TF-IDF
│   └── 03_classification_models.ipynb   # Entraînement, comparaison, sauvegarde du modèle
│
├── src/
│   ├── app.py                           # Application Streamlit (démo interactive)
│   └── config.toml                      # Thème Streamlit (couleurs, mode clair)
│
├── venv/ / .venv/                       # Environnement virtuel (ignoré par git)
├── .gitignore
├── README.md
└── requirements.txt
```

> 💡 Après avoir cloné le repo, place `reviews.csv` dans `data/raw/` et exécute les notebooks dans l'ordre (01 → 02 → 03) pour régénérer `data/processed/reviews_clean.csv` et les fichiers `models/*.pkl`. L'app (`src/app.py`) résout automatiquement le chemin vers `models/` à la racine du projet (`Path(__file__).resolve().parents[1]`), donc aucune configuration manuelle n'est nécessaire une fois les modèles générés.

---

## 📊 Dataset

- **Source** : Women's E-Commerce Clothing Reviews (Kaggle)
- **Volume** : 23 486 observations brutes, 11 variables
- **Variable cible** : `Recommended IND` (classification binaire)
- **Déséquilibre des classes** : 82,2 % recommandé (1) vs 17,8 % non recommandé (0)

| Colonne                  | Type         | Description          |
|--------------------------|--------------|----------------------|
| `Clothing ID`            | identifiant  | ID du produit        |
| `Age`                    | quantitative | Âge de la cliente    |
| `Title`                  | textuelle    | Titre de l'avis      |
| `Review Text`            | textuelle    | Corps de l'avis      |
| `Rating`                 | ordinale     | Note de 1 à 5        |
| `Recommended IND`        | cible        | 0 / 1                |
| `Positive Feedback Count`| quantitative | Nombre de retours positifs |
| `Division Name`          | catégorielle | Division (General, General Petite, Initmates) |
| `Department Name`        | catégorielle | Département (Dresses, Tops, Bottoms, Jackets, Intimate, Trend) |
| `Class Name`             | catégorielle | Sous-catégorie produit |

---

## 🔬 Démarche Data Mining

### 1️⃣ EDA & Nettoyage (`01_eda_and_cleaning.ipynb`)
- Exploration structurelle du dataset, typage des variables (quantitatives, catégorielles, textuelles, cible)
- Analyse des valeurs manquantes et des doublons
- Statistiques descriptives, distributions (histogrammes, KDE), détection des outliers (règle de Tukey / IQR)
- Analyse de la variable cible et de son déséquilibre
- Matrice de corrélation (Pearson)
- **Tests statistiques** :
  - Mann-Whitney U (`Rating` vs `Recommended IND`)
  - Chi² d'indépendance (`Department Name` vs `Recommended IND`)
- Nettoyage : suppression des doublons, espaces, chaînes vides → `Unknown`, gestion des valeurs textuelles/catégorielles manquantes, suppression des lignes à cible manquante
- Sauvegarde → `data/processed/reviews_clean.csv`

### 2️⃣ Prétraitement & Encodage (`02_preprocessing_encoding.ipynb`)
- Séparation `X` / `y`, suppression de `Clothing ID` (non numérique au sens statistique)
- **Split train/test stratifié** (80 % / 20 %)
- Construction d'un pipeline `ColumnTransformer` combinant :
  - **Numériques** (`Age`, `Rating`, `Positive Feedback Count`) → `SimpleImputer(median)` + `StandardScaler`
  - **Catégorielles** (`Division`, `Department`, `Class`) → `SimpleImputer(most_frequent)` + `OneHotEncoder`
  - **Textuelles** (`Title`, `Review Text`) → **TF-IDF** (vectorisation séparée)
- `fit_transform` sur le train uniquement, `transform` sur le test (évite toute fuite de données)
- Résultat : ~13 000 features (matrices creuses / sparse)

### 3️⃣ Modélisation & Évaluation (`03_classification_models.ipynb`)
Algorithmes comparés (tous optimisés par `GridSearchCV`, validation croisée stratifiée) :

| Modèle                    | Accuracy   | F1-Score   | ROC-AUC    |
|---------------------------|------------|------------|------------|
| **Linear SVC (champion)** | **0.9378** | **0.9622** | **0.9775** |
| Random Forest             | 0.9356     | 0.9613     | 0.9730     |
| Decision Tree             | 0.9301     | 0.9563     | 0.9684     |
| K-Nearest Neighbors       | 0.9186     | 0.9519     | 0.9642     |
| Multinomial Naïve Bayes   | 0.9052     | 0.9425     | 0.9531     |
| Zero-R (Baseline)         | 0.8223     | 0.9025     | N/A        |

- Le modèle **Linear SVC** est sélectionné comme meilleur modèle et sérialisé (`joblib`) dans `models/best_model.pkl`.
- Le baseline Zero-R (82,2 % d'accuracy) rappelle que l'accuracy seule est trompeuse sur un dataset déséquilibré → le F1-Score et le ROC-AUC sont privilégiés.

---

## 🖥️ Application Streamlit (`src/app.py`)

Interface interactive avec plusieurs onglets :

1. **🏠 Accueil** — présentation du projet
2. **🔮 Prédiction en Direct** — saisie d'un profil (département, catégorie, âge, note, avis) → prédiction en temps réel via `model.predict()` sur les données transformées par `preprocessor.pkl`
3. **📊 Performance des Modèles** — tableau comparatif + graphique F1-Score
4. **📈 Statistiques du Dataset** — répartition de la cible, répartition par département
5. **💡 Découvertes Clés** — synthèse des insights (déséquilibre des classes, modèle champion, limites de l'arbre de décision, apport du tuning)
6. **ℹ️ À propos** — stack technique et disclaimer académique

### Lancer l'application

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

L'application charge automatiquement `models/best_model.pkl` et `models/preprocessor.pkl` (mise en cache via `@st.cache_resource`), et applique le thème défini dans `src/config.toml` (rose poudré / fond clair).

---

## 🛠️ Stack technique

`Python` · `pandas` · `NumPy` · `scikit-learn` · `scipy` (tests statistiques) · `TF-IDF` · `GridSearchCV` · `joblib` · `Streamlit` · `matplotlib` / `seaborn` (visualisation notebooks)

Aucune bibliothèque de Deep Learning (pas de TensorFlow / PyTorch / Keras) — uniquement du Machine Learning classique.

---

## ⚙️ Installation

```bash
git clone <url-du-repo>
cd fashion-tech-ml-project
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Placer `reviews.csv` dans `data/raw/`, puis exécuter les notebooks dans l'ordre 01 → 02 → 03, ou directement lancer `streamlit run src/app.py` si les fichiers `.pkl` sont déjà présents dans `models/`.

---

## 🔁 Reproductibilité

Pour régénérer entièrement le projet à partir des données brutes :

| Étape            | Notebook                          | Entrée                 | Sortie    |
|------------------|-----------------------------------|------------------------|-----------|
| 1. Nettoyage     | `01_eda_and_cleaning.ipynb`       | `data/raw/reviews.csv` | `data/processed/reviews_clean.csv` |
| 2. Prétraitement | `02_preprocessing_encoding.ipynb` | `data/processed/reviews_clean.csv` | matrices sparse train/test + `models/preprocessor.pkl` |
| 3. Modélisation  | `03_classification_models.ipynb`  | matrices train/test    | `models/best_model.pkl` |
| 4. Démo          | `src/app.py`                      | `models/*.pkl`         | interface Streamlit |

Un `random_state=42` est fixé sur les modèles à composante aléatoire (Decision Tree, Random Forest, Linear SVC) et le split train/test est **stratifié** (`stratify=y`) pour garantir la reproductibilité des résultats.

---

## ⚠️ Limites

- Les avis sont en **anglais** uniquement : les prédictions ne sont fiables que sur du texte anglais.
- Dataset **déséquilibré** (82/18) : biais possible vers la classe majoritaire malgré la stratification et l'usage du F1-Score.
- Application développée à des **fins académiques** ; les prédictions sont indicatives.

---

## 👤 Contexte

Projet réalisé dans le cadre d'un module de **Data Mining / Machine Learning supervisé**.

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
- [Ensemble Learning](#-ensemble-learning-et-analyse-des-erreurs)
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
│   ├── title_tfidf.pkl                  # Vectoriseur TF-IDF (Title)
│   ├── adaboost_model.pkl               # AdaBoost optimisé (notebook 04)
│   ├── xgboost_model.pkl                # XGBoost optimisé (notebook 04)
│   ├── ensemble_gridsearch_results.csv  # Détail des combinaisons GridSearchCV (notebook 04)
│   ├── ensemble_test_scores.csv         # Score de chaque modèle sur le jeu de test (notebook 04)
│   ├── ensemble_class_metrics.csv       # Métriques par classe (notebook 05)
│   ├── error_analysis.csv               # Un avis de test par ligne : scores et métadonnées (notebook 05)
│   ├── ensemble_feature_importance.csv  # Variables les plus importantes (notebook 05)
│   └── ensemble_block_importance.csv    # Importance par bloc de variables (notebook 05)
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb        # Exploration, nettoyage, tests statistiques
│   ├── 02_preprocessing_encoding.ipynb  # Split train/test, encodage, TF-IDF
│   ├── 03_classification_models.ipynb   # Entraînement, comparaison, sauvegarde du modèle
│   ├── 04_ensemble_learning.ipynb       # AdaBoost, XGBoost, Soft Voting (GridSearchCV)
│   └── 05_error_analysis.ipynb          # Métriques par classe, analyse des erreurs, exports pour l'app
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

> 💡 Après avoir cloné le repo, place `reviews.csv` dans `data/raw/` et exécute les notebooks dans l'ordre (01 → 02 → 03 → 04 → 05) pour régénérer `data/processed/reviews_clean.csv` et les fichiers `models/*.pkl`. L'app (`src/app.py`) résout automatiquement le chemin vers `models/` à la racine du projet (`Path(__file__).resolve().parents[1]`), donc aucune configuration manuelle n'est nécessaire une fois les modèles générés.

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
  - Les résultats sont repris dans l'onglet **🧪 Tests d'hypothèses** de l'application
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
| Random Forest             | 0.9342     | 0.9604     | 0.9723     |
| Decision Tree             | 0.9301     | 0.9563     | 0.9684     |
| K-Nearest Neighbors       | 0.9186     | 0.9519     | 0.9642     |
| Multinomial Naïve Bayes   | 0.9052     | 0.9425     | 0.9531     |
| Zero-R (Baseline)         | 0.8223     | 0.9025     | N/A        |

- Le modèle **Linear SVC** est sélectionné comme meilleur modèle et sérialisé (`joblib`) dans `models/best_model.pkl`.
- Le baseline Zero-R (82,2 % d'accuracy) rappelle que l'accuracy seule est trompeuse sur un dataset déséquilibré → le F1-Score et le ROC-AUC sont privilégiés.

---

## 🌲 Ensemble Learning et analyse des erreurs

Suite aux retours d'évaluation, trois méthodes d'**Ensemble Learning** ont été ajoutées (branche `feature/siad17-ensemble-learning`). Elles combinent plusieurs modèles pour réduire les erreurs d'un modèle isolé :

| Méthode | Famille | Principe |
|---------|---------|----------|
| **AdaBoost** | Boosting | Arbres très simples construits l'un après l'autre ; les avis mal classés reçoivent plus de poids à chaque étape |
| **XGBoost** | Boosting | Chaque arbre corrige l'erreur résiduelle (gradient) des arbres précédents, avec régularisation |
| **Soft Voting (NB + RF)** *(bonus)* | Vote pondéré | Moyenne pondérée des probabilités de Naïve Bayes multinomial et de Random Forest ; les poids sont choisis par validation croisée |

### 4️⃣ Entraînement (`04_ensemble_learning.ipynb`)
- Reconstruction du **même split** (`random_state=42`, `stratify=y`) et application du `preprocessor.pkl` en `transform` seul (pas de fuite de données) ; vérification que Linear SVC retrouve bien F1 = 0.9622 sur ce jeu de test
- Même protocole que le notebook 03 : `GridSearchCV`, `cv=6`, `scoring="f1"`
- Grilles : AdaBoost (`n_estimators` 50 / 100 / 200, `learning_rate` 0.5 / 1.0, profondeur des arbres 1 / 2), XGBoost (`n_estimators` 100 / 200 / 300, `learning_rate` 0.05 / 0.1 / 0.5, `max_depth` 3 / 5), Soft Voting (poids `[1,1]`, `[1,2]`, `[1,3]`, `[2,1]`)
- Random Forest (200 arbres) réentraîné comme référence *bagging* vs *boosting*
- Sorties : `models/adaboost_model.pkl`, `models/xgboost_model.pkl`, `models/ensemble_gridsearch_results.csv` et `models/ensemble_test_scores.csv`. Le Soft Voting n'est pas sauvegardé (environ 70 Mo) : ses scores de test suffisent à l'application

### 5️⃣ Analyse des erreurs (`05_error_analysis.ipynb`)
- Métriques **par classe** (précision, rappel, F1 de la classe minoritaire *non recommandé*), matrices de confusion, courbes ROC et précision-rappel
- Effet du **seuil de décision** sur la classe minoritaire (seuil ajusté sur le jeu de test : analyse exploratoire, pas une performance finale)
- Taux d'erreur par note, par département et par longueur d'avis ; erreurs communes à tous les modèles ; avis mal classés avec forte confiance
- Importance des variables (XGBoost, AdaBoost) : texte (TF-IDF) ou variables structurées
- Sorties exploitées par l'application : `error_analysis.csv`, `ensemble_class_metrics.csv`, `ensemble_feature_importance.csv`, `ensemble_block_importance.csv`

### Résultats sur le jeu de test (4 693 avis, seuil de décision 0.5)

<!-- ensemble-results:start -->
| Modèle | Accuracy | ROC-AUC | Précision (0) | Rappel (0) | F1 (0) | F1-Score (1) |
|--------|-------|-------|-------|-------|-------|-------|
| AdaBoost | 0.9339 | 0.9743 | 0.7805 | **0.8741** | **0.8247** | 0.9593 |
| XGBoost | 0.9339 | 0.9733 | 0.7991 | 0.8393 | 0.8187 | 0.9596 |
| Soft Voting (NB + RF) | 0.9361 | 0.9737 | 0.8486 | 0.7794 | 0.8125 | 0.9615 |
| Random Forest | 0.9346 | 0.9723 | **0.8566** | 0.7590 | 0.8048 | 0.9607 |
| Linear SVC *(champion déployé)* | **0.9378** | **0.9775** | 0.8297 | 0.8177 | 0.8237 | **0.9622** |

(0) = classe minoritaire *non recommandé*, (1) = *recommandé* ; en gras : meilleure valeur de chaque colonne.

- Meilleur F1 sur la classe *non recommandé* : **AdaBoost** (F1 = 0.8247, rappel = 0.8741)
- Meilleur ROC-AUC : **Linear SVC** (0.9775)
<!-- ensemble-results:end -->

> Le modèle déployé pour la **Prédiction en Direct** reste Linear SVC (`models/best_model.pkl`). La page **🌲 Ensemble Learning** de l'application permet d'explorer ces résultats de façon interactive : choix du modèle, seuil de décision, matrices de confusion, courbes ROC et avis mal classés.

---

## 🖥️ Application Streamlit (`src/app.py`)

Interface interactive avec plusieurs onglets :

1. **🏠 Accueil** — présentation du projet
2. **🔮 Prédiction en Direct** — saisie d'un profil (département, catégorie, âge, note, avis) → prédiction en temps réel via `model.predict()` sur les données transformées par `preprocessor.pkl`
3. **📊 Performance des Modèles** — tableau comparatif + graphique F1-Score
4. **📈 Statistiques du Dataset** — répartition de la cible, répartition par département
5. **💡 Découvertes Clés** — synthèse des insights (déséquilibre des classes, modèle champion, limites de l'arbre de décision, apport du tuning)
6. **🧪 Tests d'hypothèses** — tableau récapitulatif des tests statistiques du notebook 01 (α = 0,05) : H1 Mann-Whitney U (`Rating` vs `Recommended IND`) et H2 Chi² d'indépendance (`Department Name` vs `Recommended IND`), avec statistique, p-value brute, p-value ajustée (FDR Benjamini-Hochberg), taille d'effet (r de rang bisérial, V de Cramér) et décision
7. **🌲 Ensemble Learning** — comparaison d'AdaBoost, XGBoost et Soft Voting : métriques par classe, matrice de confusion et courbes ROC interactives (choix du modèle et du seuil), analyse des erreurs par note, département et longueur d'avis, importance des variables
8. **ℹ️ À propos** — stack technique et disclaimer académique

### Lancer l'application

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

L'application charge automatiquement `models/best_model.pkl` et `models/preprocessor.pkl` (mise en cache via `@st.cache_resource`), et applique le thème défini dans `src/config.toml` (rose poudré / fond clair).

---

## 🛠️ Stack technique

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `scipy` (tests statistiques) · `TF-IDF` · `GridSearchCV` · `joblib` · `Streamlit` · `Altair` (graphiques interactifs) · `matplotlib` / `seaborn` (visualisation notebooks)

Aucune bibliothèque de Deep Learning (pas de TensorFlow / PyTorch / Keras) — uniquement du Machine Learning classique.

---

## ⚙️ Installation

```bash
git clone <url-du-repo>
cd fashion-tech-ml-project
python -m venv venv
source venv/bin/activate   
pip install -r requirements.txt
```

Placer `reviews.csv` dans `data/raw/`, puis exécuter les notebooks dans l'ordre 01 → 02 → 03 → 04 → 05, ou directement lancer `streamlit run src/app.py` si les fichiers `.pkl` sont déjà présents dans `models/`.

---

## 🔁 Reproductibilité

Pour régénérer entièrement le projet à partir des données brutes :

| Étape            | Notebook                          | Entrée                 | Sortie    |
|------------------|-----------------------------------|------------------------|-----------|
| 1. Nettoyage     | `01_eda_and_cleaning.ipynb`       | `data/raw/reviews.csv` | `data/processed/reviews_clean.csv` |
| 2. Prétraitement | `02_preprocessing_encoding.ipynb` | `data/processed/reviews_clean.csv` | matrices sparse train/test + `models/preprocessor.pkl` |
| 3. Modélisation  | `03_classification_models.ipynb`  | matrices train/test    | `models/best_model.pkl` |
| 4. Ensemble Learning | `04_ensemble_learning.ipynb`  | `data/processed/reviews_clean.csv` + `models/preprocessor.pkl` | `models/adaboost_model.pkl`, `models/xgboost_model.pkl`, `models/ensemble_test_scores.csv`, `models/ensemble_gridsearch_results.csv` |
| 5. Analyse des erreurs | `05_error_analysis.ipynb`   | `models/ensemble_test_scores.csv` + modèles | `models/error_analysis.csv`, `models/ensemble_class_metrics.csv`, `models/ensemble_feature_importance.csv`, `models/ensemble_block_importance.csv` |
| 6. Démo          | `src/app.py`                      | `models/*.pkl` et `models/*.csv` | interface Streamlit |

Un `random_state=42` est fixé sur les modèles à composante aléatoire (Decision Tree, Random Forest, Linear SVC, AdaBoost, XGBoost) et le split train/test est **stratifié** (`stratify=y`) pour garantir la reproductibilité des résultats.

---

## ⚠️ Limites

- Les avis sont en **anglais** uniquement : les prédictions ne sont fiables que sur du texte anglais.
- Dataset **déséquilibré** (82/18) : biais possible vers la classe majoritaire malgré la stratification et l'usage du F1-Score.
- Application développée à des **fins académiques** ; les prédictions sont indicatives.

---

## 🚀 Pistes d'amélioration

- Ajouter **GradientBoosting** (scikit-learn) et le comparer à AdaBoost et XGBoost
- Comparer TF-IDF à des représentations denses (**Word2Vec**, **Sentence-Transformers**)
- Choisir le seuil de décision sur un **jeu de validation** séparé pour mieux détecter la classe minoritaire

---

## 👤 Contexte

Projet réalisé dans le cadre d'un module de **Data Mining / Machine Learning supervisé**.
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_models_config():
    """Définition des modèles et de leurs grilles d'hyperparamètres."""
    return {
        "K-Nearest Neighbors": (
            KNeighborsClassifier(algorithm="brute"),
            {"n_neighbors": [5, 11, 21], "weights": ["uniform", "distance"]}
        ),
        "Multinomial Naïve Bayes": (
            MultinomialNB(),
            {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {"criterion": ["gini", "entropy"], "max_depth": [5, 7, 10]}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {"C": [0.5, 1.0, 5.0], "kernel": ["rbf", "linear"]}
        )
    }

def run_model_pipeline(X_train, y_train, X_test, y_test):
    """Entraîne et évalue tous les modèles avec Stratified 5-Fold CV."""
    models = get_models_config()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, (model, params) in models.items():
        print(f"Entraînement de : {name}...")

        if "Naive" in name or "Naïve" in name:
            X_tr = X_train.copy()
            X_tr.data = np.clip(X_tr.data, 0, None)
            X_te = X_test.copy()
            X_te.data = np.clip(X_te.data, 0, None)
        else:
            X_tr, X_te = X_train, X_test

        n_jobs = 2 if "Neighbors" in name else -1

        grid = GridSearchCV(model, params, cv=cv, scoring="f1_weighted", n_jobs=n_jobs)
        grid.fit(X_tr, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_te)
        
        try:
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_te)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_proba = best_model.decision_function(X_te)
            else:
                y_proba = None
                
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        except Exception:
            auc = np.nan

        results.append({
            "Modèle": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred, average="weighted"),
            "ROC-AUC": auc,
            "Meilleurs Paramètres": grid.best_params_
        })

    return pd.DataFrame(results).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
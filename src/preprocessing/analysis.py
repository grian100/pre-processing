"""Confronto tra strategie di imputazione e analisi della varianza spiegata dalla PCA."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .pipelines import build_pipeline_1, build_pipeline_3

SIMPLE = "SimpleImputer (media/mediana)"


def default_imputers():
    """Strategie da confrontare: None indica il SimpleImputer media/mediana delle pipeline."""
    return {
        SIMPLE: None,
        "KNNImputer (k=3)": KNNImputer(n_neighbors=3),
        "KNNImputer (k=5)": KNNImputer(n_neighbors=5),
        "KNNImputer (k=10)": KNNImputer(n_neighbors=10),
        "KNNImputer (k=5, pesato)": KNNImputer(n_neighbors=5, weights="distance"),
    }


def _classifier():
    return LogisticRegression(max_iter=5000)


def _cv(random_state):
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)


def imputation_error(X, symmetric, asymmetric, imputers=None, mask_fraction=0.1, random_state=42):
    """Nasconde una parte dei valori noti e misura quanto bene ogni imputer li ricostruisce.

    L'errore (RMSE) è calcolato sulle variabili standardizzate, così è confrontabile tra colonne:
    1.0 equivale a sbagliare di una deviazione standard.
    """
    imputers = default_imputers() if imputers is None else imputers
    numeric = list(symmetric) + list(asymmetric)
    X_true = X[numeric].to_numpy(dtype=float)

    rng = np.random.default_rng(random_state)
    hidden = ~np.isnan(X_true) & (rng.random(X_true.shape) < mask_fraction)
    X_masked = np.where(hidden, np.nan, X_true)

    scaler = StandardScaler().fit(X_masked)
    Z_masked = scaler.transform(X_masked)
    Z_true = scaler.transform(X_true)

    n_sym = len(symmetric)
    rows = []
    for name, imputer in imputers.items():
        if imputer is None:
            imputer = ColumnTransformer(transformers=[
                ("simmetriche", SimpleImputer(strategy="mean"), list(range(n_sym))),
                ("asimmetriche", SimpleImputer(strategy="median"), list(range(n_sym, len(numeric)))),
            ])
        Z_imputed = imputer.fit_transform(Z_masked)
        errors = Z_imputed[hidden] - Z_true[hidden]
        rows.append({
            "imputer": name,
            "RMSE": np.sqrt(np.mean(errors ** 2)),
            "MAE": np.mean(np.abs(errors)),
        })
    return pd.DataFrame(rows).set_index("imputer").sort_values("RMSE")


def compare_imputers(X, y, symmetric, asymmetric, categorical, imputers=None, random_state=42):
    """Valuta ogni imputer con una cross-validation a 5 fold di una regressione logistica.

    Il pre-processing è quello della pipeline 1, rifittato in ogni fold insieme al modello.
    """
    imputers = default_imputers() if imputers is None else imputers
    rows = []
    for name, imputer in imputers.items():
        model = Pipeline([
            ("preprocessing", build_pipeline_1(symmetric, asymmetric, categorical, numeric_imputer=imputer)),
            ("model", _classifier()),
        ])
        scores = cross_validate(model, X, y, cv=_cv(random_state), scoring=["roc_auc", "accuracy", "f1"])
        rows.append({
            "imputer": name,
            "ROC AUC": scores["test_roc_auc"].mean(),
            "ROC AUC std": scores["test_roc_auc"].std(),
            "accuracy": scores["test_accuracy"].mean(),
            "F1": scores["test_f1"].mean(),
        })
    return pd.DataFrame(rows).set_index("imputer").sort_values("ROC AUC", ascending=False)


def pca_explained_variance(X, symmetric, asymmetric, numeric_imputer=None):
    """Varianza spiegata da ciascuna componente, fittando la PCA con tutte le componenti."""
    pipeline = build_pipeline_3(symmetric, asymmetric, n_components=None, numeric_imputer=numeric_imputer)
    pipeline.fit(X)
    ratio = pipeline.named_steps["PCA"].explained_variance_ratio_
    return pd.DataFrame(
        {"varianza spiegata": ratio, "varianza cumulata": np.cumsum(ratio)},
        index=pd.RangeIndex(1, len(ratio) + 1, name="componenti"),
    )


def components_for_variance(variance, thresholds=(0.8, 0.9, 0.95, 0.99)):
    """Numero minimo di componenti necessario per raggiungere ogni soglia di varianza cumulata."""
    cumulative = variance["varianza cumulata"].to_numpy()
    # La tolleranza evita di mancare la soglia per errori di arrotondamento (es. 0.99999999 < 1.0)
    return {t: int(np.argmax(cumulative >= t - 1e-9) + 1) for t in thresholds}


def elbow_components(variance):
    """Componenti al "gomito" della curva cumulata: il punto più distante dalla retta tra primo e ultimo."""
    y = variance["varianza cumulata"].to_numpy()
    x = np.arange(1, len(y) + 1)
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    direction = (p2 - p1) / np.linalg.norm(p2 - p1)
    points = np.column_stack([x, y]) - p1
    distances = np.abs(points[:, 0] * direction[1] - points[:, 1] * direction[0])
    return int(x[np.argmax(distances)])


def pca_cv_scores(X, y, symmetric, asymmetric, components, numeric_imputer=None, random_state=42):
    """ROC AUC in cross-validation di una regressione logistica al variare delle componenti PCA."""
    rows = []
    for n in components:
        model = Pipeline([
            ("preprocessing", build_pipeline_3(symmetric, asymmetric, n_components=n, numeric_imputer=numeric_imputer)),
            ("model", _classifier()),
        ])
        scores = cross_validate(model, X, y, cv=_cv(random_state), scoring="roc_auc")
        rows.append({
            "componenti": n,
            "ROC AUC": scores["test_score"].mean(),
            "ROC AUC std": scores["test_score"].std(),
        })
    return pd.DataFrame(rows).set_index("componenti")

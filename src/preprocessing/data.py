"""Caricamento del dataset e individuazione dei gruppi di colonne."""

from pathlib import Path

import pandas as pd

DATASET_URL = "https://proai-datasets.s3.eu-west-3.amazonaws.com/sample_dataset.csv"
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_dataset.csv"

TARGET = "target"

# Nel dataset Breast Cancer Wisconsin (codifica di scikit-learn) il target vale:
#   0 = maligno, 1 = benigno
MALIGNANT = 0
BENIGN = 1

# Soglia oltre la quale una variabile numerica è considerata asimmetrica
SKEW_THRESHOLD = 0.5

CATEGORICAL_DTYPES = ["object", "category", "bool"]


def load_dataset(url=DATASET_URL, cache_path=CACHE_PATH):
    """Restituisce il dataset come DataFrame, scaricandolo solo la prima volta."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(url, low_memory=False).to_csv(cache_path, index=False)
    return pd.read_csv(cache_path, low_memory=False)


def split_features_target(df, target=TARGET):
    return df.drop(columns=target), df[target]


def categorical_columns(X):
    # Nota: nel dataset la colonna 'area error' contiene categorie (A, B, C) e non numeri
    return list(X.select_dtypes(include=CATEGORICAL_DTYPES).columns)


def numerical_columns(X):
    return list(X.select_dtypes(exclude=CATEGORICAL_DTYPES).columns)


def split_by_skewness(X, threshold=SKEW_THRESHOLD):
    """Divide le colonne numeriche in simmetriche e asimmetriche in base alla skewness.

    Va chiamata sui soli dati di training per non introdurre data leakage.
    """
    skew = X[numerical_columns(X)].skew()
    symmetric = list(skew[skew.abs() <= threshold].index)
    asymmetric = list(skew[skew.abs() > threshold].index)
    return symmetric, asymmetric

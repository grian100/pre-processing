"""Costruzione delle tre pipeline di pre-processing.

Ogni funzione restituisce una Pipeline NON fittata: il fit va eseguito solo sul
training set e il transform applicato poi al test set.

Il parametro ``numeric_imputer`` sceglie come riempire i valori mancanti numerici:
- ``None`` (default): SimpleImputer con media per le simmetriche e mediana per le asimmetriche
- un imputer multivariato (es. ``KNNImputer``): applicato a tutte le numeriche insieme,
  dopo averle standardizzate, perché usa le distanze tra i record
"""

import inspect

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)

CATEGORY_ORDER = ["A", "B", "C"]


def _kbins(n_bins):
    params = dict(n_bins=n_bins, encode="ordinal", strategy="quantile")
    # Da scikit-learn 1.7 va indicato il metodo di calcolo dei quantili
    if "quantile_method" in inspect.signature(KBinsDiscretizer).parameters:
        params["quantile_method"] = "averaged_inverted_cdf"
    return KBinsDiscretizer(**params)


def _numeric_transformers(symmetric, asymmetric, sym_steps, asym_steps, numeric_imputer=None):
    """Transformer per le colonne numeriche: imputazione seguita dai passi specifici di ciascun gruppo."""
    if numeric_imputer is None:
        return [
            (
                "numeriche simmetriche",
                Pipeline([("missing", SimpleImputer(strategy="mean"))] + sym_steps),
                symmetric,
            ),
            (
                "numeriche asimmetriche",
                Pipeline([("missing", SimpleImputer(strategy="median"))] + asym_steps),
                asymmetric,
            ),
        ]

    # Imputazione multivariata su tutte le numeriche, poi i passi di ciascun gruppo
    # selezionando le colonne per posizione (simmetriche prima, asimmetriche dopo)
    n_sym = len(symmetric)
    by_group = ColumnTransformer(transformers=[
        ("numeriche simmetriche", Pipeline(sym_steps), list(range(n_sym))),
        ("numeriche asimmetriche", Pipeline(asym_steps), list(range(n_sym, n_sym + len(asymmetric)))),
    ])
    return [
        (
            "numeriche",
            Pipeline([
                # StandardScaler ignora i NaN nel fit e li preserva nel transform
                ("scaler", StandardScaler()),
                ("missing", numeric_imputer),
                ("groups", by_group),
            ]),
            list(symmetric) + list(asymmetric),
        ),
    ]


def build_pipeline_1(symmetric, asymmetric, categorical, numeric_imputer=None):
    """Pipeline 1: imputazione, simmetrizzazione, one-hot encoding, standardizzazione.

    Pensata per il sottoinsieme di record con un solo valore di target.
    """
    numeric = _numeric_transformers(
        symmetric,
        asymmetric,
        sym_steps=[("scaler", StandardScaler())],
        # Yeo-Johnson simmetrizza e standardizza (media 0, dev. std 1)
        asym_steps=[("power", PowerTransformer(standardize=True))],
        numeric_imputer=numeric_imputer,
    )
    ct = ColumnTransformer(transformers=numeric + [
        (
            "categoriche",
            Pipeline([
                ("missing", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ),
    ])
    return Pipeline([("column_transformer", ct)])


def build_pipeline_2(symmetric, asymmetric, categorical, n_bins=20, k=5, numeric_imputer=None):
    """Pipeline 2: imputazione, discretizzazione in 20 bin, encoding ordinale, selezione delle k feature migliori."""
    numeric = _numeric_transformers(
        symmetric,
        asymmetric,
        sym_steps=[("bin", _kbins(n_bins))],
        asym_steps=[("bin", _kbins(n_bins))],
        numeric_imputer=numeric_imputer,
    )
    ct = ColumnTransformer(transformers=numeric + [
        (
            "categoriche",
            Pipeline([
                ("missing", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(categories=[CATEGORY_ORDER] * len(categorical))),
            ]),
            categorical,
        ),
    ])
    # Il target è binario: si usa il test F per la classificazione (ANOVA)
    return Pipeline([
        ("column_transformer", ct),
        ("F-Score", SelectKBest(f_classif, k=k)),
    ])


def build_pipeline_3(symmetric, asymmetric, n_components=0.8, numeric_imputer=None):
    """Pipeline 3 (solo numeriche): imputazione, simmetrizzazione, PCA, normalizzazione in [0, 1]."""
    numeric = _numeric_transformers(
        symmetric,
        asymmetric,
        # La PCA è sensibile alla scala: le variabili vanno standardizzate prima
        sym_steps=[("scaler", StandardScaler())],
        asym_steps=[("power", PowerTransformer(standardize=True))],
        numeric_imputer=numeric_imputer,
    )
    return Pipeline([
        ("column_transformer", ColumnTransformer(transformers=numeric)),
        ("PCA", PCA(n_components=n_components)),
        ("normal", MinMaxScaler()),
    ])

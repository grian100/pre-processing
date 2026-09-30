"""Costruzione delle tre pipeline di pre-processing.

Ogni funzione restituisce una Pipeline NON fittata: il fit va eseguito solo sul
training set e il transform applicato poi al test set.
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


def build_pipeline_1(symmetric, asymmetric, categorical):
    """Pipeline 1: imputazione, simmetrizzazione, one-hot encoding, standardizzazione.

    Pensata per il sottoinsieme di record con un solo valore di target.
    """
    ct = ColumnTransformer(transformers=[
        (
            "numeriche simmetriche",
            Pipeline([
                ("missing", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]),
            symmetric,
        ),
        (
            "numeriche asimmetriche",
            Pipeline([
                ("missing", SimpleImputer(strategy="median")),
                # Yeo-Johnson simmetrizza e standardizza (media 0, dev. std 1)
                ("power", PowerTransformer(standardize=True)),
            ]),
            asymmetric,
        ),
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


def build_pipeline_2(symmetric, asymmetric, categorical, n_bins=20, k=5):
    """Pipeline 2: imputazione, discretizzazione in 20 bin, encoding ordinale, selezione delle k feature migliori."""
    ct = ColumnTransformer(transformers=[
        (
            "numeriche simmetriche",
            Pipeline([
                ("missing", SimpleImputer(strategy="mean")),
                ("bin", _kbins(n_bins)),
            ]),
            symmetric,
        ),
        (
            "numeriche asimmetriche",
            Pipeline([
                ("missing", SimpleImputer(strategy="median")),
                ("bin", _kbins(n_bins)),
            ]),
            asymmetric,
        ),
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


def build_pipeline_3(symmetric, asymmetric, n_components=0.8):
    """Pipeline 3 (solo numeriche): imputazione, simmetrizzazione, PCA, normalizzazione in [0, 1]."""
    ct = ColumnTransformer(transformers=[
        (
            "numeriche simmetriche",
            Pipeline([
                ("missing", SimpleImputer(strategy="mean")),
                # La PCA è sensibile alla scala: le variabili vanno standardizzate prima
                ("scaler", StandardScaler()),
            ]),
            symmetric,
        ),
        (
            "numeriche asimmetriche",
            Pipeline([
                ("missing", SimpleImputer(strategy="median")),
                ("power", PowerTransformer(standardize=True)),
            ]),
            asymmetric,
        ),
    ])
    return Pipeline([
        ("column_transformer", ct),
        ("PCA", PCA(n_components=n_components)),
        ("normal", MinMaxScaler()),
    ])

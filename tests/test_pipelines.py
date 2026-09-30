import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

from preprocessing import (
    BENIGN,
    build_pipeline_1,
    build_pipeline_2,
    build_pipeline_3,
    categorical_columns,
    split_by_skewness,
    split_features_target,
)
from preprocessing.analysis import (
    compare_imputers,
    components_for_variance,
    default_imputers,
    elbow_components,
    imputation_error,
    pca_cv_scores,
    pca_explained_variance,
)


@pytest.fixture(scope="module")
def dataset():
    """Dataset con la stessa struttura di quello reale, generato offline."""
    df = load_breast_cancer(as_frame=True).frame
    rng = np.random.default_rng(0)
    df["area error"] = rng.choice(["A", "B", "C"], size=len(df), p=[0.9, 0.08, 0.02])
    features = df.columns.drop("target")
    mask = rng.random((len(df), len(features))) < 0.15
    df[features] = df[features].mask(mask)
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


@pytest.fixture(scope="module")
def columns(dataset):
    X_train = dataset[0]
    return (*split_by_skewness(X_train), categorical_columns(X_train))


def test_split_by_skewness(dataset, columns):
    symmetric, asymmetric, categorical = columns
    assert categorical == ["area error"]
    assert "mean area" in asymmetric
    assert set(symmetric) | set(asymmetric) == set(dataset[0].columns) - {"area error"}


def test_pipeline_1(dataset, columns):
    X_train, X_test, y_train, y_test = dataset
    pipeline = build_pipeline_1(*columns).fit(X_train[y_train == BENIGN])
    out = pipeline.transform(X_test[y_test == BENIGN])
    assert not np.isnan(out).any()
    # 29 numeriche + una colonna one-hot per ogni categoria vista nel training
    n_categories = X_train.loc[y_train == BENIGN, "area error"].nunique()
    assert out.shape == ((y_test == BENIGN).sum(), 29 + n_categories)


def test_pipeline_1_ignores_unseen_categories(dataset, columns):
    X_train, X_test, y_train, _ = dataset
    train = X_train[y_train == BENIGN].assign(**{"area error": "A"})
    pipeline = build_pipeline_1(*columns).fit(train)
    out = pipeline.transform(X_test.assign(**{"area error": "C"}))
    assert not np.isnan(out).any()


def test_pipeline_2(dataset, columns):
    X_train, X_test, y_train, _ = dataset
    out = build_pipeline_2(*columns).fit(X_train, y_train).transform(X_test)
    assert out.shape == (len(X_test), 5)
    assert out.min() >= 0 and out.max() <= 19


def test_pipeline_3(dataset, columns):
    X_train, X_test, _, _ = dataset
    symmetric, asymmetric, _ = columns
    pipeline = build_pipeline_3(symmetric, asymmetric).fit(X_train)
    out = pipeline.transform(X_train)
    pca = pipeline.named_steps["PCA"]
    assert pca.explained_variance_ratio_.sum() >= 0.8
    # Con le variabili scalate servono più di 2 componenti per l'80% della varianza
    assert pca.n_components_ > 2
    assert out.shape == (len(X_train), pca.n_components_)
    assert np.allclose(out.min(axis=0), 0) and np.allclose(out.max(axis=0), 1)
    assert pipeline.transform(X_test).shape == (len(X_test), pca.n_components_)


@pytest.mark.parametrize("build", ["1", "2", "3"])
def test_pipelines_with_knn_imputer(dataset, columns, build):
    X_train, X_test, y_train, _ = dataset
    symmetric, asymmetric, categorical = columns
    if build == "1":
        pipeline = build_pipeline_1(symmetric, asymmetric, categorical, numeric_imputer=KNNImputer())
    elif build == "2":
        pipeline = build_pipeline_2(symmetric, asymmetric, categorical, numeric_imputer=KNNImputer())
    else:
        pipeline = build_pipeline_3(symmetric, asymmetric, numeric_imputer=KNNImputer())
    out = pipeline.fit(X_train, y_train).transform(X_test)
    assert len(out) == len(X_test)
    assert not np.isnan(out).any()


def test_imputation_error(dataset, columns):
    X_train = dataset[0]
    symmetric, asymmetric, _ = columns
    errors = imputation_error(X_train, symmetric, asymmetric)
    assert set(errors.index) == set(default_imputers())
    assert (errors["RMSE"] > 0).all()


def test_compare_imputers(dataset, columns):
    X_train, _, y_train, _ = dataset
    imputers = {"simple": None, "knn": KNNImputer()}
    scores = compare_imputers(X_train, y_train, *columns, imputers=imputers)
    assert set(scores.index) == {"simple", "knn"}
    assert scores["ROC AUC"].between(0.5, 1).all()


def test_pca_analysis(dataset, columns):
    X_train, _, y_train, _ = dataset
    symmetric, asymmetric, _ = columns
    variance = pca_explained_variance(X_train, symmetric, asymmetric)
    cumulative = variance["varianza cumulata"]
    assert len(variance) == len(symmetric) + len(asymmetric)
    assert cumulative.is_monotonic_increasing and np.isclose(cumulative.iloc[-1], 1)

    thresholds = components_for_variance(variance, (0.8, 0.95, 1.0))
    assert thresholds[0.8] <= thresholds[0.95] <= thresholds[1.0] == len(variance)
    assert cumulative.loc[thresholds[0.8]] >= 0.8 > cumulative.loc[thresholds[0.8] - 1]
    assert 1 <= elbow_components(variance) <= len(variance)

    scores = pca_cv_scores(X_train, y_train, symmetric, asymmetric, [1, 5])
    assert list(scores.index) == [1, 5]

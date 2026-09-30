"""Esegue le tre pipeline: fit sul training set, transform sul test set."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sklearn.model_selection import train_test_split

from preprocessing import (
    BENIGN,
    build_pipeline_1,
    build_pipeline_2,
    build_pipeline_3,
    categorical_columns,
    load_dataset,
    split_by_skewness,
    split_features_target,
)

# I valori imputati (tutti uguali alla media/mediana) creano quantili coincidenti:
# KBinsDiscretizer unisce i bin troppo stretti e lo segnala con un avviso atteso
warnings.filterwarnings("ignore", message="Bins whose width are too small")

RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    df = load_dataset()
    X, y = split_features_target(df)

    # Split stratificato: il pre-processing viene fittato solo sul training set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    categorical = categorical_columns(X_train)
    symmetric, asymmetric = split_by_skewness(X_train)
    print(f"Variabili simmetriche ({len(symmetric)}): {symmetric}")
    print(f"Variabili asimmetriche ({len(asymmetric)}): {asymmetric}")
    print(f"Variabili categoriche ({len(categorical)}): {categorical}\n")

    # Pipeline 1: solo i record con target = 1 (casi benigni)
    train_1 = y_train == BENIGN
    test_1 = y_test == BENIGN
    pipeline1 = build_pipeline_1(symmetric, asymmetric, categorical)
    pipeline1.fit(X_train[train_1])
    X_test_1 = pipeline1.transform(X_test[test_1])
    print(f"Pipeline 1 (target = {BENIGN}): test {X_test_1.shape}")

    # Pipeline 2: tutti i record, 5 feature più informative rispetto al target
    pipeline2 = build_pipeline_2(symmetric, asymmetric, categorical)
    pipeline2.fit(X_train, y_train)
    X_test_2 = pipeline2.transform(X_test)
    selected = pipeline2.get_feature_names_out()
    print(f"Pipeline 2: test {X_test_2.shape}, feature selezionate: {list(selected)}")

    # Pipeline 3: solo variabili numeriche, PCA con l'80% della varianza spiegata
    pipeline3 = build_pipeline_3(symmetric, asymmetric)
    pipeline3.fit(X_train)
    X_test_3 = pipeline3.transform(X_test)
    pca = pipeline3.named_steps["PCA"]
    print(
        f"Pipeline 3: test {X_test_3.shape}, {pca.n_components_} componenti, "
        f"varianza spiegata {pca.explained_variance_ratio_.sum():.1%}"
    )


if __name__ == "__main__":
    main()

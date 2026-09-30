"""Analisi esplorativa: statistiche descrittive, istogrammi e skewness delle variabili."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt

from preprocessing import (
    categorical_columns,
    load_dataset,
    numerical_columns,
    split_by_skewness,
    split_features_target,
)


def main():
    df = load_dataset()
    df.info()
    print(f"\nRighe: {len(df)}, colonne: {len(df.columns)}")
    print(f"\nColonne per tipo:\n{df.dtypes.value_counts()}")
    print(f"\nValori mancanti per colonna:\n{df.isna().sum()}")
    print(f"\nStatistiche descrittive:\n{df.describe().T}")

    X, y = split_features_target(df)
    print(f"\nDistribuzione del target (0 = maligno, 1 = benigno):\n{y.value_counts()}")

    for col in categorical_columns(X):
        print(f"\nLa variabile '{col}' è categorica:\n{X[col].value_counts(dropna=False)}")

    print(f"\nSkewness delle variabili numeriche:\n{X[numerical_columns(X)].skew().sort_values()}")
    symmetric, asymmetric = split_by_skewness(X)
    print(f"\nSimmetriche: {symmetric}\nAsimmetriche: {asymmetric}")

    X[numerical_columns(X)].hist(bins=50, figsize=(20, 16))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

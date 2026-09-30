"""Confronto SimpleImputer / KNNImputer e analisi della varianza spiegata dalla PCA.

Le analisi usano solo il training set, perché servono a scegliere il pre-processing:
il test set resta riservato alla valutazione finale.
I risultati (tabelle CSV e grafici PNG) vengono salvati nella cartella reports/.
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from main import RANDOM_STATE, TEST_SIZE
from preprocessing import (
    categorical_columns,
    load_dataset,
    split_by_skewness,
    split_features_target,
)
from preprocessing.analysis import (
    compare_imputers,
    components_for_variance,
    elbow_components,
    imputation_error,
    pca_cv_scores,
    pca_explained_variance,
)

REPORTS = ROOT / "reports"
THRESHOLDS = (0.8, 0.9, 0.95, 0.99)


def plot_explained_variance(variance, thresholds, elbow, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = variance.index
    ax.bar(x, variance["varianza spiegata"], color="#9bb7d4", label="varianza della componente")
    ax.plot(x, variance["varianza cumulata"], marker="o", color="#1f4e79", label="varianza cumulata")
    for t, n in thresholds.items():
        ax.axhline(t, color="grey", linestyle=":", linewidth=1)
        ax.annotate(f"{t:.0%}: {n} comp.", (0.5, t), textcoords="offset points",
                    xytext=(0, 2), ha="left", fontsize=9, color="grey")
    ax.axvline(elbow, color="#c0504d", linestyle="--", linewidth=1, label=f"gomito ({elbow} comp.)")
    ax.set_xlabel("Componente principale")
    ax.set_ylabel("Varianza spiegata")
    ax.set_title("PCA: varianza spiegata (training set)")
    ax.set_xticks(list(x))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cv_scores(scores, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = scores.index
    ax.errorbar(x, scores["ROC AUC"], yerr=scores["ROC AUC std"], marker="o", capsize=3, color="#1f4e79")
    ax.set_xlabel("Numero di componenti PCA")
    ax.set_ylabel("ROC AUC (CV a 5 fold)")
    ax.set_title("Regressione logistica sulle componenti della pipeline 3")
    ax.set_xticks(list(x))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore", message="Bins whose width are too small")
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    REPORTS.mkdir(exist_ok=True)

    df = load_dataset()
    X, y = split_features_target(df)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    categorical = categorical_columns(X_train)
    symmetric, asymmetric = split_by_skewness(X_train)

    print("=== Confronto delle strategie di imputazione ===\n")
    errors = imputation_error(X_train, symmetric, asymmetric, random_state=RANDOM_STATE)
    print("Ricostruzione del 10% dei valori noti, nascosti artificialmente (errore in deviazioni standard):")
    print(errors, "\n")

    scores = compare_imputers(X_train, y_train, symmetric, asymmetric, categorical, random_state=RANDOM_STATE)
    print("Regressione logistica con il pre-processing della pipeline 1 (CV a 5 fold):")
    print(scores, "\n")
    errors.join(scores).to_csv(REPORTS / "confronto_imputer.csv")

    print("=== Analisi della PCA ===\n")
    variance = pca_explained_variance(X_train, symmetric, asymmetric)
    thresholds = components_for_variance(variance, THRESHOLDS)
    elbow = elbow_components(variance)
    print(variance.head(15), "\n")
    for t, n in thresholds.items():
        print(f"Componenti per il {t:.0%} della varianza: {n}")
    print(f"Gomito della curva cumulata: {elbow} componenti\n")
    variance.to_csv(REPORTS / "pca_varianza.csv")
    plot_explained_variance(variance, thresholds, elbow, REPORTS / "pca_varianza.png")

    components = list(range(1, max(thresholds.values()) + 1))
    cv_scores = pca_cv_scores(X_train, y_train, symmetric, asymmetric, components, random_state=RANDOM_STATE)
    best = cv_scores["ROC AUC"].idxmax()
    # Il numero minimo di componenti il cui punteggio è entro una deviazione standard dal migliore
    threshold = cv_scores.loc[best, "ROC AUC"] - cv_scores.loc[best, "ROC AUC std"]
    parsimonious = cv_scores.index[cv_scores["ROC AUC"] >= threshold][0]
    print("ROC AUC al variare delle componenti (CV a 5 fold):")
    print(cv_scores, "\n")
    print(f"Miglior ROC AUC con {best} componenti; "
          f"con la regola di una deviazione standard ne bastano {parsimonious}.")
    cv_scores.to_csv(REPORTS / "pca_roc_auc.csv")
    plot_cv_scores(cv_scores, REPORTS / "pca_roc_auc.png")

    print(f"\nRisultati salvati in {REPORTS.name}/")


if __name__ == "__main__":
    main()

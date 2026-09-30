# Pre-processing di un Dataset di Rilevazione del Tumore al Seno
Questo progetto si concentra sul pre-processing di un dataset di rilevazione del tumore al seno (Breast Cancer Wisconsin), con l'obiettivo di creare un set di dati pulito e pronto per essere utilizzato nei modelli di machine learning.

# Servizi e librerie utilizzate
- <strong>Python==3.12.11</strong>
- <strong>Scikit-learn==1.7.2</strong> è una libreria open source di apprendimento automatico per il linguaggio di programmazione Python. Contiene algoritmi di classificazione, regressione e clustering (raggruppamento) e macchine a vettori di supporto, regressione logistica, classificatore bayesiano, k-mean e DBSCAN
- <strong>Pandas==2.3.3</strong> è un pacchetto Python che fornisce strutture dati veloci, flessibili ed espressive, progettate per rendere semplice e intuitivo l'utilizzo di dati "relazionali" o "etichettati". Il suo obiettivo è quello di essere il componente fondamentale di alto livello per l'analisi pratica dei dati in Python
- <strong>Numpy==2.3.3</strong> offre funzioni matematiche complete, generatori di numeri casuali, routine di algebra lineare, trasformate di Fourier e molto altro
- <strong>Matplotlib</strong> per gli istogrammi dell'analisi esplorativa e <strong>pytest</strong> per i test

# Il dataset
- 569 record, 30 feature e la colonna `target`.
- `target`: <strong>0 = maligno</strong>, <strong>1 = benigno</strong> (codifica di scikit-learn per il dataset Breast Cancer Wisconsin).
- Tutte le feature contengono valori mancanti.
- La colonna `area error`, numerica nel dataset originale, qui contiene invece <strong>categorie (A, B, C)</strong> ed è quindi trattata come variabile categorica.
- Il CSV viene scaricato alla prima esecuzione e salvato in `data/` (cartella esclusa da git).

# Obiettivi del progetto
Risiede principalmente nella creazione di una pipeline di pre-processing solida e ben strutturata, in grado di gestire i dati in modo coerente e automatizzato. Attraverso l'uso di strumenti come <strong>Pipeline</strong> e <strong>ColumnTransformer</strong> di <strong>scikit-learn</strong>, si punta a trasformare i dati grezzi in un formato che consenta di ottenere modelli predittivi più performanti e precisi. Ogni pipeline è un unico oggetto che racchiude tutte le fasi di <strong>pre-processing</strong>: viene fittata <strong>solo sul training set</strong> e poi applicata al test set con <strong>transform</strong>, così da evitare il data leakage. Le trasformazioni si applicano a tutte le colonne del dataset, eccetto la colonna target.

La suddivisione tra variabili <strong>simmetriche</strong> e <strong>asimmetriche</strong> non è più scritta a mano ma calcolata automaticamente sul training set: una variabile è asimmetrica se il valore assoluto della sua skewness supera 0.5.

# Descrizione delle Pipeline per la modellazione
La fase di pre-processing prevede tre tipi di pipeline:

  1. <mark>Pre-processing solo per i Record con Target = 1</mark>, ovvero i casi <strong>benigni</strong>. La pipeline include:
     - <strong>Pulizia dei Valori Mancanti</strong>: media per le variabili simmetriche, mediana per quelle asimmetriche, valore più frequente per le categoriche.
     - <strong>Simmetrizzazione delle Variabili Asimmetriche</strong>: trasformazione di Yeo-Johnson (<strong>PowerTransformer</strong>) applicata solo alle variabili asimmetriche.
     - <strong>One-Hot Encoding delle Variabili Categoriche</strong>: le categorie non viste in fase di fit vengono ignorate (`handle_unknown="ignore"`).
     - <strong>Riscalatura mediante Standardizzazione</strong>: media zero e deviazione standard pari a uno (StandardScaler per le simmetriche, già inclusa nel PowerTransformer per le asimmetriche).

  2. <mark>Pre-processing per Tutti i Record del Dataset</mark> con l'obiettivo di trasformare tutte le variabili numeriche e categoriche attraverso le seguenti fasi:
     - <strong>Pulizia dei Valori Mancanti</strong>: media, mediana o valore più frequente in base al tipo di variabile.
     - <strong>Discretizzazione a 20 Bin delle Variabili Numeriche</strong>: bin per quantili, più adatti di quelli a larghezza uniforme a dati asimmetrici e con outlier.
     - <strong>Encoding Ordinale delle Variabili Categoriche</strong>: la variabile categorica è codificata in ordine crescente (A, B, C), mantenendo la semantica tra i valori.
     - <strong>Selezione delle 5 Variabili più Informative</strong>: <strong>SelectKBest</strong> con il test F per la classificazione (`f_classif`), adatto a un target binario.

  3. <mark>Pre-processing delle Variabili Numeriche</mark>: questa pipeline si concentra esclusivamente sulle variabili numeriche:
     - <strong>Pulizia dei Valori Mancanti</strong>: media per le simmetriche, mediana per le asimmetriche.
     - <strong>Simmetrizzazione e Standardizzazione</strong>: PowerTransformer per le asimmetriche e StandardScaler per le simmetriche. Scalare le variabili prima della PCA è indispensabile, altrimenti le colonne con valori grandi (es. `mean area`) dominano la varianza.
     - <strong>Principal Component Analysis (PCA)</strong>: un'unica PCA su tutte le variabili numeriche, mantenendo l'80% della varianza spiegata.
     - <strong>Riscalatura mediante Normalizzazione</strong>: le componenti sono normalizzate tra 0 e 1.

# Repository
```
pre-processing/
├── src/preprocessing/
│   ├── data.py         # caricamento del dataset e individuazione delle colonne
│   ├── pipelines.py    # costruzione delle tre pipeline
│   └── analysis.py     # confronto tra imputer e analisi della PCA
├── tests/
│   └── test_pipelines.py
├── eda.py              # analisi esplorativa (describe, istogrammi, skewness)
├── main.py             # split train/test ed esecuzione delle pipeline
├── analysis.py         # confronto SimpleImputer/KNNImputer e analisi PCA (risultati in reports/)
├── requirements.txt
└── pytest.ini
```

# Utilizzo
```bash
pip install -r requirements.txt
python eda.py      # analisi esplorativa
python main.py     # esecuzione delle tre pipeline
python analysis.py # confronto tra imputer e analisi della PCA
pytest             # test
```

# Confronto tra strategie di imputazione
Ogni pipeline accetta il parametro `numeric_imputer`. Di default usa `SimpleImputer` (media per le variabili simmetriche, mediana per le asimmetriche); passando per esempio `KNNImputer(n_neighbors=5)` l'imputazione diventa multivariata. In quel caso le variabili vengono prima standardizzate, perché il KNN si basa sulle distanze tra i record.

`analysis.py` confronta le due strategie sul solo training set, in due modi:
1. <strong>Ricostruzione dei valori</strong>: il 10% dei valori noti viene nascosto e poi imputato; l'errore è misurato in deviazioni standard.
2. <strong>Effetto sul modello</strong>: regressione logistica con il pre-processing della pipeline 1, valutata con una cross-validation a 5 fold.

| Imputer | RMSE ricostruzione | ROC AUC (CV) | Accuracy (CV) |
|---|---|---|---|
| SimpleImputer (media/mediana) | 1.127 | 0.995 ± 0.004 | 0.971 |
| KNNImputer (k=3) | 0.737 | 0.995 ± 0.004 | 0.967 |
| KNNImputer (k=5) | 0.722 | 0.995 ± 0.005 | 0.971 |
| KNNImputer (k=5, pesato per distanza) | 0.719 | 0.995 ± 0.005 | 0.969 |
| KNNImputer (k=10) | <strong>0.714</strong> | 0.995 ± 0.005 | 0.969 |

<strong>Conclusione</strong>: il KNNImputer ricostruisce i valori mancanti molto meglio (errore ridotto di circa il 35%), perché sfrutta la forte correlazione tra le variabili (es. raggio, perimetro e area). Sulla classificazione però la differenza è nulla: tutte le strategie sono entro una deviazione standard l'una dall'altra. Le pipeline mantengono quindi il `SimpleImputer`, più semplice e veloce; il KNNImputer è preferibile quando servono valori imputati realistici (es. per analisi descrittive).

# Analisi della PCA
`analysis.py` fitta la PCA della pipeline 3 con tutte le componenti e calcola la varianza cumulata, il "gomito" della curva e la ROC AUC in cross-validation di una regressione logistica al variare del numero di componenti. I grafici vengono salvati in `reports/pca_varianza.png` e `reports/pca_roc_auc.png`.

| Varianza spiegata | Componenti necessarie |
|---|---|
| 80% | 8 |
| 90% | 14 |
| 95% | 20 |
| 99% | 26 (su 29) |

- Il gomito della curva cumulata si trova a <strong>7 componenti</strong>.
- La ROC AUC passa da 0.970 con 1 componente a 0.993 con 6, poi resta stabile intorno a 0.994: il valore massimo (0.995 con 21 componenti) non è significativamente migliore.
- Con la regola di una deviazione standard (il minor numero di componenti con un punteggio entro una deviazione standard dal migliore) bastano <strong>6 componenti</strong>.

<strong>Conclusione</strong>: la soglia dell'80% (8 componenti) usata nella pipeline 3 è una buona scelta: coincide quasi con il gomito della curva e conserva tutta l'informazione utile alla classificazione, riducendo le variabili da 29 a 8.

# Possibili sviluppi
- Provare `IterativeImputer` (imputazione multivariata basata su regressione) nello stesso confronto.
- Valutare altri modelli (es. random forest) per verificare che le conclusioni non dipendano dalla regressione logistica.

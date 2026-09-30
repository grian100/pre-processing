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
│   └── pipelines.py    # costruzione delle tre pipeline
├── tests/
│   └── test_pipelines.py
├── eda.py              # analisi esplorativa (describe, istogrammi, skewness)
├── main.py             # split train/test ed esecuzione delle pipeline
├── requirements.txt
└── pytest.ini
```

# Utilizzo
```bash
pip install -r requirements.txt
python eda.py      # analisi esplorativa
python main.py     # esecuzione delle tre pipeline
pytest             # test
```

# Miglioramenti
- Testare diverse strategie di imputazione
  - Provare tecniche più avanzate come KNNImputer per confrontare i risultati rispetto alla semplice media/mediana [https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html].
- Analizzare l'impatto della PCA
  - Aggiungere una valutazione della varianza spiegata per determinare il numero ottimale di componenti da mantenere [https://towardsdatascience.com/principal-component-analysis-made-easy-a-step-by-step-tutorial-184f295e97fe/].

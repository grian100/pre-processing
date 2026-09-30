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
│   └── analysis.py     # confronto tra imputer (Simple, KNN, Iterative) e analisi della PCA
├── tests/
│   └── test_pipelines.py
├── eda.py              # analisi esplorativa (describe, istogrammi, skewness)
├── main.py             # split train/test ed esecuzione delle pipeline
├── analysis.py         # confronto tra imputer e analisi PCA con due modelli (risultati in reports/)
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
Ogni pipeline accetta il parametro `numeric_imputer`. Di default usa `SimpleImputer` (media per le variabili simmetriche, mediana per le asimmetriche); passando un imputer multivariato l'imputazione tiene conto delle altre variabili:
- `KNNImputer`: usa la media dei k record più simili. Le variabili vengono prima standardizzate, perché il KNN si basa sulle distanze tra i record.
- `IterativeImputer`: stima ogni variabile con una regressione sulle altre, ripetendo il ciclo più volte. È stato provato sia con `BayesianRidge` (regressione lineare, il default) sia con una random forest.

`analysis.py` confronta le strategie sul solo training set, in due modi:
1. <strong>Ricostruzione dei valori</strong>: il 10% dei valori noti viene nascosto e poi imputato; l'errore (RMSE) è misurato in deviazioni standard.
2. <strong>Effetto sul modello</strong>: due modelli molto diversi, una regressione logistica (lineare) e una random forest (ad alberi, 300 alberi), con il pre-processing della pipeline 1 e una cross-validation a 5 fold.

| Imputer | RMSE ricostruzione | ROC AUC regressione logistica | ROC AUC random forest |
|---|---|---|---|
| SimpleImputer (media/mediana) | 1.127 | <strong>0.995</strong> ± 0.004 | 0.990 ± 0.007 |
| KNNImputer (k=3) | 0.737 | 0.995 ± 0.004 | 0.990 ± 0.009 |
| KNNImputer (k=5) | 0.722 | 0.995 ± 0.005 | 0.990 ± 0.007 |
| KNNImputer (k=5, pesato per distanza) | 0.719 | 0.995 ± 0.005 | 0.991 ± 0.007 |
| KNNImputer (k=10) | 0.714 | 0.995 ± 0.005 | <strong>0.991</strong> ± 0.006 |
| IterativeImputer (random forest) | 0.611 | 0.994 ± 0.006 | 0.989 ± 0.008 |
| IterativeImputer (BayesianRidge) | <strong>0.537</strong> | 0.994 ± 0.005 | 0.991 ± 0.007 |

<strong>Conclusioni</strong>:
- L'<strong>IterativeImputer con BayesianRidge</strong> ricostruisce i valori mancanti meglio di tutti: errore ridotto di circa il 52% rispetto al SimpleImputer e di circa il 25% rispetto al KNN. Le variabili sono legate da relazioni quasi lineari (es. raggio, perimetro e area), che una regressione cattura bene. La versione con random forest è più lenta e meno precisa.
- Sulla classificazione le differenze sono trascurabili <strong>con entrambi i modelli</strong>: tutte le strategie sono entro una deviazione standard l'una dall'altra, e la classifica cambia da un modello all'altro. La conclusione quindi non dipende dal modello.
- Le pipeline mantengono il `SimpleImputer`, più semplice e veloce. L'IterativeImputer è la scelta migliore quando servono valori imputati realistici (es. per analisi descrittive o per pubblicare il dataset pulito).
- La regressione logistica supera la random forest (0.995 contro 0.990): dopo la simmetrizzazione e la standardizzazione le due classi sono separabili quasi linearmente.

# Analisi della PCA
`analysis.py` fitta la PCA della pipeline 3 con tutte le componenti e calcola la varianza cumulata, il "gomito" della curva e la ROC AUC in cross-validation dei due modelli al variare del numero di componenti, confrontandola con gli stessi modelli senza PCA. I grafici vengono salvati in `reports/pca_varianza.png` e `reports/pca_roc_auc.png`.

| Varianza spiegata | Componenti necessarie |
|---|---|
| 80% | 8 |
| 90% | 14 |
| 95% | 20 |
| 99% | 26 (su 29) |

Il gomito della curva cumulata si trova a <strong>7 componenti</strong>.

| | Regressione logistica | Random forest |
|---|---|---|
| ROC AUC con 1 componente | 0.970 | 0.941 |
| ROC AUC con 8 componenti (80% della varianza) | 0.994 | 0.989 |
| ROC AUC migliore | 0.995 (21 componenti) | 0.990 (9 componenti) |
| ROC AUC senza PCA (29 variabili) | 0.995 | 0.988 |
| Componenti sufficienti (regola di una deviazione standard) | 6 | 6 |

La regola di una deviazione standard sceglie il minor numero di componenti il cui punteggio è entro una deviazione standard dal migliore.

- Con entrambi i modelli la ROC AUC cresce rapidamente fino a 6–8 componenti e poi si stabilizza.
- Per la random forest aggiungere componenti oltre la 13ª peggiora leggermente il risultato (fino a 0.983): le ultime componenti contengono soprattutto rumore, su cui gli alberi tendono a fare overfitting.
- La PCA non migliora la regressione logistica (0.994 contro 0.995 senza PCA), mentre con 8–9 componenti la random forest è in linea o leggermente meglio che senza PCA.

<strong>Conclusione</strong>: la soglia dell'80% (8 componenti) usata nella pipeline 3 è una buona scelta anche con la random forest. Coincide quasi con il gomito della curva, cade nel tratto in cui entrambi i modelli raggiungono il massimo e riduce le variabili da 29 a 8 senza perdere informazione utile. La PCA serve quindi a ridurre la dimensionalità, non ad aumentare l'accuratezza.

Nota: l'esecuzione completa di `analysis.py` richiede alcuni minuti, soprattutto per la random forest e l'IterativeImputer con random forest.

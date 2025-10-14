# Pre-processing di un Dataset di Rilevazione del Tumore al Seno
Questo progetto si concentra sul pre-processing di un dataset di rilevazionedel tumore al seno, con l'obiettivo di creare un set di dati pulito e pronto per essere utilizzato nei modelli di machine learning.

# Servizie e librerie utilizzate
- Ambiente di sviluppo <strong>Google Colaboratory</strong> [https://colab.google/] piattaforma basata su cloud che consente di scrivere e eseguire codice attraverso il browser. Offre inoltre risorse computazionali gratuite, come CPU e GPU, insieme a strumenti per la scrittura, l’esecuzione e la condivisione di codice Python
- <strong>Python==3.12.11</strong>
- <strong>Scikit-learn==1.7.2</strong> è una libreria open source di apprendimento automatico per il linguaggio di programmazione Python. Contiene algoritmi di classificazione, regressione e clustering (raggruppamento) e macchine a vettori di supporto, regressione logistica, classificatore bayesiano, k-mean e DBSCAN
- <strong>Pandas==2.3.3</strong>  è un pacchetto Python che fornisce strutture dati veloci, flessibili ed espressive, progettate per rendere semplice e intuitivo l'utilizzo di dati "relazionali" o "etichettati". Il suo obiettivo è quello di essere il componente fondamentale di alto livello per l'analisi pratica dei dati in Python
- <strong>Numpy==2.3.3</strong> offre funzioni matematiche complete, generatori di numeri casuali, routine di algebra lineare, trasformate di Fourier e molto altro

# Obiettivi del progetto
Risiede principalmente nella creazione di un pipeline di pre-processing solida e ben strutturata, in grado di gestire i dati in modo coerente e automatizzato. Attraverso l'uso di strumenti come <strong>Pipeline</strong> e <strong>ColumnTransformer</strong> di <strong>scikit-learn</strong>, si punta a trasformare i dati grezzi in un formato che consenta di ottenere modelli predittivi più performanti e precisi. L'obiettivo è ottenere un unico oggetto finale che racchiuda tutte le fasi di <strong>pre-processing</strong>, il cui metodo <strong>fit_transform</strong> sarà invocato per restituire il dataset trasformato e pronto all'uso. Letrasformazioni si applicano a tutte le colonne del dataset, eccetto la colonna target.

# Descrizione delle Pipeline per la modellazone 
La fase di pre processing per analizzare il caso si passa dalla realizazione di tre tipi di pipeline possibili:

  1. <mark>Pre-processing solo per Record con Target = 1</mark> ovvero i casi positivi di rilevazione del tumore. La pipeline include:
     - <strong>Pulizia dei Valori Mancanti</strong>: la pulizia sarà distinta tra variabili simmetriche e asimmetriche. Per le variabili asimmetriche si utilizzeranno tecniche di riempimento che tengano conto delladistribuzione dei dati, mentre per quelle simmetriche si opterà per metodi di riempimento più standard.
     - <strong>Simmetrizzazione delle Variabili Asimmetriche</strong>: per garantire una distribuzione più bilanciata dei dati, verranno corretti i valori delle variabili asimmetriche mediante tecniche di simmetrizzazione.
     - <strong>One-Hot Encoding delle Variabili Categoriche</strong>: tutte le variabili categoriche saranno convertite in un formato numerico utilizzando il one-hot encoding, rendendo i dati utilizzabili nei modelli dimachine learning.
     - <strong>Riscalatura mediante Standardizzazione</strong>: le variabili numeriche saranno scalate usando la standardizzazione per garantire che tutte le variabili abbiano una distribuzione con media zero edeviazione standard pari a uno.
       
  2. <mark>Pre-processing per Tutti i Record del Dataset</mark> con l'obiettivo di trasformare tutte le variabili numeriche e categoriche attraverso le seguenti fasi:
     - <strong>Pulizia dei Valori Mancanti</strong>: sarà adottata una strategia personalizzata per riempire i valori mancanti in modo coerente con la natura delle variabili.
     - <strong>Discretizzazione a 20 Bin delle Variabili Numeriche</strong>: le variabili numeriche verranno discretizzate in 20 bin per ridurre la complessità dei dati e facilitare l'analisi.
     - <strong>Encoding Ordinale delle Variabili Categoriche</strong>: la variabile categorica sarà codificata in base a un ordine crescente (A, B, C), mantenendo la semantica tra i valori.
     - <strong>Selezione delle 5 Variabili più Informative</strong>: al termine delle trasformazioni, verranno selezionate le cinque variabili più informative rispetto al target, utilizzando una metrica appropriata,migliorando così l'efficienza e la precisione dei modelli successivi.
       
  3. <mark>Pre-processing delle Variabili Numeriche</mark> overo questa pipeline si concentra esclusivamente sulle variabili numeriche, applicando tecniche avanzate di trasformazione:
     - <strong>Pulizia dei Valori Mancanti</strong>: come nella pipeline precedente, verrà scelto un metodo di pulizia adeguato alle variabili numeriche.
     - <strong>Principal Component Analysis (PCA)</strong>: verrà applicata una PCA per ridurre la dimensionalità del dataset, mantenendo l'80% della varianza spiegata, il che permetterà di ridurre il rumore emigliorare le prestazioni dei modelli.
     - <strong>Simmetrizzazione</strong>: come nella pipeline 1, anche qui si procederà con la simmetrizzazione delle variabili asimmetriche per migliorare la distribuzione.
     - <strong>Riscalatura mediante Normalizzazione</strong>: infine, le variabili numeriche saranno normalizzate tra 0 e 1, per uniformare la scala e facilitare il processo di apprendimento dei modelli.

# Repository

# Workflow

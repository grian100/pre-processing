import pandas as pd


urlset='https://proai-datasets.s3.eu-west-3.amazonaws.com/sample_dataset.csv'

#importiamo il csv all'interno di una dataframe con pandas
df = pd.read_csv(f"{urlset}", low_memory=False)


#informazioni sul nome delle features, sul nmero di righe per ciascuna features e sulla tipologia
df.info()

#numero di righe
nr=len(df)

#numero colonne
nc=len(df.columns)

#numero colonne per tipo
nct=df.dtypes.value_counts()

'''
Prima di procedere con la pulizia del dato riempiendo i valori mancanti valutiamo utilizzando sia il metodo describe(), 
sia la visualizzazione degli istogrammi con hist() e sopratutto l'analisi della Skewness, 
quali siano le variabili da considerare simmetriche e quali siano quelle asimmetriche, 
in modo da applicare l'opportuna metodologia di riempimento differenziandola per singola variabile
'''

df.iloc[:,0:14].describe()

df.iloc[:,14:-1].describe()

'''
Procediamo alla realizzazione degli istogrammi tramite la libreria matplotlib ed il metodo hist()
'''
for i in range(len(df.columns)):
    if df[df.columns[i]].dtype != object:
        df.hist(column=df.columns[i],bins=50)
        
        
#Valutiamo più in dettaglio la simmetria e asimmetria delle vriabili numeriche usando la skewness
for i in df.columns:
    if df[i].dtype != object:
        asimmetryc = df[i].skew()
        print(f"Skew della variabile {i} = {asimmetryc}")
    else:
        print(f"La variabile {i} è categorica quindi non è possibile calcolarne la skewness")

'''        
Sulla base delle indicazioni ricevute dalle funzioni describe(), dagli istogrammi hist() e sopratutto dalla skewness assumiamo che le variabili numeriche siano divise tra 
simmetriche [simmetry_features] e asimmetriche [asymmetry_features]
'''

simmetry_features=['mean radius','mean perimeter','mean area','mean smoothness','mean symmetry','worst radius','worst texture','worst perimeter','worst area',
                     'worst smoothness','worst concave points','worst symmetry']

asymmetry_features=['mean texture','mean compactness','mean concavity','mean concave points','mean fractal dimension','radius error','texture error',
                    'perimeter error','smoothness error','compactness error','concavity error','concave points error','symmetry error','fractal dimension error',
                    'worst compactness','worst concavity','worst fractal dimension']
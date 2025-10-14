#Questa pipeline verrà applicata a tutti i record del dataset, con l'obiettivo di trasformare tutte le variabili numeriche e categoriche

#consideriamo solo le variabili numeriche
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from ..import_dataset import df

numerical = df.select_dtypes(exclude=['object','category','boolean']).columns

#consideriamo anche le variabili categoriche
categorical = df.select_dtypes(include=['object','category','boolean']).columns

#stabiliamo le variabili su cui operare la Pipeline e il target
X2 = df.drop('target',axis=1)
y2 = df['target']

#Costruiamo la pipeline che ci ritorna la 'ColumnTransformer' ovvero le operazioni che saranno eseguite sulle differenti features
ct2 = ColumnTransformer(transformers=[
    (
        'numeriche',
        Pipeline([
            ('missing',SimpleImputer(strategy='mean')),
            ('bin',KBinsDiscretizer(strategy='uniform', n_bins=20, encode="ordinal"))
            ]),
        make_column_selector(dtype_exclude=['object','category','bool'])

    ),
    (
        'categoriche',
        Pipeline([
            ('missing',SimpleImputer(strategy='most_frequent')),
            ('encoder',OrdinalEncoder(categories=[['A','B','C']]))
            ]),
        make_column_selector(dtype_include=['object','category','bool'])
    )
])
ct2

#Usando il ColumnTransformer (ct2) eseguiamo il fit() ed il transform() insieme ottenendo un array
#SelectKBest consente di restringere rapidamente il set di funzionalità a un numero gestibile, il che è particolarmente importante quando si ha a che fare con set di dati di grandi dimensioni
#ha due parametri: la funzione punteggio utilizzata per valutare l'importanza delle caratteristiche e k è il valore numerico che rappresenta il numero massimo di feature da selezionare
pipeline2 = Pipeline([
    ('column_transformer',ct2),
    ('F-Score',SelectKBest(f_regression,k = 5))
])
pipeline2.fit_transform(X2,y2)
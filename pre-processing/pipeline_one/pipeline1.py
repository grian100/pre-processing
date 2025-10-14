#Questa pipeline si concentra sul pre-processing dei soli record in cui il target è pari a 1, ovvero i casi positivi di rilevazione del tumore.

from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from ..import_dataset import df, simmetry_features, asymmetry_features



#per esigenza di risultato filtriamo il dataset includendo solo i record del target uguale ad 1 ricercando quel valore nel dataframe
tr = [1]

df1=df.loc[df['target'].isin(tr)]


#consideriamo anche le variabili categoriche presenti includendole qualora siano di tipo 'object,category,boolean'
categorical = df1.select_dtypes(include=['object','category','boolean']).columns

#stabiliamo le variabili su cui operare la Pipeline e il target (se presente)
X = df1.drop('target',axis=1)
y = df1['target']

#Costruiamo la pipeline che ci ritorna la 'ColumnTransformer' ovvero le operazioni che saranno eseguite sulle differenti features
ct = ColumnTransformer(transformers=[
    (
        'numeriche simmetriche',
        Pipeline([
            ('missing',SimpleImputer(strategy='mean')),
            ('power',PowerTransformer())]),
        simmetry_features

    ),
    (
        'numeriche asimmetriche',
        Pipeline([
            ('missing',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler()),
            ('power',PowerTransformer())]),
        asymmetry_features

    ),
    (
        'categoriche',
        Pipeline([
            ('missing',SimpleImputer(strategy='most_frequent')),
            ('encoder',OneHotEncoder(sparse_output=False))]),
        categorical
    )
])
ct

#Usando il ColumnTransformer (ct) eseguiamo il fit() ed il transform() insieme ottenendo un array
pipeline = Pipeline([
    ('column_transformer',ct)
])
pipeline.fit_transform(X,y)
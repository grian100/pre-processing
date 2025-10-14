#Questa pipeline si concentra esclusivamente sulle variabili numeriche, applicando tecniche avanzate di trasformazione

#consideriamo solo le variabili numeriche
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from ..import_dataset import df, simmetry_features, asymmetry_features

numerical = df.select_dtypes(exclude=['object','category','boolean']).columns

#stabiliamo le variabili su cui operare la Pipeline e il target
X3 = df.drop('target',axis=1)
y3 = df['target']

#La PCA (Principal Component Analysis) è una tecnica statistica di decomposizione di dati multidimensionali.
#La trasformazione applicata dalla PCA riduce le dimensioni del dataset creando delle componenti che raccolgono al meglio la varianza dei dati originali.
#Questo permette di isolare le variabili più rilevanti e di ridurre la complessità del dataset.

ct3 = ColumnTransformer(transformers=[
    (
        'numeriche simmetriche',
        Pipeline([
            ('missing',SimpleImputer(strategy='mean')),
            ('PCA',PCA(n_components=0.8)),
            ('normal',MinMaxScaler())
            ]),
        simmetry_features

    ),
    (
        'numeriche asimmetriche',
        Pipeline([
            ('missing',SimpleImputer(strategy='median')),
            ('PCA',PCA(n_components=0.8)),
            ('scaler',StandardScaler()),
            ('normal',MinMaxScaler())
            ]),
        asymmetry_features

    )
])
ct3

#Usando il ColumnTransformer (ct3) eseguiamo il fit() ed il transform() insieme ottenendo un array
pipeline3 = Pipeline([
    ('column_transformer',ct3)
])
pipeline3.fit_transform(X3,y3)
import pandas as pd

df = pd.read_csv('/home/m-fayzi/Desktop/scikit_learn/Datasets/titanic.csv')
# print(df.columns)
X = df[['Fare','Embarked','Sex','Age']]
# print(X.head())

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer


ohe = OneHotEncoder()
imp = SimpleImputer()

ct = make_column_transformer(
    (ohe, ['Embarked','Sex']),
    (imp, ['Age']),
    remainder='passthrough'
)
X_ct = ct.fit_transform(X)
print(X_ct)
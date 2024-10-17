import pandas as pd

df = pd.read_csv('/home/m-fayzi/Desktop/scikit_learn/Datasets/titanic.csv')
# print(df.info())
X = df[['Fare','Embarked','Sex','Age']]
# print(X.head())

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector


ohe = OneHotEncoder()
imp = SimpleImputer()

# ct = make_column_transformer(
#     (ohe, ['Embarked','Sex']),
#     (imp, ['Age']),
#     remainder='passthrough'
# )
# X_ct = ct.fit_transform(X)
# print(X_ct)

# [[ 0.          0.          1.         ...  1.         22.
#    7.25      ]
#  [ 1.          0.          0.         ...  0.         38.
#   71.2833    ]
#  [ 0.          0.          1.         ...  0.         26.
#    7.925     ]
#  ...
#  [ 0.          0.          1.         ...  0.         29.69911765
#   23.45      ]
#  [ 1.          0.          0.         ...  1.         26.
#   30.        ]
#  [ 0.          1.          0.         ...  1.         32.
#    7.75      ]]


# ct = make_column_transformer((ohe, ['Embarked','Sex']))
# output :
# [[0. 0. 1. 0. 0. 1.]
#  [1. 0. 0. 0. 1. 0.]
#  [0. 0. 1. 0. 1. 0.]
#  ...
#  [0. 0. 1. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 1.]
#  [0. 1. 0. 0. 0. 1.]]

# ct = make_column_transformer((imp, ['Age']))

ct = make_column_transformer((ohe, make_column_selector(pattern='S')))

# ct = make_column_transformer((ohe, make_column_selector(dtype_include=object)))
# ct = make_column_transformer((ohe, make_column_selector(dtype_include='number')))
print(ct.fit_transform(X))
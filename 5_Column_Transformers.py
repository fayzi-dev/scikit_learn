import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

# df = pd.read_csv('/home/m-fayzi/Desktop/scikit_learn/Datasets/titanic.csv')
# print(df.info())
# X = df[['Fare','Embarked','Sex','Age']]
# print(X.head())



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

# ct = make_column_transformer((ohe, make_column_selector(pattern='S')))

# ct = make_column_transformer((ohe, make_column_selector(dtype_include=object)))
# ct = make_column_transformer((ohe, make_column_selector(dtype_include='number')))
# print(ct.fit_transform(X))

data = pd.DataFrame({
    'A':[1, 2, np.nan],
    'B':[11,12,13],
    'C':[14,15,16],
    'D':[17,18,19]
})

print(data)
# output :
#      A    B     C      D
# 0  1.0  100  1000  10000
# 1  2.0  200  2000  20000
# 2  NaN  300  3000  30000

ct = make_column_transformer(
    (imp, ['A']),
    ('passthrough', ['B','C']),
    remainder='drop'
)

print(ct.fit_transform(data))
# output
# [[ 1.  11.  14. ]
#  [ 2.  12.  15. ]
#  [ 1.5 13.  16. ]]

ct = make_column_transformer(
    (imp, ['A']),
    ('drop', ['D','C']),
    remainder='passthrough'
)

print(ct.fit_transform(data))
# output:
# [[ 1.  11. ]
#  [ 2.  12. ]
#  [ 1.5 13. ]]
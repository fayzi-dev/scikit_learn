import pandas as pd
import numpy as np

X = pd.DataFrame({'Age':[100, 200, 300, np.nan, 1000]})
# print(X)
#Output
#       Age
# 0   100.0
# 1   200.0
# 2   300.0
# 3     NaN
# 4  1000.0

# 1. SimpleImputer
from sklearn.impute import SimpleImputer
# imputer = SimpleImputer()
# X = imputer.fit_transform(X)
# print(X)
# Output : impute define defult method (mean) in simple imputer
# [[ 100.]
#  [ 200.]
#  [ 300.]
#  [ 400.]  ------> mean
#  [1000.]]


# Disease and Age Scenario
imputer_ind= SimpleImputer(add_indicator= True, strategy='median')
X = imputer_ind.fit_transform(X)
# print(X)
# Output
    # [[ 100.    0.]
    #  [ 200.    0.]
    #  [ 300.    0.]
    #  [ 250.    1.]
    #  [1000.    0.]]

# 2. Iterative imputer
df = pd.read_csv('Datasets/titanic.csv')
# print(df.head())
# print(df.isnull().sum())
data = df[['Age']]
# print(data)
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# iterative_imp = IterativeImputer(initial_strategy='median').fit_transform(data)
# print(iterative_imp)

#  KNN Imputer
from sklearn.impute import KNNImputer
knn_imp = KNNImputer(n_neighbors=2).fit_transform(data)
print(knn_imp)
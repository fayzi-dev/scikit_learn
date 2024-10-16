import pandas as pd
import numpy as np

X = pd.DataFrame({'Age':[100, 200, 300, np.nan, 1000]})
print(X)
#Output
#       Age
# 0   100.0
# 1   200.0
# 2   300.0
# 3     NaN
# 4  1000.0

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
X = imputer.fit_transform(X)
print(X)
# Output : impute define defult method (mean) in simple imputer
# [[ 100.]
#  [ 200.]
#  [ 300.]
#  [ 400.]  ------> mean
#  [1000.]]


# Disease and Age Scenario
imputer = SimpleImputer(add_indicator= True)
X = imputer.fit_transform(X)
print(X)
# Output
# [[ 100.    0.]
#  [ 200.    0.]
#  [ 300.    0.]
#  [ 400.    1.]  -----> indicator
#  [1000.    0.]]

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

#Feature Selector that removes all low-variance features
from sklearn.feature_selection import VarianceThreshold


#Pipline
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

df = pd.read_csv('/home/m-fayzi/Desktop/scikit_learn/Datasets/ecoli.csv', header=None)
print(df.sample(3))
#         0     1     2    3     4     5     6    7
# 94   0.31  0.47  0.48  0.5  0.29  0.28  0.39   cp
# 221  0.63  0.49  0.48  0.5  0.54  0.76  0.79  imS
# 167  0.47  0.59  0.48  0.5  0.52  0.76  0.79   im

X = df.iloc[:, :-1]
print(X)
#         0     1     2    3     4     5     6
# 0    0.49  0.29  0.48  0.5  0.56  0.24  0.35
# 1    0.07  0.40  0.48  0.5  0.54  0.35  0.44
# 2    0.56  0.40  0.48  0.5  0.49  0.37  0.46
# 3    0.59  0.49  0.48  0.5  0.52  0.45  0.36
# 4    0.23  0.32  0.48  0.5  0.55  0.25  0.35
y = df.iloc[:, -1]
print(y)
# 0      cp
# 1      cp
# 2      cp
# 3      cp
# 4      cp
#        ..
# 331    pp
# 332    pp
# 333    pp
# 334    pp
# 335    pp

encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)
# output 
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3
#  2 2 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
#  5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 7 7 7 7 7 7 7
#  7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
#  7 7 7]

#spiliting data 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=0)
print(X_train.shape)
# (225, 7)
print(X_test.shape)
# (111, 7)

knn = KNeighborsClassifier().fit(X_train, y_train)
print('Training set score:', knn.score(X_train, y_train))
# Training set score: 0.9022222222222223
print('Test set score:', knn.score(X_test, y_test))
# Test set score: 0.8468468468468469


# Setting Up a Machin Learning Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', VarianceThreshold()),
    ('classifier', KNeighborsClassifier())
])

pipe.fit(X_train, y_train)
print('Training set score:', pipe.score(X_train, y_train))
# Training set score: 0.88
print('Test set score:', pipe.score(X_test, y_test))
# Test set score: 0.8468468468468469
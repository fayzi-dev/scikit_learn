import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Data importing 
data = pd.read_csv('Datasets/iris.csv')
data.drop('variety', inplace=True, axis=1)
# print(data.head())
# Output :
#    sepal.length  sepal.width  petal.length  petal.width
# 0           5.1          3.5           1.4          0.2
# 1           4.9          3.0           1.4          0.2
# 2           4.7          3.2           1.3          0.2
# 3           4.6          3.1           1.5          0.2
# 4           5.0          3.6           1.4          0.2

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor = 1.5):
        self.factor = factor
    
    def outlier_detector(self, X,y=None):
        X = pd.Series(X).copy()
        # X.dtype(float)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self,X,y=None):
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self.outlier_detector)
        return self
    
    def transform(self, X,y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
            X.iloc[:, i] = x
            # X = X.astype('float64')
        return X
    
outlier_remover = OutlierRemover()


test_data  = pd.DataFrame({
    'col1':[100,200,300,999],
    'col2':[0,0,1,2],
    'col3':[-10,0,1,2]
}).astype('float64')
# print(test_data) 
# Output :
#    col1  col2  col3
# 0   100     0   -10
# 1   200     0     0
# 2   300     1     1
# 3   999     2     2
#fit Outlier_remover
outlier_remover.fit(test_data)

#transform Outlier_remover
# print(outlier_remover.transform(test_data))
# Output:
#     col1  col2  col3
# 0  100.0   0.0   NaN
# 1  200.0   0.0   0.0
# 2  300.0   1.0   1.0
# 3    NaN   2.0   2.0

outlier_remover_90 = OutlierRemover(factor=90)
# print(outlier_remover_90.fit_transform(test_data))

# Output:
#     col1  col2  col3
# 0  100.0   0.0 -10.0
# 1  200.0   0.0   0.0
# 2  300.0   1.0   1.0
# 3  999.0   2.0   2.0

#plots Data with Outliers
import matplotlib.pyplot as plt
data.plot(kind='box', subplots=True, figsize=(15,5), title='Data with Outliers')
# plt.show()

outlier_remover = OutlierRemover()
# ColumnTransformer to remove outliers 
ct = ColumnTransformer(transformers=[['outlier_remover', OutlierRemover(), list(range(data.shape[1]))]], remainder='passthrough')

# iris data after outlier remove
data_with_remove_outlier = pd.DataFrame(ct.fit_transform(data), columns=data.columns)

# iris data box plot after outlier remover
data_with_remove_outlier.plot(kind='box', subplots=True, figsize=(15,5), title='Data without Outliers')
# plt.show()

# print(data_with_remove_outlier.isnull().sum())
#output :
# sepal.length    0
# sepal.width     4
# petal.length    0
# petal.width     0
# dtype: int64

#finall Code
df =  pd.read_csv('Datasets/iris.csv')
X = df.drop('variety', axis=1)
y = df['variety']
# print(X.head())
# print(y.head())

pipeline = Pipeline(steps=[['outlier_remover',ct], ['imputer',SimpleImputer()], ['regressor', LogisticRegression(max_iter=1000)]])

param_grid = {
    'outlier_remover__outlier_remover__factor' : [0,1,2,3,4],
    'imputer__strategy':['mean','median','most_frequent'],
    'regressor__C':[0.01,0.1,1,10,100]
}

gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=10)

gs.fit(X,y)
print(gs.best_params_)
#OutPut:
# {'imputer__strategy': 'mean', 'outlier_remover__outlier_remover__factor': 2, 'regressor__C': 10}

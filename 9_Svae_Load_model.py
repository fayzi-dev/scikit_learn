# pip3 install joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

df = pd.read_csv('/home/m-fayzi/Desktop/scikit_learn/Datasets/titanic.csv')
# print(df.sample(3))

X = df[['Embarked', 'Sex']]
y = df['Survived']

# ohe = OneHotEncoder()
# logreg = LogisticRegression()

# pipe = make_pipeline(ohe, logreg)

# pipe.fit(X, y)
# pred = pipe.predict(X)
# print(pred)


# # Save The Pipeline to a File
# joblib.dump(pipe, 'pipe.joblib')


# Load the pipeline from a File
pipe_load = joblib.load('pipe.joblib')
# print(pipe_load)
pipe_load.fit(X, y)
pred = pipe_load.predict(X)
print(pred)
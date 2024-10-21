"""import numpy as np
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

"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


df = pd.read_csv('Datasets/titanic.csv')
X = df[['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']]
y = df['Survived']

imput = SimpleImputer(strategy='constant')
one_hot = OneHotEncoder()

pipe = make_pipeline(imput, one_hot)
vect = CountVectorizer()
simple_imp = SimpleImputer()


# Pipeline Step 1
ct = make_column_transformer(
    (imput, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (simple_imp, ['Fare', 'Age']),
    ('passthrough', ['Parch'])
)

# Pipline Step 2
selection = SelectPercentile(chi2, percentile=50)

# Pipeline Step 3
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear')


# Display estimators as diagrams
from sklearn import set_config
set_config(display='diagram')

pipe = make_pipeline(ct, selection, logreg)
print(pipe)

# Export the diagram to html file
from sklearn.utils import estimator_html_repr
with open('pipeline.html', 'w') as f:
    f.write(estimator_html_repr(pipe)) # Output :pipeline.html the base Dir


# #Nodes
# In machine learning libraries like scikit-learn, Pipeline and make_pipeline are both used to create a sequence of preprocessing and modeling steps, but they have some differences:

# Pipeline
# Manual Definition: With Pipeline, you can manually define the various steps. For example, you can specify the name of each step.
# More Control: This method gives you more control over naming and the structure of the pipeline.
# make_pipeline
# Automatic Definition: make_pipeline automatically extracts the names of the steps from the class names, so there’s no need for manual naming.
# Simpler and Faster: It's more suitable when you want to quickly create a pipeline without needing precise control over naming.
# from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression

# # Using Pipeline
# pipeline1 = Pipeline([
#     ('scaler', StandardScaler()),
#     ('logreg', LogisticRegression())
# ])

# # Using make_pipeline
# pipeline2 = make_pipeline(StandardScaler(), LogisticRegression())

# In this example, pipeline1 has specific names for each step, while pipeline2 has names generated automatically.





                    #AND


# In scikit-learn, both make_column_transformer and ColumnTransformer are used to apply different transformations to different columns of a dataset. However, they have some key differences:

# ColumnTransformer
# Manual Definition: ColumnTransformer is used to create a column transformer by explicitly defining the transformations and the columns they should be applied to.
# More Flexibility: It allows for more complex configurations and can handle a wider range of scenarios.
# make_column_transformer
# Simpler Syntax: make_column_transformer provides a more concise and user-friendly way to create a ColumnTransformer. It automatically handles the naming of the transformations based on the provided transformations.
# Quick Setup: It is useful for quickly setting up transformations without needing to specify the column names explicitly.

# from sklearn.compose import ColumnTransformer, make_column_transformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder

# # Using ColumnTransformer
# column_transformer1 = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), ['numerical_column1', 'numerical_column2']),
#         ('cat', OneHotEncoder(), ['categorical_column'])
#     ]
# )

# # Using make_column_transformer
# column_transformer2 = make_column_transformer(
#     (StandardScaler(), ['numerical_column1', 'numerical_column2']),
#     (OneHotEncoder(), ['categorical_column'])
# )

# In this example, column_transformer1 explicitly defines the transformers and their corresponding columns, while column_transformer2 offers a more concise way to achieve the same result.
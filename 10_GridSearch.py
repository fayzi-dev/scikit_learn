import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/home/m-fayzi/Desktop/scikit_learn/Datasets/titanic.csv')
X = df[['Sex', 'Name']]
y = df['Survived']
# print(y)

one_hot = OneHotEncoder()
Vector = CountVectorizer()
ct  = make_column_transformer(
    (one_hot,['Sex']),
    (Vector, 'Name')
)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

from sklearn.pipeline import make_pipeline
make_pipe = make_pipeline(ct, classifier)

#Use Cross-Validation 
from sklearn.model_selection import cross_val_score
cross_val = cross_val_score(make_pipe, X, y, cv=5, scoring='accuracy').mean()
print(cross_val)
# 0.8024543343167408


#Specify parameter  values to search
params = {}
params['columntransformer__countvectorizer__min_df'] = [1, 2]
params['logisticregression__C'] = [0.1, 1, 10]
params['logisticregression__penalty'] = ['l1', 'l2']

#Use Grid Search
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(make_pipe, params, cv=5, scoring='accuracy')
grid.fit(X, y)

#what was the best score 
print(grid.best_score_)
# 0.8147950536689473


# what the best params
print(grid.best_params_)
#{'columntransformer__countvectorizer__min_df': 1, 'logisticregression__C': 10, 'logisticregression__penalty': 'l2'}
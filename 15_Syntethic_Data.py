import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=2,
    n_classes=2,
    random_state=22
)

df = pd.DataFrame(X)
df.columns = ['A','B','C','D','E','F','G','K','L','M']
df['y'] = y
print(df.info())

print(df['y'].value_counts())
# Output:
# y
# 0    251
# 1    249
# Name: count, dtype: int64

print(df.head())
# Output:
#           A         B         C         D         E         F         G         K         L         M  y
# 0 -2.029766 -1.058566 -0.653482 -0.603740 -0.803156 -0.772179 -0.368127  0.903268  0.736373 -1.069702  0
# 1 -0.262255  0.161890 -1.079898  0.043480 -1.263853  0.885313  0.849209 -1.200795  0.229461 -0.498300  0
# 2 -1.542607 -0.373821 -1.056594  0.077986  2.102708  0.939788  0.880503  0.743371 -0.816665  0.203094  0
# 3 -0.923595 -0.227974  0.294853 -0.210315  0.659297 -0.653668 -0.509667 -1.456635  1.282462  0.662304  1
# 4 -0.422528  0.472807 -0.899827  0.125646 -0.525059  0.923307  0.832780  0.551547 -0.188897 -0.390963  0

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

classifier = RandomForestClassifier()

scores = cross_validate(
    classifier, X,y, cv=10,
    scoring=['accuracy', 'precision', 'recall', 'f1']
)

scores = pd.DataFrame(scores)
print(scores.mean().round(4))
# Output:
# fit_time          0.1481
# score_time        0.0084
# test_accuracy     0.9780
# test_precision    0.9800
# test_recall       0.9758
# test_f1           0.9778
# dtype: float64


#Create Inbalanced Dataset

X,y = make_classification(
    n_samples=300,
    n_features=5,
    n_informative=2,
    n_classes=2,
    weights=[0.90]
)

print(pd.DataFrame(y).value_counts())
# Output:
# 0
# 0    267
# 1     33    ->>>>>>************
# Name: count, dtype: int64


# Create Multiclass Datasets

X,y = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=4,
    n_classes=3
)
print(pd.DataFrame(y).value_counts())
# Output:
# 0
# 0    334
# 1    334
# 2    332
# Name: count, dtype: int64

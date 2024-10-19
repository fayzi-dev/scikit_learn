import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Datasets/iris.csv')
# print(df.info())
data = df.loc[:,['sepal.length', 'sepal.width','variety']]
# print(data.head())
#    sepal.length  sepal.width variety
# 0           5.1          3.5  Setosa
# 1           4.9          3.0  Setosa
# 2           4.7          3.2  Setosa
# 3           4.6          3.1  Setosa
# 4           5.0          3.6  Setosa

X = data.drop(columns='variety')
y = data['variety']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=45)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=22)
dt.fit(X_train, y_train)
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

print(dt.get_depth()) #output : 10

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, y_train_pred), round(accuracy_score(y_test, y_test_pred),2))
#output : 0.95 0.67


path = dt.cost_complexity_pruning_path(X_train, y_train)
alphas = path['ccp_alphas']
print(alphas)
#output:
# [0.         0.00277778 0.00277778 0.00277778 0.00324074 0.00518519
#  0.00555556 0.00694444 0.00743464 0.01006944 0.01041667 0.01161038
#  0.01230159 0.01581699 0.02010944 0.05683866 0.06089286 0.20756944]

print(len(alphas))


print(alphas.min(), alphas.max())



accuracy_train,accuracy_test = [],[]
for i in alphas:
    tree = DecisionTreeClassifier(ccp_alpha=i) 

    tree.fit(X_train, y_train)
    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)
    accuracy_train.append(accuracy_score(y_train,y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))


sns.set()
plt.figure(figsize=(14,7))
sns.lineplot(y=accuracy_train, x=alphas, label='Train Accuracy')
sns.lineplot(y=accuracy_test, x=alphas, label='Test Accuracy')
plt.xticks(ticks=np.arange(0.00, 0.25, 0.01))
plt.show()




tree = DecisionTreeClassifier(ccp_alpha=0.02, random_state=22)
tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

print(accuracy_score(y_train, y_train_pred), round(accuracy_score(y_test, y_test_pred),2))
#output :  0.8 0.8
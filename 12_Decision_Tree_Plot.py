import pandas as pd

df = pd.read_csv('Datasets/titanic.csv')

df['Sex'] = df['Sex'].map({'male':0, 'female':1})

features = ['Pclass', 'Fare', 'Sex']
X = df[features]
y = df['Survived']

classes = ['Deceased', 'Survived']

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3, random_state=22)
dt.fit(X, y)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text 

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=features, class_names=classes, filled=True)
# plt.show()


text_tree = export_text(dt, feature_names=features, show_weights=True)
print(text_tree)
# output :
# |--- Sex <= 0.50
# |   |--- Fare <= 26.27
# |   |   |--- Fare <= 7.91
# |   |   |   |--- weights: [166.00, 14.00] class: 0
# |   |   |--- Fare >  7.91
# |   |   |   |--- weights: [195.00, 40.00] class: 0
# |   |--- Fare >  26.27
# |   |   |--- Fare <= 26.47
# |   |   |   |--- weights: [0.00, 4.00] class: 1
# |   |   |--- Fare >  26.47
# |   |   |   |--- weights: [107.00, 51.00] class: 0
# |--- Sex >  0.50
# |   |--- Pclass <= 2.50
# |   |   |--- Fare <= 28.86
# |   |   |   |--- weights: [7.00, 63.00] class: 1
# |   |   |--- Fare >  28.86
# |   |   |   |--- weights: [2.00, 98.00] class: 1
# |   |--- Pclass >  2.50
# |   |   |--- Fare <= 23.35
# |   |   |   |--- weights: [48.00, 69.00] class: 1
# |   |   |--- Fare >  23.35
# |   |   |   |--- weights: [24.00, 3.00] class: 0



#Pruning
data = pd.read_csv('Datasets/titanic.csv')
data['Sex']= data['Sex'].map({'male':0, 'female':1})


feature = ['Pclass', 'Fare', 'Sex', 'Parch']
X_n = df[feature]
y_n = df['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

dt_n = DecisionTreeClassifier(random_state=22)
print(dt_n.fit(X_n, y_n).tree_.node_count) #output: 329 default tree has nodes

print(round(cross_val_score(dt_n, X_n, y_n, cv=10, scoring='accuracy').mean(),2)) #output : 0.81

dt_s = DecisionTreeClassifier(ccp_alpha=0.002, random_state=22)
print(dt_s.fit(X_n, y_n).tree_.node_count) #output : 37 pruned tree has 37 node

print(round(cross_val_score(dt_s, X_n, y_n, cv=10, scoring='accuracy').mean(),2)) #output: 0.82 pruning improved the cross-validated accuracy

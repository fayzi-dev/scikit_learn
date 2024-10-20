import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('Datasets/iris.csv')
data = df.loc[:, ['sepal.length', 'sepal.width', 'variety']]

X = data.drop(columns='variety')
y = data['variety']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Train the initial Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=22)
dt.fit(X_train, y_train)

# Get pruning path
path = dt.cost_complexity_pruning_path(X_train, y_train)
alphas = path['ccp_alphas']

# Initialize accuracy lists
accuracy_train, accuracy_test = [], []

# Loop through alphas to train and evaluate models
for i in alphas:
    tree = DecisionTreeClassifier(ccp_alpha=i, random_state=22)
    tree.fit(X_train, y_train)

    # Use the current tree model for predictions
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))

# Plotting
sns.set()
plt.figure(figsize=(14, 7))
sns.lineplot(y=accuracy_train, x=alphas, label='Train Accuracy')
sns.lineplot(y=accuracy_test, x=alphas, label='Test Accuracy')
plt.xticks(ticks=np.arange(0.00, 0.25, 0.01))
plt.xlabel('CCP Alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. CCP Alpha')
plt.legend()
plt.show()

# Final model fitting
tree = DecisionTreeClassifier(ccp_alpha=0.02, random_state=22)
tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

print(accuracy_score(y_train, y_train_pred), round(accuracy_score(y_test, y_test_pred), 2))

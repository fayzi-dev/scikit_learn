import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv('/home/m-fayzi/Desktop/scikit_learn/Datasets/titanic.csv')

X = df[['Pclass', 'Fare']]
y = df['Survived']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=22)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues')
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=[0, 1]).plot()
plt.show()
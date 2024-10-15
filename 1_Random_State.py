# Random state
import pandas as pd
import sklearn 

# print(sklearn.__version__)


df = pd.read_csv('/home/m-fayzi/Desktop/scikit_learn/Datasets/titanic.csv')
# print(df.head())
X = df[['Fare','Embarked','Sex']]
y = df['Survived']
# print(X, y)

from sklearn.model_selection import train_test_split

#any positive integer can be used for the random_state value
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.3, random_state=22)
print(X_train.sample(3))

#      Fare Embarked   Sex
# 884  7.050        S  male
# 244  7.225        C  male
# 459  7.750        Q  male
import pandas as pd

df = pd.DataFrame({'feture':list(range(12)), 'target':['No']*10 + ['Yes']*2})
# print(df)
X = df[['feture']]
y = df['target']


# print(X)

from sklearn.model_selection import train_test_split

#Not Startify
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=0)
print(y_train)
#output
# 1    No
# 7    No
# 9    No
# 3    No
# 0    No
# 5    No
print(y_test)
#output
# 6      No
# 11    Yes
# 4      No
# 10    Yes
# 2      No
# 8      No


# Startify
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=0, stratify=y)
print(y_train)
# Output
# 9      No
# 8      No
# 4      No
# 1      No
# 11    Yes
# 2      No
print(y_test)
# Output
# 10    Yes
# 3      No
# 0      No
# 5      No
# 7      No
# 6      No
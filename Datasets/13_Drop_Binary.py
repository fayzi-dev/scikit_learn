import pandas as pd

data = pd.DataFrame({
    'Shape':['circle', 'oval', 'square', 'square'],
    'Color':['pink', 'yellow', 'pink', 'yellow']
})
# print(data)
# output
#     Shape   Color
# 0  circle    pink
# 1    oval  yellow
# 2  square    pink
# 3  square  yellow

from sklearn.preprocessing import OneHotEncoder
# drop = None (default) creates one feature column per category
one_hot = OneHotEncoder(sparse_output=False, drop=None)
print(one_hot.fit_transform(data))
#output :
# [[1. 0. 0. 1. 0.]
#  [0. 1. 0. 0. 1.]
#  [0. 0. 1. 1. 0.]
#  [0. 0. 1. 0. 1.]]

# drop = 'first'  drops the forst category in each feature
one_hot = OneHotEncoder(sparse_output=False, drop='first')
print(one_hot.fit_transform(data))
# output:
# [[0. 0. 0.]
#  [1. 0. 1.]
#  [0. 1. 0.]
#  [0. 1. 1.]]

# drop = 'if_binary' drops the first category of binary features
one_hot = OneHotEncoder(sparse_output=False, drop='if_binary')
print(one_hot.fit_transform(data))
# output:
# [[1. 0. 0. 0.]
#  [0. 1. 0. 1.]
#  [0. 0. 1. 0.]
#  [0. 0. 1. 1.]]
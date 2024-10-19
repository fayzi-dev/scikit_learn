import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

data = pd.DataFrame({
    'Name': ['Ali','Ali','Mohammad','Amin'],
    'Number':['First','Third','Second','Third'],
    'Size':['L','S','XL','S']
})
# print(data)

Encoder_ONEHot = OneHotEncoder().fit_transform(data[['Name']])
print(Encoder_ONEHot)

Encoder_Ordinal = OrdinalEncoder(categories=[['First','Second','Third'], ['S','M','L', 'XL']]).fit_transform(data[['Number','Size']])
print(Encoder_Ordinal)
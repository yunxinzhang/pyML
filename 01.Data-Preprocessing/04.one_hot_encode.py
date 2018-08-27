import numpy as np 
import pandas as pd 


data = pd.read_csv("Data.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import Imputer  # Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# LabelEncoder, OneHotEncoder  ! Label
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])
oe = OneHotEncoder(categorical_features=[0])
X = oe.fit_transform(X).toarray()
print(X)
le_y = LabelEncoder()
print(y)
y = le_y.fit_transform(y)
print(y)
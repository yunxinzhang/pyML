import numpy as np 
import pandas as pd 

data = pd.read_csv("Data.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import Imputer  # Imputer

#missing data
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
print(X[:,1:3])
imputer.fit(X[:,1:3])
print(imputer.transform(X[:,1:3]))


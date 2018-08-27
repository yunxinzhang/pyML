import numpy as np 
import pandas as pd 


data = pd.read_csv("Data.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import Imputer  # Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# train_test_split
from sklearn.model_selection import train_test_split # train_test_split
xtr,xt, ytr, yt = train_test_split(X, y, test_size=0.4)
print(xtr)
print(xt)
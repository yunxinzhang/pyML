# import moduler
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# read data
data = pd.read_csv("Data.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# train_test_split
from sklearn.model_selection import train_test_split # train_test_split
xtr,xt, ytr, yt = train_test_split(X, y, test_size=0.3)

#feature scaling
#optional
'''
from sklearn.preprocessing import StandardScaler  # StandardScaler
sc = StandardScaler()
xtr[:,1:] = sc.fit_transform(xtr[:,1:])
xt[:,1:] = sc.transform(xt[:,1:])
'''
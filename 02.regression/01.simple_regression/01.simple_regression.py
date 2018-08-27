# import moduler
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# read data
data = pd.read_csv("Salary_Data.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# train_test_split
from sklearn.model_selection import train_test_split # train_test_split
xtr,xt, ytr, yt = train_test_split(X, y, test_size=0.3)
plt.scatter(X,y)

from sklearn.linear_model import LinearRegression

slc = LinearRegression()
slc.fit(xtr, ytr)
yp = slc.predict(xt)
print(yp)
plt.plot(X, slc.predict(X))
print(slc.coef_, slc.intercept_)
plt.show()

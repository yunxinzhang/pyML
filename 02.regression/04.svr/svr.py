# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 16:59:37 2018

@author: zyx
"""


import pandas as pd
import numpy as np

data = pd.read_csv("Position_Salaries.csv")
x = data.iloc[:,1].values.reshape(-1,1)
y = data.iloc[:,-1].values

from sklearn.linear_model import LinearRegression


lr = LinearRegression()
lr.fit(x,y)
yp = lr.predict(x)

import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.plot(x,yp)



from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2)
ploy_x = pf.fit_transform(x)
lr2 = LinearRegression()
lr2.fit(ploy_x, y)
temp = np.linspace(min(x),max(x),100).reshape(-1,1)
plt.plot(temp, lr2.predict(pf.transform(temp)))

# =============================================================================
# # svr
# =============================================================================
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
# both x and y need feature scaling
sc = StandardScaler()
sc_y = StandardScaler()
x_st = sc.fit_transform(x)
y_st = sc_y.fit_transform(y.reshape(-1,1))
svr = SVR(kernel='rbf')
svr.fit(x_st,y_st)
temp = np.linspace(min(x),max(x),100).reshape(-1,1)
yp_svr = svr.predict(sc.transform(temp))
plt.plot(temp, sc_y.inverse_transform(yp_svr))
plt.show()
